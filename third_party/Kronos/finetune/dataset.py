import pickle
import random
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import Config

PROJECT_ROOT = Path(
    os.getenv("KRONOS_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_layer.a_share_rules import pressure_tokenizer_edges


def sample_without_replacement(total: int, count: int, seed: int) -> list[int]:
    """Return a deterministic epoch sample without duplicated windows."""
    if total < 0 or count < 0 or count > total:
        raise ValueError("sample count must be between zero and total")
    return random.Random(seed).sample(range(total), count)


def balanced_direction_sample(
    up_indices: list[int],
    down_indices: list[int],
    count: int,
    seed: int,
) -> list[int]:
    """Sample approximately equal up/down windows without replacement."""
    if count < 0 or count > len(up_indices) + len(down_indices):
        raise ValueError("sample count must be between zero and the pool size")
    rng = random.Random(seed)
    up_count = min(count // 2, len(up_indices))
    down_count = min(count - up_count, len(down_indices))
    remaining = count - up_count - down_count
    if remaining:
        add_up = min(remaining, len(up_indices) - up_count)
        up_count += add_up
        down_count += remaining - add_up
    sampled = rng.sample(up_indices, up_count) + rng.sample(down_indices, down_count)
    rng.shuffle(sampled)
    return sampled


def load_split_target_range(dataset_path: str, data_type: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Load the dates that are allowed to appear as forecast targets."""
    metadata_path = Path(dataset_path) / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    values = metadata.get("split_target_ranges", {}).get(data_type)
    if not values or len(values) != 2:
        return None
    start, end = pd.Timestamp(values[0]), pd.Timestamp(values[1])
    if start > end:
        raise ValueError(f"invalid {data_type} target range: {values}")
    return start, end


def slice_frame_with_context(
    frame: pd.DataFrame,
    target_range: tuple[str, str] | list[str],
    lookback: int,
    horizon: int,
) -> pd.DataFrame | None:
    """Keep target rows plus preceding context without extending target end."""
    target_start, target_end = map(pd.Timestamp, target_range)
    target_mask = (frame.index >= target_start) & (frame.index <= target_end)
    target_positions = np.flatnonzero(target_mask)
    if len(target_positions) < horizon:
        return None
    context_start = max(0, int(target_positions[0]) - lookback)
    result = frame.iloc[context_start:int(target_positions[-1]) + 1]
    if len(result) < lookback + horizon:
        return None
    return result


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        # A training sequence contains the exact context and forecast horizon.
        # After the one-token shift, forecast_start=lookback-1 therefore covers
        # exactly predict_window targets, matching autoregressive inference.
        self.window = self.config.lookback_window + self.config.predict_window
        self.target_range = load_split_target_range(self.config.dataset_path, data_type)

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        self.limit_columns = ['limit_up', 'limit_down']

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        self.up_indices = []
        self.down_indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                if self.config.tokenizer_edge_pressure:
                    missing = [column for column in self.limit_columns if column not in df]
                    if missing:
                        raise ValueError(
                            f"Tokenizer edge pressure requires A-share limit flags: {missing}"
                        )
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                retained_limits = [column for column in self.limit_columns if column in df]
                self.data[symbol] = df[
                    self.feature_list + self.time_feature_list + retained_limits
                ]

                first_start, stop_start = 0, num_samples
                if self.target_range is not None:
                    datetimes = pd.DatetimeIndex(df['datetime'])
                    first_target_pos = int(
                        datetimes.searchsorted(self.target_range[0], side='left')
                    )
                    last_target_pos = int(
                        datetimes.searchsorted(self.target_range[1], side='right') - 1
                    )
                    first_start = max(
                        0, first_target_pos - self.config.lookback_window
                    )
                    stop_start = min(
                        num_samples,
                        last_target_pos
                        - self.config.lookback_window
                        - self.config.predict_window
                        + 2,
                    )
                close_values = df['close'].to_numpy(dtype=float, copy=False)

                # Add all valid starting indices for this symbol to the global list.
                for i in range(first_start, max(first_start, stop_start)):
                    last_target_pos = (
                        i + self.config.lookback_window + self.config.predict_window - 1
                    )
                    global_index = len(self.indices)
                    self.indices.append((symbol, i))
                    origin_close = close_values[i + self.config.lookback_window - 1]
                    target_close = close_values[last_target_pos]
                    if target_close > origin_close:
                        self.up_indices.append(global_index)
                    else:
                        self.down_indices.append(global_index)

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        self.sampled_indices = []
        self.set_epoch_seed(0)
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        if self.data_type == 'train' and self.config.balance_train_directions:
            self.sampled_indices = balanced_direction_sample(
                self.up_indices, self.down_indices, self.n_samples, epoch_seed
            )
        else:
            self.sampled_indices = sample_without_replacement(
                len(self.indices), self.n_samples, epoch_seed
            )

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # ``idx`` is supplied by DistributedSampler.  Mapping it through a
        # precomputed epoch sample keeps workers/ranks deterministic and avoids
        # the duplicate streams produced when each worker owns the same RNG.
        sample_idx = self.sampled_indices[idx]
        symbol, start_idx = self.indices[sample_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        if self.config.tokenizer_edge_pressure:
            x = pressure_tokenizer_edges(
                x,
                self.feature_list,
                win_df['limit_up'].to_numpy() > 0.5,
                win_df['limit_down'].to_numpy() > 0.5,
            )
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
