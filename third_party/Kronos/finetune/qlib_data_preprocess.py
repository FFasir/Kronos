import os
import json
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange

from config import Config
from dataset import slice_frame_with_context


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = [
            'open', 'close', 'high', 'low', 'volume', 'vwap', 'limit_up', 'limit_down'
        ]
        self.data = {}  # A dictionary to store processed data for each symbol.
        self.source_manifest = None

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        manifest_path = os.path.join(self.config.qlib_data_path, 'manifest.json')
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Qlib source manifest is missing: {manifest_path}")
        with open(manifest_path, encoding='utf-8') as handle:
            self.source_manifest = json.load(handle)
        if self.source_manifest.get('adjustment') not in ('hfq', 'qfq'):
            raise ValueError("A-share fine-tuning requires adjustment=hfq or qfq in the Qlib manifest")
        if not str(self.source_manifest.get('first_listing_day_policy', '')).lower().startswith('drop'):
            raise ValueError("A-share Qlib data must drop the first listing day")
        print("Initializing Qlib...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def load_qlib_data(self):
        """
        Loads raw data from Qlib, processes it symbol by symbol, and stores
        it in the `self.data` attribute.
        """
        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        cal: np.ndarray = D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # Check if start_index lookbackw_window will cause negative index
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = cal[adjusted_start_index]

        # Check if end_index exceeds the range of the array
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # Check if end_index+predictw_window will exceed the range of the array
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = cal[adjusted_end_index]

        # Load data using Qlib's data loader.
        data_df = QlibDataLoader(config=data_fields_qlib).load(
            self.config.instrument, real_start_time, real_end_time
        )
        data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.

        symbol_list = list(data_df.columns)
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            symbol_df = data_df[symbol]

            # Pivot the table to have features as columns and datetime as index.
            symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
            symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
            symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

            # Calculate amount and select final features.
            symbol_df['vol'] = symbol_df['volume']
            symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            symbol_df = symbol_df[
                self.config.feature_list + ['limit_up', 'limit_down']
            ]

            # Filter out symbols with insufficient data.
            symbol_df = symbol_df.dropna()
            if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                continue

            self.data[symbol] = symbol_df

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        split_ranges = {
            'train': self.config.train_time_range,
            'val': self.config.val_time_range,
            'test': self.config.test_time_range,
        }
        split_data = {name: {} for name in split_ranges}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Keep lookback rows before each target period. QlibDataset filters
            # windows by target dates, so the buffer is context only and cannot
            # leak earlier targets into validation or test samples.
            for split_name, target_range in split_ranges.items():
                split_frame = slice_frame_with_context(
                    symbol_df,
                    target_range,
                    self.config.lookback_window,
                    self.config.predict_window,
                )
                if split_frame is not None:
                    split_data[split_name][symbol] = split_frame

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        for split_name, values in split_data.items():
            with open(f"{self.config.dataset_path}/{split_name}_data.pkl", 'wb') as f:
                pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            'adjustment': self.source_manifest.get('adjustment'),
            'limit_features': ['limit_up', 'limit_down'],
            'first_listing_day_policy': self.source_manifest.get('first_listing_day_policy'),
            'split_context_policy': 'lookback buffer; forecast targets strictly inside split_target_ranges',
            'split_target_ranges': split_ranges,
            'lookback': self.config.lookback_window,
            'horizon': self.config.predict_window,
            'source_manifest': os.path.join(self.config.qlib_data_path, 'manifest.json'),
            'source_archive': self.source_manifest.get('archive'),
            'symbol_count': len(self.data),
            'split_symbol_counts': {
                name: len(values) for name, values in split_data.items()
            },
        }
        with open(f"{self.config.dataset_path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()

