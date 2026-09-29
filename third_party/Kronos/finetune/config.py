import os
from pathlib import Path

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # This source bundle is extracted into <project>/third_party/Kronos.
        # Environment variables make the same config portable across servers.
        self.project_root = Path(
            os.getenv("KRONOS_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
        ).expanduser()
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = os.getenv(
            "KRONOS_QLIB_DATA_PATH", str(self.project_root / "data" / "processed" / "qlib" / "cn_data")
        )
        self.instrument = os.getenv("KRONOS_INSTRUMENT", "csi1000")

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2010-01-01"
        self.dataset_end_time = "2025-09-30"

        # Sliding window parameters for creating samples.
        self.lookback_window = int(os.getenv("KRONOS_LOOKBACK", "250"))
        self.predict_window = int(os.getenv("KRONOS_PRED_LEN", "5"))
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Paper: CSI 1000 constituents, train 2010-2022 and validate 2023-2024.
        self.train_time_range = ["2010-01-01", "2022-12-31"]
        self.val_time_range = ["2023-01-01", "2024-12-31"]
        self.test_time_range = ["2025-01-01", "2025-09-30"]
        self.backtest_time_range = ["2025-01-01", "2025-09-30"]

        self.dataset_path = os.getenv(
            "KRONOS_DATASET_PATH", str(self.project_root / "data" / "official_processed_datasets")
        )

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        # Paper table 16.  The RTX 4090 has the same 24 GB VRAM as the report's RTX 3090.
        self.epochs = int(os.getenv("KRONOS_EPOCHS", "10"))
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = int(os.getenv("KRONOS_BATCH_SIZE", "30"))

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = int(os.getenv("KRONOS_TRAIN_SAMPLES", "60000"))
        # The report does not state n_val_iter; 6,000 gives a stable 10% validation sample.
        self.n_val_iter = int(os.getenv("KRONOS_VAL_SAMPLES", "6000"))
        self.num_workers = int(os.getenv("KRONOS_NUM_WORKERS", "2"))

        # Stop after validation has not improved for this many epochs.  The
        # best checkpoint is always retained; set to 0 to disable early stop.
        self.early_stopping_patience = int(os.getenv("KRONOS_EARLY_STOPPING_PATIENCE", "3"))
        self.early_stopping_min_delta = float(os.getenv("KRONOS_EARLY_STOPPING_MIN_DELTA", "0"))
        self.resume_training = os.getenv("KRONOS_RESUME", "0").lower() in {"1", "true", "yes"}
        self.forecast_loss_weight = float(os.getenv("KRONOS_FORECAST_LOSS_WEIGHT", "1.0"))
        self.balance_train_directions = os.getenv(
            "KRONOS_BALANCE_TRAIN_DIRECTIONS", "0"
        ).lower() in {"1", "true", "yes"}
        self.tokenizer_edge_pressure = os.getenv(
            "KRONOS_TOKENIZER_EDGE_PRESSURE", "0"
        ).lower() in {"1", "true", "yes"}

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 1e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = int(os.getenv("KRONOS_SEED", "100"))
        self.experiment_id = os.getenv("KRONOS_EXPERIMENT_ID", "kronos_finetune")

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'csi1000_daily'
        self.comet_name = 'csi1000_daily_paper_replication'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = os.getenv(
            "KRONOS_SAVE_PATH", str(self.project_root / "models" / "official_finetune")
        )
        self.tokenizer_save_folder_name = os.getenv('KRONOS_TOKENIZER_SAVE_NAME', 'csi1000_tokenizer')
        self.predictor_save_folder_name = os.getenv('KRONOS_PREDICTOR_SAVE_NAME', 'csi1000_predictor')
        self.backtest_save_folder_name = os.getenv('KRONOS_BACKTEST_SAVE_NAME', 'csi1000_backtest')

        # Path for backtesting results.
        self.backtest_result_path = str(self.project_root / "data" / "outputs" / "official_backtest")

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        self.pretrained_tokenizer_path = os.getenv(
            "KRONOS_TOKENIZER_PATH", str(self.project_root / "models" / "kronos_tokenizer")
        )
        self.pretrained_predictor_path = os.getenv(
            "KRONOS_MODEL_PATH", str(self.project_root / "models" / "kronos_base")
        )

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = os.getenv(
            "KRONOS_FINETUNED_TOKENIZER_PATH",
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model",
        )
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 10
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
            # Training universe only; CSI 1000 remains the evaluation reference.
            'all_a_share': "SH000852",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
