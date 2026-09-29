#!/usr/bin/env python3
"""Run sequential 3-month/1-month Kronos predictor fine-tuning windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monthly rolling Kronos fine-tuning")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--source", default="data/official_processed_datasets_all_a_share")
    parser.add_argument("--dataset-root", default="data/rolling_kronos")
    parser.add_argument("--model-root", default="models/rolling_kronos")
    parser.add_argument("--base-tokenizer", default="models/kronos_tokenizer")
    parser.add_argument("--base-predictor", default="models/kronos_base")
    parser.add_argument("--lookback", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--validation-days", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--val-samples", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecast-loss-weight", type=float, default=4.0)
    parser.add_argument(
        "--balance-train-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--version-prefix", default="rolling-v2")
    parser.add_argument("--calibration-windows", type=int, default=1024)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--force-datasets", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--no-database", action="store_true")
    parser.add_argument("--activate-model", action="store_true")
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--prediction-samples", type=int, default=5)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def month_range(start: str, end: str) -> list[str]:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    if end_period < start_period:
        raise ValueError("end month must not precede start month")
    return [str(period) for period in pd.period_range(start_period, end_period, freq="M")]


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")


def storage_layer(enabled: bool):
    if not enabled:
        return None
    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(load_config())
    storage.init_database()
    return storage


def start_training_log(storage, month: str) -> int | None:
    if storage is None:
        return None
    storage.execute_write(
        """INSERT INTO training_log (window_id, model_type, start_time, status)
           VALUES (?, 'kronos_rolling', ?, 'running')""",
        (month, datetime.now(timezone.utc).isoformat()),
    )
    row = storage.execute_sql(
        """SELECT id FROM training_log
           WHERE window_id=? AND model_type='kronos_rolling' AND status='running'
           ORDER BY id DESC LIMIT 1""",
        (month,),
    )
    return int(row[0]["id"])


def finish_training_log(storage, log_id: int | None, summary: dict, checkpoint: Path) -> None:
    if storage is None or log_id is None:
        return
    result = summary["final_result"]
    storage.execute_write(
        """UPDATE training_log SET end_time=?, status='completed', epochs_completed=?,
              best_val_loss=?, checkpoint_path=? WHERE id=?""",
        (
            datetime.now(timezone.utc).isoformat(),
            result.get("epochs_completed"),
            result.get("best_val_loss"),
            str(checkpoint),
            log_id,
        ),
    )


def fail_training_log(storage, log_id: int | None, error: Exception) -> None:
    if storage is None or log_id is None:
        return
    storage.execute_write(
        "UPDATE training_log SET end_time=?, status='failed', error_message=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), str(error), log_id),
    )


def register_model(
    storage, month: str, checkpoint: Path, active: bool, version_prefix: str
) -> None:
    if storage is None:
        return
    version = f"{version_prefix}-{month}"
    if active:
        storage.execute_write("UPDATE model_version SET is_active=0")
    storage.execute_write(
        """INSERT INTO model_version (version, model_type, model_path, is_active, description)
           VALUES (?, 'kronos_rolling', ?, ?, ?)
           ON CONFLICT(version) DO UPDATE SET model_path=excluded.model_path,
             is_active=excluded.is_active, description=excluded.description""",
        (
            version,
            str(checkpoint),
            int(active),
            "65-day context, 20-day horizon, forecast-weighted monthly rolling fine-tune",
        ),
    )


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    dataset_root = resolve_path(args.dataset_root)
    model_root = resolve_path(args.model_root)
    tokenizer_path = resolve_path(args.base_tokenizer)
    predictor_path = resolve_path(args.base_predictor)
    storage = storage_layer(not args.no_database)

    run_manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "windows": [],
    }
    manifest_path = model_root / "rolling_manifest.json"
    model_root.mkdir(parents=True, exist_ok=True)

    try:
        for month in month_range(args.start_month, args.end_month):
            model_version = f"{args.version_prefix}-{month}"
            experiment_id = model_version.upper()
            dataset_path = dataset_root / month
            model_path = model_root / month
            checkpoint = model_path / "predictor/checkpoints/best_model"
            summary_path = model_path / "predictor/summary.json"
            calibration_path = model_path / "probability_calibration.json"
            complete_marker = model_path / "window_complete.json"
            prediction_summary = (
                PROJECT_ROOT / "data/outputs/monthly_predictions"
                / f"{model_version}_summary.json"
            )
            window_record = {"target_month": month, "status": "running"}
            run_manifest["windows"].append(window_record)
            manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            legacy_complete = (
                checkpoint.joinpath("model.safetensors").is_file()
                and summary_path.is_file()
                and (args.no_predict or prediction_summary.is_file())
            )
            if (complete_marker.is_file() or legacy_complete) and not args.rerun_completed:
                window_record["status"] = "reused"
                window_record["checkpoint"] = str(checkpoint)
                predictor_path = checkpoint
                continue

            training_reused = (
                checkpoint.joinpath("model.safetensors").is_file()
                and summary_path.is_file()
                and not args.rerun_completed
            )

            metadata_path = dataset_path / "metadata.json"
            if not metadata_path.is_file() or args.force_datasets:
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/build_monthly_kronos_dataset.py"),
                    "--source", str(source),
                    "--target-month", month,
                    "--output", str(dataset_path),
                    "--validation-days", str(args.validation_days),
                    "--lookback", str(args.lookback),
                    "--horizon", str(args.horizon),
                ]
                if args.force_datasets:
                    command.append("--force")
                run(command, PROJECT_ROOT / "logs" / f"rolling-{month}-dataset.log")

            log_id = None if training_reused else start_training_log(storage, month)
            try:
                if not training_reused:
                    run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts/run_official_finetune.py"),
                            "--experiment-id", experiment_id,
                            "--dataset-path", str(dataset_path),
                            "--save-path", str(model_path),
                            "--pretrained-tokenizer-path", str(tokenizer_path),
                            "--pretrained-predictor-path", str(predictor_path),
                            "--lookback", str(args.lookback),
                            "--horizon", str(args.horizon),
                            "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size),
                            "--train-samples", str(args.train_samples),
                            "--val-samples", str(args.val_samples),
                            "--num-workers", str(args.num_workers),
                            "--early-stopping-patience", str(args.early_stopping_patience),
                            "--forecast-loss-weight", str(args.forecast_loss_weight),
                            "--seed", str(args.seed),
                            "--skip-tokenizer",
                            "--resume",
                        ] + (["--balance-train-directions"] if args.balance_train_directions else []),
                        PROJECT_ROOT / "logs" / f"{model_version}-training.log",
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not training_reused:
                    finish_training_log(storage, log_id, summary, checkpoint)
                if not args.skip_calibration and (
                    not calibration_path.is_file() or args.rerun_completed
                ):
                    run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts/calibrate_monthly_probabilities.py"),
                            "--dataset-path", str(dataset_path),
                            "--tokenizer-path", str(tokenizer_path),
                            "--predictor-path", str(checkpoint),
                            "--output", str(calibration_path),
                            "--lookback", str(args.lookback),
                            "--horizon", str(args.horizon),
                            "--samples", str(args.prediction_samples),
                            "--batch-size", str(args.prediction_batch_size),
                            "--max-windows", str(args.calibration_windows),
                            "--device", args.device,
                            "--seed", str(args.seed),
                        ],
                        PROJECT_ROOT / "logs" / f"{model_version}-calibration.log",
                    )
                register_model(
                    storage, month, checkpoint, args.activate_model, args.version_prefix
                )
                if not args.no_predict:
                    as_of_date = (
                        pd.Period(month, freq="M").start_time - pd.Timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    prediction_command = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/predict_monthly_kronos.py"),
                        "--tokenizer-path", str(tokenizer_path),
                        "--predictor-path", str(checkpoint),
                        "--model-version", model_version,
                        "--as-of-date", as_of_date,
                        "--lookback", str(args.lookback),
                        "--horizon", str(args.horizon),
                        "--samples", str(args.prediction_samples),
                        "--batch-size", str(args.prediction_batch_size),
                        "--device", args.device,
                        "--seed", str(args.seed),
                    ]
                    if calibration_path.is_file() and not args.skip_calibration:
                        prediction_command.extend(["--calibration", str(calibration_path)])
                    if args.no_database:
                        prediction_command.append("--no-database")
                    run(
                        prediction_command,
                        PROJECT_ROOT / "logs" / f"{model_version}-prediction.log",
                    )
                training_state = model_path / "predictor/checkpoints/training_state.pt"
                training_state.unlink(missing_ok=True)
                complete_marker.write_text(json.dumps({
                    "target_month": month,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": str(checkpoint),
                    "model_version": model_version,
                    "calibration": str(calibration_path) if calibration_path.is_file() else None,
                    "prediction_completed": not args.no_predict,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                fail_training_log(storage, log_id, exc)
                raise

            predictor_path = checkpoint
            window_record.update({
                "status": "completed",
                "dataset_metadata": str(metadata_path),
                "checkpoint": str(checkpoint),
                "summary": str(summary_path),
                "best_val_loss": summary["final_result"]["best_val_loss"],
                "best_epoch": summary["final_result"]["best_epoch"],
                "calibration": str(calibration_path) if calibration_path.is_file() else None,
            })
            manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = str(exc)
        raise
    else:
        run_manifest["status"] = "completed"
    finally:
        run_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rolling fine-tuning completed: {manifest_path}")


if __name__ == "__main__":
    main()
