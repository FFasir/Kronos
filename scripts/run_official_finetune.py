#!/usr/bin/env python3
"""Run the official two-stage Kronos fine-tuning pipeline reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Kronos tokenizer and predictor fine-tuning")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--tokenizer-name", default="tokenizer")
    parser.add_argument("--predictor-name", default="predictor")
    parser.add_argument("--official-repo", default="third_party/Kronos")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--val-samples", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--forecast-loss-weight", type=float, default=1.0)
    parser.add_argument("--balance-train-directions", action="store_true")
    parser.add_argument("--pretrained-tokenizer-path", default="models/kronos_tokenizer")
    parser.add_argument("--pretrained-predictor-path", default="models/kronos_base")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_metadata(
    dataset_metadata: dict,
    lookback: int,
    horizon: int,
) -> None:
    """Reject stale or leakage-prone datasets before allocating a GPU."""
    if dataset_metadata.get("adjustment") not in ("hfq", "qfq"):
        raise ValueError("A-share fine-tuning requires an HFQ or QFQ dataset")
    if dataset_metadata.get("limit_features") != ["limit_up", "limit_down"]:
        raise ValueError("A-share fine-tuning dataset is missing directional limit flags")
    if not str(dataset_metadata.get("first_listing_day_policy", "")).lower().startswith("drop"):
        raise ValueError("A-share fine-tuning dataset must drop the first listing day")

    expected_full_ranges = {
        "train": ["2010-01-01", "2022-12-31"],
        "val": ["2023-01-01", "2024-12-31"],
    }
    split_ranges = dataset_metadata.get("split_target_ranges")
    if not isinstance(split_ranges, dict) or not all(
        split in split_ranges and len(split_ranges[split]) == 2
        for split in ("train", "val")
    ):
        raise ValueError("dataset metadata is missing train/val target ranges")
    if (
        "target_month" not in dataset_metadata
        and any(split_ranges.get(split) != values for split, values in expected_full_ranges.items())
    ):
        raise ValueError(
            "dataset split targets must be train=2010-2022 and val=2023-2024 "
            "with separate lookback context"
        )
    if "lookback buffer" not in dataset_metadata.get("split_context_policy", ""):
        raise ValueError("dataset metadata does not guarantee leakage-safe lookback context")
    if dataset_metadata.get("lookback") != lookback:
        raise ValueError("dataset lookback does not match the training configuration")
    if dataset_metadata.get("horizon") != horizon:
        raise ValueError("dataset horizon does not match the training configuration")


def run_stage(name: str, module: str, repo: Path, env: dict[str, str], log_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        module,
    ]
    print(f"Starting {name}: {' '.join(command)}", flush=True)
    log_mode = "a" if env.get("KRONOS_RESUME") == "1" else "w"
    with log_path.open(log_mode, encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}; see {log_path}")


def main() -> None:
    args = parse_args()
    repo = resolve_path(args.official_repo)
    dataset_path = resolve_path(args.dataset_path)
    save_path = resolve_path(args.save_path)
    pretrained_tokenizer_path = resolve_path(args.pretrained_tokenizer_path)
    pretrained_predictor_path = resolve_path(args.pretrained_predictor_path)
    logs_path = PROJECT_ROOT / "logs"
    result_path = PROJECT_ROOT / "data" / "outputs" / "training" / args.experiment_id
    logs_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    required = [
        repo / "finetune/train_tokenizer.py",
        repo / "finetune/train_predictor.py",
        dataset_path / "train_data.pkl",
        dataset_path / "val_data.pkl",
        dataset_path / "metadata.json",
        pretrained_tokenizer_path / "model.safetensors",
        pretrained_predictor_path / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required training inputs are missing: {missing}")
    dataset_metadata = json.loads(
        (dataset_path / "metadata.json").read_text(encoding="utf-8")
    )
    validate_dataset_metadata(dataset_metadata, args.lookback, args.horizon)

    env = os.environ.copy()
    env.update({
        "KRONOS_PROJECT_ROOT": str(PROJECT_ROOT),
        "KRONOS_DATASET_PATH": str(dataset_path),
        "KRONOS_SAVE_PATH": str(save_path),
        "KRONOS_TOKENIZER_SAVE_NAME": args.tokenizer_name,
        "KRONOS_PREDICTOR_SAVE_NAME": args.predictor_name,
        "KRONOS_TOKENIZER_PATH": str(pretrained_tokenizer_path),
        "KRONOS_MODEL_PATH": str(pretrained_predictor_path),
        "KRONOS_EXPERIMENT_ID": args.experiment_id,
        "KRONOS_EPOCHS": str(args.epochs),
        "KRONOS_BATCH_SIZE": str(args.batch_size),
        "KRONOS_TRAIN_SAMPLES": str(args.train_samples),
        "KRONOS_VAL_SAMPLES": str(args.val_samples),
        "KRONOS_NUM_WORKERS": str(args.num_workers),
        "KRONOS_EARLY_STOPPING_PATIENCE": str(args.early_stopping_patience),
        "KRONOS_SEED": str(args.seed),
        "KRONOS_LOOKBACK": str(args.lookback),
        "KRONOS_PRED_LEN": str(args.horizon),
        "KRONOS_RESUME": "1" if args.resume else "0",
        "KRONOS_FORECAST_LOSS_WEIGHT": str(args.forecast_loss_weight),
        "KRONOS_BALANCE_TRAIN_DIRECTIONS": "1" if args.balance_train_directions else "0",
        "PYTHONPATH": str(repo),
        "PYTHONHASHSEED": str(args.seed),
    })

    manifest = {
        "experiment_id": args.experiment_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "save_path": str(save_path),
        "settings": vars(args),
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in required
            if path.is_file() and path.is_relative_to(PROJECT_ROOT)
        },
        "a_share_rules": {
            "adjustment": dataset_metadata.get("adjustment", "hfq"),
            "drop_first_listing_day": True,
            "tokenizer_edge_pressure": not args.skip_tokenizer,
            "limit_up_high_multiplier": 1.02,
            "limit_down_low_multiplier": 0.98,
            "limit_up_blocks_buy": True,
            "limit_down_blocks_sell": True,
        },
    }
    manifest_path = result_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        tokenizer_checkpoint = save_path / args.tokenizer_name / "checkpoints/best_model/model.safetensors"
        if not args.skip_tokenizer:
            env["KRONOS_TOKENIZER_EDGE_PRESSURE"] = "1"
            run_stage(
                "tokenizer",
                "finetune/train_tokenizer.py",
                repo,
                env,
                logs_path / f"{args.experiment_id}_tokenizer.log",
            )
        else:
            tokenizer_checkpoint = pretrained_tokenizer_path / "model.safetensors"
            env["KRONOS_FINETUNED_TOKENIZER_PATH"] = str(pretrained_tokenizer_path)
        if not tokenizer_checkpoint.is_file():
            raise FileNotFoundError(f"tokenizer checkpoint is missing: {tokenizer_checkpoint}")

        env["KRONOS_TOKENIZER_EDGE_PRESSURE"] = "0"
        run_stage(
            "predictor",
            "finetune/train_predictor.py",
            repo,
            env,
            logs_path / f"{args.experiment_id}_predictor.log",
        )
        predictor_checkpoint = save_path / args.predictor_name / "checkpoints/best_model/model.safetensors"
        if not predictor_checkpoint.is_file():
            raise FileNotFoundError(f"predictor checkpoint was not produced: {predictor_checkpoint}")

        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs"] = {
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
            "tokenizer_sha256": sha256(tokenizer_checkpoint),
            "predictor_checkpoint": str(predictor_checkpoint),
            "predictor_sha256": sha256(predictor_checkpoint),
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(exc)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training completed: {manifest_path}")


if __name__ == "__main__":
    main()
