#!/usr/bin/env python3
"""Download complete Kronos model and tokenizer snapshots with validation."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "NeoQuasar/Kronos-base"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "*.ckpt")
MIN_MODEL_WEIGHT_BYTES = 10 * 1024 * 1024
MIN_TOKENIZER_ASSET_BYTES = 1024 * 1024


def has_weights(path: Path) -> bool:
    return path.is_dir() and any(
        file.stat().st_size >= MIN_MODEL_WEIGHT_BYTES
        for pattern in WEIGHT_PATTERNS
        for file in path.glob(pattern)
    )


def has_tokenizer_assets(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    patterns = ("*.safetensors", "*.bin", "*.model", "*.pt", "tokenizer.json", "vocab.json")
    return any(
        file.stat().st_size >= MIN_TOKENIZER_ASSET_BYTES
        for pattern in patterns
        for file in path.glob(pattern)
    )


def download(repo_id: str, target: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        token=token,
        max_workers=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate Kronos-base snapshots")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "kronos_base")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "models" / "kronos_tokenizer")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="optional Hugging Face read token")
    parser.add_argument("--endpoint", help="optional Hugging Face endpoint, for example https://hf-mirror.com")
    parser.add_argument("--use-xet", action="store_true", help="use Xet transfer instead of standard HTTPS")
    args = parser.parse_args()

    # Some mirrors cannot authorize the Xet CAS reconstruction endpoint. Standard
    # HTTPS is slower but reliable and resumes files in the local .cache folder.
    if not args.use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        print(f"Downloading model: {args.model_id}")
        download(args.model_id, args.model_dir, args.token)
        if not has_weights(args.model_dir):
            raise RuntimeError(f"model weights missing after download: {args.model_dir}")

        print(f"Downloading tokenizer: {args.tokenizer_id}")
        download(args.tokenizer_id, args.tokenizer_dir, args.token)
        if not has_tokenizer_assets(args.tokenizer_dir):
            raise RuntimeError(f"tokenizer assets missing after download: {args.tokenizer_dir}")
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        message = str(exc).lower()
        if "network is unreachable" in message or "connecterror" in message:
            print("The server has no route to the model host. A token cannot fix this.", file=sys.stderr)
            print("Try a reachable mirror: python scripts/download_kronos.py --endpoint https://hf-mirror.com", file=sys.stderr)
            print("Or download on a networked machine and upload models/kronos_base and models/kronos_tokenizer.", file=sys.stderr)
        elif "401" in message or "unauthorized" in message:
            print("Authentication or Xet CAS access failed. Disable Xet and use an official endpoint:", file=sys.stderr)
            print("  unset HF_ENDPOINT; export HF_HUB_DISABLE_XET=1; hf auth login", file=sys.stderr)
            print("  python scripts/download_kronos.py", file=sys.stderr)
        else:
            print("Retry after checking the network, endpoint, and optional Hugging Face read token.", file=sys.stderr)
        return 2

    print("Download complete.")
    print(f"Model: {args.model_dir}")
    print(f"Tokenizer: {args.tokenizer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
