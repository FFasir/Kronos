#!/usr/bin/env bash
set -euo pipefail

project_root="${KRONOS_PROJECT_ROOT:-/root/kronos-strategy}"
save_path="${KRONOS_SAVE_PATH:-${project_root}/models/official_finetune/all_a_share}"
checkpoint="${save_path}/all_a_share_tokenizer/checkpoints/best_model/model.safetensors"
tokenizer_log="${project_root}/logs/all_a_share_tokenizer.log"

while ! grep -q "Training finished. Summary file saved." "${tokenizer_log}" 2>/dev/null; do
  if ! pgrep -f '[t]rain_tokenizer.py' >/dev/null; then
    echo "Tokenizer stopped before its completion marker; predictor was not started." >&2
    exit 1
  fi
  sleep 30
done

if [[ ! -s "${checkpoint}" ]]; then
  echo "Tokenizer completion marker exists but best checkpoint is missing." >&2
  exit 1
fi

cd "${project_root}/third_party/Kronos"
env \
  KRONOS_PROJECT_ROOT="${project_root}" \
  KRONOS_QLIB_DATA_PATH="${project_root}/data/processed/qlib/all_a_share_20260715" \
  KRONOS_INSTRUMENT="all_a_share" \
  KRONOS_DATASET_PATH="${project_root}/data/official_processed_datasets_all_a_share" \
  KRONOS_SAVE_PATH="${save_path}" \
  KRONOS_TOKENIZER_SAVE_NAME="all_a_share_tokenizer" \
  KRONOS_PREDICTOR_SAVE_NAME="all_a_share_predictor" \
  PYTHONPATH="${project_root}/third_party/Kronos" \
  /usr/local/miniconda3/envs/py312/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=1 finetune/train_predictor.py

/usr/local/miniconda3/envs/py312/bin/python \
  "${project_root}/scripts/finalize_all_a_share_report.py"
