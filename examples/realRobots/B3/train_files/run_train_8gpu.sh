#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_b3.yaml}"
CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
  shift
fi
OVERRIDES=("$@")
if [[ -n "${BASE_VLM:-}" ]]; then
  OVERRIDES=(--framework.qwenvl.base_vlm "${BASE_VLM}" "${OVERRIDES[@]}")
fi

# Run once before distributed startup to install metadata and populate caches.
# The same overrides are checked here and used for training below.
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_dataset.py" \
  --config_yaml "${CONFIG_YAML}" --install-modality "${OVERRIDES[@]}"
if [[ "${CHECK_ONLY}" == true ]]; then
  exit 0
fi
if [[ "${STARVLA_DISABLE_DEEPSPEED:-0}" == 1 ]]; then
  echo 'Eight-GPU training requires DeepSpeed; unset STARVLA_DISABLE_DEEPSPEED.' >&2
  exit 1
fi
"${PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 8 or not torch.cuda.is_available():
    raise SystemExit(f"Expected 8 visible CUDA GPUs, found {count}. Check CUDA_VISIBLE_DEVICES and the driver.")
for index in range(count):
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"GPU {index} does not support the BF16 ZeRO-2 recipe.")
print("Starting B3 training on 8 GPUs.")
PY

exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" "${OVERRIDES[@]}"
