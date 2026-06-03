#!/usr/bin/env bash
# Freeze MixVPR, validate before/after 1 training epoch; recalls should match.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON="${ROOT}/.venv-mixvpr/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/mixvpr_frozen_sanity_${TS}.log"
mkdir -p logs

EXTRA_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--resume_ckpt "$RESUME_CKPT")
fi

echo "=== Frozen sanity (GPU ${CUDA_VISIBLE_DEVICES}) ===" | tee "$LOG"
"$PYTHON" train_mixvpr.py \
  --freeze_model \
  --max_epochs 1 \
  --validate_before_fit \
  --no_checkpoint \
  --gpu 0 \
  --num_workers 8 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

echo "Log: $LOG"
