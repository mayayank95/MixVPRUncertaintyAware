#!/usr/bin/env bash
# Train with train_mixvpr.py (.venv-mixvpr), then eval.py (local_venv_alt) on best checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MIXVPR_VENV="${MIXVPR_VENV:-$ROOT/.venv-mixvpr}"
EVAL_VENV="${EVAL_VENV:-$ROOT/local_venv_alt}"

if [[ ! -x "$MIXVPR_VENV/bin/python" ]]; then
  echo "ERROR: MixVPR venv not found at $MIXVPR_VENV (create with python3.9 -m venv .venv-mixvpr)"
  exit 1
fi
if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
  echo "ERROR: Eval venv not found at $EVAL_VENV"
  exit 1
fi

TRAIN_PYTHON="$MIXVPR_VENV/bin/python"
EVAL_PYTHON="$EVAL_VENV/bin/python"

TS="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="logs/mixvpr_train_${TS}.log"
EVAL_LOG="logs/mixvpr_eval_${TS}.log"
mkdir -p logs

echo "=== [$(date)] train_mixvpr.py (GPU ${CUDA_VISIBLE_DEVICES}, ${MIXVPR_VENV}) ===" | tee -a "$TRAIN_LOG"
"$TRAIN_PYTHON" train_mixvpr.py 2>&1 | tee -a "$TRAIN_LOG"

mapfile -t CKPTS < <(find LOGS/resnet50 -name '*.ckpt' -type f 2>/dev/null | sort)
if ((${#CKPTS[@]} == 0)); then
  echo "ERROR: No .ckpt files under LOGS/resnet50" | tee -a "$TRAIN_LOG"
  exit 1
fi

BEST_CKPT="$(printf '%s\n' "${CKPTS[@]}" | "$TRAIN_PYTHON" -c "
import re, sys
files = [line.strip() for line in sys.stdin if line.strip()]
def score(path):
    m = re.search(r'R1\[([0-9.]+)\]', path)
    return float(m.group(1)) if m else -1.0
files.sort(key=score, reverse=True)
print(files[0])
")"

echo "=== [$(date)] eval.py on ${BEST_CKPT} (${EVAL_VENV}) ===" | tee -a "$EVAL_LOG"
"$EVAL_PYTHON" eval.py \
  --config configs/datasets.json \
  --datasets pitts30k,msls-val \
  --datasets_type test \
  --eval_query_folders query \
  --method mixvpr \
  --descriptors_dimension 512 \
  --resume_model "$BEST_CKPT" \
  --ckpt_state_dict_key state_dict \
  --model_mode basic \
  --use_labels \
  --num_workers 8 \
  --backbone ResNet50 \
  2>&1 | tee -a "$EVAL_LOG"

echo "=== [$(date)] Done. Train log: ${TRAIN_LOG}  Eval log: ${EVAL_LOG} ==="
