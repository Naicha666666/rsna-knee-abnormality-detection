#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES=0
EPOCHS=${1:-6}
OUT="/home/eason/rsna-knee/experiments/s6_convnextv2_base_fold0"
mkdir -p "$OUT"
nohup /home/eason/umud/venv-rocm/bin/python -u /home/eason/rsna-knee/scratchpad/train_s6_convnextv2_base_r9700.py --epochs "$EPOCHS" --fold 0 --batch 1 --grad-accum 4 --n-win 20 --chunk-size 8 > "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "[R9700] S6 ConvNeXt-V2-Base Retrain Fold 0 launched with PID: $PID"
echo "Log: $OUT/train.log"
