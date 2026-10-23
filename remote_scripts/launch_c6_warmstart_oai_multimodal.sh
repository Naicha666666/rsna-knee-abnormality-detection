#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES=0
EPOCHS=${1:-4}
OUT="/home/eason/rsna-knee/experiments/local_c6_coatnet_warmstart_oai_multimodal"
mkdir -p "$OUT"
nohup /home/eason/umud/venv-rocm/bin/python -u /home/eason/rsna-knee/src/train_c6_coatnet_warmstart_oai_multimodal.py --epochs "$EPOCHS" > "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "[R9700] Warm-Start CoAtNet-2 384 (OAI 0.949 Base + Multimodal Refined Labels) launched with PID: $PID"
echo "Log: $OUT/train.log"
