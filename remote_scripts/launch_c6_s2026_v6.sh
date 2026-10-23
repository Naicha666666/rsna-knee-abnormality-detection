#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES=0
OUT="/home/eason/rsna-knee/experiments/local_c6_coatnet_s2026_v6"
mkdir -p "$OUT"
nohup /home/eason/umud/venv-rocm/bin/python -u /home/eason/rsna-knee/src/train_c6_coatnet_s2026_v6.py > "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "[R9700] CoAtNet-2 384 Seed 2026 launched with PID: $PID"
echo "Log: $OUT/train.log"
