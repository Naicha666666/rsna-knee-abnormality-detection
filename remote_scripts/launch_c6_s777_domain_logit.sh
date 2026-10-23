#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES=0
SEED=${1:-777}
OUT="/home/eason/rsna-knee/experiments/local_c6_coatnet_s${SEED}_domain_logit"
mkdir -p "$OUT"
nohup /home/eason/umud/venv-rocm/bin/python -u /home/eason/rsna-knee/src/train_c6_coatnet_s777_domain_logit.py --seed "$SEED" > "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "[R9700] CoAtNet-2 384 Seed $SEED + Domain-Logit launched with PID: $PID"
echo "Log: $OUT/train.log"
