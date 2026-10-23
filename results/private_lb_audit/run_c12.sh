#!/bin/bash
set -e
cd /home/eason/rsna-knee
export RSNA_DATA_ROOT=/home/eason/rsna-knee/data
export RSNA_CORPUS_DIR=/home/eason/rsna-knee/data/corpus64
export RSNA_CORPUS_SLOTS=64
export RSNA_SEED=42
export PYTHONUNBUFFERED=1

OUT_DIR=/home/eason/rsna-knee/experiments/local_c12_64_k12_ep16_convnext_small_384_calibrated
mkdir -p "$OUT_DIR"

echo "Starting C12 ConvNeXt-Small 384 Calibrated at $(date)" > "$OUT_DIR/run.log"
nohup /home/eason/umud/venv-rocm/bin/python src/train_c12_local.py > "$OUT_DIR/train.log" 2>&1 &
PID=$!
echo $PID > "$OUT_DIR/train.pid"
echo "Launched C12 training with PID $PID, logs at $OUT_DIR/train.log"
