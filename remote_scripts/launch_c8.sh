#!/usr/bin/env bash
set -u
cd /home/eason/rsna-knee
O=experiments/local_c8_64_k12_ep16_radimagenet
mkdir -p $O
PAT="train_c"
PAT="${PAT}8_local.py"
running=$(pgrep -f "$PAT" | grep -v "^$$\$" || true)
if [ -n "$running" ]; then
  echo "ABORT: already running:"; ps -o pid=,cmd= -p $running; exit 3
fi
gpu=$(/opt/rocm/bin/rocm-smi --showuse 2>/dev/null | grep -o "GPU use (%): [0-9]*" | grep -o "[0-9]*$")
echo "gpu use before launch: ${gpu:-unknown}%"
nohup /home/eason/umud/venv-rocm/bin/python -u src/train_c8_local.py > $O/train.log 2>&1 < /dev/null &
pid=$!
echo "launched pid $pid"
echo $pid > $O/train.pid
sleep 60
echo "--- first lines of train.log"
head -32 $O/train.log
