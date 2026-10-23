#!/usr/bin/env bash
set -u
cd /home/eason/rsna-knee
O=experiments/local_c9b_64_k12_ep16_dinov2b_llrd
mkdir -p $O
PAT="train_c"
PAT="${PAT}9b_local.py"
running=$(pgrep -f "$PAT" | grep -v "^$$\$" || true)
if [ -n "$running" ]; then
  echo "ABORT: already running:"; ps -o pid=,cmd= -p $running; exit 3
fi
newest=$(ls -t experiments/*/train.log 2>/dev/null | head -1)
age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
echo "newest train.log: $newest (last written ${age}s ago)"
if [ "$age" -lt 300 ]; then echo "ABORT: a train.log was written in the last 5 min"; exit 4; fi
gpu=$(/opt/rocm/bin/rocm-smi --showuse 2>/dev/null | grep -o "GPU use (%): [0-9]*" | grep -o "[0-9]*$")
echo "gpu use before launch: ${gpu:-unknown}%"
nohup /home/eason/umud/venv-rocm/bin/python -u src/train_c9b_local.py > $O/train.log 2>&1 < /dev/null &
pid=$!
echo "launched pid $pid"
echo $pid > $O/train.pid
sleep 75
echo "--- first lines of train.log"
head -24 $O/train.log
