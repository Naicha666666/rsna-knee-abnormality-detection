#!/usr/bin/env bash
set -u
cd /home/eason/rsna-knee
O=experiments/local_c6s_64_k12_ep16_new4labels
mkdir -p $O
PAT="train_c"
PAT="${PAT}6s_local.py"
running=$(pgrep -f "$PAT" | grep -v "^$$\$" || true)
if [ -n "$running" ]; then
  echo "ABORT: already running:"; ps -o pid=,cmd= -p $running; exit 3
fi
newest=$(ls -t experiments/*/train.log 2>/dev/null | head -1)
age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
echo "newest train.log: $newest (last written ${age}s ago)"
if [ "$age" -lt 300 ]; then echo "ABORT: a train.log was written in the last 5 min"; exit 4; fi
if /opt/rocm/bin/rocm-smi --showpids 2>/dev/null | grep -q "^[0-9]"; then
  echo "ABORT: a KFD process is using the GPU"; /opt/rocm/bin/rocm-smi --showpids; exit 5
fi
nohup /home/eason/umud/venv-rocm/bin/python -u src/train_c6s_local.py > $O/train.log 2>&1 < /dev/null &
pid=$!
echo "launched pid $pid"
echo $pid > $O/train.pid
sleep 75
echo "--- first lines of train.log"
head -24 $O/train.log
