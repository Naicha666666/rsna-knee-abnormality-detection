#!/bin/bash
# Retry Kaggle kernel-output download until vols64 exist.
set -u
KAG=/home/eason/umud/venv-rocm/bin/kaggle
OUT=/home/eason/rsna-knee/data/corpus64
mkdir -p "$OUT"
export PATH="/home/eason/umud/venv-rocm/bin:$PATH"
download_one () {
  local slug="$1"
  local n=1
  while [ $n -le 8 ]; do
    echo "[$(date +%H:%M:%S)] try $n $slug"
    if $KAG kernels output "$slug" -p "$OUT"; then
      echo "OK $slug"
      return 0
    fi
    echo "FAIL $slug attempt $n"
    n=$((n+1))
    sleep 15
  done
  return 1
}
download_one easonyanyan/rsna-knee-c4-dump-64slice-p0
download_one easonyanyan/rsna-knee-c4-dump-64slice-p1
ls -lh "$OUT"
echo DONE_DL
