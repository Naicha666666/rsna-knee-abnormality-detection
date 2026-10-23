#!/bin/bash
cd /home/eason/rsna-knee
mkdir -p /home/eason/rsna-knee/experiments/local_ortho_specialist_pfoa_latoa_ep16
nohup /home/eason/umud/venv-rocm/bin/python src/train_ortho_specialist_local.py --epochs 16 > /home/eason/rsna-knee/experiments/local_ortho_specialist_pfoa_latoa_ep16/train.log 2>&1 &
echo \$! > /home/eason/rsna-knee/experiments/local_ortho_specialist_pfoa_latoa_ep16/train.pid
echo " Orthopedic Specialist Training launched with PID \\
