#!/usr/bin/env python3
import subprocess, time, os, json, sys

SLUG = "easonyanyan/rsna-knee-screen-dinov2-b-subset600-5ep"
KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"
OUT = r"C:/Users/Administrator/kout_base"
INTERVAL = 180

def status():
    r = subprocess.run([KAGGLE, "kernels", "status", SLUG],
                        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "status" in line:
            return line.split('"')[-2] if '"' in line else line.strip()
    return "UNKNOWN:" + r.stdout.strip()

def download():
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", "-p", OUT, SLUG],
                   capture_output=True, text=True)
    # print metrics if present
    mp = os.path.join(OUT, "experiments", "dinov2_base_600_ep5", "metrics.json")
    if os.path.exists(mp):
        print("METRICS:", open(mp).read())
    else:
        print("metrics.json NOT found after download; listing:")
        for root, _, files in os.walk(OUT):
            for f in files:
                print(os.path.join(root, f))

print(f"monitoring {SLUG} every {INTERVAL}s", flush=True)
while True:
    s = status()
    print(time.strftime("%H:%M:%S"), "status=", s, flush=True)
    if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
        print("terminal -> downloading", flush=True)
        download()
        break
    time.sleep(INTERVAL)
