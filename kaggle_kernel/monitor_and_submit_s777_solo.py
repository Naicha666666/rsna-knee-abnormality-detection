"""自动监听并提交 Seed 777 Solo (CoAtNet-2 384) 提交内核：
1. 推送 easonyanyan/rsna-knee-c6-s777-solo
2. 轮询等待其在 T4 GPU 上完成 (COMPLETE)
3. 自动提交到竞赛 rsna-knee-abnormality-detection
4. 跟踪官方评分进度并输出出分结果
"""
import os
import sys
import time
import subprocess
from datetime import datetime

KAGGLE = r"C:\Users\vyanx\AppData\Local\Programs\Python\Python312\Scripts\kaggle.exe"
SLUG = "easonyanyan/rsna-knee-c6-s777-solo-coatnet-384"
COMP = "rsna-knee-abnormality-detection"
VERSION = "2"
MESSAGE = "Seed 777 Solo: CoAtNet-2 384 (Domain-Logit Labels SWA, Gold 0.9057; LOG #610)"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def push_kernel():
    log(f"Pushing kernel from kaggle_kernel/submissions/rsna-knee-c6-s777-solo...")
    cmd = [KAGGLE, "kernels", "push", "-p", "kaggle_kernel/submissions/rsna-knee-c6-s777-solo"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    log(f"Push output:\n{out}")
    if r.returncode != 0:
        raise RuntimeError(f"Failed to push kernel: {out}")

def get_kernel_status():
    r = subprocess.run([KAGGLE, "kernels", "status", SLUG], capture_output=True, text=True)
    blob = (r.stdout or "") + (r.stderr or "")
    for status in ["COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"]:
        if status in blob:
            return status
    return "UNKNOWN: " + blob.strip()

def submit_to_competition():
    log("Kernel is COMPLETE! Submitting Version 1 to competition...")
    cmd = [
        KAGGLE, "competitions", "submit",
        "-c", COMP,
        "-k", SLUG,
        "-v", VERSION,
        "-f", "submission.csv",
        "-m", MESSAGE
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    log(f"Submit output:\n{out}")
    if r.returncode != 0:
        return None
    time.sleep(5)
    try:
        r2 = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMP], capture_output=True, text=True)
        for line in (r2.stdout or "").splitlines()[:10]:
            if "Seed 777 Solo" in line or "LOG #610" in line:
                log(f"Detected submission entry: {line}")
                parts = line.split()
                if parts and parts[0].isdigit():
                    return parts[0]
    except Exception as e:
        log(f"Warning parsing submission ID: {e}")
    return "SUBMITTED"

def watch_submission(sub_id):
    if not sub_id or not sub_id.isdigit():
        log(f"Cannot watch non-numeric sub_id: {sub_id}")
        return
    log(f"Watching official submission {sub_id} on hidden test set...")
    while True:
        try:
            r = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMP], capture_output=True, text=True)
            for line in (r.stdout or "").splitlines():
                if sub_id in line:
                    log(f"[{sub_id}]: {line}")
                    if "COMPLETE" in line:
                        log(f"SUCCESS! Submission {sub_id} finished scoring!")
                        return
                    elif "ERROR" in line:
                        log(f"FAILED! Submission {sub_id} encountered an error.")
                        return
                    break
        except Exception as e:
            log(f"Watch error: {e}")
        time.sleep(60)

def main():
    push_kernel()
    log(f"Starting monitor for kernel {SLUG} (Version {VERSION})...")
    poll_count = 0
    while True:
        status = get_kernel_status()
        poll_count += 1
        if poll_count % 2 == 1 or status != "RUNNING":
            log(f"Kernel Status: {status}")
        
        if status == "COMPLETE":
            sub_id = submit_to_competition()
            if sub_id:
                watch_submission(sub_id)
            break
        elif status in ["ERROR", "CANCEL"]:
            log(f"Kernel failed with status: {status}")
            r = subprocess.run([KAGGLE, "kernels", "output", SLUG], capture_output=True, text=True)
            log(f"Kernel output logs:\n{(r.stdout or '')[:1000]}")
            break
        
        time.sleep(60)

if __name__ == "__main__":
    main()
