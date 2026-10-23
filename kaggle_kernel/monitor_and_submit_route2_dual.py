"""自动监听并提交线路 2（双读取器终极融合）内核 (Version 1)：
1. 轮询等待 easonyanyan/rsna-knee-warmstart-ep8-convnext-dual 达到 COMPLETE
2. 调用 Kaggle API 提交到竞赛 rsna-knee-abnormality-detection（消耗今日第 5 次/最后 1 次配额）
3. 跟踪官方评分进度并输出出分结果
"""
import os
import sys
import time
import subprocess
from datetime import datetime

KAGGLE = r"C:\Users\vyanx\AppData\Local\Programs\Python\Python312\Scripts\kaggle.exe"
SLUG = "easonyanyan/rsna-knee-warmstart-ep8-convnext-dual"
COMP = "rsna-knee-abnormality-detection"
VERSION = "1"
MESSAGE = "Grand Dual Reader 0.952+ (CoAtNet Ep8 Solo 0.948 + ConvNeXt 2.5D MIL x Dual Logit Quantile & Anti-Tie; LOG #619)"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def get_kernel_status():
    r = subprocess.run([KAGGLE, "kernels", "status", SLUG], capture_output=True, text=True)
    blob = (r.stdout or "") + (r.stderr or "")
    for status in ["COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"]:
        if status in blob:
            return status
    return "UNKNOWN: " + blob.strip()

def submit_to_competition():
    log("Kernel is COMPLETE! Submitting Version 1 to competition (Final quota of today)...")
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
            if "LOG #619" in line or "0.952+" in line:
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
    log(f"Watching official competition scoring for submission {sub_id}...")
    while True:
        r = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMP], capture_output=True, text=True)
        for line in (r.stdout or "").splitlines():
            if sub_id in line:
                log(f"Scoring update: {line}")
                if "COMPLETE" in line:
                    log(f"Scoring COMPLETE! Final score in line above.")
                    return
                elif "ERROR" in line:
                    log(f"Scoring FAILED!")
                    return
                break
        time.sleep(60)

def main():
    log(f"Starting watcher for kernel: {SLUG}")
    while True:
        status = get_kernel_status()
        log(f"Current kernel status: {status}")
        if status == "COMPLETE":
            sub_id = submit_to_competition()
            if sub_id and sub_id.isdigit():
                watch_submission(sub_id)
            break
        elif status == "ERROR":
            log("Kernel failed with ERROR! Fetching kernel output...")
            r_out = subprocess.run([KAGGLE, "kernels", "output", SLUG], capture_output=True, text=True)
            log(f"Kernel output logs:\n{(r_out.stdout or '')[:2000]}")
            break
        elif status == "CANCEL":
            log("Kernel was CANCELLED!")
            break
        time.sleep(60)

if __name__ == "__main__":
    main()
