import os
import sys
import time
import subprocess
from datetime import datetime

KAGGLE = r"C:\Users\vyanx\AppData\Local\Programs\Python\Python312\Scripts\kaggle.exe"
SLUG = "easonyanyan/rsna-knee-dual-seed-coatnet-0950"
COMP = "rsna-knee-abnormality-detection"
VERSION = "1"
MESSAGE = "Dual-Seed CoAtNet-2 384 (Seed 42 Base 0.949 + Seed 2026 WangPai 2) x 12-Label Optimized Schedule (LOG #599)"

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
            if "Dual-Seed" in line or "LOG #599" in line:
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
                log(f"Submission successful! ID = {sub_id}")
                watch_submission(sub_id)
            else:
                log("Submission command encountered an issue.")
            break
        elif status in ("ERROR", "CANCEL"):
            log(f"Kernel failed with status {status}. Aborting submission.")
            break
        
        time.sleep(15)

if __name__ == "__main__":
    main()
