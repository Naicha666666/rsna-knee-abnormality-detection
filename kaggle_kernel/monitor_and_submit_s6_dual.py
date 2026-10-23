import os
import sys
import time
import subprocess
from datetime import datetime

SLUG = "easonyanyan/rsna-knee-s6-helios-dual-reader"
COMP = "rsna-knee-abnormality-detection"
VERSION = "2"
MESSAGE = "Dual-Reader S6 Helios Blend: Stage 1 Ep8 CoAtNet-2 384 (Solo 0.948) + Stage 2 S6 ConvNeXt-V2-Base 336 (0.70:0.30 Rank Blend)"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def get_kernel_status():
    r = subprocess.run(["kaggle", "kernels", "status", SLUG], capture_output=True, text=True, shell=True)
    blob = (r.stdout or "") + (r.stderr or "")
    for status in ["COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"]:
        if status in blob:
            return status
    return "UNKNOWN: " + blob.strip()

def submit_to_competition():
    log("Kernel is COMPLETE! Submitting to competition...")
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMP,
        "-k", SLUG,
        "-v", VERSION,
        "-f", "submission.csv",
        "-m", MESSAGE
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    out = (r.stdout or "") + (r.stderr or "")
    log(f"Submit output:\n{out}")
    if r.returncode != 0:
        return None
    # Find new submission ID
    time.sleep(5)
    try:
        r2 = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP], capture_output=True, text=True, shell=True)
        for line in (r2.stdout or "").splitlines()[:10]:
            if "Dual-Reader" in line or "S6 Helios" in line or VERSION in line:
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
            r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP], capture_output=True, text=True, shell=True)
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
        time.sleep(45)

def main():
    log(f"Starting monitor for kernel {SLUG} (Version {VERSION})...")
    poll_count = 0
    while True:
        status = get_kernel_status()
        poll_count += 1
        if poll_count % 3 == 1 or status != "RUNNING":
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
            # Fetch output log
            r_out = subprocess.run(["kaggle", "kernels", "output", SLUG], capture_output=True, text=True, shell=True)
            log(f"Kernel output log:\n{r_out.stdout}\n{r_out.stderr}")
            break
        
        time.sleep(30)

if __name__ == "__main__":
    main()
