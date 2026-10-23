import os
import sys
import time
import subprocess
from datetime import datetime

KAGGLE = r"C:\Users\vyanx\AppData\Local\Programs\Python\Python312\Scripts\kaggle.exe"
SLUG = "easonyanyan/rsna-knee-c6-warmstart-ep12-solo-coatnet-384"
COMP = "rsna-knee-abnormality-detection"
VERSION = "1"
MESSAGE = "CoAtNet-2 384 WarmStart Ep12 Solo (Deep Basin SWA, Gold 0.9203, Loss 0.7112; LOG #645)"

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
    log("Kernel is COMPLETE! Submitting to competition...")
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
            if "Ep12" in line or SLUG in line or "WarmStart" in line:
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
                    log(f"Status update: {line}")
                    if "COMPLETE" in line:
                        log(f"*** FINAL OFFICIAL LB SCORE: Ep12 Solo ***")
                        return
                    elif "ERROR" in line or "FAIL" in line:
                        log(f"*** SUBMISSION FAILED ***")
                        return
        except Exception as e:
            log(f"Error querying submissions: {e}")
        time.sleep(30)

def main():
    log(f"Starting monitor for {SLUG} (Version {VERSION})...")
    poll_count = 0
    while True:
        poll_count += 1
        st = get_kernel_status()
        log(f"Poll #{poll_count}: Kernel status = {st}")
        if st == "COMPLETE":
            sub_id = submit_to_competition()
            if sub_id:
                watch_submission(sub_id)
            break
        elif st in ["ERROR", "CANCEL"]:
            log(f"Kernel terminated with status: {st}")
            break
        time.sleep(30)

if __name__ == "__main__":
    main()
