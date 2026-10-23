"""Watch the v3 submit kernel; on COMPLETE, submit to the competition.

Stdout is only DONE / FAILED / CANCELLED (monitor-friendly).
"""
import os
import subprocess
import sys
import time

KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"
SLUG = "easonyanyan/rsna-knee-dinov2-s-v3-extractor-unfreeze6"
COMP = "rsna-knee-abnormality-detection"
OUT = r"C:/Users/Administrator/kout_v3"
INTERVAL = 180


def _run(args):
    return subprocess.run(args, capture_output=True, text=True)


def status():
    r = _run([KAGGLE, "kernels", "status", SLUG])
    blob = (r.stdout or "") + (r.stderr or "")
    for token in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if token in blob.upper() or token in blob:
            return token
    return "UNKNOWN"


def submit_on_complete():
    os.makedirs(OUT, exist_ok=True)
    _run([KAGGLE, "kernels", "output", "-p", OUT, SLUG])
    csv = os.path.join(OUT, "submission.csv")
    if not os.path.exists(csv):
        # walk for it
        for root, _, files in os.walk(OUT):
            if "submission.csv" in files:
                csv = os.path.join(root, "submission.csv")
                break
    if not os.path.exists(csv):
        return False
    # Code competition: -f is the kernel OUTPUT NAME, not a local path.
    r = _run([
        KAGGLE, "competitions", "submit",
        "-c", COMP, "-k", SLUG, "-v", "1",
        "-f", "submission.csv",
        "-m", "DINOv2-S v3: community extractor + unfreeze 6 + OneCycle, full 4407, 12ep",
    ])
    return r.returncode == 0


while True:
    s = status()
    if s == "COMPLETE":
        ok = submit_on_complete()
        print("DONE" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if s == "ERROR":
        print("FAILED")
        sys.exit(1)
    if s == "CANCEL":
        print("CANCELLED")
        sys.exit(1)
    time.sleep(INTERVAL)
