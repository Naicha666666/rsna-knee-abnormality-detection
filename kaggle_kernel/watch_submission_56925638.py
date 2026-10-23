import subprocess
import time
import sys
from datetime import datetime

REF = "56925638"
KAGGLE = r"C:\Users\vyanx\AppData\Local\Programs\Python\Python312\Scripts\kaggle.exe"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

log(f"Starting watcher for official submission {REF}...")
while True:
    try:
        out = subprocess.check_output(
            [KAGGLE, "competitions", "submissions", "-c", "rsna-knee-abnormality-detection"],
            text=True
        )
        for line in out.splitlines():
            if REF in line:
                log(f"Current Row: {line}")
                if "COMPLETE" in line:
                    log("SUCCESS! Submission scored!")
                    sys.exit(0)
                elif "ERROR" in line:
                    log("FAILED! Submission encountered error.")
                    sys.exit(1)
                break
    except Exception as e:
        log(f"Error checking status: {e}")
    time.sleep(60)
