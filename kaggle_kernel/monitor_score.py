"""Poll the official v2 submission until it has a publicScore."""
import subprocess
import sys
import time

KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"
REF = "55612746"
INTERVAL = 180


def row():
    r = subprocess.run(
        [KAGGLE, "competitions", "submissions", "-c",
         "rsna-knee-abnormality-detection"],
        capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if REF in line:
            return line
    return ""


while True:
    line = row()
    up = line.upper()
    if "COMPLETE" in up and REF in line:
        print("DONE")
        sys.exit(0)
    if "ERROR" in up or "FAIL" in up:
        print("FAILED")
        sys.exit(1)
    time.sleep(INTERVAL)
