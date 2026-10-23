"""Poll one official submission until it leaves PENDING; print its row.

Usage:  python kaggle_kernel/monitor_submission.py <ref> [interval_s]
"""
import subprocess
import sys
import time

KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"
COMP = "rsna-knee-abnormality-detection"


def row(ref):
    r = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if line.startswith(ref):
            return line
    return ""


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ref = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    while True:
        line = row(ref)
        up = line.upper()
        if "COMPLETE" in up:
            print(line.strip())
            sys.exit(0)
        if "ERROR" in up or "FAIL" in up:
            print(line.strip())
            sys.exit(1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
