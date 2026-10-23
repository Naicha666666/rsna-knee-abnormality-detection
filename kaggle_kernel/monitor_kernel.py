"""Poll one kernel until it leaves RUNNING; print the terminal status.

Usage:  python kaggle_kernel/monitor_kernel.py <slug> [interval_s]
"""
import subprocess
import sys
import time

KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    slug = sys.argv[1]
    if "/" not in slug:
        slug = "easonyanyan/" + slug
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0

    while True:
        r = subprocess.run([KAGGLE, "kernels", "status", slug],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        up = out.upper()
        if "COMPLETE" in up:
            print(f"{slug} COMPLETE")
            sys.exit(0)
        if "ERROR" in up or "CANCEL" in up:
            print(out.strip())
            sys.exit(1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
