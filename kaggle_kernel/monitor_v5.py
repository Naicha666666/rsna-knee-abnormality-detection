"""Poll the v5 kernel until it leaves RUNNING; print the terminal status."""
import subprocess, sys, time

KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"
SLUG = "easonyanyan/rsna-knee-dinov2-s-v5-ep20"
INTERVAL = 600

while True:
    r = subprocess.run([KAGGLE, "kernels", "status", SLUG],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    up = out.upper()
    if "COMPLETE" in up:
        print("COMPLETE"); sys.exit(0)
    if "ERROR" in up or "CANCEL" in up:
        print(out.strip()); sys.exit(1)
    time.sleep(INTERVAL)
