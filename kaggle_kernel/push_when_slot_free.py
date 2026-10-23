"""Retry `push_exp.py` until Kaggle has a free GPU session slot.

Kaggle allows two concurrent batch GPU sessions per account. A push made while
both are busy fails with `Maximum batch GPU session count of 2 reached` (and the
CLI still exits 0, which push_exp.py now detects). This wrapper simply retries
the same push on an interval so the experiment starts the moment a slot frees,
instead of the queue depending on someone watching it.

Usage:
  python kaggle_kernel/push_when_slot_free.py <slug> <title> <backbone> \
      <max_studies> <epochs> <folds> <do_submit> [interval_s] [max_hours]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PUSH = os.path.join(HERE, "push_exp.py")


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        sys.exit(2)
    argv = sys.argv[1:]
    # trailing numeric pair = [interval_s] [max_hours]; everything before it goes to
    # push_exp.py verbatim, so new push_exp arguments need no change here.
    interval, max_h = 600.0, 12.0
    tail_numeric = all(a.replace(".", "", 1).isdigit() for a in argv[-2:])
    if len(argv) > 8 and tail_numeric:
        interval, max_h = float(argv[-2]), float(argv[-1])
        argv = argv[:-2]
    args = argv

    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        r = subprocess.run([sys.executable, PUSH] + args,
                           capture_output=True, text=True,
                           cwd=os.path.dirname(HERE))
        out = (r.stdout or "") + (r.stderr or "")
        stamp = time.strftime("%H:%M:%S")
        if r.returncode == 0 and "PUSHED" in out:
            print(f"[{stamp}] attempt {attempt}: PUSHED")
            print(out.strip()[-600:])
            sys.exit(0)
        busy = "maximum batch gpu session" in out.lower()
        print(f"[{stamp}] attempt {attempt}: "
              f"{'GPU slots busy, will retry' if busy else 'push failed'}")
        if not busy:
            print(out.strip()[-1500:])
            sys.exit(1)
        if time.time() - t0 > max_h * 3600:
            print("gave up after", max_h, "h")
            sys.exit(1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
