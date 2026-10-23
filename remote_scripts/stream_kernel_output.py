#!/usr/bin/env python3
"""Stream kernel output files to disk with resume. Avoids kaggle CLI loading
the whole 15 GB npy into RAM (IncompleteRead)."""
from __future__ import annotations
import os
import sys
import time

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

CHUNK = 8 * 1024 * 1024
MIN_VOL = 14 * 1024**3


def list_files(kernel: str):
    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(kernel)
    files = []
    token = None
    with api.build_kaggle_client() as kg:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = 50
            if token:
                req.page_token = token
            resp = kg.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in resp.files or []:
                files.append((item.file_name, item.url))
            token = resp.next_page_token
            if not token:
                break
    return files


def download(url: str, dest: str, force: bool = False):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    existing = 0 if force else (os.path.getsize(dest) if os.path.isfile(dest) else 0)
    headers = {}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resume {dest} from {existing}", flush=True)
    for attempt in range(1, 12):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=120) as r:
                if r.status_code == 416:
                    print(f"already complete {dest}", flush=True)
                    return
                r.raise_for_status()
                # If server ignores Range and sends 200, restart the file.
                if existing and r.status_code == 200:
                    print("server ignored Range; rewriting", flush=True)
                    existing = 0
                    mode = "wb"
                n = existing
                t0 = time.time()
                last = t0
                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        n += len(chunk)
                        now = time.time()
                        if now - last >= 10:
                            mb = n / 1024**2
                            sp = (n - existing) / max(now - t0, 1) / 1024**2
                            print(f"  {os.path.basename(dest)} {mb:.0f} MB  {sp:.1f} MB/s",
                                  flush=True)
                            last = now
                print(f"wrote {dest} ({n} bytes)", flush=True)
                return
        except Exception as e:
            print(f"attempt {attempt} failed: {e}", flush=True)
            existing = os.path.getsize(dest) if os.path.isfile(dest) else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"
            time.sleep(min(30, 2 * attempt))
    raise SystemExit(f"gave up on {dest}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/eason/rsna-knee/data/corpus64"
    slugs = sys.argv[2:] or [
        "easonyanyan/rsna-knee-c4-dump-64slice-p0",
        "easonyanyan/rsna-knee-c4-dump-64slice-p1",
    ]
    os.makedirs(out, exist_ok=True)
    want = {"vols64_p0.npy", "vols64_p1.npy", "masks64_p0.npy", "masks64_p1.npy",
            "ids64_p0.npy", "ids64_p1.npy", "manifest64_p0.json", "manifest64_p1.json"}
    for slug in slugs:
        print("listing", slug, flush=True)
        files = list_files(slug)
        print("files:", [n for n, _ in files], flush=True)
        for name, url in files:
            if name not in want:
                print("skip", name, flush=True)
                continue
            dest = os.path.join(out, name)
            download(url, dest)
    for vol in ("vols64_p0.npy", "vols64_p1.npy"):
        p = os.path.join(out, vol)
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        print(vol, sz, "OK" if sz > MIN_VOL else "TOO_SMALL")
        if sz < MIN_VOL:
            raise SystemExit(1)
    print("DONE_DL", flush=True)


if __name__ == "__main__":
    main()
