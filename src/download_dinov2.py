"""ONE-TIME helper: download DINOv2 weights to a local folder.

We pull the official `.pth` files directly from Meta's public file host (NOT
torch.hub, which fails in some sandboxes). Then create a Kaggle dataset from the
folder and attach it to the submission kernel so the model loads OFFLINE (Code
Competition = no internet at submit time).

Run:
    python download_dinov2.py --out data/dinov2_weights
Then:
    kaggle datasets create -p data/dinov2_weights
and attach `easonyanyan/rsna-dinov2-weights` to the submission kernel (or set
DINOV2_SMALL_PATH / DINOV2_BASE_PATH to the mounted .pth paths).
"""
from __future__ import annotations
import os
import argparse
import urllib.request

URLS = {
    # Public 518-resolution "pretrain" weights (pos_embed = 1370 tokens). We feed
    # 224px slices and INTERPOLATE the positional embedding down to 257 tokens in
    # backbones/dinov2.py -- the standard DINOv2 recipe (224 eval from 518-pretrained
    # backbone). The 224-only .pth URLs currently return 403, so use these.
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
}


def _download(url: str, dst: str):
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"  -> {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dinov2_weights")
    ap.add_argument("--variant", default="all",
                    choices=["all", "dinov2_vits14", "dinov2_vitb14"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    want = list(URLS) if args.variant == "all" else [args.variant]
    for name in want:
        _download(URLS[name], os.path.join(args.out, f"{name}.pth"))
    print("done. Upload this folder as a Kaggle dataset and attach it to the "
          "submission kernel.")


if __name__ == "__main__":
    main()
