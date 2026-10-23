"""Build + push a single Kaggle script kernel for one experiment.

Each experiment = its own kernel slug so we get independent, auditable runs.
The experiment config is BAKED into run.py at build time (Kaggle script kernels
cannot receive CLI env vars through `kaggle kernels push`), so this helper sets
the env vars, rebuilds run.py via _build.py, rewrites kernel-metadata.json with
the experiment's slug/title, then pushes.

Usage:
  python kaggle_kernel/push_exp.py <slug> <title> <backbone> <max_studies> <epochs> <folds> [do_submit]
Example:
  python kaggle_kernel/push_exp.py rsna-knee-scr-dinov2-small \
      "RSNA Knee screen DINOv2-S (subset600, 5ep)" dinov2_small 600 5 5 0
"""
import os
import sys
import json
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(REPO, "kaggle_kernel", "_build.py")
META = os.path.join(REPO, "kaggle_kernel", "kernel-metadata.json")
KERNEL_DIR = os.path.join(REPO, "kaggle_kernel")
KAGGLE = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/Scripts/kaggle"


def slugify_title(title: str) -> str:
    """Mimic Kaggle's title->slug rule so the metadata `id` matches the kernel
    slug Kaggle actually assigns (otherwise re-pushes hit 409 Conflict)."""
    s = title.lower()
    s = "".join(c if (c.isalnum() or c == "-") else "-" for c in s)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        sys.exit(1)
    slug = sys.argv[1]          # label only; real id is the slugified title
    title = sys.argv[2]
    backbone = sys.argv[3]
    max_studies = sys.argv[4]
    epochs = sys.argv[5]
    folds = sys.argv[6]
    do_submit = sys.argv[7] if len(sys.argv) > 7 else "0"
    seed = sys.argv[8] if len(sys.argv) > 8 else "42"
    mode = sys.argv[9] if len(sys.argv) > 9 else "train"
    shard = sys.argv[10] if len(sys.argv) > 10 else "0"
    n_shards = sys.argv[11] if len(sys.argv) > 11 else "2"

    assert backbone in ("tiny_cnn", "dinov2_small", "dinov2_base"), backbone
    assert do_submit in ("0", "1"), do_submit
    assert mode in ("train", "infer", "diag", "raptor", "corpus", "corpus_dump",
                    "corpus_infer"), mode

    kernel_id = f"easonyanyan/{slugify_title(title)}"

    # 1) bake config into run.py
    env = dict(os.environ)
    env["RSNA_BACKBONE"] = backbone
    env["RSNA_DO_SUBMIT"] = do_submit
    env["RSNA_MAX_STUDIES"] = max_studies
    env["RSNA_EPOCHS"] = epochs
    env["RSNA_FOLDS"] = folds
    env["RSNA_SEED"] = seed
    env["RSNA_MODE"] = mode
    env["RSNA_SHARD"] = shard
    env["RSNA_N_SHARDS"] = n_shards
    # corpus_infer only: which checkpoint file(s) to look for, and the eval window
    # count the arm was validated with. Both are pass-through env, not new argv
    # positions, so the existing call sites keep working.
    if os.environ.get("RSNA_K_EVAL"):
        env["RSNA_K_EVAL"] = os.environ["RSNA_K_EVAL"]
    if os.environ.get("RSNA_CKPT"):
        env["RSNA_CKPT"] = os.environ["RSNA_CKPT"]
    print(f"[build] backbone={backbone} ms={max_studies} ep={epochs} folds={folds} "
          f"submit={do_submit} seed={seed} mode={mode} shard={shard}/{n_shards}")
    subprocess.run([sys.executable, BUILD], cwd=REPO, env=env, check=True)

    # 2) rewrite metadata with the experiment id/title
    with open(META) as f:
        meta = json.load(f)
    meta["id"] = kernel_id
    meta["title"] = title
    if mode == "corpus_dump":
        # CPU-only decode: no GPU quota, no extra datasets (competition DICOMs only).
        # machine_shape=NvidiaTeslaT4 with enable_gpu=false still ate a GPU session
        # on p0, so drop the shape key entirely.
        meta["enable_gpu"] = False
        meta.pop("machine_shape", None)
        meta["dataset_sources"] = []
    elif mode == "corpus_infer":
        # Inference only: mount the WEIGHTS and the competition DICOMs. No train
        # corpus, no dump kernels -- run_corpus_infer never calls open_corpus, it
        # decodes the hidden test itself. Keeping the mounts minimal is the point:
        # this kernel is meant to be cheap enough to submit freely.
        meta["enable_gpu"] = True
        meta["machine_shape"] = "NvidiaTeslaT4"
        meta["kernel_sources"] = []
        meta["dataset_sources"] = [x.strip() for x in
                                   os.environ.get("RSNA_WEIGHT_DATASETS", "").split(",")
                                   if x.strip()] or meta.get("dataset_sources", [])
    else:
        meta["enable_gpu"] = True
        meta["machine_shape"] = "NvidiaTeslaT4"
        # corpus-mode sources are owned by kernel-metadata.json (C4 mounts dump
        # kernels + coatnet weights + LLM labels, NOT the 44-slice public corpus).
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] id={kernel_id} title={title!r} gpu={meta.get('enable_gpu')}")

    # 3) push
    # run.py is written UTF-8 (report_extractor vocab has accented literals);
    # force the kaggle CLI (Python) into UTF-8 mode so it does not read the
    # script with the Windows GBK default and choke on non-ASCII bytes.
    push_env = dict(os.environ)
    push_env["PYTHONUTF8"] = "1"
    r = subprocess.run([KAGGLE, "kernels", "push", "-p", KERNEL_DIR],
                       capture_output=True, text=True, env=push_env)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    # The kaggle CLI prints "Kernel push error: ..." and still exits 0 (seen with
    # "Maximum batch GPU session count of 2 reached"), so returncode alone is not a
    # success signal -- a failed push must not be logged as a push.
    if r.returncode != 0 or "error" in out.lower() or "successfully pushed" not in out.lower():
        print("PUSH FAILED", r.returncode)
        sys.exit(1)
    print(f"PUSHED {kernel_id}")


if __name__ == "__main__":
    main()
