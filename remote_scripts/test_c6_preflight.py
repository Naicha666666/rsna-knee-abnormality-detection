"""Preflight verification for C6 training on R9700 (ROCm).
Tests data loading, memmap, 4-LLM soft labels, and CoAtNet backbone.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src") if os.path.isdir(os.path.join(HERE, "src")) else HERE
sys.path.insert(0, SRC)

os.environ["RSNA_DATA_ROOT"] = "/home/eason/rsna-knee/data"
os.environ["RSNA_CORPUS_DIR"] = "/home/eason/rsna-knee/data/corpus64"
os.environ["RSNA_CORPUS_SLOTS"] = "64"

import tv_stub
tv_stub.install()

import config
config.SOFT_LABEL_PARQUETS = [
    "labels_kimi_soft.parquet",
    "labels_qwen_soft.parquet",
    "labels_grok_soft.parquet",
    "labels_gemma4_soft.parquet",
]

import torch
print(f"PyTorch version: {torch.__version__}, ROCm device: {torch.cuda.get_device_name(0)}")

from corpus import open_corpus, CORPUS_SLOTS, CORPUS_D
print(f"Corpus D={CORPUS_D}, slots={len(CORPUS_SLOTS)}: {CORPUS_SLOTS}")
vols, masks, ids = open_corpus()
print(f"Opened corpus shape: {vols.shape}, ids count: {len(ids)}")
del vols, masks, ids

from dataset import RSNADataset
ds = RSNADataset("train")
print(f"Dataset uids: {len(ds.uids)}, targets shape: {ds.targets.shape}")
print(f"Target sample 0: {ds.targets[0][:4]}")

from raptor import build_raptor_backbone
m = build_raptor_backbone()
print(f"Backbone loaded successfully: {m.__class__.__name__}")
del m

print("=== ALL PREFLIGHT CHECKS PASSED SUCCESSFULLY ===")
