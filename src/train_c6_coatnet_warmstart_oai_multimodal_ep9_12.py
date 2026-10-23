"""CoAtNet-2 384 Warm-Start Post-Distillation Stage 3 (Epochs 9-12) on AMD R9700.

Pretrained Base:
- Checkpoint from Stage 2 Peak (Gold-58 AUC = 0.9186, Loss = 0.7226):
  experiments/local_c6_coatnet_warmstart_oai_multimodal_ep5_8/corpus0_best_epoch.pt

Supervision:
- labels_4407_multimodal_refined.parquet (Claude Haiku 5.5 Audited Targets)

Training Protocol:
- Epochs: 4 Additional Epochs (representing total epochs 9-12, ~70 min)
- Learning Rates: Backbone 2.5e-6, Head 2.5e-5 (fine cosine basin refinement)
- Corpus64: 4,407 studies, 64 slices, 384px, batch=2, k_train=12, k_eval=42
- Output: experiments/local_c6_coatnet_warmstart_oai_multimodal_ep9_12/
"""
from __future__ import annotations
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# AMD Radeon AI PRO R9700 (GPU 0)
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")

import tv_stub
tv_stub.install()

import config

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Additional 4 Epochs Fine-Tuning for CoAtNet-2 384 (Epochs 9-12)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of fine-tuning epochs (default: 4)")
    parser.add_argument("--seed", type=int, default=44, help="Seed for data ordering (default: 44)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch quick check")
    parser.add_argument("--budget", type=float, default=6.0, help="Time budget in hours")
    args = parser.parse_args()

    out_dir = "/home/eason/rsna-knee/experiments/local_c6_coatnet_warmstart_oai_multimodal_ep9_12"
    if args.smoke:
        out_dir += "_smoke"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 模型架构配置（对齐 Nartaa 0.949 Base）
    config.RAPTOR_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    config.RAPTOR_RES = 384
    config.CALIBRATE_SYNOVITIS = False

    # 2. 热启动权重路径 (使用 Stage 2 峰值 checkpoint: Ep8 best_epoch)
    config.RAPTOR_INIT_CKPT = "/home/eason/rsna-knee/experiments/local_c6_coatnet_warmstart_oai_multimodal_ep5_8/corpus0_best_epoch.pt"

    # 3. 监督目标：载入全新生成的多模态微调全量标签
    config.SOFT_LABEL_PARQUETS = []
    config.TEACHER_PARQUET = "labels_4407_multimodal_refined.parquet"
    config.TEACHER_LAMBDA = 1.0

    # 4. 微调超参数：更温和的学习率，进入平坦最优盆地
    config.RAPTOR_LR_BACKBONE = 2.5e-6
    config.RAPTOR_LR_HEAD = 2.5e-5

    config.CORPUS_K_TRAIN = 12
    config.CORPUS_K_EVAL = 42
    config.CORPUS_EPOCHS = 1 if args.smoke else args.epochs
    config.CORPUS_BATCH = 2
    config.CORPUS_LEGAL_WINDOWS = False

    from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS

    print("=" * 85, flush=True)
    print("LAUNCHING CoAtNet-2 384 STAGE 3 FINE-TUNING (EPOCHS 9-12) ON AMD R9700", flush=True)
    print(f"Pretrained Weights: {config.RAPTOR_INIT_CKPT}", flush=True)
    print(f"Refined Supervised Labels: {config.TEACHER_PARQUET}", flush=True)
    print(f"Backbone LR: {config.RAPTOR_LR_BACKBONE} | Head LR: {config.RAPTOR_LR_HEAD}", flush=True)
    print(f"Fine-Tuning Epochs: {config.CORPUS_EPOCHS} (Total Epochs 9-12) | Batch Size: {config.CORPUS_BATCH}", flush=True)
    print(f"Output Directory: {out_dir}", flush=True)
    print("=" * 85, flush=True)

    # 校验数据集
    vols, masks, ids = open_corpus()
    assert vols.shape == (4407, 64, 336, 336), f"Corpus shape mismatch: {vols.shape}"
    print(f"Corpus verified: {vols.shape} uint8 memmap, 4407 studies", flush=True)
    del vols, masks, ids

    run_corpus_submit(
        epochs=config.CORPUS_EPOCHS,
        max_studies=40 if args.smoke else None,
        batch=config.CORPUS_BATCH,
        k_train=config.CORPUS_K_TRAIN,
        k_eval=12 if args.smoke else config.CORPUS_K_EVAL,
        out_dir=out_dir,
        seed=args.seed,
        do_submit=False,
        time_budget_h=args.budget,
    )
    print("=" * 85, flush=True)
    print(f"Stage 3 (Epochs 9-12) 加训完成！产物保存在: {out_dir}", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()
