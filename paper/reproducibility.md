# 复现信息

> 2026-10-04 更新到 LOG #539（B3 / B2 / S4 进行中）。

## 自研出分核与提交

| 实验 | 出分核 | 版本 | 提交 ref | LB |
|---|---|---|---|---|
| C6 | `easonyanyan/rsna-knee-c6-infer-64slice` | v1 | 56101443 | 0.935 |
| C7（完整 16 轮） | `easonyanyan/rsna-knee-c7-infer-64slice` | v3 | 56188738 | 0.928 |
| C6 × 公开 fork | `easonyanyan/rsna-knee-0941-c6-rankfuse` | v1 | 56157012 | 0.942 |
| C6 + C8 | `easonyanyan/rsna-knee-c6-c8-heterogeneous-fuse` | v1 | 56191506 | 0.930 |
| C6 + C9b | `easonyanyan/rsna-knee-c6-c9-fuse` | v1 | 56197449 | 0.931 |
| (C6 + C9b) × 公开 fork | `easonyanyan/rsna-knee-0941-c69-rankfuse` | v1 | 56199028 | 0.939 |
| C6 + 合法评估窗口（实验 2a） | `easonyanyan/rsna-knee-c6-legalwin-infer` | v1 | 56199367 | 0.933 |
| C6r（方案 2 对照） | `easonyanyan/rsna-knee-c6r-infer` | v1 | 56208686 | 0.933 |
| C6w（实验 2，判据：top-3 平均） | `easonyanyan/rsna-knee-c6w-topk-infer` | v1 | 56214265 | 0.932 |
| C6c（实验 3，判据：top-3 平均） | `easonyanyan/rsna-knee-c6c-topk-infer` | v1 | 56217799 | 0.927 |
| C6c（实验 3，记录：最佳单轮） | `easonyanyan/rsna-knee-c6c-infer` | v1 | 56217800 | 0.928 |
| C6s（实验 5，判据：top-3 平均） | `easonyanyan/rsna-knee-c6s-topk-infer` | v1 | 56444939 | 0.933 |
| C6s（实验 5，记录：最佳单轮） | `easonyanyan/rsna-knee-c6s-infer` | v1 | 56444958 | 0.933 |
| 0.943+ SOTA 整合核（参照，非单变量） | `easonyanyan/rsna-knee-sota-0943-speedy-raptors` | v1 | 56440322 | 0.942 |
| SOTA 整合核 × C6 rank 融合（策略 1） | `easonyanyan/rsna-knee-sota-0943-c6-rankfuse` | v1 | 56441002 | 0.942 |
| Dinosaurs v38 原样复现（公开核，非单变量） | `easonyanyan/rsna-knee-dinosaurs-v38` | v1 | 56452824 | 0.940 |
| 两线融合（策略 2，**已构建未推送**：9 h 上限不成立） | `easonyanyan/rsna-knee-sota-0943-dino-rankfuse`（本地目录） | — | — | — |
| **0.943 基底：Speedy Raptors 原版**（3 读者） | `easonyanyan/rsna-knee-speedy-original-0922` | — | 56469339 | 0.943 |
| 基底 + RadImageNet 放回 Fracture | `easonyanyan/rsna-knee-sota-0943-frac-include` | — | 56469365 | 0.942 |
| 基底，外层默认 0.60 | `easonyanyan/rsna-knee-speedy-outer-060` | — | 56469491 | 0.942 |
| 基底，平权（去掉 5 处覆盖） | `easonyanyan/rsna-knee-speedy-flat-065` | — | 56469522 | 0.941 |
| 4 读者（+Repair-v1），外层 0.70 | `easonyanyan/rsna-knee-speedy-repairv1-0925` | — | 56558062 | 0.943 |
| 基底 + 无重训 A5 深度 TTA | `easonyanyan/rsna-knee-dinosaur-v5-0925` | — | 56558284 | ERROR（> 9 h） |
| 4 读者，外层 0.65（原主选，2026-10-03 起被 56786075 取代） | `easonyanyan/rsna-knee-speedy-repairv1-outer-065` | — | 56558421 | 0.943 |
| **4 读者，平权（最终对冲）** | `easonyanyan/rsna-knee-speedy-repairv1-flat-065` | — | 56558589 | 0.942 |
| 5 读者（+C6），族归约**被静默降级** | `easonyanyan/rsna-knee-speedy-5-way-consensus-0926` | v1 | 56569299 | 0.943（混淆） |
| 5 读者平权，族归约**被静默降级** | `easonyanyan/rsna-knee-speedy-5-way-flat-065` | v2 | 56570186 | 0.935（混淆） |
| 5 读者（修正 npz dtype） | `easonyanyan/rsna-knee-speedy-5-way-consensus-fix-0926` | v1 | 56591530 | 0.943 |
| 5 读者平权（修正） | `easonyanyan/rsna-knee-speedy-5-way-flat-065-fix-0926` | v1 | 56591535 | 0.942 |
| 4 读者平权 + 强制旧式秩混合（对照） | `easonyanyan/rsna-knee-speedy-flat065-rankmix-ctl-0926` | v1 | 56591631 | 0.941 |
| C6k（图像老师蒸馏，top-3 平均） | `easonyanyan/rsna-knee-c6k-topk-infer` | v1 | 56628896 | 0.931 |
| C10 × 0.943（W = 0.10，fail-closed） | `easonyanyan/rsna-knee-speedy-c10-fuse010` | v1 | 56741332 | 0.942 |
| 探测 P1：保留 4 个弱标签 | `easonyanyan/rsna-knee-speedy-probe-p1-weak4` | v1 | 56742278 | 0.636 |
| 探测 P2：只保留 Synovitis | `easonyanyan/rsna-knee-speedy-probe-p2-synovitis` | v1 | 56742314 | 0.534 |
| 探测 P3：只保留 PF OA | `easonyanyan/rsna-knee-speedy-probe-p3-pfoa` | v1 | 56742378 | 0.533 |
| PF-v1 单列探测（PF OA 专科） | `easonyanyan/rsna-knee-pf1-probe` | v1 | 56772338 | 0.532 |
| C10 PF 单列对照 | `easonyanyan/rsna-knee-c10-pf-probe` | v1 | 56776462 | 0.530 |
| PF 列融合探测（栈 ⊕ PF-v1，各 0.5） | `easonyanyan/rsna-knee-speedy-pf1-fuse050-probe` | v1 | 56778078 | 0.534 |
| PF-v2 单列探测（只训 PF，全部窗口） | `easonyanyan/rsna-knee-pf2-probe` | v1 | 56780996 | 0.531 |
| **PF 融合整版（最终主选）** | `easonyanyan/rsna-knee-speedy-pf1-fuse050` | v1 | 56786075 | 0.943 |
| C10 单模（换骨干） | `easonyanyan/rsna-knee-c10-solo` | v1 | 56801517 | 0.912 |
| B3：C8 单模 | `easonyanyan/rsna-knee-c8-solo` | v1 | 56813677 | 0.834 |
| B3：C9b 单模 | `easonyanyan/rsna-knee-c9b-solo` | v1 | 56813681 | 待出分 |
| B3：C11 单模（平均池化，核内断言） | `easonyanyan/rsna-knee-c11-solo` | v1 | 56813796 | 待出分 |

**不提交的诊断核**（只读比赛数据，CPU）：`rsna-knee-gold-redecode` v2（58 金标按测试路径重解码，与训练语料逐字节比对，#515）、`rsna-knee-sag-geometry` v1（矢状几何，#522）、`rsna-knee-laterality-headers` v1（4407 个 study 的膝别头信息，#531）。

版本一栏为"—"的，以 Kaggle 提交描述里写的版本为准。构建脚本：5 读者修正版与对照见 `scratchpad/build_0926_batch.py`，C10 融合见 `scratchpad/build_c10_fuse.py`，探测见 `scratchpad/build_label_probes.py`；PF 融合探测 / 整版见 `scratchpad/build_pf1_fuse_probe.py` / `build_pf1_fuse_full.py`；B3 三个单模见 `scratchpad/build_b3_solo_kernels.py`（由 `rsna-knee-c10-solo` 派生）。单列 / C10 单模类核由 `kaggle_kernel/_build.py` 打包当前 `src`，再追加后处理函数。

## 权重数据集

- `easonyanyan/rsna-knee-c6-weights`、`rsna-knee-c7-weights`
- `easonyanyan/rsna-knee-c6s-weights`：NEW4 软标签重训模型权重（含 c6s_topk_avg.pt 与 c6s_corpus0.pt）
- `easonyanyan/rsna-knee-c8-weights`：含 `resnet50_radimagenet.pt`，由 `marwanmath/resnet-50-radimagenet-marwan` 重映射键名得到
- `easonyanyan/rsna-knee-c9-weights`：含官方 `dinov2_vitb14.pth` 与 C9b 权重（文件名 `c9_corpus0.pt`）
- `easonyanyan/timm-coatnet-rmlp2-384`
- `easonyanyan/rsna-knee-c6k-weights`：`c6k_topk_avg.pt`（`8ce66895…`）、`c6k_corpus0.pt`（`a0a8b58a…`）
- `easonyanyan/rsna-knee-c10-weights`：`c10_topk_avg.pt`（`9fbba7c7…`）。Kaggle 端不需要 ConvNeXt 的 timm 预训练权重，因为完整的 state_dict 会被严格加载；R9700 本地存于 `data/timm_weights/convnext_small_in22k_ft_in1k_384.pt`（`0d61401f…`）
- `easonyanyan/rsna-knee-c6r-weights`：C6r 对照权重（B5 的对照探测直接用）
- `easonyanyan/rsna-knee-pf1-weights`：`pf1_topk_avg.pt`（`ef779c2b…`），checkpoint 记录 `centre_allow`（20 个中央矢状中心）与 `train_labels`
- `easonyanyan/rsna-knee-pf2-weights`：`pf2_topk_avg.pt`（`1be682e7…`）
- `easonyanyan/rsna-knee-c11-weights`：`c11_topk_avg.pt`（`90364e43…`）。**checkpoint 未记录池化方式**，推理时必须设 `RAPTOR_POOL = "mean"`（`rsna-knee-c11-solo` 核内设置并断言）
- 蒸馏老师：`data/labels/teacher_dino_rad_oof.parquet`（`b046f4f7…`），由 `pilkwang/rsna-knee-weights/oof.npz` 与 `antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv` 生成

## 关键哈希（SHA-256 前缀）

| 文件 | 前缀 | 用途 |
|---|---|---|
| C6 推理脚本 | `3ce7f829…` | 0.935 出分代码，嵌入 C6 × fork 融合核 |
| 公开 Probe #22 fork 笔记本 | `01f2d834…` | 0.941 原样复现 |
| C6 + C9b 融合核 | `c00a7514…` | 训练前预注册的等权融合 |
| (C6 + C9b) × fork 融合笔记本 | `47031338…` | 方案 3 |
| `src/corpus.py`（C9b 训练时） | `95141ce6…` | 含 `build_param_groups` 与按 arch 建骨干 |
| `src/r9700_opt.py` | `ed0a1d48…` | NativeDW 深度卷积 |
| 0.943+ SOTA 整合核输出（策略 1 的 SOTA 臂） | `81a4a26c…` | 融合输入之一 |
| C6 臂输出（策略 1） | `f6c197ee…` | 融合输入之二 |
| 策略 1 融合输出 submission.csv | `5ec73a0a…` | 提交 56441002 的内容 |
| Speedy Raptors 原版笔记本 | `60bbe1af…` | 0.943 基底（56469339） |
| 4 读者外层 0.65 笔记本 | `59e34d72…` | 最终主选（56558421） |
| 4 读者平权笔记本 | `4fda44e9…` | 最终对冲（56558589） |
| 5 读者修正版笔记本 | `9320a80a…` | 56591530 |
| C10 融合笔记本 / 内嵌 C10 推理脚本 | `7109b4ed…` / `78399b24…` | 56741332 |
| PF 融合整版笔记本 / 内嵌 PF-v1 臂 | `54d263c7…` / `5ee0ef18…` | 56786075（最终主选） |
| PF 融合探测笔记本 | `20dfc9aa…` | 56778078 |
| C10 PF 探测 / PF-v2 探测脚本 | `13d1047a…` / `063a4e86…` | 56776462 / 56780996 |
| C10 / C8 / C9b / C11 单模脚本 | `0ee18b20…` / `02a2978d…` / `4c1cefd8…` / `75193088…` | 56801517 / 56813677 / 56813681 / 56813796 |
| 诊断核：重解码 / 矢状几何 / 膝别头信息 | `6d2467fa…` / `de851276…` / `29dfbf55…` | 不提交 |
| S4 训练代码（两端一致） | `train_s4_local.py` `8c91f236…`，`corpus.py` `e2c61da1…`，`raptor.py` `14a5d1bb…`，`config.py` `3bf8cc88…` | #525 |

## 本地训练入口

`src/train_c6_local.py`、`train_c7_local.py`、`train_c8_local.py`、`train_c9_local.py`（失败配置，保留作记录）、`train_c9b_local.py`、`train_c6r_local.py`、`train_c6s_local.py`、`train_c6k_local.py`（蒸馏）、`train_c10_local.py`（ConvNeXt）、`train_c11_local.py`（ConvNeXt + 全局平均池化）、`train_pf1_local.py`（PF OA 专科：只训 PF + 中央矢状窗口）、`train_pf2_local.py`（只训 PF，全部窗口）、`train_s4_local.py`（S4 有序窗口，`RSNA_S4_ARM=T|K`）、`train_c6b_local.py`（B5：二值化目标）。`train_c10_local.py` / `train_pf1_local.py` 支持 `RSNA_SEED`（B4，默认 42）。所有脚本在启动时断言关键超参，并把实际解析到的配置打印进 `train.log`。
