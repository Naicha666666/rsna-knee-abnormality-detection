# RSNA Knee — 最新核心 Notebook 指南与交付清单

> 更新时间：2026-10-14 (UTC 2026-10-15) ｜ 比赛截止：2026-10-22 23:59 UTC (剩余 8 天) ｜ 主账号：`easonyanyan`

本文档整理当前赛题在公开榜与私榜对冲场景下的**最新核心 Notebook 资产**。所有 Notebook 均经过严格的 22 项离线/在线门禁断言自检，确保无静默回退（0 FALLBACK）、进程零异常、逐标签单调性及双阶段/多阶段逻辑实装。

---

## 1. 核心 Notebook 清单

| Notebook 文件 | Kaggle Kernel Ref | 架构定位 | 核心亮点 | 推荐用途 |
|---|---|---|---|---|
| [`rsna-knee-09573-replicate.ipynb`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/notebooks/rsna-knee-09573-replicate.ipynb) | `easonyanyan/rsna-knee-09573-replicate` | **0.9573 官方对齐核** (双尺度公开 SOTA 复刻) | - **Stage 1 (384)**: CoAtNet-2 384 原生<br>- **Stage 1b (224)**: CoAtNet-2 224 独立进程 (`CROP_FRAC=0.8` 裁剪缩放)<br>- **Stage 2**: R7 ConvNeXt 4 折 + 双 Stacker<br>- **Stage 3**: 秩空间融合 $0.5 \times R7 + 0.5 \times (0.65 R_{384} + 0.35 R_{224})$ | **主版候选 1 / 跑分核**：完全对齐 heliosli 0.9573 真实增量 |
| [`rsna-knee-0957-hedge-ms-ep8-dualgroup.ipynb`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/notebooks/rsna-knee-0957-hedge-ms-ep8-dualgroup.ipynb) | `easonyanyan/rsna-knee-0957-hedge-ms-ep8-dualgroup` | **私榜去同质化对冲核** (两阶段结构异构) | - **Stage 1**: 三臂多尺度融合 (Raptor 384@320 w0.55 + c08 224@224 w0.15 + 自研 Ep8 CoAtNet-2 w0.30)<br>- **Stage 2**: 双组独立网络 (`r7` + `compact-v1`) 原生 logit 混合，**剥离脆弱拟合式 Stackers**<br>- **Stage 3**: 0.50 / 0.50 稳健融合 | **Final 2 对冲必备**：打破公开同分 fork 墙，规避私榜过拟合风险 |

---

## 2. 详细配置与数据集挂载要求

### A. `rsna-knee-09573-replicate.ipynb`
- **代码文件路径**: [`notebooks/rsna-knee-09573-replicate.ipynb`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/notebooks/rsna-knee-09573-replicate.ipynb)
- **Kaggle 对应目录**: [`kaggle_kernel/submissions/easonyanyan-rsna-knee-09573-replicate/`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/kaggle_kernel/submissions/easonyanyan-rsna-knee-09573-replicate/)
- **所需数据集**:
  1. `heliosli/rsna-knee-compact-r7cnxfsss` (R7 ConvNeXt 4 折权重与 Stackers)
  2. `nartaa/rsna-knee-publication-swa-weights-20261007` (含 384 与 224 SWA 权重)
  3. `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (预训练底座)
  4. `rsna-knee-abnormality-detection` (官方竞赛数据)
- **硬件配置**: GPU (T4 x 2), Internet: Off

### B. `rsna-knee-0957-hedge-ms-ep8-dualgroup.ipynb`
- **代码文件路径**: [`notebooks/rsna-knee-0957-hedge-ms-ep8-dualgroup.ipynb`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/notebooks/rsna-knee-0957-hedge-ms-ep8-dualgroup.ipynb)
- **Kaggle 对应目录**: [`kaggle_kernel/submissions/easonyanyan-rsna-knee-0957-hedge-ms-ep8-dualgroup/`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/kaggle_kernel/submissions/easonyanyan-rsna-knee-0957-hedge-ms-ep8-dualgroup/)
- **所需数据集**:
  1. `heliosli/rsna-knee-compact-r7cnxfsss`
  2. `nartaa/rsna-knee-publication-swa-weights-20261007`
  3. `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`
  4. `easonyanyan/rsna-knee-c6-warmstart-ep8-weights` (自研 0.948 单模 Ep8 权重)
  5. `heliosli/rsna-knee-compact-v1` (第二组独立训练模型)
  6. `rsna-knee-abnormality-detection` (官方竞赛数据)
- **硬件配置**: GPU (T4 x 2), Internet: Off

---

## 3. 对应文档与归因分析
- **0.9573 逐行 Diff 与几何归因**: [`RSNA_Knee_09573_Replication_2026-10-14.md`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/RSNA_Knee_09573_Replication_2026-10-14.md)
- **13 笔探针中性归因与 Final-2 收官计划**: [`RSNA_Knee_Saturation_Attribution_Final2_Plan_2026-10-14.md`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/RSNA_Knee_Saturation_Attribution_Final2_Plan_2026-10-14.md)
- **整体赛题推进日志**: [`LOG.md`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/LOG.md)
