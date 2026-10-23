# 最终提交资产快照（2026-10-03，LOG #508）

> 冻结两份最终提交所对应的全部本地资产，**之后不再修改**。本地文件已设为只读（Windows `attrib +R`）；确需改动时，先在 LOG 里写明理由。
> 已出分的提交在 Code Competition 里同时算好了公榜和私榜，所以这些资产只用于复现和审计，不影响分数。

## 主选 56786075（公榜 0.943）

| 项 | 值 |
|---|---|
| 核 | `easonyanyan/rsna-knee-speedy-pf1-fuse050` v1，notebook，`machine_shape` NvidiaTeslaT4，断网 |
| 本地目录 | `kaggle_kernel/submissions/rsna-knee-speedy-pf1-fuse050/` |
| notebook SHA-256 | `54d263c7ce0979bdb6e03a7925b855eaed6994b2d6303aa3b839659009ed4bd0` |
| 构建脚本 | `scratchpad/build_pf1_fuse_full.py`（SHA `e7e65f2d…`） |
| 父本 | `rsna-knee-speedy-repairv1-outer065`（56558421），24 格逐字节保留 |
| PF-v1 臂 payload | `kaggle_kernel/submissions/rsna-knee-pf1-probe/rsna-knee-pf1-probe.py`，SHA `5ee0ef18b8524610cf5318364dfa319e4487117903d3fdca1ed4b95d838061cf`（gzip + b64 内嵌，运行时校验） |
| PF-v1 权重 | `data/pf1_weights/pf1_topk_avg.pt`，SHA `ef779c2bd91e090d102f0b412a9c8dfcd8402358497b01879d8d4504b598c8ff` → `easonyanyan/rsna-knee-pf1-weights` |
| PF-v1 训练 | `src/train_pf1_local.py`；`src/corpus.py` SHA `1f35c298…`，`src/config.py` `98275139…`（训练时两端一致） |
| 融合规则 | PF OA = 0.50·pct_rank(父本) + 0.50·pct_rank(PF-v1)；其余 11 列逐值等于父本 |
| 外部依赖 | 14 个公开数据集（另加自有的 `rsna-knee-pf1-weights`）+ `sofiaanjenje/rsna-knee-e11-train`、`e13-train` 两个公开核输出 + `metaresearch/dinov2` small |
| 隐藏集用时 | 约 8 h（2026-10-03 00:56 UTC 提交） |
| 可见集输出 SHA | `8f0c5e7b…`（3 行，与父本相同） |

## 对冲 56558589（公榜 0.942）

| 项 | 值 |
|---|---|
| 核 | `easonyanyan/rsna-knee-speedy-repairv1-flat-065` v1，notebook，`machine_shape` NvidiaTeslaT4，断网 |
| 本地目录 | `kaggle_kernel/submissions/rsna-knee-speedy-repairv1-flat065/` |
| notebook SHA-256 | `4fda44e9dbc2e1faa1e4f11347923874a96934ccd58f8357ce318822c79d8e04` |
| 内容 | 4 个 CoAt 读者 + DINOv2 / RadImageNet，12 个标签外层全为 0.65（无 5 处覆盖、无 PF 融合） |
| 外部依赖 | 14 个公开数据集 + 同上两个公开核 + `metaresearch/dinov2` small |
| 记录 | LOG #444–#447 |

## 被取代的原主选 56558421（公榜 0.943，作对照保留）

| 项 | 值 |
|---|---|
| 核 | `easonyanyan/rsna-knee-speedy-repairv1-outer-065` v1 |
| notebook SHA-256 | `59e34d72dbe90905b1cc1aa9dc99956c63c377c1d2d87a61f10be20a943ea75e` |

## 已知风险（写明）

- **外部资产不在我们控制之内**：公开数据集和公开核输出若被作者删除或更新，这两份核将无法逐字节重跑。分数不受影响，因为已经算好。复现时以本地目录与 SHA 为准。
- **勾选状态无法用 API 核对**；用户已于 2026-10-03 在 Kaggle 提交页确认勾选 56786075 和 56558589（LOG #510）。
