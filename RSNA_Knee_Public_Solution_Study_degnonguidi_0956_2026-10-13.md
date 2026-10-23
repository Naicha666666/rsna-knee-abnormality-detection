# 公开方案解剖：`degnonguidi/rsna-knee-abnormality-detection`（公开 LB **0.956**）

> 分析时间 2026-10-13。目的：判断这篇公开 notebook 是否含**我们尚未掌握的技术**，以及它对我们 0.956 位置的含义。
> 拉取源：`scratchpad/study_degnonguidi/`（`kaggle kernels pull degnonguidi/rsna-knee-abnormality-detection`）。对比基准：我们的私有逐字复刻 `snowy615/rsna-knee-heliosli-0957-verbatim`（= heliosli 的私有 candidate，sha256 `053d933027ab718d`）。

---

## 1. 结论速览（一句话）

**这不是一个新方案，而是 heliosli 那篇私有 0.957-candidate notebook 的「公开再发布版」——功能代码逐字节相同，公开得分 0.956。**

- 它**没有引入任何我们没见过的新技术**。README 里吹的 "M4 OA extent heads / P2 spatial attention priors / synovitis decomposition / Bayesian calibration stackers" 全都是 **r7 权重包内部本来就有的东西**，不是这篇新加的方法。
- 它是**同一套 pipeline 的第 3 次独立运行**：heliosli 原版宣称 0.957 → 我们逐字复刻 **0.956** → degnonguidi 公开版 **0.956**。→ **该 pipeline 在公开榜的真实上限 ≈ 0.956。**

---

## 2. 与我们的逐字复刻版逐单元 diff（唯一差异只有 1 处代码 + README）

| 单元 | 我们的 verbatim | degnonguidi | 是否相同 |
|---|---|---|---|
| cell 0/1 markdown | heliosli 原版短头（"RSNA Knee: R7 + Raptor / Private candidate"） | 重写的 "Dual-Engine SOTA Ensemble (LB 0.956)" 营销式 README | ✗ 仅文案 |
| cell 2 `raptor_run.py`（96,957 字符） | — | — | **✓ 逐字节相同** |
| cell 3 Raptor 驱动 | — | — | **✓ 逐字节相同** |
| cell 4 配置头（`ARM`/`BUNDLE_ID`/`CODE_SHA`/`SOURCE_SHA`/`SOURCES`） | — | — | **✓ 逐字节相同** |
| cell 5 ours stage（`probe_dual.py` 驱动 + receipt） | — | — | **✓ 逐字节相同** |
| cell 6 Stage-3 融合 | — | — | **✓ 逐字节相同** |
| `dataset_sources` | `r7cnxfsss` + `nartaa SWA` + `orthofoundation-fold0-ep7` | 同 | **✓ 相同** |
| `enable_gpu/internet/machine` | True/False/T4 | 同 | **✓ 相同** |

**配置逐项核对（两边完全一致）**
- `ARM = 'E_lrprior'`
- `BUNDLE_ID = '8c7b666e27dc0e8f'` ← **r7 bundle**（与 heliosli 0.957 同一枚）
- `C_SLUG = 'rsna-knee-compact-r7cnxfsss'`
- `SOURCE_SHA` 7 个 worker 文件 hash **全等**（含 `probe_dual.py = a6155eb0…`）
- `CODE_SHA` 8 个 orthofoundation 文件 hash **全等**
- Stage-3：`W_RAPTOR, W_OURS = 0.5, 0.5`；`W_RAPTOR_C = {'Medial OA':0.5,'Lateral OA':0.5,'PF OA':0.5}`（**与默认同值 → 仍是空操作**）
- 两个 stacker 走 `r7` 包里的 `fluid_stacker.json` / `state_stacker.json`，**与我们所读的是同一份**

---

## 3. 唯一的代码改动：`find_mount` 改成接受「slug 列表」

cell 4 尾部 179 字符的差异，全部集中在这里：

```diff
-def find_mount(slug, required):
-    cands = [INPUT / slug, INPUT / 'competitions' / slug] ...
-    if len(found) != 1: raise RuntimeError(...)
+def find_mount(slugs, required):
+    if isinstance(slugs, str): slugs = [slugs]
+    for slug in slugs: ...          # 依次尝试每个 slug
+    if len(found) < 1: raise RuntimeError(...)

-M_SRC = find_mount(C_SLUG, 'manifest.json')
+M_SRC = find_mount([C_SLUG, 'rsna-knee-compact-r7cnxfsss', 'rsna-knee-compact-v1'], 'manifest.json')
```

**判读：这是「挂载鲁棒性」补丁，不是模型改动。**
- 动机：公开 notebook 的挂载路径因数据集嵌套方式不同而异，作者加了多候选 slug 兜底。
- **但 `rsna-knee-compact-v1`（= r6）这条兜底是死代码**：紧随其后有
  `assert manifest['bundle_id'] == BUNDLE_ID`，而 `BUNDLE_ID='8c7b666e27dc0e8f'` 是 r7 的 bundle_id。
  若真回退到 compact-v1（bundle_id `209a32c8b50a6b1c`）→ **断言直接崩**。所以它只可能加载 r7。
- 结论：**行为与我们完全相同。**

---

## 4. 对我们策略的三点含义

1. **技术上零增量。** 不需要移植任何东西——我们手上的 verbatim 已经是同一份代码，且我们还有它没有的**可打补丁能力**（cell 6 是活代码、builder 支持 4 个 knob）。
2. **0.956 是这套公开 pipeline 的诚实上限。** 三个独立运行点：0.957(宣称)/0.956(我们)/0.956(degnonguidi)。**继续在同一套 r7 权重上做融合/调权重已被反复证伪**（见 MEMORY「已判定死亡路线」），这次是第 4 个佐证。
3. **⚠ 奖牌线会收紧。** 完整配方（r7 包 + 逐字 pipeline）现在**公开可 fork**：任何人 attach `heliosli/rsna-knee-compact-r7cnxfsss` 跑这篇 → 0.956。这会**推高银牌阈值**（当前银 0.954 / 铜 0.954 / 金 0.963）。我们的相对位置不会自动变差，但「0.956 就安全」的假设要打折。

---

## 5. 行动建议（不变）+ 出分更新（10-13 10:05 PDT）

**我们的增量必须来自「配方之外」，而不是重跑配方。** 目前唯一有统计证据支撑的公开内杠杆曾是这个：

- **`stk-latoa`**（翻 `state_stacker['classes']['Lateral OA']['adopt']` False→True）：corrected-ruler 交叉拟合增益 **+0.0064**（CI90 [+0.0037,+0.0094]，2000-bootstrap 从未变负），仅因预登记 explicit 尺子门槛被误杀；行已拟合齐备，零重训。**已提交 `57127697`（snowy615）→ 出分 `0.956`（中性）。**
  - 若 +0.0064 兑现，`Δmacro = +0.0064/12 = +0.00053` → LB 应变 0.9565 → 显示 **0.957**。实测 **0.956** → **增益未兑现**。
  - **结论：预登记的 explicit-ruler 门禁是对的；corrected-ruler 增益是过拟合。最后一个公开内杠杆关闭。**
- 配套诊断 `ablate-latoa`（`57127701`）→ **0.922** → `AUC(Lateral OA) = 0.5+12×(0.956−0.922) =` **0.908**（真值 ∈ [0.902,0.914]）。独立验证同一列。
- `stk-latoa-medoa`（`57150216`，PENDING）；`bakersoff`（snowy615 新建，已推 v1，守护 `Ipc9ev`）。

**不要做**：不要再尝试「换 r6 / 三路 compact-v1 融合 / 调 0.5:0.5 权重 / 嫁接 0.954 骨架 / 用 corrected-ruler 增益翻 adopt 门」——全部为空操作、负收益或已实测中性。

---

## 6. 附：本篇相对 heliosli 原文的「公开化」意义

- heliosli 的原 notebook 标为 **private candidate**（我们的 cell 0 就是它的原文）；degnonguidi 把它**改头换面公开**。
- 也就是说 **0.957 配方现在已完全公开**。这解释了为什么「公开侧已枯竭」——不是没东西可抄，而是**能抄的都已经被抄完了**，天花板就是 0.956。
- 副产品：这也意味着我们**不必担心自己的 verbatim 私有**——同样的代码已公开存在，我们的价值在于**在其之上做增量**（stacker 门禁、D4 空间掩码等），而非隐藏配方。
