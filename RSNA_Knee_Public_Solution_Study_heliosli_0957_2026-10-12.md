# 公开方案解剖：`heliosli/0-957-latest-dataset-score-in-one-click`（宣称 ≈0.957）

> 分析时间 2026-10-12。目的：搞清楚 0.954 → 0.957 的 +0.003 到底来自什么，并做原样复刻提交。

---

## 1. 结论速览

**这不是「换个权重包」那么简单。r7 相比 r6/compact-v1 同时多了三样东西：**

| 新增项 | 内容 | 意义 |
|---|---|---|
| **重训的四折权重** | `cnx_r7`（vs `cnx_s6`），4 个 fold checkpoint sha256 全部不同 | Stage-2 读者真的重训过 |
| **`fluid_stacker.json`** | 拟合式后处理 stacker，特征来自**积液体积测量**（`D2 sag_p99_open`） | **自带准入规则：cross-fitted AUC 增益 ≥ +0.003 才采纳** |
| **`state_stacker.json`** | 第二个拟合式 stacker（state 相关） | 同类后处理 |

外加 manifest 新增字段：`compile: True`、`data_policy: "No KneeCoT inputs or derivatives; gold never trained"`、`flip_tta: False`。

**关键点：notebook 代码在 0.956 版本里就已经支持这两个 stacker 了**（两版代码里 `fluid_stacker` 各出现 14 次、`state_stacker` 各 19 次，**完全相同**），但 **r6 数据包里没有这两个文件** → 代码走 `if manifest.get('fluid_stacker')` 判断为假 → **stacker 被静默跳过**。r7 把文件补上，stacker 才真正生效。

→ **所以 +0.003 = 「重训的 Stage-2」+「两个拟合式 stacker 被激活」**，而不是单纯的权重更新。

---

## 2. 与 0.956 版的逐行 diff（只有 3 处变化）

对比 `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`（0.956）与本篇（0.957）：

```
-  BUNDLE_ID = '209a32c8b50a6b1c'        →  BUNDLE_ID = '8c7b666e27dc0e8f'
-  C_SLUG    = 'rsna-knee-compact-r6cnxfsss'  →  'rsna-knee-compact-r7cnxfsss'
-  （删除了 r6 版的 _checked_mount 预检块：校验 bundle 209a32c8b50a6b1c / 四个 E_lrprior fold）
```

其余 **49.7 万字符代码逐字节相同**（除上述 34 行 diff）。

**融合配置未变**：`W_RAPTOR, W_OURS = 0.5, 0.5`；`W_RAPTOR_C = {'Medial OA':0.5,'Lateral OA':0.5,'PF OA':0.5}`（与默认同值 → **仍是空操作**）；`ARM = 'E_lrprior'`。

→ **这是一个干净的单变量升级：只有 Stage-2 权重包变了。**

---

## 3. 证明 r7 是真新权重（不是像 r6 那样的字节级重传）

| | compact-v1（0.954 基线） | r6cnxfsss | **r7cnxfsss（0.957）** |
|---|---|---|---|
| manifest sha256 | `1c1211e64b73f1f6433ade804816572d` | `1c1211e64b73f1f6433ade804816572d` | **`c4bb0f7399ada9a020e0c15cae625203`** |
| bundle_id | `209a32c8b50a6b1c` | `209a32c8b50a6b1c` | **`8c7b666e27dc0e8f`** |
| tag | `cnx_s6` | `cnx_s6` | **`cnx_r7`** |
| `E_lrprior_s0_f0.pt` sha256 | `55983f47c25f78e8cd3b9f3d…` | 同左（**重传**） | **`ca8a76e024828793e0663e5a…`** |
| `E_lrprior_s0_f1..f3.pt` | — | 全部同左 | **全部不同** |
| source_run | `D:\rsna_align_probe\runs\cnx_s6` | 同左 | `/root/autodl-tmp/rsna_align_s0/runs/cnx_r7` |

**r6 是 compact-v1 的字节级重传（#657 已证）；r7 是全新的一次训练。** 三个包共用同一骨干（`convnext_small`）、同一 `crop 352 / size 384`、同一 `arm E_lrprior`、同一 `folds=4 / epochs=16`——**架构与配方没变，数据/流程变了**。

r7 数据包文件：`E_lrprior_s0_f0..f3.pt`（各 200,379,391 B）、`side_cls.pt`（85 MB）、`state_spec.json`、`manifest.json`、**`fluid_stacker.json`（新）**。

---

## 4. `fluid_stacker.json` 是什么（最有价值的部分）

```
version: 1        created: 2026-10-12 03:48:52
script:  /root/autodl-tmp/rsna_align_s0/analysis/fit_fluid_stacker.py   md5 a3a1b512…
rule:    "ADOPT iff cross-fitted AUC gain >= +0.003 on the explicit-only ruler
          AND >= 0 on the corrected ruler"
feature: {combine: 'logit',
          lines: [{run: …/runs/cnx_r7, oof: 'oof_E_lrprior_s0.npy', pool: 4344, config_crop: 352}],
          weights: [1.0], …}
volume:  {definition: 'D2 sag_p99_open (scratch_fluid_1009/fluid_measure.py):
          best fluid-sensitive sagittal series (META fluid == 1, lowest median of the
          full-field foreground u > 10); region view rows u >= 229 (series p99); …'}
fit_cells: {targets: 'probe_targets_opusv11b_rr4e.csv', mask: 'silent_mask_opusv11b_rr4e.json', …}
```

**读法**：
- 它是一个**在 OOF 预测上交叉拟合出来的线性/逻辑 stacker**，把「**积液体积测量**」当作额外特征，叠加到读者的 logits 上（`combine: logit`）。
- 它**自带准入规则**：只有在 **cross-fitted AUC 增益 ≥ +0.003** 且校正后 ruler 上 ≥ 0 时才采纳。
- 特征来源是一次**几何测量**（`fluid_measure.py` 的 `D2 sag_p99_open`：取最佳 fluid-sensitive 矢状序列、在膝关节区域行 `u ≥ 229` 上量积液），**不是又一个神经网络**。
- 拟合目标用的是 `probe_targets_opusv11b_rr4e.csv` + 一个 `silent_mask`（大概是「静默/可信」样本掩码）。

**含义（对我们的直接启示）**：
> 这 +0.003 很可能**主要不是来自重训**，而是来自**「一个廉价的几何测量特征 + 交叉拟合 stacker」**这条我们从未走过的路。
> 这与我们突破清单里的 **D2（跨标签 stacking，58 金标 +0.0116）**、**D4（逐标签空间掩码，+0.0052 零重训）** 属于**同一类杠杆**——**用辅助信号重排既有模型的输出，而不是再训一个模型**。
> 而「**带准入规则的交叉拟合**」这个做法本身也值得抄：它把「后处理是否真的有用」变成了一个**可判定的、不依赖公开榜的**门槛。

---

## 5. 复刻（已执行）

| 步骤 | 结果 |
|---|---|
| 拉取内核 | `scratchpad/pub6/heliosli0957/` |
| 构建原样复刻核 | `kaggle_kernel/submissions/rsna-knee-heliosli-0957-verbatim/`，核 `easonyanyan/rsna-knee-heliosli-0957-verbatim` |
| **代码一致性** | 与作者代码**逐字节相同**，sha256 **`053d933027ab718d87b847cc4e919361`** |
| 元数据 | 挂载 `heliosli/rsna-knee-compact-r7cnxfsss` + nartaa + orthofoundation；`enable_gpu=True`、`enable_internet=False`；保留作者的 `docker_image` 与 `machine_shape=NvidiaTeslaT4`；**已删除 `id_no`**（否则 403） |
| 推送 | v1 成功 |
| **远端校验（铁律）** | 6/6 通过：代码逐字节相同 / `C_SLUG=r7cnxfsss` / `BUNDLE_ID=8c7b666e…` / `W=0.5,0.5` / **无残留 r6 引用** / r7 数据集已挂载 |
| 提交 | 守护脚本 `scratchpad/submit_0957_when_ready.py`（task `q5ML5z`）：等内核 COMPLETE → 校验输出 → **等配额恢复后自动提交** |

**配额**：10-12 UTC 已用 5/5，**约 17:00 PDT（00:00 UTC）重置为 5/5**，守护脚本会自动重试提交。

---

## 6. 判据（出分前定死）

| 结果 | 判读 | 动作 |
|---|---|---|
| **≥ 0.956** | 公开前沿真的推进了，我们复刻成功 | 立即把 r7 作为新基线；**重点转向研究 fluid_stacker / state_stacker 的做法**（几何测量 + 交叉拟合准入），看能否自造同类组件 |
| **= 0.955** | 部分复现 | r7 有效但幅度减半；仍换新基线 |
| **= 0.954** | 与 r6 同分 | 说明 +0.003 可能来自 stacker 之外（例如他的私榜运气/口径），需回到单变量排查 |
| **< 0.954** | 复刻失败 | 检查 r7 挂载与 stacker 是否真的生效（日志里找 `fluid-stacker` worker 参数） |

---

## 7. 对我们的战略含义

1. **公开前沿没有枯竭** —— 我们昨天（#657）判定的「公开侧已无独立信息」**被推翻**：heliosli 通过**重训 + 补上两个 stacker 文件**又拿到了 +0.003。**结论要修正为「同一份权重上做融合是空操作，但重训与后处理 stacker 不是」。**
2. **一条我们从未走过的路被点亮**：**几何测量特征 + 交叉拟合 stacker + 显式准入规则**。我们的 D2/D4 正是同类，且**离线已验证有增益（+0.0116 / +0.0052）却从未上过榜**。
3. **「准入规则」这个工程习惯值得立刻采纳**：任何后处理都必须先在离线 ruler 上达到明确门槛（如 +0.003）才允许进入提交核，**杜绝「凭感觉调权重」**。
4. **我们自己的读者路线仍成立但优先级下降**：既然 heliosli 用「同架构重训 + stacker」就能拿 +0.003，而我们训了三个弱读者全部有害（−0.003），**投入产出比明显不利于「从零造一个 0.90 的读者」**。下一轮应优先做**后处理/stacker 方向**。
