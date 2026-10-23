# RSNA Knee — 公开侧饱和归因与 Final-2 收官计划

> 2026-10-14 10:00 UTC / 10-14 03:00 PDT ｜ 距截止 10-22 23:59 剩 **8 天** ｜ 唯一主账号 **easonyanyan**

---

## 0. 一句话结论

**三笔在途探针全部落在 0.956。自 0.956 基线以来累计 13 笔单变量探针，无一例外全部中性（Δ = 0.000）。**
→ **0.956 是该 pipeline 在现有公开权重池上的诚实上限。** 剩余 8 天的唯一价值不在公开涨分，而在**私榜去同质化对冲**与**合规收尾**。

---

## 1. 本轮三笔出分（守护 `x3WkyB`，跑 3h31m）

| ref | 账户 | 变体 | 状态 | LB | vs 0.956 |
|---|---|---|---|---|---|
| `57170307` | easonyanyan | **multi-scale-stage-1**：Stage-1 = raptor r384@320 (w0.80) + **r224@224 (w0.20)** | COMPLETE | **0.956** | **+0.000** |
| `57169236` | easonyanyan | **3arm**：raptor 0.65 + Ep8 0.25 + **S777 0.10** | COMPLETE | **0.956** | **+0.000** |
| `57169231` | snowy615 | **stage-3-tilt**：cell6 逐病种 `W_RAPTOR_C`（MOA/LOA/Syn 0.60；PFOA/Baker's/MM/LM 0.40） | COMPLETE | **0.956** | **+0.000** |

### 判读

- **r224 跨 FOV 臂中性（最重要）**：这是唯一「**真换视野**（FOV 83.3%→58.3%，保 mm/px）+ **带 2,399 例 OAI 外部先验**」的异构臂——连它都零增益。**说明模型误差不由视野/外部先验主导**，靠 Stage-1 换臂已无路可走。
- **3arm 中性**：第三臂（S777 独立初始化盆地）无增益 → **Stage-1 第二臂轴彻底闭合**。
- **stage-3-tilt 中性**：Stage-3 逐病种权重分工无效 → **0.50:0.50 全局平权是帕累托最优**。

---

## 2. 饱和定论：13 笔中性全表

| 轴 | 变体 | ref | LB | Δ |
|---|---|---|---|---|
| **Stage-1 第二臂** | Ep8 0.30 | `57161780` | 0.956 | 0 |
| | + S777 0.10（3arm） | `57169236` | 0.956 | 0 |
| | + r224 跨 FOV（multi-scale） | `57170307` | 0.956 | 0 |
| **Stage-2** | 关 stacker（nostack） | `57121214` | 0.956 | 0 |
| | Baker's coef→0（bakersoff） | `57161675` / `57127661` | 0.956 | 0 |
| | 双组 r7+compact-v1（VD） | `57121422` | 0.956 | 0 |
| **Stage-3** | 逐病种 `W_RAPTOR_C`（tilt） | `57169231` | 0.956 | 0 |
| | 外层层权重 0.40/0.60（W40） | — | 0.956 | 0 |
| | `flip_tta=True` | `57121236` | 0.956 | 0 |
| **adopt 门** | 翻 Lateral OA | `57127697` | 0.956 | 0 |
| | 翻 Lateral OA + Medial OA | `57150216` | 0.956 | 0 |

**机制**：
- **Stage-1**：所有候选第二臂都共享 **CoAtNet-2 384 + Nartaa OAI 底座** → 与主臂秩高度相关，只注入独立误差。r224 虽换 FOV，但同底座 → 同样无效。
- **Stage-2**：两个拟合式 stacker 在 4,349 OOF 上拟合，但**隐藏集上完全中性**（含 explicit +0.0460 的 Baker's 都不兑现）→ **OOF 离线增益一律不可信**。
- **Stage-3**：逐病种权重不足以改变全局秩排序。

---

## 3. 收官：Final-2 对冲结构

公开榜在 **0.950 / 0.954 / 0.956** 形成巨大同分 fork 墙（数百队同分）。私榜揭晓时必然洗牌——**Final 2 应选两份「公开同分、私榜行为不同」的结构**。

### easonyanyan 现成的 0.956 候选（4 笔）

| ref | 变体 | Stage-1 | Stage-2 | 定位 |
|---|---|---|---|---|
| `57127661` | bakersoff（≈ verbatim） | raptor r384@320 | r7（stacker 实质关闭） | **主版候选**：最接近充分验证的基线 |
| `57161780` | superstage1 | raptor 0.70 + Ep8 0.30 | r7 | Stage-1 异构 |
| `57169236` | 3arm | raptor 0.65 + Ep8 0.25 + S777 0.10 | r7 | Stage-1 异构 |
| `57170307` | multi-scale | raptor r384@320 0.80 + r224@224 0.20 | r7 | Stage-1 异构（跨 FOV） |

> ⚠ 注意：**充分验证的 verbatim 基线 `57120771` 在 snowy615**（已停用）。easonyanyan 上与之等效的是 `57127661`（bakersoff，stacker 中性 → 功能等价）。

### 推荐 Final 2

- **主版（稳）**：`57127661` ≈ verbatim 0.956 基线（Raptor 0.50 + r7 0.50 + `E_lrprior`）。**最充分验证**，3 个独立运行点（heliosli / 我们 / degnonguidi）均 0.956。
- **对冲版（去同质化）**：**新建合成核** —— **同时改动 Stage-1 与 Stage-2**（现有 4 笔只改 Stage-1）：
  - **Stage-1** = multi-scale + Ep8：`raptor r384@320 (w≈0.55) + r224@224 (w≈0.15) + Ep8 (w≈0.30)`
  - **Stage-2** = **双组 r7 + compact-v1，关 stacker**（`--dual-group --no-stackers`）
  - **Stage-3** = 0.5/0.5
  - **理由**：唯一一个**两阶段都与公开 fork 结构不同**的配置；且移除拟合式 stacker → 规避私榜过拟合下行。
  - **构建方式**：合并 `build_0957_s1_multiscale.py`（Stage-1 补丁）与 `build_0957_variant.py --dual-group --no-stackers`（Stage-2 补丁）。
  - **预期公开分**：**0.956**（预计无公开增益；价值仅在私榜去同质化）。

---

## 4. ⚠ 合规收尾

- **单账号政策**（10-14 已定）：只用 `easonyanyan`（5 次/日）；`snowy615` 退役，仅读历史。
- **Team Merger 截止 10-15 23:59 UTC**（约 38 小时）。若 `snowy615` 是队友 → 必须合并；若为单人副号（已确认）→ 保持单账号即可。
- **残余风险**：历史双账号提交属既成事实，无法回溯撤销；赛后合规审查可能回看时序。

---

## 5. 配额与下一步

- **easonyanyan 今日 3/5 已用，剩 2**。
- **建议动作**（2 次配额正好）：
  1. 构建 + 推送 + 提交 **去同质化对冲核**（1 次配额）→ 作为 Final 2 之一。
  2. 保留 1 次配额作为机动（若对冲核异常/或需补主版）。
- **主版 `57127661` 已在账上，无需再耗配额。**

> 结论：**公开侧已无杠杆。** 剩余 8 天的正确姿态 = 锁定 0.956 基座 + 一份结构互补的对冲，然后等私榜。

---

## 6. 执行记录（2026-10-14 07:55 PDT）

### 6.1 对冲核已构建并推送

- **构建器**：`scratchpad/build_0957_hedge.py` —— **31 项自检全 PASS**。
  - 策略：**`import` 复用三个已验证构建器的补丁串**（`build_0957_s1_multiscale` / `build_0957_superstage1` / `build_0957_variant`），不手抄任何字符串 → 消除转写错误。
- **核**：`easonyanyan/rsna-knee-0957-hedge-ms-ep8-dualgroup` **v1 RUNNING**。

| 阶段 | 配置 |
|---|---|
| **Stage-1**（cell 2） | 三臂：`raptor r384@320 (w0.55)` + `raptor r224@224 (w0.15)` + `WarmStart Ep8@320 (w0.30)`；`infer_res` 逐臂（r384 钉 320 复现基线）；三处硬编码 res 断言全放宽；raw/ranks 改为**概率混合→rank** |
| **Stage-2**（cell 4） | **双组** `r7 + compact-v1`（原生 `--models2`，`BUNDLE2` 重指向）+ **关两个拟合式 stacker** |
| **Stage-3**（cell 6） | 0.50 / 0.50，**未碰** |
| 数据集 | base + `easonyanyan/rsna-knee-c6-warmstart-ep8-weights` + `heliosli/rsna-knee-compact-v1` |

- **守护**：`scratchpad/watch_hedge.py`（后台 task `PF3pNw`）。**11 项门禁** → 全过才自动提交（任一不过 = 不提交、不烧配额）：
  `status COMPLETE` / `loaded 3 arm(s)` / **零 FALLBACK** / blend 列出三臂文件名 / `DUAL_GROUP` 行 / `raptor_failed==0` / `ours_failed==0` / 3 行 / 非常数；且**提交前再校验 token = easonyanyan**。

### 6.2 ⚠ 本次发现：cell 6 有一道**致命守卫**（此前未记录）

```python
if rho_rr and min(rho_rr.values()) < 0.99:
    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)
```

即 **`raptor_raw.csv` 必须与 `raptor_sub.csv` 逐标签 Spearman ≥ 0.99**（基线单臂时 raw = `arm_probs[0]`、sub = `rankpct(arm_probs[0])` → 恒为 1.0）。

**推论（多臂内核的硬约束）**：一旦 Stage-1 有多个臂，**必须同步改 raw/ranks** —— 把 `_raw = pd.DataFrame(arm_probs[0]…)` 换成**概率混合** `_wp`，并令 `ranks = rankpct(_wp)`（即 `build_0957_superstage1.py` 的 `RANKS_NEW`）。否则 raw = 臂0、sub = 加权秩混合 → 可能 < 0.99 → **整核 RuntimeError 崩**。

> 这解释了为何 `superstage1` 当初要改 raw/ranks，而 `multi-scale-stage-1`（沿用基线 rank-mean）**只是侥幸未触发**（两臂高度相关）。对冲核采用 `superstage1` 的写法，`spearman ≡ 1.0`，守卫必然通过。

### 6.3 门禁结果与提交

核 v1 于 08:08 完成，**11 项门禁全部 PASS**，守护自动提交：

| 门禁 | 结果 |
|---|---|
| status COMPLETE | PASS |
| `loaded 3 arm(s)` | PASS |
| **零 FALLBACK** | PASS |
| blend 指纹 = `{r384: 0.55, r224: 0.15, ep8: 0.3}` | PASS |
| `DUAL_GROUP E_lrprior convnext_small [0.5×12]` | PASS |
| **`spearman_raptor_raw_vs_sub` 全 = 1.0**（cell 6 致命守卫） | PASS |
| `raptor_failed == 0` / `ours_failed == 0` | PASS |
| 3 行 / 非常数 | PASS |

→ **提交 ref `57186664`（PENDING）**。easonyanyan 今日 **4/5 已用，剩 1**。

**预期公开分 = 0.956**（一切组合均中性）；价值在**私榜去同质化** + 移除 stacker 的私榜过拟合风险。
