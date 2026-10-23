# 0.9573 复刻：源码核对、偏差归因与已推送配置

**日期**：2026-10-14（下午，PDT）　**作者操作**：Agent
**触发**：用户转发 heliosli 新发布的 `0-9573-latest-public-dataset-score-in-one-click` 拆解

---

## 1. 结论先说

heliosli 的 0.9573 已拉到本地并逐字核对（`scratchpad/heliosli_09573/`，9 cells）。

**它的唯一功能增量就是那颗 r224 权重**——这一点由源码 diff 直接证实，不是我方的推断：

| 组件 | 0.9573 vs 我们的 0.956 基线 | 判定 |
|---|---|---|
| `raptor_run.py`（384 臂，cell 2） | **diff = 0 行** | 逐字节相同 |
| Stage-1 runner（cell 3） | 相同 | — |
| Stage 2（R7） | 仅 `C_SLUG` 换了数据集名 + 新增 `FLIP_TTA=False` 开关 + 一行 print | **无操作**：manifest sha256 同为 `c4bb0f73…`、bundle 同为 `8c7b666e`，而 R7 manifest 的 `flip_tta` 键本来就是 false |
| **Stage 1b（新增，cell 4+5）** | **新增 r224 臂，独立进程** | **唯一新增信息源** |
| **Stage 3（cell 8）** | 秩空间加入 r224，`A224 = 0.35` | 配合项 |

→ **0.9573 与我们的 0.956 是干净的单变量差异**：一颗权重 + 一个新融合系数。这是 13 笔中性探针之后第一个有作者实证支撑的新杠杆。

---

## 2. r224 臂的精确配方（源码逐字摘录）

`raptor224_run.py`（由 384 版改写，diff 224 行）：

```python
_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
# 权重候选首个路径即 /kaggle/input/rsna-knee-publication-swa-weights-20261007 —— 我们基线已挂载，无需新增
assert sha256 == '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'

IMG = 384            # native 384 corpus
CROP_MM = 140.0
NORM = "imagenet"
K_EVAL = 94
CROP_FRAC = 0.8      # centre crop of the field before the resize
def _crop(t):
    H, W = t.shape[-2], t.shape[-1]; ch, cw = int(round(H * CROP_FRAC)), int(round(W * CROP_FRAC))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    return t[..., y0:y0+ch, x0:x0+cw]
# eval_windows: t = _crop(...)  →  if t.shape[-1] != res: F.interpolate(t[None], size=(res,res), 'bilinear')
# gpu_windows: if x.shape[-1] != res or CROP_FRAC < 1.0: ... x = _crop(x); x = F.interpolate(...)

ARMS = [{"file": "..._c08_r224_swa.pt", "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224, "w": 1.0}]
```

四个此前未被充分强调的细节：

1. **几何是「0.8 裁剪 + 缩放」**：`round(384×0.8)=307` 的中心裁剪，再双线性插值到 224。作者 markdown 原文 "0.8 centre crop of the 384-pixel field resized to 224" 与此完全一致。**不是** `(384−224)//2` 的 58.3% 紧裁。
2. **架构是 `coatnet_rmlp_2_rw_224`**：`build_backbone` 在该文件中**不传 `img_size`**，用 timm 原生默认（224）。我们多尺度构建用的是 `_rw_384` + `img_size=ck_res`。
3. **单视角，无镜像 TTA**：`infer_pair_probs` / `_mirror_tri` / SLOT_SIZES 在该文件中被整段删除；`infer_probs` 另带 fp16→fp32 兜底。
4. **独立进程 + 独立 csv**：写 `raptor224_raw.csv` / `raptor224_sub.csv`，**384 臂的 `raptor_raw.csv` / `raptor_sub.csv` 保持纯净** → 那道 `rho_rr ≥ 0.99` 的致命守卫恒过。

## Stage 3 融合公式

```python
A224 = 0.35                                  # 预先固定，不拟合
r384 = pct_on(raw[c].values,  ~r384_fail)
r224 = pct_on(raw224[c].values, ~r224_fail)
rr   = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0-A224)*r384 + A224*r224))
b    = wr * rr + (1.0 - wr) * ro             # wr = 0.5 for every finding (MOA/LOA/PFOA override 同为 0.5)
```

即 **`0.5 × R7 + 0.5 × [0.65·rank(384) + 0.35·rank(224)]`**，且每个臂**各自在自身非失败行上取百分位秩**（`pct_on`），彻底消掉两臂的校准尺度差异。新增 `rho_rr224` 守卫与 `spearman_raptor224_vs_raptor384` 记录。

---

## 3. 为什么我们的 `multi-scale-stage-1`（`57170307` = 0.956）没能拿到这 +0.0013

四处差异，按重要性排序：

| # | 维度 | 我们的 v2 | heliosli 0.9573 |
|---|---|---|---|
| 1 | **几何** | `lo=(384−224)//2=80` → 224 直切，**58.3% FOV，不缩放** | 0.8 裁剪（307 px）→ **插值到 224** |
| 2 | **融合空间** | `raptor_run.py` 内对**概率**加权平均（ARMS `w=0.80/0.20`） | Stage-3 对**秩** 0.65/0.35 |
| 3 | **权重** | r224 占 0.20 | A224 = 0.35 |
| 4 | **架构** | `coatnet_rmlp_2_rw_384` + `img_size=224` | `coatnet_rmlp_2_rw_224` 原生 |

**第 1 条的根因值得固化**：`(IMG - res)//2` 这段中心裁剪代码**只存在于 384 的 `raptor_run.py`**（服务 `res < INPUT_RES` 的情形，即 res=320）；224 臂在他手里有**独立的 `CROP_FRAC` 路径**。我们的多尺度构建器把 r224 臂接到了 384 的窗口路径上——这才是真正的几何偏差来源，而不是简单的"少打一个补丁"。

第 2、3 条同样关键：概率平均会让开了镜像 TTA、logit 更锐利的一侧主导；秩混合则不受校准尺度影响。

**行动含义**：不要试图在单次解码的 `multi-scale` 构建器上继续改补丁。正确做法是把作者已验证的 Stage 1b 整个搬过来。

---

## 4. 已构建并推送：精确复刻核

**`scratchpad/build_0957_replicate_09573.py`** — 35 项自检全 PASS。

设计原则：**零手工转写**。基线 4 个 cell（两份 markdown、`raptor_run.py`、Stage-1 runner）与 Stage-2、deadline cell **逐字节保持原样**；作者的 `raptor224_run.py` cell、Stage-1b runner cell、Stage-3 cell **整段复制**。元数据挂载与基线一致（r224 权重就在我们已挂的 `nartaa/rsna-knee-publication-swa-weights-20261007` 里，无需新增）。

最终 10 个 cell：

```
[0] markdown 溯源说明（新增）
[1] markdown 基线致谢（不变）
[2] %%writefile /tmp/raptor_run.py      ← 384 臂，字节不变
[3] Stage-1 runner                     ← 不变
[4] %%writefile /tmp/raptor224_run.py  ← 作者原文（含 sha256 绑定）
[5] Stage-1b runner                    ← 作者原文
[6] Stage 2 (R7, bundle 8c7b666e)      ← 不变
[7] deadline                           ← 不变
[8] Stage-3 秩融合 A224=0.35            ← 作者原文
```

门禁（`scratchpad/watch_09573.py`，task `QLLiGo`，后台）：
1. status COMPLETE
2. `RAPTOR224_STAGE_DONE`（224 子进程退出码 0 且两个 csv 落盘）
3. `RAPTOR_STAGE_DONE`
4. **零 `FALLBACK`**（吸取 v1 教训：不能只看 `raptor_failed`）
5. 无 `BLEND_FALLBACK` / `RuntimeError` / `Traceback`
6. `raptor224_raw.csv` 与 `raptor224_sub.csv` 存在
7. **`raptor_raw.csv` 与 `raptor224_raw.csv` 内容不相同** ← 证明 224 臂真合入、未被静默丢空
8. `blend_receipt.json` 的 `a224 == 0.35`
9. `raptor224_failed == raptor384_failed == raptor_failed == ours_failed == 0`
10. `spearman_raptor224_raw_vs_sub` 全 ≥ 0.99
11. `spearman_raptor224_vs_raptor384` 已记录
12. submission 3 行 12 列且非常数
13. **提交前再校验 token = easonyanyan**（防串号）

**已推**：`easonyanyan/rsna-knee-09573-replicate` v1，RUNNING。过门禁自动提交 = easonyanyan 今日 **5/5**（当日最后一格；00:00 UTC 后刷新，即约 3.3 小时后）。

---

## 5. 判读预案

| 结果 | 判读 | 下一步 |
|---|---|---|
| **≥ 0.957** | 新杠杆成立：跨视野 + 外部 OAI 先验的异构臂 + 秩空间 0.65/0.35 是有效组合 | 次日优先：把 A224 与权重做成小网格（0.30/0.35/0.40、r224 0.15/0.35），并在 hedge 核里把 224 臂几何也改成 CROP_FRAC 路径 |
| **= 0.956** | 该增量不超三位小数波动；0.956 确为公开池上限 | 维持 Final-2：主版 `57127661` + 对冲版 `57186664`（hedge） |
| < 0.956 | r224 权重在隐藏集上有害 | 立即弃用该轴；确认 `57170307` 已用错几何这一点，不再纠缠 |

**对 hedge 核 (`57186664`) 的影响预告**：它的 Stage-1 也含 r224 臂，但用的是错误的 58.3% 直裁 + 概率混合 w=0.15。若本次复刻证实几何是关键，hedge 的 r224 分量应当按 `CROP_FRAC` 路径重建——但它作为对冲核的价值主要来自"双阶段结构异构 + 无 stacker"，不受这一点否定。

---

## 6. 附：本次核对顺带修正的两处认知

1. 转发的拆解说"train_knee RAPTOR_CROP_FRAC" —— 源码注释确认 `CROP_FRAC = 0.8 # centre crop of the field before the resize (train_knee RAPTOR_CROP_FRAC)`，**作者的训练侧确实是 0.8 裁剪**，转发分析的推断这次是对的。
2. 我们上一轮"死代码考古"得到的结论（`lo=(384-224)//2` 时 `F.interpolate` 不触发）**只对 384 的 `raptor_run.py` 成立**；224 臂的 `_crop` 路径下 `F.interpolate` 是**活代码**。上一条分析把 384 文件的死代码性质外推到了 224 臂，这是偏差 #1 的思想源头。
