# 公开方案解剖：`bliverrigpdoge/velci-kcd25-20`（公开 0.955）

> 分析日期 2026-10-12。核：https://www.kaggle.com/code/bliverrigpdoge/velci-kcd25-20
> 代码 SHA-256（code cells 拼接）：`18c67a09e843be3d125187e9…`，300,455 字符 / 1,136 行 / 10 个单元。
> 作者 Mikkel Chávez Petersen（`bliverrigpdoge`），另一个核 `rsna-knee-raptor-student-blend-lb-0-944`。

---

## 1. 结论速览

**这份 0.955 = 公开 0.954 配方原样不动（80%）+ 作者自己的私有 12 标签读者 KCD25（20%）。**

- **不可直接复现**：私有腿来自数据集 `bliverrigpdoge/rsna-knee-kcd25leg`（含 `leg_main.py` + `members.json`），**实测 403 Forbidden，是私有数据集**。KCD25 权重拿不到。
- **但它给出了一个极重要的第二数据点**：证明「在 0.954 公开骨架上加一个**独立的** 12 标签读者、只给 20% 权重」就能拿 **+0.001**。

---

## 2. 结构解剖（10 个单元）

| 单元 | 内容 | 角色 |
|---|---|---|
| 0 | markdown：自述「公开 0.954 配方（未改动）+ KCD25」 | — |
| 1 | 记录 notebook 起始时间 `_VP_NB_T0` | 给末尾的私有腿算**剩余时间预算** |
| 2 | `%%writefile /tmp/raptor_run.py`（96,957 字符） | Stage 1 Raptor 推理源码 |
| 3 | 子进程跑 Raptor（失败即停，**不允许静默单模出分**） | Stage 1 |
| 4 | 装 compact-v1 权重包（182,167 字符，含 `E_lrprior` 四折） | Stage 2 准备 |
| 5 | 子进程跑 `probe_dual.py` → `ours_sub.csv` | Stage 2 |
| 6 | **公开 0.954 的 Stage 3**：`W_RAPTOR, W_OURS = 0.7, 0.3` 分位秩融合 → `submission.csv` | **Apex** |
| 7 | **KCD25 私有腿**：子进程跑 `leg_main.py` → `_dm.csv`（+ `_dm_report.json`） | **私有读者** |
| 8 | Fast-stack 融合库（`fs_*` 函数，只依赖 numpy/pandas） | 融合引擎 |
| 9 | **最终混合**：`ADM_WEIGHTS = {'apex': 0.8, 'dm': 0.2}` | 出分 |

### 融合公式
```
final[finding] = 0.80 · rank01(apex[finding]) + 0.20 · rank01(KCD25[finding])
```
- `rank01`：平均秩缩放到 [0,1]；NaN 保持 NaN；单个值映射为 0.5。
- 权重**在任何 Submit 之前写死**，不做任何 leaderboard 调参。
- **逐 study 重归一化**：某个成员对某个 study 没有值（或该成员整体失败），它的权重按比例分给其余成员。

### KCD25 是什么
从单元 7 的注释与 markdown：**knee-centred compartment view（膝关节中心 + 间室局部视野），五个 ImageNet 成员的集成，训练在 d25 软报告标签上，权重私有。**
→ 注意它与我们的方向**高度一致**：解剖局部变焦 + 多成员集成 + 软标签。这正是我们在做的「解剖锚点银行 + 专科」。

---

## 3. 七个值得直接抄的工程模式

1. **Fail-open（永不失败）**：每个可选单元都用 `try/except BaseException` 包住，任何异常只写进报告 JSON，**绝不 raise**。私有腿缺失/失败/超时 → `_dm.csv` 删除 → 最终混合检测不到成员 → **apex 的 `submission.csv` 原样保留**。→ 内核永远产出合法提交，不会因为附加件挂掉而白跑一次配额。
2. **时间预算传递**：`ADM_CAP_S = max(60, min(14400, (NB_T0 + 8.5·3600) − now))` —— 私有腿只拿到**9 小时限制的剩余部分**，跑不完就丢弃，而不是拖垮整本。
3. **成员契约校验**（`fs_check_member`）：列名必须**恰好**是 `[StudyInstanceUID] + 12 标签`；study 集合必须与 `test.csv` **完全一致**；值必须**有限**且**非常数**（`nunique() > 1`）。任一条不满足就丢弃该成员。→ 这正是本项目反复踩的「静默降级」的解药。
4. **逐 study 权重重归一化**（而非固定覆盖）：某成员缺值时按比例分权重，数学上等价于「在可用子集上重新归一」。
5. **原子落盘**：`tmp 文件 → flush → os.fsync → os.replace`，杜绝半截 `submission.csv`。
6. **Copy group vs Readers 教义**（`FS_READER_TOTAL_ONE = 0.30`、`FS_READER_TOTAL_MANY = 0.40`）：**独立读者合计只拿 0.30（一个）或 0.40（≥2 个），其余全给「copy group」（公共强基底）**。作者把「公共基底」和「自己的读者」在权重上严格分层，不让自研件反客为主。
7. **支持逐病种权重**（`fs_weight_of` 接受 dict），但明确写「no tuning on the leaderboard」—— 权重必须在提交前定死。

---

## 4. 科学结论：三个方案对齐，指向同一条定律

| 方案 | Stage 1（Raptor 权重） | Stage 2 | 公榜 |
|---|---|---|---|
| `heliosli/rsna-knee-blend-gold` | 0.70 | `compact-v1` 0.30 | **0.954** |
| **`velci-kcd25-20`** | 0.70 | `compact-v1` 0.30（= 上一条，80%）+ **私有 KCD25 20%** | **0.955** |
| `heliosli/blend-mine-in-…` | **0.50** | **`compact-r6cnxfsss` 0.50** | **0.956** |

**三个方案的增益来源完全一致：都是「换一个更强的 / 再加一个更独立的 Stage-2 读者」。**
- 从 `compact-v1`（0.30）换成 `r6cnxfsss`（0.50）→ **+0.002**
- 在 0.954 骨架上叠加一个私有独立读者 20% → **+0.001**

**没有一例**来自后处理技巧、后校准、外层权重微调 —— 与本项目十余次实测（融合权重 / 外层层权重 / 换 Stage-1 底座 / PF 单列外科融合全部零效应或负效应）**完全一致**。

---

## 5. 我们能复用什么，不能复用什么

| 项 | 可用性 |
|---|---|
| **工程模式 1–7** | ✅ **全部可复用**，与本项目已有工程（熔断、回执、SHA 校验）互补；第 3、4 条尤其值得吸收 |
| 公开 0.954 骨架 | ✅ 已有（我们复刻过，= 0.954） |
| `compact-r6cnxfsss` | ✅ 公开，**已在我们三路核里用上** |
| **KCD25 私有腿** | ❌ **403 Forbidden，私有数据集，拿不到** |

**所以我们不能"抄"这 0.955，但可以"造"等价的私有腿。** 我们手上已有的自研件：
- 本地 R9700 双病种专修（PF OA + Lateral OA，convnext_small，53 解剖中心，金标 **0.8551**）—— 但只覆盖 **2 个标签**；
- S6 ConvNeXt-V2-Base（corpus64 从零训，6 轮欠拟合 Gold 0.7548）；
- Ep8 CoAtNet-2 384 solo（0.948，但与 Raptor 同族、高度相关）。

**缺口很明确：我们缺一个「12 标签、且与 Raptor 信息正交」的自研读者。** 而 KCD25 的做法（knee-centred compartment view + 5 成员集成 + 软标签）正是把「解剖局部变焦」这一条做成 12 标签读者 —— 与我们 `RSNA_Knee_Breakthrough_Directions_2026-10-10.md` 里的 **D5（薄层语料重抽）/ D6（LM 专科）** 是同一个方向。

---

## 6. 对我们的行动含义

1. **短期（今天就可能拿到 +0.002）**：`57095718` / `rsna-knee-heliosli-0956-verbatim` 就是那条 **0.956 配方**，出分即知。
2. **中期（+0.001 ~ +0.002）**：三路核 `0.50·Raptor + 0.25·compact-v1 + 0.25·r6cnxfsss`（已构建，待推送）—— 用**两个独立 Stage-2** 替代「二选一」，是这条定律下最省成本的下一格。
3. **中期（真正的私有腿）**：把自研件做成 **12 标签读者**再按 **0.80 / 0.20** 挂上去。最低成本的路径是**把本地双病种专修从 2 标签扩到 12 标签**（同一套 53 解剖中心 + convnext_small 管线，只改标签数），金标 0.8551 的 2 标签版本已经证明这条管线有效。
4. **工程上立刻吸收**：把模式 3（成员契约校验）与模式 4（逐 study 重归一化）加进我们的融合核 —— 我们的三路核已经做了第 4 条，但第 3 条（**非常数**检查）还没加。
5. **不要做**：任何形式的公开榜后处理调参（这份 notebook 自己也写明 "No tuning on the leaderboard"）。
