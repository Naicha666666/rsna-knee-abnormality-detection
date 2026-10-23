# RSNA Knee Abnormality Detection — 瓶颈审计对话记录

> 日期：2026-10-01  
> 主题：Kaggle RSNA Knee Abnormality Detection 竞赛瓶颈审计与下一阶段实验设计  
> 说明：本文整理本轮对话的完整实质性内容；工具调用等内部过程不包含在内。

---

## 1. 用户：项目遇到瓶颈

> [RSNA Knee Abnormality Detection Kaggle Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) 一个我参加的 kaggle 竞赛，遇到瓶颈了。

同时提供项目 `LOG.md` 与 `README.md`。

## 2. ChatGPT：初步判断

从项目日志和 README 看，目前已经不是“刚遇到瓶颈”，而是已经把瓶颈定位到了相当细的层面。

当前大致状态：

- 自研单模最好：**C6 = 0.935**
- 账户公开榜最好：约 **0.943**
- 当时榜面前沿：约 **0.959**
- DINOv2 六槽路线基本确认饱和在 **0.855 左右**
- 已系统做过训练轮数、slot、分辨率、折数、骨干、epoch selection、标签源、异构融合等消融，大多数方向已经明确证伪。
- 当前剩余差距不太像“再调几个超参”能解决，更像输入表征、数据选择、推理策略、模型家族层面的结构性差距。
- Kaggle GPU quota 应尽量留给最终推理/出分，训练已经主要转移到 R9700。

建议把 **0.935 → 0.95+** 当作正式 research problem 拆解，不再从“还有哪些 trick 可以试”出发。

---

## 3. 用户：开始审计

> 开始审计

---

# 4. 瓶颈审计

## 4.1 总结论

> **当前 0.943 瓶颈，不是“模型还不够大”、不是“再加一个标签源”，也不是“再加一个普通异构 backbone”。真正剩下的变量是：如何给已经很强的公共 ensemble 提供一个“同样强、但看到的是不同证据”的第二条腿，并且最好直接攻击 PF OA / Lateral OA / Lateral Meniscus / Synovitis 这组公榜弱项。**

因此，下一阶段不应该继续围绕 C6/C10 做小修小补，而应该从 **建立独立信息源** 重新设计实验。

## 4.2 已经基本关闭的方向

### DINOv2 六槽路线

DINOv2-S 六槽路线从 v8 到 v14 的连续消融基本全部落在噪声范围：增加折数、换分辨率、ViT-S → ViT-B、修 slot、改 epoch selection、header sequence flag recovery、同构 seed ensemble，均没有形成稳定 LB 增益。因此这条路线可以视为基本饱和。

### 标签侧

标签侧也已经系统测试：

- 更强报告标注器：曾产生约 **+0.006**。
- 继续增加或替换 LLM：没有持续收益。
- C6k 使用 DINO/Rad OOF teacher 蒸馏：58 gold 指标从约 0.904 提高到约 0.916，但 LB 只有 **0.931**，改善没有转化为 hidden-test 收益。

这说明继续提高 teacher/report agreement 并不等价于提高最终排行榜表现。

### 普通“异构 backbone”

C10 是关键反例：同一语料、同一软标签、同一 MIL pipeline，只把 CoAtNet 换成 ConvNeXt-Small，结果 gold ≈ **0.902**，与 public CoAt family 的相关仍约 **0.906**，没有真正形成独立排序信息；与 0.943 parent 做 10% rank fusion 后得到 **0.942**。

因此：

> **换 architecture ≠ 产生 diversity。**

---

# 5. 真正的靶心：4 个弱标签

10 月 1 日做的 leaderboard column probe 是目前最重要的诊断之一。

以 0.943 public stack 为基底，反推出：

| 标签/标签组 | 估计 public AUC |
|---|---:|
| 其余 8 个标签平均 | **~0.961** |
| Synovitis | **~0.908** |
| PF OA | **~0.896** |
| Lateral OA + Lateral Meniscus 平均 | **~0.914** |
| 四个弱标签平均 | **~0.908** |

因此 0.943 并不是“12 个标签整体都差一点”，而更接近：

> **8 个标签已经达到约 0.96 水平，主要剩余误差集中在少数 4 个标签。**

这改变了下一阶段问题的定义。下一次训练不应该继续优化一个完整的 12-head generalist，更合理的是：

> **specialist：只负责一个或几个弱标签，其他标签继续使用 0.943 public ensemble。**

---

# 6. 为什么不再做完整第二模型

已有实验说明：

> 强模型只要训练在大致相同的信息和目标上，就会收敛到相似排序。

C6/C6r/C10 与 public CoAt family 的相关都很高。进一步做 class-conditional correlation 后结论仍成立：强 reader 即便 architecture 不同，也趋于高度同源；弱模型可能看起来更 decorrelated，但很大程度是因为它本身没学好。

因此真正需要的是：

> **heterogeneous information，而不只是 heterogeneous architecture。**

---

# 7. 真正有价值的 private leg 应改变什么

一个值得融合的 private leg 至少应该改变一个“信息维度”。

### A. 输入空间

- 不同 crop geometry
- 不同物理尺度
- 不同 depth sampling
- 不同 anatomical centering
- 专门围绕 patellofemoral / lateral compartment 组织输入

### B. 数据选择

- 只训练特定序列
- 只训练特定 plane
- 只训练 lesion-relevant windows
- 不让 12 个 diagnosis 共享同一个信息瓶颈

### C. 目标函数

- 只优化 PF OA
- 只优化 Lateral Meniscus
- pairwise ranking
- ordinal severity
- lesion-presence 与 report-derived score 使用不同 target

### D. 表征性质

- texture / radiomics
- cartilage-specific descriptors
- 传统 image statistics
- 3D / 2.5D spatial features
- morphology / geometry

而不是简单继续换 ImageNet backbone。

---

# 8. 新主线：标签专科化

公共 0.943 ensemble 可以看成当前 generalist。新模型没有必要重新负责全部 12 标签。

可以定义：

$$
\hat y_j =
\begin{cases}
y^{specialist}_j, & j \in S \\
y^{public}_j, & j \notin S
\end{cases}
$$

这样实验的因果归因会非常干净。

---

# 9. 第一优先级：PF OA Specialist

PF OA 是最值得先攻击的目标。

原因：

1. leaderboard probe 直接反推其 public AUC 约 **0.896**，是弱标签中最低的。
2. 报告侧并没有显示 PF OA 的监督特别差：report target 约 **0.912**，image model gold 约 **0.865**。
3. 因此 PF OA 更像是 representation bottleneck / spatial coverage / localization bottleneck，而不是单纯 label-quality bottleneck。

这使 PF OA 成为非常适合 specialist 的目标。

---

# 10. PF OA 不应该简单增加 axial

已有 slot/plane ablation 显示：

- PF OA 只用 axial：明显下降
- 去掉 axial：几乎不下降
- 模型注意力主要落在 sagittal

因此：

> **“给 PF OA 多塞 axial slices”不是当前证据支持的方向。**

更值得测试的是：

> **PF-specific spatial crop / sagittal depth selection / local anatomical reader。**

Generalist 的问题是：“整个膝关节里有什么异常？”

PF specialist 的问题应该变成：“只看 patellofemoral cartilage / patella / trochlea，是否存在 PF OA 证据？”

目标是提高局部空间密度，同时减少无关解剖结构。

---

# 11. 第二优先级：Lateral Meniscus

Lateral Meniscus 也适合 specialist。

潜在方向：

- lateral-oriented crop
- sagittal + coronal specialist
- side-aware local model
- lateral compartment depth selection

公共基底已经对 LM 使用较高外层权重，说明它并非简单 calibration 问题，更可能涉及 information allocation。

---

# 12. 第三优先级：Lateral OA

Lateral OA 更适合做 **compartment-specific model**。

输入不再问整个 knee，而只关注 lateral tibiofemoral compartment。

因为：

$$
\Delta_{macro}=\frac{\Delta_{label}}{12}
$$

| 单标签提升 | Macro-AUC 理论提升 |
|---:|---:|
| +0.02 | +0.00167 |
| +0.03 | +0.00250 |
| +0.05 | +0.00417 |
| +0.10 | +0.00833 |

因此不需要再训练一个“整体达到 0.95 的第二模型”。一个真正强的 PF OA specialist 就可能产生足够明显的 overall gain。

---

# 13. Synovitis 单独处理

Synovitis 的特点不同：

- 报告 recall 很差
- report-derived supervision 不完整
- 继续增加 LLM 没有稳定 LB 收益
- teacher distillation 也没有解决

因此不应该继续假设“更聪明的报告标注器能解决 Synovitis”。更合理的方向是 **image-only local specialist**，重点关注 fluid / synovium / joint recess / suprapatellar region 等视觉证据。

---

# 14. 推荐的新架构思想：Label-specific Spatial Reader

```text
                 Public 0.943 Base
                        |
          +-------------+-------------+
          |                           |
    normal 11 labels              specialist
          |                           |
      keep public               one target only
                                      |
                              PF OA / LM / Lat OA
                                      |
                            replace one column
```

优势：

1. 不需要 specialist 击败整个 0.943 generalist。
2. 不要求它对所有病种都强。
3. 只要某一标签有真正的新信息，就能产生宏平均收益。
4. LB 变化容易做单变量归因。

---

# 15. 现有 dense-window 资产应该重新利用

项目已经导出了 C6r / C10 在 58 gold 上的 dense-window 信息，包括约 62 windows/study 的逐窗信息与 attention/logit。

下一步最值得回答的问题不是“哪个 backbone 更强？”，而是：

> **“哪些 spatial windows 对某个 diagnosis 真正提供判别信息？”**

可以为不同弱标签设计独立 window policy：

```text
PF OA:
    sagittal PF-relevant windows
    + 少量 axial supporting windows
    + local aggregation

Lateral Meniscus:
    lateral sagittal windows
    + coronal lateral-compartment windows

Lateral OA:
    lateral compartment depth zone

Synovitis:
    fluid-sensitive windows
    + joint recess / suprapatellar region
```

---

# 16. 第二梯队：真正的 3D / spatial representation

目前大多数强模型本质上仍然是：

> 2D/2.5D slice → feature → MIL aggregation

即使处理多切片，也没有真正完整建模 3D anatomy。

因此未来一个可能真正提供独立信息的方向是：

- lightweight 3D
- 2.5D temporal convolution
- cross-slice attention
- depth-aware transformer
- spatial continuity modeling

但不建议直接训练一个巨大 12-label 3D 模型。更好的试法是先只针对 PF OA 做一个单标签 spatial/3D specialist。

---

# 17. 新实验矩阵

| Experiment | 唯一主要变化 | 回答的问题 | 成本 | 优先级 |
|---|---|---|---:|---:|
| **S1** | PF OA specialist spatial reader | PF OA 是否主要是 representation bottleneck | 中 | ★★★★★ |
| **S2** | LM specialist | lateral meniscus 是否可通过局部输入改善 | 中 | ★★★★☆ |
| **S3** | Lateral OA specialist | compartment-specific input 是否有效 | 中 | ★★★★☆ |
| **S4** | Synovitis image specialist | image-only 是否能突破 report ceiling | 中 | ★★★☆☆ |
| **S5** | 真正 3D / spatial model | 是否存在 2D-MIL 的结构性上限 | 高 | ★★★★☆ |
| ConvNeXt 全 12-label | 只换 backbone | 已被 C10 基本否定 | 中 | 不做 |
| 新 LLM | 换报告 teacher | 已基本收口 | 高 | 不做 |
| 更多 fold | ensemble size | 已平台 | 中 | 不做 |
| 更大 DINO | capacity | 已基本否定 | 高 | 不做 |
| 新 rank mix | 外层微调 | 已平台 | 低 | 不做 |

---

# 18. 新研究假设

> **H1 — The remaining leaderboard gap is concentrated in a small number of diagnosis-specific labels, and diagnosis-specific spatial readers can improve those labels without sacrificing the strong public ensemble on the remaining targets.**

中文：

> **假设：0.943 之后的主要剩余误差集中在少数诊断标签；针对诊断专门设计的空间读者可以改善这些标签，而不需要替换已经很强的其余标签。**

---

## 19. 用户确认继续

> 好的

---

# 20. 下一实验：PF OA Specialist v1

接下来实验收敛到 **PF OA Specialist v1**。

核心原则：

> **0.943 public stack 完全冻结，只训练一个 PF OA 专科读者，最终 submission 只允许修改 `PF OA` 一列。**

这样 leaderboard 变化可以直接归因于 specialist，不混入 ensemble 改动、outer weights、其他 11 个标签、新标签源或新 rank mix。

## 20.1 当前基准

Leaderboard probe 反推：

> public PF OA AUC ≈ **0.896**

大致区间约：

> **0.890–0.902**

因此 specialist 的任务非常明确：不是提高整体模型，而是提高 PF OA 的排序。

## 20.2 实验顺序

### Step 1 — 零训练诊断

利用已有 dense-window / plane-ablation 数据，分析：

- PF OA 阳性 vs 阴性
- sagittal depth distribution
- slot distribution
- 哪些 spatial windows 判别力最高
- attention 是否集中在稳定 anatomical zone
- 是否存在明显的局部窗口群

目的：在训练前确定 specialist 的 spatial sampling policy。

### Step 2 — PF-v1 训练

初始建议：

- **单标签：PF OA only**
- backbone：ConvNeXt-Small
- resolution：384
- 2.5D
- 只取 PF-relevant sagittal + supporting axial windows
- 尽量复用现有 64-window corpus
- 其他训练条件尽量冻结
- 不重新优化 12 个 head

### Step 3 — 离线 sanity check

重点不是单纯追 gold58 AUC，而是确认：

1. specialist 确实学到了 PF OA。
2. 预测不是常数或退化。
3. 与 generalist 的排序有合理差异。
4. 差异来自可解释 spatial evidence。
5. 不用 58 gold 调融合权重。

### Step 4 — 干净的 leaderboard 实验

构建 submission：

```text
ACL                  = public
MCL                  = public
Medial Meniscus      = public
Lateral Meniscus     = public
Medial OA            = public
Lateral OA           = public
PF OA                = SPECIALIST
Effusion             = public
Synovitis            = public
Baker's              = public
Contusion            = public
Fracture             = public
```

即：

> **11 列逐值复制 0.943 parent，只替换 PF OA。**

不做：

- post-hoc weight tuning
- 多列同时替换
- 新 outer routing
- 事后挑融合比例

这样下一次 Kaggle 完整提交只回答一个问题：

> **Does a diagnosis-specific image reader beat the public PF OA ranking?**

## 20.3 成功后的扩展顺序

若 PF OA Specialist 成功：

1. **Lateral Meniscus**
2. **Lateral OA**
3. **Synovitis**

逐个复制 specialist methodology。

如果 PF OA specialist 失败：

> 不继续微调 PF-v1 的大量超参数，而是重新审视“diagnosis-specific spatial specialization”这个假设本身。

---

# 21. 当前最终判断

整个项目已经从：

> “怎么再找一个更好的模型？”

转变成：

> **“公共 generalist 已经很强，剩余误差究竟集中在哪里，以及怎样为这些特定病种提供 generalist 没看到的信息？”**

目前最有证据支持的下一步是：

> **PF OA Specialist v1：单标签、诊断特异空间采样、冻结 public 0.943 的其余 11 列，用一次干净 leaderboard 实验直接验证。**

这也是当前审计后优先级最高的实验。
