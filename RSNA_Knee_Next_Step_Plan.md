# RSNA Knee Abnormality Detection — 下一步研究与竞赛计划

> 核心原则：**先定位瓶颈，再花 GPU；先做可归因实验，再看 leaderboard；Competition Track 与 Research Track 并行但不混淆。**

## 1. 总体目标

未来阶段优先完成三件事：

1. 建立 12 个 diagnosis 的 **error / bottleneck map**
2. 找出当前 0.94x 区间剩余误差主要来自哪里
3. 从误差分解中提出至少 1 个可形成 Research Project / Paper 的问题

同时维护两条线：

- **Competition Track**：LB、runtime、稳定提交、异构 ensemble
- **Research Track**：weak supervision、label uncertainty、MIL、error decomposition

---

## 2. P0：建立每标签 Bottleneck Map

停止只看 Macro-AUC。建立主表：

| Label | 当前 AUC | 最强 reader | 次强 reader | 模型相关性 | Weak-label 质量 | 主要 sequence | 疑似瓶颈 | 下一实验 |
|---|---:|---|---|---:|---:|---|---|---|
| ACL | | | | | | Sagittal | | |
| MCL | | | | | | Coronal | | |
| Medial Meniscus | | | | | | Sag/Cor | | |
| Lateral Meniscus | | | | | | Sag/Cor | | |
| Medial OA | | | | | | Coronal | | |
| Lateral OA | | | | | | Coronal | | |
| PF OA | | | | | | Axial/Sag | | |
| Effusion | | | | | | Fluid-sensitive | | |
| Synovitis | | | | | | Fluid-sensitive | | |
| Baker's | | | | | | Sag/Ax | | |
| Contusion | | | | | | Multi-plane | | |
| Fracture | | | | | | Multi-plane | | |

每个标签至少统计：

- 单模型 AUC
- ensemble AUC
- 各 reader 预测 Spearman correlation
- 高置信 disagreement 数量
- Gold vs weak-label agreement
- Missing-series prevalence
- 各 plane / sequence availability
- seed / fold 稳定性

**完成标准：**
- 明确哪 3–5 个 diagnosis 对 Macro-AUC 的剩余损失贡献最大。
- 明确主要属于 supervision / coverage / representation / aggregation / ensemble 中哪一类。

---

## 3. P1：Weak Supervision

核心研究问题：

> **How should privileged radiology reports supervise image-only MRI models when expert labels are scarce?**

报告不仅是 pseudo-label 来源，还可以作为：

- Privileged information
- Weak label source
- Sample weighting signal
- Distillation teacher
- Uncertainty estimator

建议把每个诊断的报告状态拆成：

- explicit positive
- explicit negative
- uncertain / possible
- mild / trace
- moderate
- severe
- not mentioned
- laterality
- confidence

优先实验：

1. 当前 soft-label baseline
2. uncertainty-aware loss weighting
3. `not mentioned -> low-confidence negative`
4. `not mentioned -> ignore loss`
5. teacher-generated soft target
6. per-label supervision policy

评估同时看：

- Gold agreement AUC
- Derived holdout AUC
- Per-label change
- Multi-seed consistency
- LB

---

## 4. P1：Model Disagreement / Residual Diversity

已知同构 seed ensemble 没有稳定收益，因此以后 ensemble 按 **error diversity** 设计。

分析对象：

- DINOv2
- CoAtNet
- RadImageNet family
- Raptor family
- 其他异构 readers

每标签计算：

```text
Spearman(pred_A, pred_B)
rank disagreement
top-k disagreement
high-confidence contradictory cases
```

优先测试：

1. Per-label rank average
2. Greedy forward selection
3. Caruana-style repeated selection
4. Correlation-filtered / residual-aware ensemble

暂停：

- 更多 seed
- 更多 fold
- 相同模型重复平均

---

## 5. P1：Lesion-Aware / MIL

假设：局部病灶信号可能被 Study-level pooling 稀释。

重点对象：

- Meniscus tear
- ACL injury
- Contusion
- Small fracture

最小实验：

```text
2.5D groups
 -> encoder
 -> diagnosis-specific attention
 -> MIL/top-k pooling
 -> 12 outputs
```

只比较：

- current pooling
- MIL pooling

如果有效，再继续 token-level MIL：

```text
Study -> Series -> Slice -> Region -> Diagnosis score
```

---

## 6. 医学错误分析

每个标签建立：

- High-confidence FP
- High-confidence FN
- Model disagreement cases
- Weak-label / model conflict cases

错误分类：

```text
Error
├── Supervision
│   ├── report ambiguity
│   ├── negation
│   ├── severity threshold
│   └── not-mentioned ambiguity
├── Sequence / acquisition
│   ├── missing useful sequence
│   ├── wrong slot assignment
│   └── insufficient depth coverage
├── Spatial
│   ├── laterality normalization
│   └── slice ordering
├── Representation
└── Aggregation
```

目标不是只看几个案例，而是得到类似：

```text
Top 100 FN
31% supervision
27% coverage
18% aggregation
14% representation
10% uncertain
```

---

## 7. Competition Track 提交纪律

每次实验必须记录：

```text
Question:
Hypothesis:
Single variable:
Frozen components:
Offline evidence:
Noise floor:
Runtime:
LB result:
Interpretation:
Decision:
```

### 可以提交
- 单变量变化
- 明确 hypothesis
- offline evidence 强
- 输出确实变化
- runtime 可控

### 不提交
- no-op
- 多变量混改
- 纯 seed lottery
- 同构 ensemble 堆数量
- 只因为“还有额度”

---

## 8. GPU 升级条件

现在不要仅因为“更快”升级。

只有满足以下条件之一再买：

1. **VRAM 是硬限制**：token MIL、multi-series attention、teacher-student、medical foundation model 无法运行。
2. **迭代速度成为真实瓶颈**：实验矩阵已经明确，但当前硬件导致一周无法完成关键实验。
3. **新 GPU 能解锁新实验**，而不是只让旧实验更快。

好的理由：

> “24GB VRAM 让我能同时跑 6 series × 24 slices × token MIL。”

不充分理由：

> “新 GPU 快 2 倍。”

---

## 9. 四周执行顺序

### Week 1 — Measurement

输出：
`results/bottleneck_map.md`

完成：
- per-label score table
- disagreement matrix
- weak-label agreement
- FP/FN case bank
- sequence availability

### Week 2 — Weak Supervision

输出：
`results/weak_supervision_ablation.md`

做 2–3 个严格单变量实验：
- baseline
- ignore-not-mentioned
- uncertainty-aware weighting

### Week 3 — Lesion-aware MIL

输出：
`results/mil_ablation.md`

只比较：
- current pooling
- MIL pooling

### Week 4 — Heterogeneous Ensemble

输出：
`results/ensemble_diversity.md`

比较：
- current ensemble
- per-label greedy ensemble
- correlation-filtered ensemble

---

## 10. Stop Rules

立即停止某方向，如果：

1. 连续 3 个干净实验都落在 **±0.004 LB** 噪声内
2. offline metric 没有改善，也没有机制性解释
3. 成本增加 >2×，质量收益 <0.005
4. 新 reader 与现有最强模型高度相关且 solo 不优
5. 实验不能回答一个明确问题

---

## 11. Research Questions

### RQ1 — Weak supervision
**How can multilingual radiology reports provide uncertainty-aware privileged supervision for image-only knee MRI diagnosis?**

### RQ2 — Lesion-aware aggregation
**Does diagnosis-specific multiple-instance learning improve localized knee abnormality detection compared with global study pooling?**

### RQ3 — Ensemble diversity
**When do heterogeneous MRI readers provide complementary diagnostic information, and when does ensembling merely average correlated errors?**

---

## 12. 每次实验模板

```markdown
## Experiment ID

### Question
这个实验到底想回答什么？

### Hypothesis
为什么认为它可能有效？

### Single Variable
只改了什么？

### Frozen Components
哪些东西完全不动？

### Offline Metric
用什么判断？

### Noise Floor
当前可分辨的最小变化是多少？

### Runtime
GPU / 时间 / 显存。

### Result
Offline:
LB:

### Interpretation
支持 / 不支持 hypothesis？

### Decision
Adopt / Reject / Repeat / Needs more seeds
```

---

## 13. 最终项目叙事

不要写成：

> “我参加 RSNA Kaggle，试了很多模型。”

而应形成：

> **Learning Knee MRI Abnormalities from Sparse Expert Labels and Privileged Radiology Reports**

主线：

```text
Sparse gold labels
        ↓
Radiology report weak supervision
        ↓
MRI representation learning
        ↓
Diagnosis-specific sequence/slice aggregation
        ↓
Error decomposition
        ↓
Heterogeneous reader analysis
        ↓
Robust image-only inference
```

---

## 14. 最重要的原则

Biohub 后期学到的是：

> **系统为什么错？**

RSNA 下一阶段要进一步问：

> **不同类型的错误，分别由谁造成？**

先分清：

- supervision
- sequence selection
- coverage
- representation
- aggregation
- ensemble

再决定下一张 GPU-hour 花在哪里。

**目标不是跑更多实验，而是让每个实验产生更多信息。**
