# 方法

## 1. 任务与数据

- RSNA Knee Abnormality Detection：每个 study 含多个 MRI 序列，预测 12 种膝关节异常；评估为 12 标签 macro-AUC。代码竞赛，断网 Notebook，时限 9 小时。
- 训练集 4,407 个 study，**仅 58 个有专家标签**，其余只有放射科报告文本；测试集没有报告。

## 2. 标签：报告 → 软标签

- 多语言正则抽取器给出每个标签的分级概率与置信度（在 58 金标上 GAUGE 1 = 0.865）。
- LLM 逐报告按 12 标签 rubric 输出 (p, conf)。C6 使用正则 + Kimi + Qwen + Grok + Gemma4，**逐源做概率平均**，而不是 rank 平均——后者会把每列均值钉在 0.5，不适合作为 BCE 目标。
- 58 个金标作为验证集留出，不参与训练。

## 3. 影像语料

- 预解码 uint8 memmap：4,407 × 64 × 336 × 336，分两片，各约 15.9 GB。
- 五个槽：矢状流体 18、矢状结构 14、冠状流体 12、冠状结构 8、轴位 12（切片数）。每个序列取 6%–94% 跨度，槽内做 2–98 百分位归一，按 140 mm 物理尺寸中心裁剪；不做左右翻转。
- 测试集在推理核内按同一规则现场解码（`corpus.build_test_volume`），保证与训练分布一致。

## 4. 模型与训练（C 线）

- **骨干**：CoAtNet-2（`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`），输入 384（由 336 双线性上采样）。
- **2.5D 窗口**：三张连续切片叠为 RGB；训练时每个 study 随机取 k_train = 12 个窗口，评估时确定性取 k_eval = 42 个。
- **聚合**：逐标签注意力池化——每种异常一套独立的注意力分布——后接线性分类头。
- **损失**：BCEWithLogits，按患病率设 pos_weight（裁剪到 1–10）。
- **优化**：AdamW（骨干 2e-5、头 1e-3、weight decay 0.02），OneCycle（pct_start 0.15），16 轮，batch 2，梯度裁剪 3.0，bf16 autocast；**bf16 下 GradScaler 不生效，手工跳过梯度非有限的步**。
- **选轮**：每轮在 58 金标上评估；训练结束时比较最佳单轮与 top-3 轮权重平均（平均后重算 BN 统计），取金标更高者。
- **异构臂**：
  - C8：RadImageNet ResNet-50（2048 维特征），权重来自公开数据集，重映射键名后载入。
  - C9b：DINOv2 ViT-B/14（768 维），patch 嵌入内自动补齐到 392 以保持 384 输入；骨干峰值 LR 5e-6，逐层衰减 0.75，头 LR 1e-3。

## 5. 评估协议

- **单变量纪律**：每次提交相对父版本只改一个变量；改了多个变量的提交单独标注，不参与归因。
- **预注册**：判据（以及融合权重）在训练或出分之前写入 LOG，出分后不改；融合核文件在出分前记录 SHA-256，提交时复核。
- **判定阈值**：|ΔLB| ≤ 0.004 不下结论。
- **推理核核对清单**：严格加载日志；无中途快照警告；启动时的占位文件已被真实结果覆盖；可见测试集上的取值集合与所声明的融合权重一致。

## 6. 硬件与工程

- 训练：AMD Radeon AI PRO R9700（32 GB，ROCm 7.2，torch 2.14）。出分：Kaggle T4 纯推理核，挂载训练好的权重。
- R9700 优化：8 个深度卷积改走 PyTorch 原生核（自定义 autograd，前向与反向都绕开 MIOpen），加 fused AdamW。

## 7. 2026-09-22 之后新增的方法

### 7.1 公开方案基底（0.943）

- 原样复现公开 Speedy Raptors 线：20 个 DINOv2 member、5 折 A5、RadImageNet ResNet-50 头，加 4 视图 Raptor CoAtNet 与 CoAt 读者家族（resgated / global96 / d4，后加 Repair-v1）。最后按标签做外层加权：`(1 − w)·rank(DINO/A5/Rad 栈) + w·rank(Raptor/CoAt 侧)`，默认 w = 0.65，另有 5 处按标签覆盖。
- 在它上面的所有实验都只追加或替换单元格；父本单元格逐字节保留，并用单元格 md5 核对。
- 隐藏集一次运行约 5 h 20 min；Code Competition 上限 9 h。

### 7.2 图像老师蒸馏（C6k）

- **老师**：公开 DINOv2×20 的 5 折 OOF 与 RadImageNet e11 的 5 折 OOF 的概率均值，均为折外预测，每个 study 的老师都没见过它。
- **目标**：非金标 study 的 BCE 目标 = (1 − λ)·报告源均值 + λ·老师，λ = 0.5，训练前定死。金标照常被覆盖并留作验证集。
- **实现**：`config.TEACHER_PARQUET / TEACHER_LAMBDA`，默认关闭；老师文件缺任何 study 直接报错。

### 7.3 换骨干与池化对照（C10、C11）

- **C10**：只把骨干换成 `convnext_small.fb_in22k_ft_in1k_384`，学习率沿用 CoAtNet 的值，其余全同 C6r。
- **C11**：在 C10 基础上只把逐标签注意力池化换成**全局平均池化**（`config.RAPTOR_POOL = "mean"`），作为 RQ2 的对照。checkpoint 不记录池化方式，加载时必须先设开关。

### 7.4 融合核构建（fail-closed）

- 自研臂以子进程运行：推理脚本经 gzip + base64 嵌入笔记本，并校验 SHA-256；在百分位秩空间与父本输出融合，权重在训练前定死。
- 遇到任何异常都报错，**不回退到父本**：study 集不一致、NaN、常数列、列序漂移、npz dtype 不符都算。原因是一次静默降级曾把实验结果混淆（`findings.md` §10）。

### 7.5 公榜按标签探测

- 常数列在 AUC 下恰为 0.500。只保留 k 个标签的真实预测、其余填 0.5，就能由 `LB = ((12 − k)·0.5 + Σ AUC_k)/12` 反推出这 k 个标签的公榜 AUC。
- **分辨率**：k = 4 时 ±0.0015，k = 1 时 ±0.006，来自 LB 只显示 3 位小数。
- **纪律**：结果只用于诊断，不用来调权重，探测提交也不进入最终候选。

### 7.6 金标上的分析方法

- **样本外预测来源**：
  - 公开 CoAt 读者自带的金标预测，训练脚本里断言金标不进训练，但它们按金标选轮，因此**偏乐观**；
  - DINOv2 / RadImageNet 的 5 折 OOF；
  - 自研臂把金标当验证集，从不在金标上训练。
  - 汇总在 `results/gold58_preds/`。
- **基底代理**：CoAt 家族秩均值与 DINO+Rad 秩均值，按基底真实的外层权重混合。缺 Raptor 与 A5。
- **读者相关**：
  - 整体相关 = 逐标签 Spearman 的中位数；
  - 类内相关 = 在专家阳性内、阴性内分别计算 Spearman，按类大小加权后取中位数；
  - 融合红利 = 等权秩融合的 AUC 减去两者 AUC 的平均。
- **报告四态**：由 `report_extractor.extract_raw` 的阳性 / 阴性计数、分数和置信度判定，分为明确阳性 / 可疑（置信度恰为 0.55）/ 明确阴性 / 未提及。**召回**取 4 个 LLM 均值 ≥ 0.5 的比例。
- **稠密窗口导出**：注意力池化头是线性的，`logit = Σ softmax(a_k)·(h_k·w) + b`。导出每个窗口的注意力 logit a_k 与得分 h_k·w 后，任何槽位或池化消融都可以离线重算（`scratchpad/window_dump_gold.py`）。
- **区间估计**：一律在 58 个 study 上 bootstrap（1000–2000 次）。单标签区间很宽，正文只解读区间不含 0 的差异。

## 8. 2026-10-01 之后新增的方法（LOG #490–#539）

### 8.1 诊断专科与窗口策略

- `config.CORPUS_TRAIN_LABELS`：只对指定标签计 BCE、只按这些标签选轮，其余标签的损失权重为 0、金标列置为 NaN。
- `config.CORPUS_CENTRE_ALLOW`：训练、金标评估、测试推理三处统一限定 MIL 窗口中心。允许集合写入 checkpoint（`centre_allow`），推理时恢复；集成成员的策略不一致直接报错。
- PF-v1 = 只训 PF OA + 中央矢状 20 个中心（SagF 4–13、SagS 20–29）；PF-v2 = 只训 PF OA、全部窗口。

### 8.2 多列探测的分辨率

- 保留 k 列、其余填 0.5：k 列平均 AUC = (12·LB − (12 − k)·0.5) / k，**1 个舍入单位 = 0.012 / k**（k = 1：0.012；k = 2：0.006；k = 4：0.003）。
- 单列判定一律以"2 个单位"为实际效应阈值（预注册，不称统计显著）。

### 8.3 测量与验证（S2、H-DICOM、精度）

- **金标 → 公榜可迁移性**：单变量父 → 子对，配对 bootstrap（2000 次）给出 Δ 金标的 95% 区间；公榜 Δ 落在区间外 = 金标误判（`scratchpad/gold_lb_transfer.py`、`s2_extension_predictions.py`）。
- **推理精度检验**：同一 checkpoint 在 bf16 / fp16 / fp32 下的金标宏 AUC（`scratchpad/gold_precision_check.py`，RTX 3070）。
- **解码一致性**：用当前 Kaggle 镜像按测试路径（`build_test_volume`）重解码 58 金标，与训练语料逐片比对 md5（`rsna-knee-gold-redecode` v2）。

### 8.4 强度—多样性前沿（S3）

- 由逐标签恒等式"融合 − 较强者 = 红利 − |差距| / 2"导出：红利 = 融合 − 两者均值；逐标签平均 |差距| = 2 ×（红利 − 融合增益）。
- 红利对类内多样性 1 − ρ_w 过原点拟合（k = 0.053），得到拟合前沿 gap < 2k(1 − ρ_w)（`scratchpad/strength_diversity_frontier.py`）。

### 8.5 报告噪声分类（S6）

- 8 个 LLM 的 (p, conf) 统一转为四态：conf < 0.25 → 未提及；否则 p ≥ 0.6 阳性、p ≤ 0.4 阴性、其余可疑。报告态取多数。
- 漏报（金标阳性中报告非阳性）、阈值错位（金标阴性中报告阳性）、二值 Fleiss κ、跨标签报告 AUC；Wilson 区间（`scratchpad/report_noise_taxonomy.py`）。

### 8.6 空间归因（S7、B2）

- 由稠密窗口导出计算每个标签的 softmax 注意力质量，按切面、槽 × 5 个深度箱汇总；阳性 − 阴性差用分层 study bootstrap，要求 C6r 与 C10 同号复现（`scratchpad/spatial_attribution.py`）。
- **膝别**：矢状序列全部切片 x 坐标的中位数（LPS，+x 指向患者左侧）< 0 → 右膝；与头标签一致率 98.7%。所有矢状语料行都朝患者右侧递增，所以展开深度 u = d（右膝）或 1 − d（左膝），0 = 内侧端，1 = 外侧端（`scratchpad/b2_laterality.py`、`b2_unfolded_attribution.py`）。

### 8.7 有序窗口模型（S4）

- `raptor.S4SequenceClassifier`：编码器输出 + 可学习位置嵌入（20 个位置）→ 2 层 Transformer（d 768，8 头，pre-norm）→ 父类的按诊断注意力池化。每次前向前必须 `set_positions()`，缺失或形状不符时报错。
- T：位置 = 窗口中心在允许集合中的真实索引；K：每个 study 一个固定置换（SHA-256(`S4-perm-v1:` + uid) 作种子），清单写入 `s4_perm_manifest.npz`。两臂参数逐名相同。
- 训练截取：两个序列（SagF、SagS）取同一段 6 个连续深度，共 12 个窗口，两臂同一 RNG 序列；评估 / 推理用全部 20 个（`config.RAPTOR_HEAD = "s4seq"`、`S4_ORDER`、`CORPUS_TRAIN_CONTIGUOUS`）。

### 8.8 目标二值化（B5）

- `config.CORPUS_BINARIZE_TARGETS`：在报告源平均与教师混合之后、金标覆盖之前，把非金标目标阈值化为 1[p ≥ 0.5]，默认关闭。
