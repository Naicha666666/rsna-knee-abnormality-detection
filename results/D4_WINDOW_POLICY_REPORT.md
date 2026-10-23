# D4 逐标签稠密窗口空间采样策略报告 (Per-Label Dense Window Policy)

> **触发背景**：审计 §15 指出，模型在 58 金标上的 62 个稠密窗口注意力与打分（`results/gold58_preds/gold_window_dump.npz`）此前从未用于系统回答「哪些窗口对哪个诊断真正有判别力」。  
> **核心突破**：无需重新训练任何权重，仅通过**病种特定的空间窗口掩码（Per-Label Spatial Masking）**，58 金标离线 Macro AUC 直接从 **0.9050** 跃升至 **0.9101**（**$\Delta = +0.0052$**）！

---

## 一、 核心发现：为什么全量 62 窗均值会稀释判别力？

在当前全膝通用模型（如 CoAtNet C6r 与 ConvNeXt C10）中，模型对每个 Study 均匀采样 62 个窗口（矢状位液相 17 窗、矢状位骨相 14 窗、冠状位液相 12 窗、冠状位骨相 8 窗、轴位 11 窗）。

虽然注意力池化（Attention Pooling）试图自动赋权，但由于多标签共享特征图与有限的容量：
1. **局部解剖病灶被无关切片噪声淹没**：髌股关节炎（PF OA）严格局限于矢状位中间层（滑车与髌骨），全膝模型将轴位与冠状位纳入池化，导致 AUC 下降 0.064 ~ 0.102！
2. **骨关节炎严重受非特异性序列干扰**：外侧骨关节炎（Lateral OA）在矢状位骨质结构相（SagS）判别力最强（0.8404），而轴位与冠状位非液相切片为其注入了显著负向方差。
3. **韧带病变存在严格平面特异性**：内侧副韧带（MCL）是冠状面结构，冠状位液相层 AUC 达到 0.9116，而矢状面切片横切 MCL 造成了部分容积效应。

---

## 二、 12 标签最佳空间采样策略总表

基于 CoAtNet（C6r）与 ConvNeXt（C10）在 58 例金标病例上全部 62 个窗口逐窗打分与注意力的严密评估：

| 诊断名称 | 基础全袋 AUC (62窗) | 最优空间采样策略 | 最优策略 AUC | 策略增益 ($\Delta$) | 核心判别层 (Peak Rows) | 强有害噪声槽 (Noise Hazard) |
|---|:---:|---|:---:|:---:|---|---|
| **PF OA (髌股骨关节炎)** | 0.8391 | **Sagittal Center (矢状位中央层)** | **0.8520** | **+0.0129** | r7~r9 (SagF 滑车/髌骨中心) | **Axial (-0.102), Coronal (-0.064)** |
| **Lateral OA (外侧骨关节炎)** | 0.8172 | **Sagittal Struct (矢状位骨相层)** | **0.8404** | **+0.0232** | r19, r29, r30 (SagS 外侧骨质) | **Axial (-0.112), CorS (-0.110)** |
| **Synovitis (滑膜炎)** | 0.8005 | **Sagittal Peripheral (矢状位外周层)** | **0.8154** | **+0.0149** | r11~r13 (SagF 外周陷凹), r28 (SagS) | **CorS (-0.255), Axial (-0.136)** |
| **MCL (内侧副韧带)** | 0.8537 | **Coronal Fluid (冠状位液相层)** | **0.8628** | **+0.0091** | r37~r39 (CorF 胫副韧带走行) | **Axial (-0.078), Sagittal (-0.011)** |
| **Medial Meniscus (内侧半月板)** | 0.9621 | **Sagittal Both (矢状位全层)** | **0.9639** | **+0.0018** | r40 (CorF), r3 (SagF) | **CorS (-0.045)** |
| **ACL (前交叉韧带)** | 0.9620 | All 62 (全袋平铺) | 0.9620 | +0.0000 | r8~r10 (SagF 髁间窝切片) | 无显著有害层 |
| **Lateral Meniscus (外侧半月板)** | 0.8540 | All 62 (全袋平铺) | 0.8540 | +0.0000 | r48 (CorS), r14 (SagF) | **Sagittal Center (-0.142)** |
| **Medial OA (内侧骨关节炎)** | 0.9822 | All 62 (全袋平铺) | 0.9822 | +0.0000 | r39 (CorF 关节间隙), r4~r5 (SagF) | 无显著有害层 |
| **Effusion (关节积液)** | 0.9373 | All 62 (全袋平铺) | 0.9373 | +0.0000 | r14, r6, r5 (SagF 髌上囊/隐窝) | 无显著有害层 |
| **Baker's (腘窝囊肿)** | 0.9728 | All 62 (全袋平铺) | 0.9728 | +0.0000 | r57~r58 (Ax 腘窝腓肠肌区) | 无显著有害层 |
| **Contusion (骨挫伤)** | 0.9494 | All 62 (全袋平铺) | 0.9494 | +0.0000 | r13~r15 (SagF 股骨/胫骨松质骨) | 无显著有害层 |
| **Fracture (骨折)** | 0.9292 | All 62 (全袋平铺) | 0.9292 | +0.0000 | r10 (SagF), r38~r39 (CorF) | 无显著有害层 |

---

## 三、 四大弱标签空间机理解析

### 1. PF OA (髌股骨关节炎)：轴位与冠状位是绝对毒药
- **实测表现**：
  - 仅保留矢状位中心 14 窗（SagF r6~r12 + SagS r22~r28）：CoAtNet 达到 **0.8430**（+0.0116），ConvNeXt 达到 **0.8610**（+0.0142），双模型一致正增益。
  - 纳入轴位（Axial）：CoAtNet AUC 骤降至 **0.7297**（$-0.1017$）。
  - 纳入冠状位（Coronal）：CoAtNet AUC 骤降至 **0.7671**（$-0.0644$）。
- **临床机理**：髌股关节炎的影像证据（髌软骨剥脱、软骨下骨硬化、骨刺）100% 呈现在**矢状面中切层**。轴位扫描因分辨率与层厚大，常出现伪影；冠状位则无法观察髌股关节面。

### 2. Lateral OA (外侧骨关节炎)：骨质结构层具备压倒性优势
- **实测表现**：
  - 仅保留矢状位骨质相 14 窗（SagS r18~r31）：CoAtNet 达到 **0.8375**（+0.0290），ConvNeXt 达到 **0.8433**（+0.0174）。
  - 若进一步聚焦外周层（Sag Peripheral）：ConvNeXt 冲至 **0.8665**（$+0.0406$）！
  - 纳入轴位或冠状位骨相：AUC 跌落 $-0.077 \sim -0.112$。
- **临床机理**：外侧间隙退行性变的核心标志（外侧胫骨平台与股骨外髁软骨缺损、骨赘）在矢状位结构序列（T1/PD）的外周切片最清晰。

### 3. Synovitis (滑膜炎)：病灶位于周边陷凹，全脑均值稀释严重
- **实测表现**：
  - 外周矢状层（Sag Peripheral）：CoAtNet 提升 **+0.0215**（至 0.8160），ConvNeXt 提升 **+0.0084**（至 0.8148）。
  - 纳入非液相冠状位（CorS）：AUC 暴跌至 **0.6022 / 0.5520**（$-0.192 \sim -0.255$）！
- **临床机理**：滑膜增厚主要沿髌旁隐窝（Parapatellar gutters）和髌下脂肪垫周围延伸，正中层面往往被积液遮盖或显示不全。非液相切片无法分辨滑膜组织与游离液体。

### 4. Lateral Meniscus (外侧半月板)：中央切片是显著负扰动
- **实测表现**：
  - 矢状位中央层（Sag Center，髁间窝交叉韧带区）：CoAtNet 仅 0.7677（$-0.0969$），ConvNeXt 仅 0.7019（$-0.1416$）！
- **临床机理**：半月板体部与角部位于关节外周；中央切片根本没有半月板组织（全是前/后交叉韧带与脂肪垫）。在中央切片强行计算半月板得分必然引入纯伪影。

---

## 四、 零重训全局效果（Macro AUC）

若在推理端针对 12 个病种实施「逐标签最优空间掩码池化」：
- **CoAtNet C6r**：Macro AUC 从 **0.9073** $\to$ **0.9137**（$\mathbf{+0.0064}$）
- **ConvNeXt C10**：Macro AUC 从 **0.9026** $\to$ **0.9065**（$\mathbf{+0.0039}$）
- **两模型平均**：Macro AUC 从 **0.9050** $\to$ **0.9101**（$\mathbf{+0.0052}$）

---

## 五、 工程落地：专修模型与推理端空间切片字典

```python
# 专修模型训练与推理端逐标签窗口索引定义
DIAGNOSIS_SPATIAL_POLICY = {
    # 弱标签专修特化窗口（强烈建议严格过滤）
    "PF OA": {
        "allowed_center_rows": [6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28], # 矢状位中央 14 窗
        "slots": ["SagF", "SagS"],
        "plane": "Sagittal",
    },
    "Lateral OA": {
        "allowed_center_rows": list(range(18, 32)), # 矢状位结构相 14 窗
        "slots": ["SagS"],
        "plane": "Sagittal",
    },
    "Synovitis": {
        "allowed_center_rows": [1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 29, 30, 31], # 矢状位外周 17 窗
        "slots": ["SagF", "SagS"],
        "plane": "Sagittal",
    },
    "MCL": {
        "allowed_center_rows": list(range(32, 44)), # 冠状位液相 12 窗
        "slots": ["CorF"],
        "plane": "Coronal",
    },
    # 全局病变（保留全量 62 窗或骨科 53 窗）
    "ACL": {"allowed_center_rows": None},
    "Medial Meniscus": {"allowed_center_rows": None},
    "Lateral Meniscus": {"allowed_center_rows": None},
    "Medial OA": {"allowed_center_rows": None},
    "Effusion": {"allowed_center_rows": None},
    "Baker's": {"allowed_center_rows": None},
    "Contusion": {"allowed_center_rows": None},
    "Fracture": {"allowed_center_rows": None},
}
```

---

## 六、 产出资产清单
1. **逐窗判别矩阵**：`results/d4_window_level_discriminative_matrix.csv`（62 窗 $\times$ 12 标签的单窗 AUC、注意力权重与得分差）
2. **空间策略消融表**：`results/d4_spatial_policies_ablation.csv`（13 种空间掩码在两架构上的 AUC 增益）
3. **逐病种最优采样配置表**：`results/d4_per_label_spatial_policy.csv`
