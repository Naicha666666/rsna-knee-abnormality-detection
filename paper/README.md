# 论文素材（RSNA Knee Abnormality Detection）

> 整理于 2026-09-22，2026-10-01 补充 §9–§16，2026-10-03 补充 §17（PF OA 专科化）。所有数字都能追溯到 `LOG.md` 条目、Kaggle 提交 ref 或本目录下的数据文件；**这里不产生新结论**，只把分散在日志里的证据按论文结构归拢。

| 文件 | 内容 |
|---|---|
| [`findings.md`](findings.md) | 主要发现与负结果，每条附证据表和 LOG 编号 |
| [`branch_research_plan.md`](branch_research_plan.md) | 项目梳理与后续分支研究计划（B0–B10、E1；2026-10-03） |
| [`methods.md`](methods.md) | 数据、标签、模型、训练、推理、评估协议 |
| [`reproducibility.md`](reproducibility.md) | 数据集、出分核、提交 ref、关键文件哈希 |
| `data/experiment_registry.csv` | 全部已出分的 Kaggle 提交（截至 2026-10-03）：父版本、改动、变量个数、58 金标、公开 LB、判定、LOG 编号 |
| `data/run_summary.csv` | C5–C6s 每次本地训练：最佳轮、best gold、top-3 平均、最终权重类型、耗时、跳步数 |
| `data/gold_curves.csv` | 同上各次训练的逐轮 loss / gold / 累计分钟 / 跳步（长表，可直接画图） |
| `data/train_logs/` | 上述训练的原始日志（含 C7 第 9 轮崩溃、C9 第 2 轮崩溃两份失败运行） |
| `data/kaggle_submissions_raw.csv` | `kaggle competitions submissions --csv` 原始导出 |
| `../results/profile_20260911/` | R9700 分段计时与内核核查的原始 JSON / 日志 |
| `../results/gold58_preds/` | 58 金标上的样本外预测：公开 CoAt 三读者、DINOv2×20 OOF、RadImageNet OOF、C6 / C8 / C6r / C6s / C6k / C10（`findings.md` §12–§14 的输入） |
| `../results/bottleneck_map.md` + `_v1.csv` | 每标签瓶颈表与公榜按标签探测结果（§13） |
| `../results/report_recall.md` + `_v1.csv` + `_cases_v1.csv` | 报告召回表（§14），含逐病例状态 |
| `../results/diversity.md` + `diversity_within_class_v1.csv` | 读者多样性：整体相关、类内相关、融合红利（§12） |
| `../results/final_selection.md` | 最终两份提交的选择与覆盖的金标核查（§15） |
| `../results/pfoa_window_profile_v1.csv`、`weak_window_policies_v1.csv`、`weak_window_profile_v1.csv` | PF OA 与其余弱标签的逐窗口诊断（§17，LOG #490、#504） |

## 论文主图（2026-10-03，`scratchpad/make_main_figures.py`，LOG #516）

| 图 | 文件 | 说明 |
|---|---|---|
| 主图 1 | `figures/main1_score_timeline.png` | 全部已出分提交的时间线（不含探测和 #453 的两份混淆项），按自研 DINOv2 线 / 自研 MIL 线 / 公开栈着色；标出 C2 标签台阶、C6 0.935、0.943、C10 0.912 和榜首 0.961 |
| 主图 2 | `figures/main2_label_bottleneck.png` | 12 个标签的金标代理 AUC（95% CI），叠加公榜探测（PF 0.896、Synovitis 0.908、外侧两标签均值 0.914、其余 8 个均值 0.961）；弱标签标出剩余损失占比 |
| 主图 3 | `figures/main3_gold_vs_lb.png` | 金标 vs 公榜，按骨干着色：CoAtNet 偏移 +0.015…+0.029、DINOv2-B +0.020、ConvNeXt +0.008 / +0.010、ResNet-50 −0.005；同骨干内偏移抵消（C11 − C10：金标 −0.014、公榜 −0.015）（§23） |
| 主图 4 | `figures/main4_strength_diversity_frontier.png` | 强度—多样性前沿（S3）：红利 ≈ 0.053 ×（1 − ρ_within），拟合前沿 gap < 0.106 ×（1 − ρ_within），"强且独立"区为空（§19） |
| 补图 S7a / S7b | `figures/s7a_plane_mass.png`、`figures/s7b_pos_minus_neg.png` | 诊断 × 切面注意力；阳性 − 阴性（槽 × 深度，两模型复现标 + / −）（§20） |
| 补图 S7c | `figures/s7c_unfolded_laterality.png` | 按膝别展开的矢状注意力（内侧 → 外侧，阳性 − 阴性）：ACL 看外侧，Medial / Lateral OA 看本侧（§22，B2） |
| 补图 S6 | `figures/s6_report_noise.png` | 诊断 × 报告噪声机制（漏报 / 阈值错位 / LLM 分歧 / 概念重叠）（§21） |
| 强度—多样性—融合 | `figures/fig1_fusion_dividend.png` | 沿用；这是**金标**上的结论，C10 说明金标上的强读者不一定是隐藏集上的强读者（§18） |

## 建议的图（数据都已就绪）

1. **C 线阶梯图**：C1 → C2 → C3 → C5 → C6 的 LB（`experiment_registry.csv`）。
2. **逐轮 gold 曲线**：C6 / C8 / C9 / C9b 同图（`gold_curves.csv`），竖线标出 OneCycle 峰值（第 2.4 轮）。
3. **强度 vs 解耦散点**：横轴单臂 gold，纵轴与 C6 的逐标签 Spearman 中位数，点旁标注融合 ΔLB（`findings.md` §4）。
4. **金标 Δ vs LB Δ 离散图**：展示 58 金标的分辨率问题（`findings.md` §5）。
5. **训练步耗时分解**：条形图（`findings.md` §6）。
6. **强度 vs 多样性（扩充版）**：11 对读者，横轴两者 AUC 差距，纵轴类内相关，点的颜色表示"融合 − 平均"红利，并画出"差距 = 2 × 红利"这条线（`results/diversity_within_class_v1.csv`，§12）。
7. **金标 vs 公榜的单标签对照**：4 个弱标签，金标代理 AUC 与公榜探测 AUC 并排，带舍入区间（§13）。
8. **报告召回 vs 模型 AUC**：12 个标签的散点，标出公榜确认的 4 个弱标签和 Fracture 这个例外（`results/report_recall_v1.csv` + `bottleneck_map_v1.csv`，§14）。
9. **PF OA 专科化阶梯**：已画（`figures/fig6_pf_ladder.png`），5 个读者的单列公榜 AUC 带舍入区间（§17）。

## 现状与尚未完成（2026-10-03 更新）

- **账户最好 0.943**（Speedy Raptors 线，56558421 等），排名约 326 / 4,670；榜首 0.961。自研单模最好仍是 C6 = 0.935（§8–§9）。**最终提交（2026-10-03 起）**：主选 56786075（0.943，含 PF 专科融合），对冲 56558589（0.942）。
- **已收口的方向**：标签侧（加源、换组合、置信度加权、蒸馏，§1、§7、§11）、自研臂融合（7 次全在噪声内，§4、§8、§12）、MIL 袋大小（§2）、窗口规则（§7）、在 0.943 基底上微调外层权重（§9）。
- **研究线的三个问题已有初步回答**：
  - RQ1 弱监督天花板由报告漏报率决定（§14）；
  - RQ3 强读者不论架构都收敛，多样性只来自弱读者（§12）；
  - 58 金标的单标签排序不可靠，公榜探测确认了弱标签所在（§13）；
  - RQ2 按诊断注意力 vs 全局平均：差约 0.01，预注册判定无可测效应，局灶病变不更依赖注意力（§16）；
  - 单诊断专科化（PF OA）：相对同骨干通才 +0.024，公榜可测；拆成"只训 PF"与"窗口限定"两个变量各 +0.012，在分辨率边缘；与公开集成融合后 PF 0.896 → 0.908（§17）。整版融合提交 56786075 = 0.943，与父本相同（预期内）。
- **未做**：
  - "在不同信息上训练的强读者"这个推论（§12）未经验证。
  - 0.943 → 0.961 的差距仍无可复现的公开来源（#465、#466）。
- 早期记录：实验 4（C6n）已放弃（LOG #414）；公开 Dinosaurs v38 复现 0.940，两线融合超 9 h 上限（§8.1）。
