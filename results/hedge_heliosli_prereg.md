# 预注册:跨家族异构对冲 —— heliosli 0.9586 配方复刻(决赛最后一张牌)

日期:2026-10-19(LOG #698)。状态:**待构建,未提交**。

## 背景

- 我方 0.958 基底(v1/v3/v4)= goodpjw 核(R6 fe055406 + Part B 2.5D MIL + SWA + orthofoundation),挂载 4 份数据集。
- 高分内核侦察(16 核 source+metadata)结论:scoreDescending 排序不可靠;**lavinwins gold-medal 实为 0.943 家族组合层改良,其自述预期仅 +0.001~0.002,不构成 0.958 段威胁**;真正的公开顶是 heliosli `this-is-best-public-dataset-new-version-0-9586`(标题声称 0.9586)。
- heliosli 配方(cell 0/8 已逐行解码):
  - Raptor 半臂 = 0.65×rank(r384) + 0.35×rank(r224),各自在自身非失败行上取百分位秩;
  - 我方半臂 = R6(fe055406,4 折,自带 Effusion fluid + Baker's stackers)+ flip TTA(logit 域平均,MCL 取原始 pass);
  - Stage 3 逐标签秩混合:默认 0.5/0.5,逐标签 Raptor 权重覆盖 **{Medial OA 0.50, Lateral OA 0.15, PF OA 0.65, Synovitis 0.85, Baker's 0.30}**;
  - 失败回退:单臂失败取另一臂秩,双臂失败填 0.5;Raptor 失败率 >3% 或我方 >50% 未评分即中止;fallback 打印 BLEND_FALLBACK;
  - 外加 Part B(goodpjw 2.5D MIL,cell 9)。
- **依赖核查(全部本地已有)**:`publication-swa` 内含 `..._d96_r384_swa.pt` 与 `..._d96_c08_r224_swa.pt`(292.8MB×2,即配方所需两个 Raptor checkpoint);r6-fe055406、2-5d-convnext-reader、orthofoundation 均在 v3 现有挂载内。零新数据集。

## 为什么只能走 LB(不可离线验证)

- Raptor checkpoint 本身是在 58 gold 上选择的(heliosli cell 8 自述),gold58 对其**污染**;
- BEE 内核审计同样指出 residual-CoAt(mattiaangeli)训练器用 gold58 选 epoch → CoAt 臂对 gold58 亦污染;
- 故任何含 Raptor/CoAt 臂的新混合,58 gold 离线评估只有偏乐观的方向性参考价值,不作为决策依据。LB 是唯一诚实检验。

## 单变量定义

相对我方 v3 基底,唯一变更 = **混合层替换为 heliosli 公开逐标签权重表 + Raptor 384/224 半臂**(含其失败回退规则);保留我方 Anti-Tie 90/10 与 Part B 不动。不引入任何额外 TTA(BEE 负结果:强度增益 TTA 集成 −0.018)。

## 决策规则(预注册,提交前锁定)

| LB 结果 | 判定 | 动作 |
|---|---|---|
| ≥ 0.959 | 命中(+0.001 以上,超出同分带) | 取代一个决赛席位;复跑一次确认非抖动 |
| = 0.958 | 平 | 保留为**决赛多样性候选**(家族构成与 v1/v3/v4 不同:Raptor 半臂显式入混),2 选 1 时按私有震荡对冲逻辑择优 |
| ≤ 0.957 | 负 | 丢弃,记负结果,不动现有候选 |

- 噪声带纪律不变:±0.003~0.004;0.958 平分不视为证据。
- 运行时护栏:若 Raptor 384 阶段实测投影 >8.5h,预注册降级路径 = 弃 224 阶段(A224→0,纯 384 半臂),此为**唯一**允许的配方偏离,且须在日志声明。

## 诚实预期

heliosli 标题 0.9586 无法第三方验证(可能是内部估计而非 LB 显示值)。复刻的期望增益 ±0.001,主要价值是:(a) 若真 0.9586 则 +0.0006~0.001;(b) 决赛第二席位的家族多样性对冲。预期为平或小赢,不期待决定性突破——与 GOLD cell 0 的自述一致:聚合层改动不产生大位移。

## 关联负结果档案(BEE cell 23/24,引以为戒)

- V17(0.940 完整配方)+ 残差头 A/B → 三次 0.940 无位移;强度增益 TTA G → **0.922(−0.018)**,重复复现;
- evgendvorkin V13 与 V17 差异(Raptor 55/10/15/20 vs 60/10/10/20;rank-H)→ 均 0.940,不可分辨;
- 报告标签实验:未知码被当观测标签的设计缺陷;Steven/Pilkwang 标签分歧加权 ±0.002 不显著。
