# D3 Technical Report: Implementation & Verification of Bayesian Likelihood-Ratio Prior (`E_lrprior`)

**Author / Component**: Core Modeling Infrastructure  
**Target Milestone**: D3 (Porting heliosli Stage-2 Missing +0.005 Component)  
**Deliverable Assets**:
- Source Module: [`src/lr_prior.py`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/src/lr_prior.py)
- Remote Mirror: `eason@192.168.1.177:/home/eason/rsna-knee/src/lr_prior.py`
- Test Suite: [`tests/test_lr_prior.py`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/tests/test_lr_prior.py)
- Remote Verification: `eason@192.168.1.177:/home/eason/rsna-knee/tests/test_lr_prior.py`

---

## 1. Executive Summary

In the forensic audit of the top solution architectures (specifically heliosli's Stage-2 training pipeline), one critical training signal was identified as missing from our local codebase (`src/`): the **Bayesian Likelihood-Ratio Prior (`E_lrprior`)** and its associated **`StateEvidenceHead`**. 

heliosli attributed **+0.005 LB** gain to the joint adoption of dynamic thin-slab sampling and `E_lrprior`. While standard pipelines optimize Binary Cross Entropy (BCE) against noisy soft labels or pseudo-labels, `E_lrprior` reframes multi-label abnormality prediction under **Bayesian Evidence Theory**, directly optimizing the fold-change in odds relative to unconditional population priors.

We have fully implemented, unit-tested, and verified this component:
1. **Mathematical equivalence**: Discrepancy against heliosli's decompiled reference implementation (`compact_train.py:L503`) is strictly **$0.00 \times 10^0$** (bitwise identical output).
2. **Invertibility**: Roundtrip transformation between posterior probabilities and likelihood ratios achieves numerical reconstruction error $< 3 \times 10^{-8}$.
3. **Hardware verification**: Verified directly on AMD Radeon AI 9 700 (ROCm / PyTorch 2.6 bfloat16 environment) with **5/5 unit tests passing**.

---

## 2. Mathematical Principles

### 2.1 From Posterior Probabilities to Likelihood Ratios (Bayes Factor)
Given:
- A report soft target or pseudo-label probability $p_c \in (0, 1)$ for class $c$,
- The unconditional population base rate $p_{0, c} = \mathbb{E}[y_c]$,

The unconditional prior odds and target posterior odds are:
$$\text{odds}_{0, c} = \frac{p_{0, c}}{1 - p_{0, c}}, \quad \text{odds}_c = \frac{p_c}{1 - p_c}$$

The **Likelihood Ratio (Bayes Factor)** is defined as:
$$\text{LR}_c = \frac{\text{odds}_c}{\text{odds}_{0, c}} = \left(\frac{p_c}{1 - p_c}\right) \bigg/ \left(\frac{p_{0, c}}{1 - p_{0, c}}\right)$$

- If $p_c = p_{0, c}$, $\text{LR}_c = 1.0$ (no evidence either way; odds match population average).
- If $p_c > p_{0, c}$, $\text{LR}_c > 1.0$ (positive evidence; odds scaled up).
- If $p_c < p_{0, c}$, $\text{LR}_c < 1.0$ (negative evidence; odds scaled down).

### 2.2 Likelihood Ratio Loss (`lr_loss`)
Standard BCE with soft targets $\hat{p}$ suffers from gradient vanishing near saturated regions ($p \to 0$ or $1$) and fails to account for class imbalance when pseudo-labels drift. In contrast, `lr_loss` treats model logit $z$ as generating an evidence term:

$$\sigma(z) = P(\text{Positive} \mid \text{Image})$$

The loss is formulated as:
$$\mathcal{L}(z, \text{LR}) = -\text{logaddexp}\big(\log \sigma(z) + \log \text{LR}, \;\log \sigma(-z)\big)$$

Using the identity $\sigma(-z) = 1 - \sigma(z)$, this simplifies algebraically to:
$$\mathcal{L}(z, \text{LR}) = -\log \Big( (1 - \sigma(z)) + \text{LR} \cdot \sigma(z) \Big) = -\log \Big( 1 + (\text{LR} - 1)\sigma(z) \Big)$$

#### Gradient Dynamics:
The gradient with respect to logit $z$ is:
$$\frac{\partial \mathcal{L}}{\partial z} = -\frac{(\text{LR} - 1)\sigma(z)(1 - \sigma(z))}{1 + (\text{LR} - 1)\sigma(z)}$$
- When $\text{LR} > 1$ (evidence favors disease), $\frac{\partial \mathcal{L}}{\partial z} < 0$. Gradient descent ($z \leftarrow z - \eta \frac{\partial \mathcal{L}}{\partial z}$) drives the logit $z$ upwards.
- When $\text{LR} < 1$ (evidence rules out disease), $\frac{\partial \mathcal{L}}{\partial z} > 0$. Gradient descent drives the logit $z$ downwards.
- When $\text{LR} = 1$, $\frac{\partial \mathcal{L}}{\partial z} = 0$. The model receives zero gradient penalty, naturally respecting uninformative priors.

---

## 3. Architecture & Implementation in `src/lr_prior.py`

### 3.1 Modular Structure
[`src/lr_prior.py`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/src/lr_prior.py) provides 5 core components:

| Component | Signature | Description |
|---|---|---|
| `compute_likelihood_ratios` | `(soft_targets, base_rates, eps, clamp_range)` | Vectorized conversion with dual NumPy / PyTorch support. Handles arbitrary batch dimensions. |
| `lr_to_posterior` | `(lr, base_rates, eps)` | Analytical inverse mapping to reconstruct posterior probabilities from LRs. |
| `lr_loss` | `(z, lr, reduction="mean")` | Numerically stable PyTorch functional loss using `F.logsigmoid` and `torch.logaddexp`. |
| `LRPriorLoss` | `nn.Module(base_rates, auto_compute_lr)` | Drop-in PyTorch loss module compatible with existing training loops. |
| `StateEvidenceHead` | `nn.Module(hidden_dim, state_spec)` | Multi-state categorical evidence projection head with learnable zero-init residual gating. |

### 3.2 StateEvidenceHead Integration
In heliosli's design, each finding is modeled as having discrete clinical states (e.g. `[absent, equivocal, present]`). `StateEvidenceHead`:
1. Projects class tokens $H \in \mathbb{R}^{B \times C \times D}$ into categorical state logits $S \in \mathbb{R}^{B \times C \times K}$.
2. Normalizes state probabilities via Softmax: $P(s) = \text{Softmax}(S)$.
3. Computes expected prior evidence: $E = \sum_k P(s_k) \cdot \text{Prior}_k$.
4. Computes logit offset relative to baseline: $\Delta z = \text{logit}(E) - \text{logit}(p_0)$.
5. Employs a learnable gate initialized to zero (`self.state_gate = nn.Parameter(torch.zeros(C))`), ensuring that at the start of training, the model's baseline logit output is 100% preserved.

---

## 4. Test & Verification Suite

The verification suite [`tests/test_lr_prior.py`](file:///d:/RSNA%20Knee%20Abnormality%20Detection/tests/test_lr_prior.py) was executed on the AMD R9700 ROCm platform:

```bash
cd /home/eason/rsna-knee && /home/eason/umud/venv-rocm/bin/python tests/test_lr_prior.py
```

### Test Results Summary:

```
============================================================
Running D3 (E_lrprior) Unit Test Suite
============================================================
[1/5] Testing Likelihood Ratio conversion round-trip...
  PASS: Max roundtrip difference = 2.98e-08
[2/5] Testing Bayesian gradient direction...
  PASS: LR=5.0 -> dL/dz = -0.3333 (< 0, drives logit up)
  PASS: LR=0.2 -> dL/dz = 0.3333 (> 0, drives logit down)
[3/5] Testing numerical stability under extreme logits...
  PASS: Extreme z in [-50, 50] produced loss=-1.8421, zero NaNs/Infs.
[4/5] Testing exact equivalence with heliosli reference implementation...
  PASS: Exact match with heliosli reference (diff = 0.00e+00)
[5/5] Testing StateEvidenceHead initialization and forward pass...
  PASS: StateEvidenceHead forward pass valid, shape=torch.Size([4, 12]), init gate=0 intact.
============================================================
ALL 5 UNIT TESTS PASSED SUCCESSFULLY!
============================================================
```

All 5 core requirements are formally validated:
- **Round-trip fidelity**: Inversion is lossless within float32 precision ($2.98 \times 10^{-8}$).
- **Gradient correctness**: Directional derivatives match Bayesian belief updates.
- **Numerical robustness**: Logits from $-50$ to $+50$ produce stable gradients with zero overflow/underflow.
- **Equivalence**: Exact match ($\Delta = 0.00$) against heliosli reference code.
- **Head compatibility**: Forward pass and zero-gate initialization verified.

---

## 5. Usage in Training Loops

To integrate `E_lrprior` into ongoing or upcoming training scripts:

```python
from src.lr_prior import LRPriorLoss

# Define empirical population base rates across the 12 RSNA knee labels
BASE_RATES = torch.tensor([
    0.189, # ACL
    0.121, # MCL
    0.441, # Medial Meniscus
    0.201, # Lateral Meniscus
    0.365, # Medial OA
    0.225, # Lateral OA
    0.369, # PF OA
    0.485, # Effusion
    0.278, # Synovitis
    0.246, # Baker's
    0.249, # Contusion
    0.122  # Fracture
], dtype=torch.float32)

# Option A: Train directly with soft targets [0, 1] (e.g. from VLM teacher or multi-modal consensus)
criterion = LRPriorLoss(base_rates=BASE_RATES, auto_compute_lr=True)

# In training loop:
# logits: (B, 12), soft_targets: (B, 12)
loss = criterion(logits, soft_targets)
loss.backward()
```

---

## 6. Progress on Today's Action Plan

| Task ID | Item | Status | Key Metric / Deliverable |
|:---:|:---|:---:|:---|
| **D1** | 跨标签秩传播 (Effusion $\to$ Synovitis) | **COMPLETED** | Ref `57051453` on Kaggle (`snowy615`)，公榜 `0.534`（Synovitis AUC = 0.908，$\Delta = 0.000$ 证伪简单线性传播） |
| **D4** | 逐标签稠密窗口空间策略 | **COMPLETED** | 58 金标 Macro AUC **0.9050 $\to$ 0.9101** ($\Delta = \mathbf{+0.0052}$) |
| **D2** | 跨标签 12$\to$12 GBDT Stacking | **COMPLETED** | 58 金标 Macro AUC **0.8545 $\to$ 0.8661** ($\Delta = \mathbf{+0.0116}$) |
| **D3** | 补齐 `E_lrprior` 似然比先验与头网络 | **COMPLETED** | 模块已入库，5/5 单元测试通过，与 heliosli 源码差异 **$0.00$** |

