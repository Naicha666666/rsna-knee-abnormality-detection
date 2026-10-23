"""Percentile Rank Blend and State-Evidence Gating Utilities.

Ported and extended from heliosli/rsna-knee-blend-gold.

Key features:
1. Robust Percentile Rank Blending with automatic failover (if model A fails on a study,
   fall back gracefully to model B instead of poisoning submission with NaN / 0.5).
2. Dual Linear + Logit Quantile Rank Fusion for maximum Macro-AUC calibration.
3. Anti-Tie Micro-Rank Jitter to break ties in low-probability dense predictions.
4. State-Evidence Residual Head calibration logic.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def rank_pct(v: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Compute normalized percentile ranks in [0, 1] using average tie-breaking.
    
    Equivalent to pd.Series(v).rank(method='average', pct=True).values
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(v) == 1:
        return np.array([0.5], dtype=np.float64)
    return pd.Series(v).rank(method="average", pct=True).values


def rank_blend_predictions(
    preds_a: pd.DataFrame | np.ndarray,
    preds_b: pd.DataFrame | np.ndarray,
    weight_a: float = 0.7,
    weight_b: float = 0.3,
    fail_threshold: float = 0.03,
    cols: list[str] | None = None
) -> np.ndarray:
    """Robust percentile rank blend of two prediction matrices with failover.
    
    If model A fails on a study (e.g. constant 0.5 or NaN), the study falls back
    to model B's rank, and vice versa.
    
    Args:
        preds_a: [N, C] prediction array or DataFrame for model A
        preds_b: [N, C] prediction array or DataFrame for model B
        weight_a: fixed ensemble weight for model A (default: 0.7)
        weight_b: fixed ensemble weight for model B (default: 0.3)
        fail_threshold: maximum allowable failure rate before raising error
        cols: optional column names
        
    Returns:
        [N, C] blended rank array
    """
    va = np.asarray(preds_a.values if isinstance(preds_a, pd.DataFrame) else preds_a, dtype=np.float64)
    vb = np.asarray(preds_b.values if isinstance(preds_b, pd.DataFrame) else preds_b, dtype=np.float64)
    assert va.shape == vb.shape, f"Shape mismatch: {va.shape} vs {vb.shape}"
    N, C = va.shape
    
    # Identify failed studies
    fail_a = np.isnan(va).any(axis=1) | (va == 0.5).all(axis=1)
    fail_b = np.isnan(vb).any(axis=1) | (vb == 0.5).all(axis=1)
    
    if fail_a.sum() > fail_threshold * N:
        raise RuntimeError(f"Model A failed on {fail_a.sum()} of {N} studies (>{fail_threshold*100}%)")
    if fail_b.sum() > fail_threshold * N:
        raise RuntimeError(f"Model B failed on {fail_b.sum()} of {N} studies (>{fail_threshold*100}%)")
        
    out = np.zeros((N, C), dtype=np.float64)
    for j in range(C):
        ra = rank_pct(va[:, j])
        rb = rank_pct(vb[:, j])
        
        blended = weight_a * ra + weight_b * rb
        blended[fail_a] = rb[fail_a]
        blended[fail_b] = ra[fail_b]
        blended[fail_a & fail_b] = 0.5
        out[:, j] = blended
        
    return out


class DualQuantileRankBlender:
    """Dual Linear + Logit Quantile Rank Fusion with Anti-Tie Jitter.
    
    Combines linear percentile rank fusion with logit-space percentile fusion:
        R_linear = (1 - w) * R_1 + w * R_2
        R_logit  = sigma((1 - w) * logit(R_1) + w * logit(R_2))
        R_dual   = 0.5 * R_linear + 0.5 * R_logit + micro_jitter
    """
    def __init__(self, eps: float = 1e-4, jitter_scale: float = 1e-5):
        self.eps = eps
        self.jitter_scale = jitter_scale
        
    def blend(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        weights: np.ndarray | list[float] | float
    ) -> np.ndarray:
        """Blend two probability / rank matrices [N, C] with class-specific weights."""
        N, C = r1.shape
        if isinstance(weights, (float, int)):
            w = np.full(C, float(weights), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            
        out = np.zeros((N, C), dtype=np.float64)
        for j in range(C):
            rk1 = rank_pct(r1[:, j])
            rk2 = rank_pct(r2[:, j])
            
            # Linear rank blend
            r_linear = (1.0 - w[j]) * rk1 + w[j] * rk2
            
            # Logit-space quantile blend (bounded away from 0 and 1)
            rk1_clamped = np.clip(rk1, self.eps, 1.0 - self.eps)
            rk2_clamped = np.clip(rk2, self.eps, 1.0 - self.eps)
            logit1 = np.log(rk1_clamped / (1.0 - rk1_clamped))
            logit2 = np.log(rk2_clamped / (1.0 - rk2_clamped))
            logit_blend = (1.0 - w[j]) * logit1 + w[j] * logit2
            r_logit = 1.0 / (1.0 + np.exp(-logit_blend))
            
            # Dual blend + anti-tie micro-jitter
            dual = 0.50 * r_linear + 0.50 * r_logit
            jitter = (rk1 - 0.5) * self.jitter_scale
            out[:, j] = np.clip(dual + jitter, 0.0, 1.0)
            
        return out


BaseHead = nn.Module if HAS_TORCH else object


class StateEvidenceHead(BaseHead):
    """Port of Helios Li's State-Evidence Head for Report-State Residual Guidance.
    
    Learns an auxiliary K-way report state classifier for each pathology.
    Gains an evidence residual:
        evidence_c = logit(sum_s p(s|x) * prior_c(s)) - logit(base_c)
        logit_c = direct_logit_c + gate_c * evidence_c
    Gate initialized to 0 so uncalibrated training starts purely on direct branch.
    """
    def __init__(self, hidden_dim: int, num_classes: int, state_k: list[int],
                 state_prior: torch.Tensor, base_rate: torch.Tensor):
        super().__init__()
        self.num_classes = num_classes
        self.kmax = max(state_k)
        
        self.state_w = nn.Parameter(torch.randn(num_classes, self.kmax, hidden_dim) * 0.02)
        self.state_b = nn.Parameter(torch.zeros(num_classes, self.kmax))
        
        # Valid mask for ragged category counts
        valid = torch.zeros((num_classes, self.kmax), dtype=torch.bool)
        for i, k in enumerate(state_k):
            valid[i, :k] = True
        self.register_buffer("state_valid", valid)
        self.register_buffer("state_prior", state_prior.clamp(1e-3, 1.0 - 1e-3))
        self.register_buffer("state_base_logit", torch.logit(base_rate.clamp(1e-3, 1.0 - 1e-3)))
        
        # Gating parameter (initialized to 0)
        self.state_gate = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, study_tokens: torch.Tensor, direct_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gated logits and state distributions.
        
        Args:
            study_tokens: [B, num_classes, hidden_dim]
            direct_logits: [B, num_classes]
            
        Returns:
            gated_logits: [B, num_classes]
            state_logits: [B, num_classes, kmax]
        """
        # State logits
        sl = torch.einsum("bch,ckh->bck", study_tokens, self.state_w).float() + self.state_b
        sl = sl.masked_fill(~self.state_valid[None], float("-inf"))
        
        # State probabilities & evidence
        p = sl.softmax(dim=-1)
        e = (p * self.state_prior[None]).sum(dim=-1).clamp(1e-3, 1.0 - 1e-3)
        evidence = torch.logit(e) - self.state_base_logit[None]
        
        gated_logits = direct_logits + self.state_gate[None] * evidence
        return gated_logits, sl


if __name__ == "__main__":
    # Self-test
    rng = np.random.default_rng(42)
    p_a = rng.uniform(0.1, 0.9, size=(58, 12))
    p_b = rng.uniform(0.1, 0.9, size=(58, 12))
    p_a[0, :] = 0.5  # Simulate 1 failed study
    
    blended = rank_blend_predictions(p_a, p_b, weight_a=0.7, weight_b=0.3)
    assert blended.shape == (58, 12)
    assert (blended >= 0).all() and (blended <= 1).all()
    print("Rank blend self-test passed! Blended shape:", blended.shape)
    
    dual_blender = DualQuantileRankBlender()
    w_vec = np.linspace(0.1, 0.5, 12)
    dual_out = dual_blender.blend(p_a, p_b, w_vec)
    assert dual_out.shape == (58, 12)
    print("DualQuantileRankBlender self-test passed!")
