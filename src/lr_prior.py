"""
Likelihood-Ratio Prior (E_lrprior) and Evidence Head Module for RSNA Knee Abnormality Detection.
Extracted and ported from heliosli Stage-2 compact_train architecture.

Key Mathematical Principles:
1. Likelihood Ratio (Bayes Factor) Transformation:
   Given report soft label probability p in (0, 1) and population base rate p0:
       odds = p / (1 - p)
       base_odds = p0 / (1 - p0)
       LR = odds / base_odds = (p / (1 - p)) / (p0 / (1 - p0))
   LR measures the fold-change in odds relative to the unconditional population prior.
   Unlike BCE which saturates in high confidence regions, LR scales proportionally to evidence.

2. Likelihood Ratio Loss (lr_loss):
   Given model logit z (where sigma(z) is image evidence) and target Likelihood Ratio LR:
       loss(z, LR) = -logaddexp(log(sigma(z)) + log(LR), log(sigma(-z)))
                   = -log(1 + (LR - 1) * sigma(z))
   This directly optimizes the marginal joint agreement under Bayesian evidence theory.
"""

from typing import Optional, Union, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =========================================================================
# 1. NumPy / PyTorch Likelihood Ratio Conversion Utilities
# =========================================================================

def compute_likelihood_ratios(
    soft_targets: Union[np.ndarray, "torch.Tensor"],
    base_rates: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    eps: float = 1e-4,
    clamp_range: Tuple[float, float] = (1e-3, 1e3)
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convert soft label probabilities into Likelihood Ratios (LR = odds / base_odds).
    
    Args:
        soft_targets: Array or Tensor of probabilities in [0, 1], shape (N, C)
        base_rates: Array or Tensor of base rates in (0, 1), shape (C,). If None, calculated from mean(0).
        eps: Minimum distance from 0 and 1 before computing odds.
        clamp_range: (min_lr, max_lr) to prevent numerical divergence.
        
    Returns:
        Likelihood ratios with same shape and type as soft_targets.
    """
    is_torch = HAS_TORCH and isinstance(soft_targets, torch.Tensor)
    
    if is_torch:
        p_clamped = torch.clamp(soft_targets, eps, 1.0 - eps)
        odds = p_clamped / (1.0 - p_clamped)
        
        if base_rates is None:
            base_p = torch.clamp(soft_targets.mean(dim=0, keepdim=True), eps, 1.0 - eps)
        else:
            base_p = torch.clamp(torch.as_tensor(base_rates, dtype=soft_targets.dtype, device=soft_targets.device), eps, 1.0 - eps)
            if base_p.ndim == 1:
                base_p = base_p.unsqueeze(0)
                
        base_odds = base_p / (1.0 - base_p)
        lr = torch.clamp(odds / base_odds, clamp_range[0], clamp_range[1])
        return lr
    else:
        p_clamped = np.clip(np.asarray(soft_targets, dtype=np.float32), eps, 1.0 - eps)
        odds = p_clamped / (1.0 - p_clamped)
        
        if base_rates is None:
            base_p = np.clip(p_clamped.mean(axis=0, keepdims=True), eps, 1.0 - eps)
        else:
            base_p = np.clip(np.asarray(base_rates, dtype=np.float32), eps, 1.0 - eps)
            if base_p.ndim == 1:
                base_p = np.expand_dims(base_p, 0)
                
        base_odds = base_p / (1.0 - base_p)
        lr = np.clip(odds / base_odds, clamp_range[0], clamp_range[1])
        return lr


def lr_to_posterior(
    lr: Union[np.ndarray, "torch.Tensor"],
    base_rates: Union[np.ndarray, "torch.Tensor"],
    eps: float = 1e-4
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Inverse transform: Reconstruct posterior probabilities from Likelihood Ratio and base rates.
    """
    is_torch = HAS_TORCH and isinstance(lr, torch.Tensor)
    if is_torch:
        base_p = torch.clamp(torch.as_tensor(base_rates, dtype=lr.dtype, device=lr.device), eps, 1.0 - eps)
        base_odds = base_p / (1.0 - base_p)
        post_odds = lr * base_odds
        return post_odds / (1.0 + post_odds)
    else:
        base_p = np.clip(np.asarray(base_rates, dtype=np.float32), eps, 1.0 - eps)
        base_odds = base_p / (1.0 - base_p)
        post_odds = lr * base_odds
        return post_odds / (1.0 + post_odds)


# =========================================================================
# 2. PyTorch Loss Implementation (E_lrprior)
# =========================================================================

if HAS_TORCH:
    def lr_loss(z: torch.Tensor, lr: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Likelihood-Ratio loss as defined in heliosli compact_train.py:
        loss(z, lr) = -logaddexp(logsigmoid(z) + log(lr), logsigmoid(-z))
        
        Args:
            z: Logits from model, shape (B, C)
            lr: Likelihood Ratios > 0, shape (B, C)
            reduction: "mean", "sum", or "none"
            
        Returns:
            Scalar loss or loss tensor of shape (B, C)
        """
        # logsigmoid(z) = log(sigma(z)), logsigmoid(-z) = log(1 - sigma(z))
        log_term_pos = F.logsigmoid(z) + torch.log(lr.clamp_min(1e-6))
        log_term_neg = F.logsigmoid(-z)
        loss = -torch.logaddexp(log_term_pos, log_term_neg)
        
        if reduction == "mean":
            return loss.mean(0).mean() # mean over classes, then mean over batch
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    class LRPriorLoss(nn.Module):
        """
        Loss module implementing E_lrprior training objective.
        Supports both precomputed Likelihood Ratio targets, or computes them from soft targets on the fly.
        """
        def __init__(
            self,
            base_rates: Optional[torch.Tensor] = None,
            auto_compute_lr: bool = False,
            reduction: str = "mean",
            clamp_range: Tuple[float, float] = (1e-3, 1e3)
        ):
            super().__init__()
            self.auto_compute_lr = auto_compute_lr
            self.reduction = reduction
            self.clamp_range = clamp_range
            
            if base_rates is not None:
                self.register_buffer("base_rates", torch.as_tensor(base_rates, dtype=torch.float32))
            else:
                self.base_rates = None

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: Predicted logits of shape (B, C)
                targets: Likelihood ratios if auto_compute_lr=False, or soft probabilities in [0, 1] if auto_compute_lr=True.
            """
            if self.auto_compute_lr:
                lr = compute_likelihood_ratios(targets, self.base_rates, clamp_range=self.clamp_range)
            else:
                lr = targets
            return lr_loss(logits, lr, reduction=self.reduction)


    class StateEvidenceHead(nn.Module):
        """
        Report-State Evidence Alignment Head.
        Projects class token representation into discrete categorical states (e.g. absent, mild, severe)
        and aligns them with official labels via precomputed prior vectors.
        """
        def __init__(self, hidden_dim: int, state_spec: dict, num_classes: int = 12):
            super().__init__()
            self.num_classes = num_classes
            cats = state_spec["categories"]
            self.state_k = [len(cats[l]) for l in state_spec["labels"]]
            kmax = max(self.state_k)
            
            self.state_w = nn.Parameter(torch.randn(num_classes, kmax, hidden_dim) * 0.02)
            self.state_b = nn.Parameter(torch.zeros(num_classes, kmax))
            
            prior = torch.full((num_classes, kmax), float("nan"))
            valid = torch.zeros((num_classes, kmax), dtype=torch.bool)
            for i, l in enumerate(state_spec["labels"]):
                for j, pv in enumerate(state_spec["prior"][l]):
                    prior[i, j] = float(pv)
                    valid[i, j] = True
                    
            self.register_buffer("state_prior", torch.nan_to_num(prior, nan=0.5).clamp(1e-3, 1 - 1e-3))
            self.register_buffer("state_valid", valid)
            base_vals = [float(state_spec["base"][l]) for l in state_spec["labels"]]
            self.register_buffer("state_base_logit", torch.logit(torch.tensor(base_vals).clamp(1e-3, 1 - 1e-3)))
            
            # Learnable gate initialized to 0 so direct prediction path is unchanged at init
            self.state_gate = nn.Parameter(torch.zeros(num_classes))

        def forward(self, study_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                study_tokens: Tensor of shape (B, num_classes, hidden_dim)
            Returns:
                evidence_logits: Additional logit evidence of shape (B, num_classes)
                state_probs: Softmax state distribution of shape (B, num_classes, kmax)
            """
            # (B, C, H) x (C, K, H) -> (B, C, K)
            sl = torch.einsum("bch,ckh->bck", study_tokens, self.state_w) + self.state_b
            sl = sl.masked_fill(~self.state_valid.unsqueeze(0), float("-inf"))
            
            p = F.softmax(sl, dim=-1)
            e = (p * self.state_prior.unsqueeze(0)).sum(dim=-1).clamp(1e-3, 1 - 1e-3)
            evidence = torch.logit(e) - self.state_base_logit.unsqueeze(0)
            
            gated_evidence = self.state_gate.unsqueeze(0) * evidence
            return gated_evidence, p
