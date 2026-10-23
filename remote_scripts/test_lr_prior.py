"""
Comprehensive Unit Tests for E_lrprior and StateEvidenceHead (D3 Deliverable)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from src.lr_prior import (
    compute_likelihood_ratios,
    lr_to_posterior,
    lr_loss,
    LRPriorLoss,
    StateEvidenceHead
)

def test_lr_conversion_roundtrip():
    print("[1/5] Testing Likelihood Ratio conversion round-trip...")
    p = torch.tensor([[0.05, 0.20, 0.50, 0.85, 0.95],
                      [0.10, 0.40, 0.60, 0.70, 0.90]], dtype=torch.float32)
    p0 = torch.tensor([0.15, 0.30, 0.45, 0.60, 0.75], dtype=torch.float32)
    
    lr = compute_likelihood_ratios(p, p0)
    reconstructed_p = lr_to_posterior(lr, p0)
    
    diff = torch.abs(p - reconstructed_p).max().item()
    assert diff < 1e-4, f"Roundtrip error too high: {diff}"
    print(f"  PASS: Max roundtrip difference = {diff:.2e}")


def test_lr_gradient_direction():
    print("[2/5] Testing Bayesian gradient direction...")
    # Positive case: LR > 1 (report supports positive)
    z_pos = torch.zeros(1, 1, requires_grad=True)
    lr_high = torch.tensor([[5.0]]) # 5x odds of positive
    loss_pos = lr_loss(z_pos, lr_high)
    loss_pos.backward()
    
    # Negative loss gradient means moving in -grad direction INCREASES z (pushes it positive)
    assert z_pos.grad.item() < 0, f"Expected negative gradient for LR > 1, got {z_pos.grad.item()}"
    print(f"  PASS: LR=5.0 -> dL/dz = {z_pos.grad.item():.4f} (< 0, drives logit up)")
    
    # Negative case: LR < 1 (report supports negative)
    z_neg = torch.zeros(1, 1, requires_grad=True)
    lr_low = torch.tensor([[0.2]]) # 0.2x odds
    loss_neg = lr_loss(z_neg, lr_low)
    loss_neg.backward()
    assert z_neg.grad.item() > 0, f"Expected positive gradient for LR < 1, got {z_neg.grad.item()}"
    print(f"  PASS: LR=0.2 -> dL/dz = {z_neg.grad.item():.4f} (> 0, drives logit down)")


def test_numerical_stability():
    print("[3/5] Testing numerical stability under extreme logits...")
    extreme_z = torch.tensor([[-50.0, -20.0, 0.0, 20.0, 50.0]], requires_grad=True)
    extreme_lr = torch.tensor([[1e-3, 0.1, 1.0, 10.0, 1e3]])
    
    loss = lr_loss(extreme_z, extreme_lr)
    loss.backward()
    
    assert torch.isfinite(loss), "Loss is non-finite!"
    assert torch.isfinite(extreme_z.grad).all(), "Gradients contain NaN/Inf!"
    print(f"  PASS: Extreme z in [-50, 50] produced loss={loss.item():.4f}, zero NaNs/Infs.")


def test_equivalence_with_heliosli_reference():
    print("[4/5] Testing exact equivalence with heliosli reference implementation...")
    # Reference function from extracted_sources/heliosli_abnormality/compact_train.py line 503
    def ref_lr_loss(z, lr):
        return (-torch.logaddexp(F.logsigmoid(z) + torch.log(lr), F.logsigmoid(-z))).mean(0).mean()
        
    torch.manual_seed(42)
    z = torch.randn(16, 12, requires_grad=True)
    lr = torch.rand(16, 12).clamp(1e-2, 1e2)
    
    l_our = lr_loss(z, lr)
    l_ref = ref_lr_loss(z, lr)
    
    diff = abs(l_our.item() - l_ref.item())
    assert diff < 1e-7, f"Discrepancy with heliosli reference: {diff}"
    print(f"  PASS: Exact match with heliosli reference (diff = {diff:.2e})")


def test_state_evidence_head():
    print("[5/5] Testing StateEvidenceHead initialization and forward pass...")
    labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", 
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    spec = {
        "labels": labels,
        "categories": {l: ["absent", "equivocal", "present"] for l in labels},
        "prior": {l: [0.05, 0.50, 0.95] for l in labels},
        "base": {l: 0.25 for l in labels}
    }
    
    head = StateEvidenceHead(hidden_dim=64, state_spec=spec, num_classes=12)
    tokens = torch.randn(4, 12, 64)
    evidence, probs = head(tokens)
    
    assert evidence.shape == (4, 12), f"Wrong evidence shape: {evidence.shape}"
    assert probs.shape == (4, 12, 3), f"Wrong probs shape: {probs.shape}"
    # Gate initialized to 0 -> evidence should be 0 at init
    assert torch.allclose(evidence, torch.zeros_like(evidence)), "Evidence not zero at init!"
    print(f"  PASS: StateEvidenceHead forward pass valid, shape={evidence.shape}, init gate=0 intact.")

if __name__ == "__main__":
    print("=" * 60)
    print("Running D3 (E_lrprior) Unit Test Suite")
    print("=" * 60)
    test_lr_conversion_roundtrip()
    test_lr_gradient_direction()
    test_numerical_stability()
    test_equivalence_with_heliosli_reference()
    test_state_evidence_head()
    print("=" * 60)
    print("ALL 5 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
