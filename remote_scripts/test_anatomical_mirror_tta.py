import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anatomical_mirror_tta import compute_slot_indices, mirror_triplets, AnatomicalMirrorTTA

def test_slot_indices():
    # 64-slice test
    s64 = compute_slot_indices(64)
    assert len(s64) == 64
    assert (s64[:18] == 0).all()
    assert (s64[18:32] == 1).all()
    assert (s64[32:44] == 2).all()
    assert (s64[44:52] == 3).all()
    assert (s64[52:64] == 4).all()
    print("Slot indices 64-slice test PASSED!")

    # 96-slice test
    s96 = compute_slot_indices(96)
    assert len(s96) == 96
    assert (s96[:26] == 0).all()
    assert (s96[26:48] == 1).all()
    assert (s96[48:66] == 2).all()
    assert (s96[66:78] == 3).all()
    assert (s96[78:96] == 4).all()
    print("Slot indices 96-slice test PASSED!")

def test_mirror_triplets_4d():
    # K=4, 3 channels, 8x8
    K, C, H, W = 4, 3, 8, 8
    # Center slices: 2 from sagittal (slots 0, 1), 2 from coronal/axial (slots 2, 4)
    cs = [5, 25, 35, 55]  # for 64 slices: 5 in s0, 25 in s1, 35 in s2, 55 in s4
    
    t = torch.zeros(K, C, H, W)
    # Channel gradient for sagittal:
    t[0, 0, :, :] = 1.0
    t[0, 1, :, :] = 2.0
    t[0, 2, :, :] = 3.0

    # Width gradient for coronal/axial:
    t[2, :, :, :] = torch.arange(W).view(1, 1, W).expand(C, H, W).float()

    mir = mirror_triplets(t, cs, total_slices=64)

    # For window 0 (sagittal): dim 1 should be flipped [3, 2, 1]
    assert (mir[0, 0] == 3.0).all()
    assert (mir[0, 1] == 2.0).all()
    assert (mir[0, 2] == 1.0).all()
    print("Window 0 (Sagittal channel flip) PASSED!")

    # For window 2 (coronal): width dim -1 should be flipped
    expected_w = torch.flip(torch.arange(W), dims=[0]).float()
    for c in range(C):
        for h in range(H):
            assert torch.allclose(mir[2, c, h, :], expected_w)
    print("Window 2 (Coronal width flip) PASSED!")

def test_mirror_triplets_5d():
    # B=2, K=4, 3, 8, 8
    B, K, C, H, W = 2, 4, 3, 8, 8
    cs = [10, 20, 40, 60]
    t = torch.randn(B, K, C, H, W)
    mir = mirror_triplets(t, cs, total_slices=64)
    assert mir.shape == t.shape

    # Double mirror should restore original
    mir2 = mirror_triplets(mir, cs, total_slices=64)
    assert torch.allclose(t, mir2)
    print("5D involution property (mirror(mirror(x)) == x) PASSED!")

def test_tta_wrapper():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8 * 8, 12)
        def forward(self, x, chunk=24):
            # x: (B, K, 3, H, W)
            feats = x[:, :, 0].mean(dim=1).flatten(1)
            return self.linear(feats)

    base = DummyModel().eval()
    tta = AnatomicalMirrorTTA(base, total_slices=64)
    
    x = torch.randn(2, 4, 3, 8, 8)
    cs = [5, 20, 35, 55]
    prob = tta(x, center_slices=cs)
    assert prob.shape == (2, 12)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    print("AnatomicalMirrorTTA end-to-end wrapper PASSED!")

if __name__ == "__main__":
    test_slot_indices()
    test_mirror_triplets_4d()
    test_mirror_triplets_5d()
    test_tta_wrapper()
    print("\nALL ANATOMICAL MIRROR TTA TESTS PASSED SUCCESSFULLY!")
