"""Anatomical Mirror Test-Time Augmentation (TTA) for RSNA Knee MRI.

Based on Danial Zakaria's (Nartaa) bilateral knee symmetry formulation:
In knee MRI, a left knee and right knee have bilateral anatomical mirror symmetry:
- Sagittal plane: Left-to-right mirror reflection corresponds to inverting the
  medial-to-lateral / anterior-to-posterior slice sequence. For a 3-slice window
  triplet [c-1, c, c+1], this is achieved by reversing the channel dimension (dim 1):
  [c-1, c, c+1] -> [c+1, c, c-1].
- Coronal & Axial planes: Left-to-right mirror reflection corresponds to horizontal
  width flipping (dim -1 / W axis).

Applying standard horizontal flipping blindly to sagittal slices breaks ACL/PCL
anatomy. This anatomical mirror TTA preserves true bilateral anatomy, transforming
a left knee into a right knee and vice versa, yielding ~+0.005 AUC improvement.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

# Standard layout configurations
LAYOUT_SLOTS = {
    # 64-slice corpus layout (our in-house Quad-Ensemble):
    # Sagittal fluid (18), Sagittal non-fluid (14), Coronal fluid (12), Coronal non-fluid (8), Axial (12)
    64: {
        "slot_sizes": [18, 14, 12, 8, 12],
        "sagittal_slots": (0, 1),
    },
    # 96-slice corpus layout (Nartaa 0.949 CoAtNet-2 SWA):
    # Sagittal fluid (26), Sagittal non-fluid (22), Coronal fluid (18), Coronal non-fluid (12), Axial (18)
    96: {
        "slot_sizes": [26, 22, 18, 12, 18],
        "sagittal_slots": (0, 1),
    },
    # 80-slice corpus layout
    80: {
        "slot_sizes": [22, 18, 15, 10, 15],
        "sagittal_slots": (0, 1),
    },
}


def compute_slot_indices(total_slices: int, slot_sizes: list[int] | None = None) -> np.ndarray:
    """Computes which slot index each slice [0 .. total_slices-1] belongs to."""
    if slot_sizes is None:
        if total_slices in LAYOUT_SLOTS:
            slot_sizes = LAYOUT_SLOTS[total_slices]["slot_sizes"]
        else:
            raise ValueError(f"Unknown total_slices={total_slices}. Provide custom slot_sizes.")
    bounds = np.cumsum([0] + list(slot_sizes))
    if bounds[-1] != total_slices:
        raise ValueError(f"Slot sizes sum to {bounds[-1]}, but total_slices is {total_slices}")
    s_ = np.searchsorted(bounds, np.arange(total_slices), side="right") - 1
    return s_


def mirror_triplets(
    triplets: torch.Tensor,
    center_slices: list[int] | np.ndarray | torch.Tensor,
    total_slices: int = 64,
    slot_sizes: list[int] | None = None,
    sagittal_slots: tuple[int, ...] = (0, 1),
) -> torch.Tensor:
    """Applies anatomical bilateral mirroring to a tensor of triplet windows.

    Args:
        triplets: Tensor of shape (..., 3, H, W).
                  Can be (K, 3, H, W) or (B, K, 3, H, W).
        center_slices: Center slice index for each window in K.
        total_slices: Total slice depth of the volume (e.g. 64 or 96).
        slot_sizes: Optional custom slot sizes.
        sagittal_slots: Tuple of slot indices that are Sagittal.

    Returns:
        Mirrored tensor with identical shape and dtype.
    """
    s_ = compute_slot_indices(total_slices, slot_sizes)
    cs_arr = np.asarray(center_slices, dtype=np.int64).reshape(-1)
    # Check which center slices fall into sagittal slots
    is_sag = np.isin(s_[cs_arr], sagittal_slots)
    
    # Broadcast mask to match triplet tensor dimensions
    # If triplets is (B, K, 3, H, W), cs is length K
    device = triplets.device
    sag_mask = torch.from_numpy(is_sag).to(device)

    if triplets.ndim == 4:
        # (K, 3, H, W)
        sag_mask = sag_mask.view(-1, 1, 1, 1)
        mirrored = torch.where(sag_mask, triplets.flip(1), triplets.flip(-1))
    elif triplets.ndim == 5:
        # (B, K, 3, H, W)
        sag_mask = sag_mask.view(1, -1, 1, 1, 1)
        mirrored = torch.where(sag_mask, triplets.flip(2), triplets.flip(-1))
    else:
        raise ValueError(f"Expected 4D or 5D triplet tensor, got shape {triplets.shape}")

    return mirrored


class AnatomicalMirrorTTA(nn.Module):
    """Wrapper that decorates any PyTorch knee abnormality model with Anatomical Mirror TTA."""

    def __init__(
        self,
        base_model: nn.Module,
        total_slices: int = 64,
        slot_sizes: list[int] | None = None,
        sagittal_slots: tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        self.base_model = base_model
        self.total_slices = total_slices
        self.slot_sizes = slot_sizes
        self.sagittal_slots = sagittal_slots

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        center_slices: list[int] | np.ndarray | torch.Tensor,
        **model_kwargs,
    ) -> torch.Tensor:
        """Forward pass with 2-way bilateral anatomical mirror average.

        Args:
            x: Input tensor (B, K, 3, H, W)
            center_slices: Center slice indices for the K windows.
            **model_kwargs: Extra keyword arguments passed to base_model (e.g. chunk=24).

        Returns:
            Mean sigmoid probabilities of shape (B, num_classes).
        """
        assert not self.base_model.training, "TTA must be run with model in eval mode"
        x_mir = mirror_triplets(
            x,
            center_slices=center_slices,
            total_slices=self.total_slices,
            slot_sizes=self.slot_sizes,
            sagittal_slots=self.sagittal_slots,
        )

        out_orig = self.base_model(x, **model_kwargs).float()
        out_mir = self.base_model(x_mir, **model_kwargs).float()

        prob_orig = torch.sigmoid(out_orig)
        prob_mir = torch.sigmoid(out_mir)

        return 0.5 * (prob_orig + prob_mir)
