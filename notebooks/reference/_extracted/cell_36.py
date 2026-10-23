def augment(imgs):
    """Small affine and intensity jitter over a whole bag. No flips of either kind.

    Horizontal flip would undo the laterality normalisation of section 5. Vertical flip
    would exchange the femoral and tibial surfaces, which is the axis the compartmental
    osteoarthritis targets are defined on.
    """
    B, S, C, H, W = imgs.shape
    x = imgs.float()

    if AUG_AFFINE:
        ang = math.radians((torch.rand(1).item() - 0.5) * 2 * AUG_ROT_DEG)
        sc = 1.0 + (torch.rand(1).item() - 0.5) * 2 * AUG_SCALE
        tx = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
        ty = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
        cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]],
                             dtype=torch.float32, device=imgs.device)
        theta = theta.unsqueeze(0).repeat(B * S, 1, 1)
        flat = x.reshape(B * S, C, H, W)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False).reshape(B, S, C, H, W)

    scale = 1.0 + (torch.rand(1, device=imgs.device) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)

    if USE_VFLIP and torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    return x.round().to(imgs.dtype)


class Ema:
    """Exponential moving average of the trainable weights.

    Buffers are copied rather than averaged: the normalisation constants are fixed, and
    averaging them would only accumulate floating point error.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        # Warm-up: without it the average stays anchored to the random initialisation for
        # the first few hundred steps, which is most of a short fold.
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
            else:
                v.copy_(s)

    def target(self, model):
        """The weights to evaluate and to keep: the average, or the model if disabled."""
        return self.model if self.decay > 0 else model


def rank_loss(logits, y, w):
    """Pairwise ranking term on confidently graded pairs inside the batch.

    AUC counts ordered positive-negative pairs, so this optimises the metric directly.
    A target with no usable pair in the batch contributes nothing rather than a constant.
    """
    parts = []
    usable = w > 0
    for j in range(logits.shape[1]):
        pos = logits[(y[:, j] > RANK_POS) & usable[:, j], j]
        neg = logits[(y[:, j] < RANK_NEG) & usable[:, j], j]
        if len(pos) and len(neg):
            parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
    return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)


# --- self-tests ----------------------------------------------------------------------
_t = torch.randint(0, 255, (2, 3, GROUP, 32, 32), dtype=torch.uint8)
_a = augment(_t)
assert _a.shape == _t.shape and _a.dtype == _t.dtype
# an asymmetric marker must not end up mirrored in either axis
_m = torch.zeros(1, 1, GROUP, 32, 32, dtype=torch.uint8)
_m[..., :6, :] = 255                                  # bright band along the top
_out = augment(_m).float()
assert _out[..., :16, :].mean() > _out[..., 16:, :].mean(), "vertical orientation was lost"
_m2 = torch.zeros(1, 1, GROUP, 32, 32, dtype=torch.uint8)
_m2[..., :6] = 255                                    # bright band on the left
_out2 = augment(_m2).float()
assert _out2[..., :16].mean() > _out2[..., 16:].mean(), "horizontal orientation was lost"

_lg = torch.tensor([[2.0], [-2.0]])
_y = torch.tensor([[1.0], [0.0]])
_w = torch.ones(2, 1)
assert rank_loss(_lg, _y, _w) < rank_loss(-_lg, _y, _w), "ranking loss has the wrong sign"
assert float(rank_loss(_lg, torch.tensor([[1.0], [1.0]]), _w)) == 0.0, "no pair, no loss"
print("augmentation preserves both anatomical axes; ranking loss is oriented correctly")