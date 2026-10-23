class Model(nn.Module):
    """Encoder plus head, trained end to end.

    A study arrives as a bag of slot images. The bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.
    """

    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim, N_SLOT, len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        # The mean is stable for diffuse disease; top-k pooling preserves focal high
        # responses from tears, contusions and fractures that a global mean dilutes.
        k = max(1, patch.shape[1] // 8)
        focal = patch.topk(k, dim=1).values.mean(1)
        feat = torch.cat([out[:, 0], patch.mean(1), focal], dim=1).reshape(B, S, -1)
        return self.head(feat, mask)
