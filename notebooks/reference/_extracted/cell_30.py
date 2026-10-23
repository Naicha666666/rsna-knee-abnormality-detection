class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences - cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially - so pooling
    the slots identically would dilute the one that carries the evidence with the rest.

    The aggregation is deliberately this simple. Richer alternatives were measured
    against it and lost: with a study-level label there is no signal telling the model
    which part of a study matters, so extra attention parameters have nothing to learn
    from and spend their capacity fitting noise.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

        # A weak prior keeps noisy study-level supervision from learning obviously
        # implausible view assignments. A value of 0.55 is only a 1.7x softmax tilt,
        # so the learned query can override it when the data disagrees.
        prior = torch.zeros(n_out, n_slot)
        if SLOT_SCHEME == "recovered" and n_slot == 6 and n_out == len(TARGETS):
            preferred = {
                "ACL": (0, 3, 5), "MCL": (1, 4),
                "Medial Meniscus": (0, 1, 3, 4),
                "Lateral Meniscus": (0, 1, 3, 4),
                "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
                "PF OA": (0, 2, 5), "Effusion": (0, 2),
                "Synovitis": (0, 2), "Baker's": (0,),
                "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
            }
            for target, slots in preferred.items():
                prior[TARGETS.index(target), list(slots)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = (torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
               + self.slot_prior.unsqueeze(0))
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
