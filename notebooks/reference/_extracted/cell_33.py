def build_model():
    """Load the encoder and open the last few blocks for training.

    Only the last UNFREEZE_LAST blocks and the final norm are trainable. The early
    blocks of a self-supervised transformer are generic edge and texture filters; there
    is not enough supervision here to improve them and quite enough to damage them.
    """
    from transformers import AutoModel
    p = find_dinov2(BACKBONE_VARIANT)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - UNFREEZE_LAST):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size * 3
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {UNFREEZE_LAST} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim}")
    return Model(bb, dim)
