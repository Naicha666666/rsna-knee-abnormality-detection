def take_group(cache_rows, g):
    """Slice GROUP consecutive channels out of the cached slices."""
    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]


# `augment` is defined further down, after the reasoning for dropping the flip.


@torch.no_grad()
def predict(model, cache, mask, idx, dev):
    """Average the logits over the groups of each slot.

    Training sees one group at a time, which acts as augmentation along the stack;
    inference averages over all of them, which is the same aggregation the frozen
    pipeline used and the one that measured best.
    """
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        rows = torch.from_numpy(cache[sel]).to(dev)
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(take_group(rows, g), m).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / N_GROUP).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.nanmean([roc_auc_score(y[:, j], p[:, j])
                             if len(set(y[:, j])) > 1 else np.nan
                             for j in range(y.shape[1])]))
