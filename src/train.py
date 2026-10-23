"""Training / CV for RSNA Knee (run inside Kaggle Notebook, internet OFF).

Key practices wired in from top public notebooks (see LOG.md):
- GroupKFold grouped by report-text hash so identical/duplicated reports never
  leak across folds.
- Per-label confidence weights (from report_extractor) so a study silent on a
  finding is a weak, not confident, negative; gold labels get weight 1.0.
- BCE (weighted) + pairwise ranking loss; EMA shadow weights for inference.
- Epoch selection: each epoch is scored on the holdout as min(derived-AUC,
  gold-AUC) and the best epoch's EMA state is restored ("worse_of_two").
- Macro-AUC evaluated only on the held-out GOLD-labeled portion (NaN-aware).
- Rank-averaged OOF across folds for the final ensembled prediction.

This module is the single source of truth for the training protocol. The
comparison study (experiments.py) calls run_cv() with different BACKBONEs while
keeping EVERYTHING else fixed, so differences in score reflect the backbone only.
"""
from __future__ import annotations
import os
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from config import (LABELS, N_FOLDS, BATCH_SIZE, EPOCHS, LR, SEED, PROJECT_ROOT,
                    IMG_SIZE,
                    USE_EMA, RANK_LOSS_WEIGHT, ENCODER_UNFREEZE_LAST,
                    TRAIN_TIME_BUDGET_H, WEIGHT_DECAY, ONECYCLE_PCT_START,
                    SELECTION)
from dataset import RSNADataset, build_slot_cache, take_group
import model as M

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ---------------- offline metrics (no sklearn dependency) ----------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC; y_true binary (0/1), y_score continuous."""
    yt = y_true.astype(float)
    n_pos = yt.sum(); n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_r = ranks[yt == 1].sum()
    return (sum_r - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ok = np.isfinite(yt)
        if ok.sum() < 2:
            aucs.append(float("nan")); continue
        yt, yp = yt[ok], yp[ok]
        if len(np.unique(yt)) < 2:
            aucs.append(float("nan")); continue
        aucs.append(roc_auc(yt, yp))
    return np.nanmean(aucs), aucs


def rank_average(fold_preds: list[np.ndarray]) -> np.ndarray:
    """Average predictions across folds in rank space (robust to per-fold drift)."""
    out = np.zeros_like(fold_preds[0])
    N = fold_preds[0].shape[0]
    for j in range(fold_preds[0].shape[1]):
        ranks = np.zeros(N)
        for p in fold_preds:
            ranks += _rankdata(p[:, j])
        ranks /= len(fold_preds)
        pooled = np.concatenate([p[:, j] for p in fold_preds])
        q = (ranks - 0.5) / N
        out[:, j] = np.quantile(pooled, q)
    return out


def _report_hash(report: str) -> int:
    return int(hashlib.md5(report.encode("utf-8", "ignore")).hexdigest(), 16)


# ---------------- training on an in-memory cache ----------------
def _prepare_imgs(cache, idx, g, device, do_aug=False):
    rows = torch.from_numpy(take_group(cache[idx], g))  # [B, NS, 3, H, W] uint8
    rows = rows.to(device)
    x = rows.float().div_(255.0)
    if do_aug:
        x = M.augment(x)
    return x


@torch.no_grad()
def predict_groups(model, cache, mask, idx, n_groups, device, batch=16):
    """Average logits over the 2.5D groups of each slot."""
    model.eval()
    out = []
    idx = np.asarray(idx)
    for b0 in range(0, len(idx), batch):
        sel = idx[b0:b0 + batch]
        m = torch.from_numpy(mask[sel]).to(device)
        acc = None
        use_amp = device == "cuda"
        for g in range(n_groups):
            x = _prepare_imgs(cache, sel, g, device, do_aug=False)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = model(x, m).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / n_groups).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, len(LABELS)), np.float32)


def train_on_cache(cache, mask, y, w, tr_idx, epochs, backbone_name,
                   unfreeze_last, lr, batch, n_groups, device=None,
                   va_idx=None, y_gold_va=None):
    """Train one model on cached 2.5D groups. Returns (model, ema).

    Epoch selection (config.SELECTION): after each
    epoch the EMA weights are evaluated on the holdout twice — against the
    binarised derived targets (all holdout studies with weight>0) and, when
    >=6 gold studies are held out, against the gold labels. The epoch whose
    min(derived, gold) macro-AUC is best is restored at the end, so a bad
    final epoch cannot ship. Without a holdout the final EMA state is used.
    """
    device = device or DEVICE
    model = M.build_model(backbone_name, unfreeze_last)
    model.to(device)
    groups = M.get_param_groups(model, lr)
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=WEIGHT_DECAY)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    rankloss = M.PairwiseRankingLoss()
    ema = M.EMA(model) if USE_EMA else None
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tr_idx = np.asarray(tr_idx)
    n = len(tr_idx)
    rng = np.random.RandomState(SEED)
    # Drop the leftover incomplete batch so OneCycleLR step counts match
    # and rank-loss pairs are not computed on a 1-2 study remainder.
    steps_per = max(n // batch, 1)
    total_steps = max(epochs * steps_per, 1)
    max_lrs = [g.get("lr", lr) for g in groups]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START)

    do_select = (ema is not None and va_idx is not None and len(va_idx) > 0)
    if do_select:
        va_idx = np.asarray(va_idx)
        yd = y[va_idx].astype(float).copy()
        yd[w[va_idx] <= 0] = np.nan          # silent targets are not evidence
        yd_bin = np.where(np.isfinite(yd), (yd > 0.5).astype(float), np.nan)
        has_gold = (y_gold_va is not None and
                    np.isfinite(y_gold_va).any(axis=1).sum() >= 6)
    sel_state, sel_score = None, -1.0

    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        nb, loss_sum = 0, 0.0
        for b0 in range(0, n - batch + 1, batch):
            sel = tr_idx[order[b0:b0 + batch]]
            g = int(rng.randint(0, n_groups))
            imgs = _prepare_imgs(cache, sel, g, device, do_aug=True)
            masks = torch.from_numpy(mask[sel]).to(device)
            yt = torch.from_numpy(y[sel]).to(device)
            wt = torch.from_numpy(w[sel]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs, masks)
                loss_bce = (bce(logits, yt) * wt).sum() / wt.sum().clamp(min=1.0)
                loss = loss_bce + RANK_LOSS_WEIGHT * rankloss(logits, yt, wt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            loss_sum += float(loss.detach())
            nb += 1
        msg = f"    epoch {ep + 1}/{epochs}  loss={loss_sum / max(nb, 1):.4f}"
        if do_select:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
            pred = predict_groups(model, cache, mask, va_idx, n_groups, device)
            der, _ = macro_auc(yd_bin, pred)
            gld, _ = (macro_auc(y_gold_va, pred) if has_gold
                      else (der, None))
            score = min(der, gld) if SELECTION == "worse_of_two" else der
            if np.isfinite(score) and score > sel_score:
                sel_score = score
                sel_state = {k: v.detach().clone()
                             for k, v in ema.state.items()}
            msg += (f"  sel[{SELECTION}]={score:.4f} (der={der:.4f}"
                    + (f", gold={gld:.4f}" if has_gold else "") + ")")
            model.load_state_dict(raw)
        print(msg, flush=True)

    if ema:
        if sel_state is not None:
            ema.state = sel_state            # checkpoint saves the best epoch
        ema.apply_to(model)
    model.eval()
    return model, ema


def _subset_indices(ds, max_studies: int | None):
    """If max_studies given, keep ALL gold studies + randomly sample the rest so
    the CV validation stays meaningful (and reproducible via SEED)."""
    idx = np.arange(len(ds))
    if max_studies is None or max_studies >= len(ds):
        return idx
    rng = np.random.RandomState(SEED)
    gold = np.where(ds.has_gold)[0]
    if max_studies <= len(gold):
        return rng.choice(gold, size=max_studies, replace=False)
    rest = np.setdiff1d(idx, gold)
    take = rng.choice(rest, size=min(max_studies - len(gold), len(rest)),
                      replace=False)
    return np.concatenate([gold, take])


def run_cv(backbone_name: str = "tiny_cnn", unfreeze_last: int = 2,
           max_studies: int | None = None, epochs: int = EPOCHS,
           folds: int = N_FOLDS, batch: int = BATCH_SIZE, lr: float = LR):
    """Run the fixed-protocol 5-fold CV for one backbone. Returns a dict with
    OOF predictions, fold scores, per-label AUCs and the split assignment
    (needed for reproducibility logging)."""
    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    reports = [ds.reports[i] for i in idx]
    strat = np.nansum(ds.y_raw[idx], axis=1).astype(int).clip(0, 5)
    groups = np.array([_report_hash(r) for r in reports])

    if folds == 1:
        # Single fixed group-disjoint 80/20 split (controlled comparison mode).
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
        tr, va = next(gss.split(idx, strat, groups))
        splits = [(tr, va)]
    else:
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(idx, strat, groups))

    cache, cmask, n_groups = build_slot_cache(ds, idx)
    y = ds.targets[idx]
    w = ds.weights[idx]
    oof = np.full((len(idx), len(LABELS)), np.nan, dtype=np.float32)
    fold_preds, fold_scores, fold_splits = [], [], []
    for fold, (tr, va) in enumerate(splits):
        fold_splits.append((idx[tr].tolist(), idx[va].tolist()))
        print(f"Fold {fold}: train={len(tr)} val={len(va)}", flush=True)
        model, _ema = train_on_cache(
            cache, cmask, y, w, tr, epochs, backbone_name, unfreeze_last,
            lr, batch, n_groups, va_idx=va, y_gold_va=ds.y_raw[idx[va]])
        pred = predict_groups(model, cache, cmask, va, n_groups, DEVICE)
        oof[va] = pred
        fold_preds.append(pred)
        golds = ds.y_raw[idx[va]]
        _, per = macro_auc(golds, pred)
        macro = np.nanmean([a for a in per if np.isfinite(a)])
        fold_scores.append(macro)
        print(f"Fold {fold}: macro-AUC={macro:.4f}  " +
              ", ".join(f"{l[:4]}:{a:.2f}" for l, a in zip(LABELS, per)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall, _ = macro_auc(ds.y_raw[idx], oof)
    per_label = []
    for j in range(len(LABELS)):
        yt = ds.y_raw[idx][:, j]; yp = oof[:, j]
        ok = np.isfinite(yt)
        per_label.append(roc_auc(yt[ok], yp[ok]) if ok.sum() >= 2 and
                         len(np.unique(yt[ok])) >= 2 else float("nan"))
    return dict(oof=oof, fold_preds=fold_preds, fold_scores=fold_scores,
                macro=float(overall), per_label=per_label,
                idx=idx.tolist(), fold_splits=fold_splits,
                strat=strat.tolist(), groups=groups.tolist())


def _write_submission(uids, preds, path):
    import pandas as pd
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for i, lab in enumerate(LABELS):
        sub[lab] = preds[:, i]
    sub = sub[["StudyInstanceUID"] + LABELS]
    sub.to_csv(path, index=False)
    print(f"wrote {path} ({len(sub)} rows)", flush=True)


def infer_test_ensemble(models, n_groups, device, series_root=None):
    """Decode the test set once, run each fold model, rank-average."""
    test = RSNADataset("test", series_root=series_root)
    uids = test.uids
    if not uids:
        return uids, np.zeros((0, len(LABELS)), np.float32)
    idx = np.arange(len(uids))
    cache, mask, ng = build_slot_cache(test, idx, n_groups=n_groups)
    fold_preds = []
    for mi, model in enumerate(models):
        print(f"infer test model {mi + 1}/{len(models)} ({len(uids)} studies)",
              flush=True)
        fold_preds.append(predict_groups(model, cache, mask, idx, ng, device))
    del cache, mask
    if len(fold_preds) == 1:
        return uids, fold_preds[0]
    return uids, rank_average(fold_preds)


def find_checkpoints(root: str = "/kaggle/input", max_depth: int = 4):
    """Every `*.pt` fold checkpoint mounted as a Kaggle dataset input.

    The DINOv2 backbone weights are `.pth`, so a `.pt` filter separates trained
    fold checkpoints from pretrained encoder weights without naming either one.
    """
    found = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, files in os.walk(root):
        if dirpath[len(root):].count(os.sep) > max_depth:
            dirnames[:] = []
            continue
        for f in files:
            if f.endswith(".pt"):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def run_infer_only(out_dir: str = ".", device: str | None = None):
    """Rank-average every mounted fold checkpoint over the test set. No training.

    Used to ensemble folds from SEPARATE training kernels (a code competition
    reruns the kernel on the hidden test, so cross-run ensembling cannot be done
    by combining local CSVs -- the checkpoints have to be carried into one
    inference kernel as a dataset).
    """
    import time
    t0 = time.time()
    dev = device or DEVICE
    ckpts = find_checkpoints()
    print(f"ensemble: {len(ckpts)} checkpoint(s) found", flush=True)
    for c in ckpts:
        print("   ", c, flush=True)
    if not ckpts:
        raise RuntimeError("no *.pt checkpoints under /kaggle/input")

    models, n_groups = [], None
    for c in ckpts:
        blob = torch.load(c, map_location="cpu", weights_only=False)
        bb = blob.get("backbone", "dinov2_small")
        ul = int(blob.get("unfreeze_last", ENCODER_UNFREEZE_LAST))
        ng = int(blob.get("n_groups", 1))
        # The cache is built once for all models, so the ensemble runs at the
        # smallest group count any member was trained with.
        n_groups = ng if n_groups is None else min(n_groups, ng)
        ck_img = int(blob.get("img_size", 224))
        # The cache is decoded once at config.IMG_SIZE, and the ViT grid is built
        # from it, so a checkpoint trained at another size would be fed the wrong
        # token count with the wrong pos_embed rows -- silently, since the slice
        # `pos_embed[:, :n_token]` always succeeds. Fail loudly instead.
        if ck_img != IMG_SIZE:
            raise RuntimeError(
                f"{os.path.basename(c)} was trained at img_size={ck_img} but this "
                f"kernel decodes at {IMG_SIZE}; ensemble one resolution per kernel")
        m = M.build_model(bb, unfreeze_last=ul)
        m.load_state_dict(blob["state_dict"])
        m.to(dev).eval()
        models.append(m)
        print(f"    loaded {os.path.basename(c)} backbone={bb} n_groups={ng}",
              flush=True)

    uids, preds = infer_test_ensemble(models, n_groups, dev)
    out_csv = os.path.join(out_dir, "submission.csv")
    _write_submission(uids, preds, out_csv)
    print(f"infer-only finished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(models)} models", flush=True)
    return out_csv


def run_submit(backbone_name: str = "dinov2_small", unfreeze_last: int = 2,
               max_studies: int | None = None, epochs: int = EPOCHS,
               folds: int = N_FOLDS, batch: int = BATCH_SIZE, lr: float = LR,
               out_dir: str = ".", time_budget_h: float = TRAIN_TIME_BUDGET_H):
    """Train a report-hash fold ensemble and write submission.csv incrementally.

    After each finished fold we overwrite submission.csv with the rank-average
    of the models we have so far, so a time-budget stop still leaves a real
    (not placeholder) file.
    """
    import time
    t0 = time.time()
    budget = time_budget_h * 3600.0
    out_csv = os.path.join(out_dir, "submission.csv")

    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    reports = [ds.reports[i] for i in idx]
    strat = np.nansum(ds.y_raw[idx], axis=1).astype(int).clip(0, 5)
    groups = np.array([_report_hash(r) for r in reports])
    print(f"submit train: {len(idx)} studies / gold={int(ds.has_gold.sum())} "
          f"backbone={backbone_name} ep={epochs} folds={folds}", flush=True)

    if folds <= 1:
        splits = [(np.arange(len(idx)), np.array([], dtype=int))]
    else:
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(idx, strat, groups))

    cache, cmask, n_groups = build_slot_cache(ds, idx)
    y = ds.targets[idx]
    w = ds.weights[idx]

    models, fold_times = [], []
    for fold, (tr, va) in enumerate(splits):
        elapsed = time.time() - t0
        if fold > 0:
            mean_ft = float(np.mean(fold_times))
            if elapsed + 1.15 * mean_ft > budget:
                print(f"time budget: stopping after {fold} folds "
                      f"(elapsed={elapsed / 60:.1f}m, next~{mean_ft / 60:.1f}m)",
                      flush=True)
                break
        print(f"=== submit fold {fold + 1}/{len(splits)} "
              f"train={len(tr)} val={len(va)} ===", flush=True)
        ft0 = time.time()
        model, ema = train_on_cache(
            cache, cmask, y, w, tr, epochs, backbone_name, unfreeze_last,
            lr, batch, n_groups, va_idx=va, y_gold_va=ds.y_raw[idx[va]])
        if len(va):
            pred = predict_groups(model, cache, cmask, va, n_groups, DEVICE)
            golds = ds.y_raw[idx[va]]
            mac, per = macro_auc(golds, pred)
            print(f"fold {fold} val macro-AUC={mac:.4f}", flush=True)
        fold_times.append(time.time() - ft0)
        ckpt = os.path.join(out_dir, f"fold{fold}.pt")
        state = ema.state_dict() if ema else model.state_dict()
        torch.save({"state_dict": state, "backbone": backbone_name,
                    "unfreeze_last": unfreeze_last, "n_groups": n_groups,
                    "img_size": IMG_SIZE}, ckpt)
        models.append(model)
        print(f"fold {fold} done in {fold_times[-1]:.0f}s -> {ckpt}", flush=True)

    del cache, cmask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uids, preds = infer_test_ensemble(models, n_groups, DEVICE)
    _write_submission(uids, preds, out_csv)
    print(f"submit finished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(models)} folds", flush=True)
    return out_csv


def main():
    res = run_cv("tiny_cnn", unfreeze_last=2)
    print(f"\nOOF macro-AUC = {res['macro']:.4f}  "
          f"(mean fold = {np.mean(res['fold_scores']):.4f})")
    out = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "oof_preds.npy"), res["oof"])
    print("saved OOF ->", out)


if __name__ == "__main__":
    main()
