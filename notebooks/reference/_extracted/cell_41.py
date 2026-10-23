def save_bundle(fold_states, extra=None):
    """Write the weights together with the configuration they were trained under.

    The inference notebook rebuilds the architecture from these fields rather than from
    its own constants. Without them a slot list or an image size that drifted apart
    between the two notebooks would load cleanly and predict nonsense.
    """
    bundle = {
        "fold_states": fold_states,
        "targets": list(TARGETS),
        "slots": [list(s) for s in SLOTS],
        "slot_scheme": SLOT_SCHEME,
        "img": IMG,
        "group": GROUP,
        "n_group": N_GROUP,
        "crop_mm": CROP_MM,
        "model_variant": f"dinov2-{BACKBONE_VARIANT}",
        "unfreeze_last": UNFREEZE_LAST,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    bundle.update(extra or {})
    torch.save(bundle, BUNDLE_FILE)
    size = os.path.getsize(BUNDLE_FILE) / 1e6
    log(f"saved {BUNDLE_FILE}: {len(fold_states)} state(s), {size:.0f} MB")


def main():
    train_df = pd.read_csv(ROOT / "train.csv")
    train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)
    train_series = pd.read_csv(ROOT / "train_series.csv")
    train_series["StudyInstanceUID"] = train_series["StudyInstanceUID"].astype(str)
    train_series["SeriesInstanceUID"] = train_series["SeriesInstanceUID"].astype(str)
    log(f"train {train_df.shape}; train_series {train_series.shape}")

    plane_map = dict(zip(train_series["SeriesInstanceUID"], train_series["Anatomical_Plane"]))
    log("header pass: train")
    htr = annotate(walk("train_series"))
    log(f"  {len(htr)} train series")

    lat_tr, lat_info = laterality_maps(htr)
    globals()["LAT_INFO"] = lat_info

    slots_tr = pick_slots(htr, plane_map)
    cov = pd.Series([len(v) for v in slots_tr.values()]).describe()
    log(f"train slots per study: mean {cov['mean']:.2f} min {cov['min']:.0f} max {cov['max']:.0f}")
    globals()["SLOT_COVERAGE"] = pd.Series(
        {name: float(np.mean([name in v for v in slots_tr.values()])) for name, *_ in SLOTS})
    globals()["WEIGHT_TABLE"] = htr.groupby(["weight", "fatsat"]).size()

    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_tr, "train")

    # ---- targets ---------------------------------------------------------- #
    t_lab = time.time()
    lab = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    lab["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    lab = lab.set_index("StudyInstanceUID")
    log(f"derived labels for {len(lab)} studies in {time.time() - t_lab:.1f}s")

    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]

    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = gold.loc[st].values, GOLD_WEIGHT
        elif st in lab.index:
            r = lab.loc[st]
            Y[i] = r[TARGETS].values
            W[i] = 0.25 + 0.75 * r[[t + "__conf" for t in TARGETS]].values
    keep = np.where(W.sum(1) > 0)[0]
    log(f"supervised {len(keep)} of {len(st_tr)} studies (annotated {len(gold)})")

    # Folds are grouped on report text: identical reports across studies yield one target
    # vector for all of them, so splitting such a group would score a model on a target
    # whose source it had already trained on.
    import hashlib
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
                    for s in st_tr])
    gpos = {s: i for i, s in enumerate(st_tr)}
    gi_all = np.array([gpos[s] for s in gold.index if s in gpos])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_states, histories, fold_scores = [], [], []
    oof_gold = np.full((len(st_tr), len(TARGETS)), np.nan, np.float32)

    max_folds = N_FOLDS if MAX_FOLDS == "auto" else int(MAX_FOLDS)
    for fold in range(min(N_FOLDS, max_folds)):
        va = np.array([i for i in keep if grp[i] == fold])
        tr = np.array([i for i in keep if grp[i] != fold])
        if len(va) == 0 or len(tr) < BATCH_STUDIES:
            log(f"fold {fold}: not enough studies, skipped")
            continue
        gi_va = np.array([i for i in gi_all if grp[i] == fold])
        gold_y_va = (gold.loc[[st_tr[i] for i in gi_va]].values.astype(int)
                     if len(gi_va) else None)
        log(f"=== fold {fold}: train {len(tr)} / holdout {len(va)} "
            f"(annotated held out: {len(gi_va)}) ===")

        for rep_i in range(SEEDS_PER_FOLD):
            seed = SEED + 1000 * fold + rep_i
            np.random.seed(seed)
            torch.manual_seed(seed)
            t_fold = time.time()
            state, history, best = train_one_fold(fold, Ctr, Mtr, Y, W, tr, va,
                                                  gi_va, gold_y_va, dev)
            fold_time = time.time() - t_fold
            histories.extend(history)
            fold_scores.append(best)
            fold_states.append({"fold": fold, "seed": seed, "score": float(best),
                                "state_dict": state})

            # Out-of-fold predictions for the annotated studies held out of this fold.
            # Only the first seed contributes, so the estimate stays a clean average of
            # one prediction per study rather than a seed ensemble on some studies only.
            if len(gi_va) and rep_i == 0:
                model = build_model().to(dev)
                model.load_state_dict(state)
                oof_gold[gi_va] = predict(model, Ctr, Mtr, gi_va, dev)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            save_bundle(fold_states, {"fold_scores": fold_scores})
            log(f"fold {fold} seed {seed} done in {fold_time / 60:.1f} min")

            elapsed = time.time() - T0
            if elapsed + FOLD_TIME_PAD * fold_time > TIME_BUDGET:
                log(f"stopping after {len(fold_states)} state(s): another needs "
                    f"{FOLD_TIME_PAD * fold_time / 60:.0f} min and "
                    f"{(TIME_BUDGET - elapsed) / 60:.0f} min remain")
                break
        else:
            continue
        break

    globals()["HISTORY"] = pd.DataFrame(histories)
    if not fold_states:
        log("no fold completed; nothing to save")
        return

    seen = np.isfinite(oof_gold[gi_all]).all(axis=1) if len(gi_all) else np.zeros(0, bool)
    oof_auc = None
    if seen.sum() >= 8:
        y_true = gold.loc[[st_tr[i] for i in gi_all]].values.astype(int)[seen]
        oof_auc = macro_auc(y_true, oof_gold[gi_all][seen])
        globals()["OOF_GOLD"] = (y_true, oof_gold[gi_all][seen])
        log(f"out-of-fold macro AUC on {int(seen.sum())} annotated studies: {oof_auc:.4f}")

    save_bundle(fold_states, {"fold_scores": fold_scores,
                              "oof_macro_auc": oof_auc,
                              "n_train_studies": int(len(keep))})
    log(f"fold selection scores: {[round(s, 4) for s in fold_scores]}")

    if not RUN_TEST_INFERENCE:
        log("training only; run the inference notebook against the bundle to submit")
        return

    # ---- optional: the old single-notebook path ---------------------------- #
    test_df = pd.read_csv(ROOT / "test.csv")
    test_df["StudyInstanceUID"] = test_df["StudyInstanceUID"].astype(str)
    test_series = pd.read_csv(ROOT / "test_series.csv")
    test_series["StudyInstanceUID"] = test_series["StudyInstanceUID"].astype(str)
    test_series["SeriesInstanceUID"] = test_series["SeriesInstanceUID"].astype(str)
    plane_map_te = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))
    hte = annotate(walk("test_series"))
    lat_te, _ = laterality_maps(hte)
    slots_te = pick_slots(hte, plane_map_te)
    st_te, Cte, Mte = build_cache(slots_te, plane_map_te, lat_te, "test")

    rank_sum = np.zeros((len(st_te), len(TARGETS)), np.float64)
    for entry in fold_states:
        model = build_model().to(dev)
        model.load_state_dict(entry["state_dict"])
        P = predict(model, Cte, Mte, np.arange(len(st_te)), dev)
        rank_sum += pd.DataFrame(P).rank(pct=True).values
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    P = rank_sum / len(fold_states)
    sub = pd.DataFrame(P, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", st_te)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv("submission.csv", index=False)
    log(f"submission.csv {sub.shape}; models {len(fold_states)}")
    print(sub.head().to_string())