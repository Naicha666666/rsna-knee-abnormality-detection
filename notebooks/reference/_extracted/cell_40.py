def train_one_fold(fold, Ctr, Mtr, Y, W, tr, va, gi_va, gold_y_va, dev):
    """Train one fold and return (ema_state, history, val_predictions)."""
    model = build_model().to(dev)
    ema = Ema(model, EMA_DECAY)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    steps = max(EPOCHS * (len(tr) // BATCH_STUDIES), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    yv = (Y[va] > 0.5).astype(int)
    best, best_state, history = -1.0, None, []

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tot, nstep = 0.0, 0
        for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
            sel = perm[b:b + BATCH_STUDIES]
            rows = torch.from_numpy(Ctr[sel]).to(dev)
            g = int(torch.randint(N_GROUP, (1,)).item())
            imgs = augment(take_group(rows, g))
            m = torch.from_numpy(Mtr[sel]).to(dev)
            y = torch.from_numpy(Y[sel]).to(dev)
            w = torch.from_numpy(W[sel]).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(imgs, m)
                loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).mean()
                if RANK_LOSS_W > 0:
                    loss = loss + RANK_LOSS_W * rank_loss(z.float(), y, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item()
            nstep += 1

        eval_model = ema.target(model)
        pv = predict(eval_model, Ctr, Mtr, va, dev)
        d = macro_auc(yv, pv)

        # The annotated reference for this fold is only the annotated studies held out of
        # it. Scoring the ones it trained on is what made the previous rule optimistic.
        g_auc = float("nan")
        if gold_y_va is not None and len(gi_va) >= 6:
            g_auc = macro_auc(gold_y_va, predict(eval_model, Ctr, Mtr, gi_va, dev))

        history.append({"fold": fold, "epoch": ep + 1, "loss": tot / max(nstep, 1),
                        "derived": d, "annot": g_auc})
        log(f"fold {fold} epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}"
            f"  derived {d:.4f}  annot(held out) {g_auc:.4f}")

        if SELECTION == "derived" or not np.isfinite(g_auc):
            score = d
        else:
            score = min(d, g_auc)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            log(f"  best so far ({SELECTION} {score:.4f})")
        if time.time() - T0 > TIME_BUDGET:
            log("time budget reached inside fold")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in ema.target(model).state_dict().items()}
    del model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_state, history, best