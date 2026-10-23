# Pre-registration: genuine bottleneck-focused specialist (path C, on-the-fly DICOM render)

Date: 2026-10-17. Author: X&Y. Status: registered BEFORE the run.

## Why this exists

The A2 plan ("inject `specialist_auc_ranker_best.pt` into 4 bottleneck columns") was built on an
unvalidated asset. Reconstruction of its trainer (`scratchpad/build_specialist_auc_finetune_nb.py`,
cell 6) shows the "specialist" was fine-tuned for 150 steps on `torch.randn` noise, never on real
images, and its "gold holdout" evaluation also used noise. Directions 2 (`build_auc_loss_train_nb.py`)
and 3 (`build_3d_multimodal_train_nb.py`) have no training loop at all ("cell 5: Training Loop" just
writes a manifest with `status: COMPLETE`). No self-trained arm therefore has any evidence of task
skill. A2 as literally specified is falsified, not merely unverified.

This prereg replaces it: train a **genuine** compartment specialist on real competition images.

## Task and data

- Source: the competition `rsna-knee-abnormality-detection`, 4,407 train studies
  (`train.csv` + `train_series.csv` + `train_series/<uid>/<series>/*.dcm`).
- Targets: `easonyanyan/rsna-knee-s6-multimodal-targets` (`targets_s6_multimodal.csv`), soft labels
  from LLM report reading; arm column `<label>|lr_prior`. 58 studies are `is_gold == 1` and are
  **never** rendered for training nor used in any hyperparameter decision.
- Rendering: the mainline's own code (probe_infer.install_native + px_cache.StudySlices), the exact
  same path the 0.958 submission uses at inference, so train and test windows share geometry,
  intensity mapping and view set (`views == v1`, `size == 256`, 3 slices per window).

## Model

- Encoder: `timm` ConvNeXt-Small (in_chans=3), initialised from the real task-trained compact
  backbone `E_lrprior_s0_f0.pt` (R6 bundle), NOT from noise and NOT from ImageNet alone.
- Head: per-class attention pooling over the study's windows + linear classifier, 12 logits
  (the same MIL shape as mainline `compact_train.Net` with `--spatial-attn`, re-implemented
  self-contained so no meta-column contract is inherited).

## Loss

Soft-target BCE on the `|lr_prior` columns, with a fixed per-class boost concentrating capacity on
the bottleneck findings (bottleneck_map.md): Synovitis, Lateral OA, PF OA, Lateral Meniscus.

    CLASS_BOOST = {Synovitis: 3.0, Lateral OA: 3.0, PF OA: 3.0, Lateral Meniscus: 2.0, else 1.0}

No gold-derived weighting, no per-label weights fitted on labels.

## Fixed hyperparameters (no sweep on gold)

    SIZE=256  VIEWS='v1'  SLICES=3  MAX_WIN=96/study
    EPOCHS=4  LR_ENC=1e-5  LR_HEAD=1e-4  WD=0.02  AMP=fp16
    N_TRAIN cap = 1600 pool studies (deterministic: by hash-sorted UID), 4 render workers
    OPT=AdamW  GRAD_CLIP=1.0

## Readout (the only accepted evidence)

- **Primary**: macro-AUC and per-label AUC on the 58 gold, printed every epoch. The specialist is
  considered usable for an A2-style injection only if, on the 4 bottleneck labels, gold AUC is
  >= the 0.958 base reader's gold AUC on those labels (the base is dumped in the same run from
  the frozen E_lrprior windows for a like-for-like comparison).
- **Secondary (monitor, not selection)**: AUC on pool `hash(uid) % 4 == 0` (soft targets).
- Single model, single seed. No gold-based early stopping.

## Decision rule (fixed now)

1. If specialist bottleneck-label gold AUC > base by a margin outside the ±0.003 noise band on at
   least 2 of the 4 bottleneck labels -> build an A2+B1 injection kernel (rank-blend, pre-registered
   small weight) and submit as a challenger.
2. If within the noise band -> keep the 0.958 head; the specialist is not injected.
3. If below -> discard; record the negative result and close the self-trained-arm line.

## Compute envelope

One Kaggle T4 session, no internet. Cache-free (renders per study on the fly). If the training cell
does not finish 4 epochs + gold eval inside the session, the run is void (no partial weights shipped).

## Outcome (2026-10-18, LOG #695) -- decision rule #3

Run `georgexue666/rsna-knee-bottleneck-specialist` v3 COMPLETE (4/4 epochs, wall 5.85h, errs 0).

| bottleneck label | specialist (ep3) | base (same 58 gold) | delta |
|---|---:|---:|---:|
| Synovitis | 0.7915 | 0.8076 | -0.016 |
| Lateral OA | 0.8008 | 0.8491 | -0.048 |
| PF OA | 0.8327 | 0.8417 | -0.009 |
| Lateral Meniscus | 0.8509 | 0.8944 | -0.044 |

All four are below base; 0/4 exceed it by a positive margin -> rule #1 not met -> **rule #3: discard,
recorded as a negative result, self-trained-arm line closed**. A2+B1 injection cancelled.

Deviations, recorded honestly: the run did **not** dump the base AUC in-session (no `base` block in
`specialist_report.json`), so the like-for-like base column above was recomputed locally from the
existing `results/gold58_preds/gold58_top3_predictions.npz` on the identical 58 gold. MONITOR was NaN
every epoch (monitor set = pool `hash%4==0`, no ground truth), so it contributed nothing to selection.
PF OA reached 0.8764 at ep0 (+0.035 over base) but decayed with training and is a single early-epoch
label (1/4), so it does not satisfy rule #1.
