# RSNA Knee — Method Comparison (paper-style)

> Controlled experiment: the training protocol is FIXED; only the image backbone varies. CV macro-AUC is computed on the held-out GOLD portion (public labels are only 58 studies, so it is a *relative* ranking signal; the hidden test leaderboard is the final arbiter).

## Results

| exp_id | backbone | tune | data | ep | macro-AUC | mean±std fold |
|---|---|---|---|---|---|---|
| dinov2_base_600_ep5 | dinov2_base | ft | sub600 | 5 | 0.5351 | 0.5769±0.0803 |
| dinov2_small_600_ep5 | dinov2_small | ft | sub600 | 5 | 0.5390 | 0.5765±0.0404 |
| tiny_cnn_600_ep5 | tiny_cnn | ft | sub600 | 5 | 0.4483 | 0.5392±0.0741 |

## Per-label AUC (held-out gold)

| exp_id | ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|------|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_base_600_ep5 | 0.414 | 0.517 | 0.701 | 0.470 | 0.580 | 0.515 | 0.525 | 0.692 | 0.504 | 0.467 | 0.502 | 0.535 |
| dinov2_small_600_ep5 | 0.439 | 0.449 | 0.698 | 0.481 | 0.668 | 0.578 | 0.425 | 0.656 | 0.528 | 0.522 | 0.437 | 0.588 |
| tiny_cnn_600_ep5 | 0.542 | 0.243 | 0.494 | 0.470 | 0.453 | 0.393 | 0.467 | 0.619 | 0.484 | 0.353 | 0.304 | 0.560 |

## Reproducibility

Each `experiments/exp_<id>/` contains `config.json`, `metrics.json`, `oof.npy`, and `provenance.json` (env + source SHA-256 + fold assignment + CLI).

## Limitations

- Public gold labels = 58 studies; CV macro-AUC is a noisy *relative* signal, not an absolute estimate. Final ranking uses the hidden test leaderboard (public + private).
- Physical-scale crop and laterality normalization rely on DICOM headers present in the Kaggle environment; they degrade gracefully (fixed crop / no flip) if stripped.
- Screening runs use subset=600 studies / 5 epochs / 5-fold; they establish the *backbone* ranking, not an absolute score.
