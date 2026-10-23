# Head and shoulders, knees and toes

<p align="center">
  <img src="https://playbees.firstcry.com/blog/wp-content/uploads/2025/12/Head-Shoulders-Knees-and-Toes-Lyrics-Nursery-Rhymes-For-Kids.jpg" width="760" alt="Head, Shoulders, Knees and Toes nursery-rhyme illustration">
</p>
<p align="center"><sub>Illustration supplied by the user via <a href="https://www.bing.com/images/search?view=detailV2&ccid=R3yueXDs&id=615F980032BC4C1280484744A510D964D1306DA6">Bing Images / FirstCry PlayBees</a>.</sub></p>

## A 0.935 knee-MRI checkpoint you can actually start from

This notebook is the inference checkpoint, not a leaderboard scrapbook. Give
it the competition data and two T4s; it reconstructs each study from DICOM,
runs three deliberately different model families, and writes the twelve-column
submission in about the same time it takes to make coffee.

### Quick start

1. Join **RSNA Knee Abnormality Detection** so Kaggle can mount the test data.
2. Leave the attached inputs unchanged, select **GPU T4 ×2**, keep internet off,
   and run **Save Version**.
3. The visible test produces three rows. That is Kaggle's decoy; the scored
   rerun automatically expands to 1,322 hidden studies.

The exact H&S prediction version **V36** scored **0.935**
publicly. This release changes its front page only; the four prediction cells
are byte-identical to that DINOsaur V4 with the MaxSpan Raptor checkpoint.

## The signal path

| Leg | Representation | Job in the ensemble |
|---|---|---|
| DINOv2 + DINOv3 | six protocol-aware MRI slots | broad transformer consensus |
| RadImageNet | reference and fat-sensitive layouts | medical-domain error diversity |
| Raptor CoAtNet | 2.5D study windows | finding-specific spatial evidence |

The first two legs form a calibrated transformer/RadImageNet parent. Raptor is
then blended by target in rank space; the meniscus and fracture weights are
preserved from the scored source. Ranking is important here: it lets models
with very different probability calibration vote on the same scale.

## Provenance trail

This is a faithful package, not a claim that every ingredient began here.

| Layer | Public source | What is retained |
|---|---|---|
| Scored graph | [Roman Tamrazov — DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) | four-cell 0.934 inference graph, Apache-2.0 |
| CoAtNet fusion lineage | [Muhammad Aldo Fahrezy — CoAtNet Transformer Blend](https://www.kaggle.com/code/aldofhrzy/rsna-knee-coatnet-transformer-blend) | target-wise rank fusion |
| Raptor checkpoint | [Dread Development — Raptor weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) | study-level CoAtNet weights |
| Reproduction bundle | [Tony Li — RSNA Knee reproduction assets](https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets) | consolidated DINO/RadImageNet assets |
| Base encoder | [Meta DINOv2](https://github.com/facebookresearch/dinov2) | standard ViT-S/14 base |

The Kaggle **Inputs** panel is the definitive inventory and license surface.
Competition images, reports, labels, and row-wise predictions are not
redistributed by this notebook.

## Reproducibility contract

- Scored source: `romantamrazov/rsna-knee-dinosaur-v4`, V1.
- H&S inference parent: V36, four cells, unchanged after its
  scored run.
- Parent contract: only the Raptor checkpoint/span/window cell differs from V35.
- Final visible-output schema: `StudyInstanceUID` plus the twelve official
  targets, with finite cohort-rank scores.

This is a competition checkpoint, not a clinical device.
