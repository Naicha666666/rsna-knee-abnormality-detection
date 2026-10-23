<!-- Cell 1 -->
**What this is.** The community's 0.943 inference stack, run unchanged, plus one new model family that we
trained ourselves: a 2.5D ConvNeXt-Tiny reader with a small study-level transformer. The two are combined
by a single rank blend (70% stack, 30% ours). That blend scored **0.944** on the public leaderboard.

**Why share it.** Most forks of the public stack differ only in blend constants. This notebook adds a
model that was trained independently of every public checkpoint, and it shows the full inference code and
the weights, so you can check it, reuse the reader on its own, or swap in your own model at the same place.

| | Public LB |
|---|---|
| Community stack alone (verified by [skarin](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack)) | 0.943 |
| Our reader alone, 3 folds | 0.929 |
| **Stack 70% + our reader 30% (this notebook)** | **0.944** |

The 0.944 was scored on 4 October 2026 by our private run of exactly this pipeline: same code, same three
checkpoints, same weight. This public copy reads the checkpoints from a public dataset instead.

**Reading the figure at the top.** Panel 1 is the reader's data flow. Panel 2 shows the three leaderboard numbers behind
the title. Panel 3 compares the reader's AUC per finding on two different yardsticks: held-out folds scored
against labels derived from the radiology reports (blue), and the 58 studies that carry official radiologist
labels (orange). Where orange sits far to the right of blue, the model agrees with the radiologists better
than the report-derived labels suggest. Synovitis is the opposite case: most reports never mention it, so
labels built from reports teach the model the wrong thing.

## What is ours, and what is not

**Not ours (Part A of this notebook, cells unchanged).** Every model in the stack, and all of the stack's
inference code, is the work of the authors listed under *Thanks* below. We changed nothing in it.

**Ours (Part B).**
1. **Preprocessing.** Each DICOM series becomes a canonical volume: 0.4 mm per pixel, a fixed 154 mm
   field of view, one standard orientation per plane derived from `ImageOrientationPatient`, slices sorted
   by their position along the slice normal, and one intensity window per series.
2. **Model.** Up to six series per study (three planes, fat-suppressed or not). Twelve windows of three
   neighbouring slices per series go through one shared ConvNeXt-Tiny. Each window becomes a token tagged
   with its series type and slice position; a two-layer transformer reads all tokens of the study; one
   attention pooling per finding produces the twelve outputs.
3. **Training.** Five folds grouped by report text, soft targets, plain BCE, 14 epochs at 256 px. The
   official 58 gold-labelled studies were never trained on; they are only used as a check.
   The soft targets are the mean of four public label tables (see *Thanks*).
4. **The blend.** One global weight (0.30), chosen before submitting and not tuned on the leaderboard.
   If our reader fails for any reason, the stack's submission is kept.

## Things worth knowing before you fork

- **Runtime.** The stack is slow on the hidden test set: our submission needed between six and a half and
  eight hours from submit to score, out of the nine allowed. Our reader is the small part of that (alone,
  with five checkpoints, it was scored in under ninety minutes). Do not add much more on top.
- **Leaderboard fit.** The stack's blend constants were tuned by its authors on the public leaderboard
  and on the 58 gold studies. Our reader was validated by cross-validation only. Expect the two parts to
  behave differently on the private leaderboard.
- **Small margin.** +0.001 is one leaderboard tick. The reader is useful mainly as an independent model
  family; it is not a large gain on its own.

## Thanks

This notebook exists because of work that others shared openly. Thank you.

- **[skarin](https://www.kaggle.com/skarin)** for
  [Reproducing the 0.943 Public Stack](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack),
  the clean, fully public reproduction that Part A is taken from without modification.
- **[Mattia Angeli](https://www.kaggle.com/mattiaangeli)** for
  [Bend the Knee to Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original),
  the design of the stack, the A5 folds and the CoAtNet readers.
- **[pjmathematician](https://www.kaggle.com/pjmathematician)** for the
  [D4 blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend), from which the idea of adding
  a leg with a safe fallback comes, and **[Jiwei Liu](https://www.kaggle.com/jiweiliu)** for the
  [fast 2xT4 inference](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference).
- **[Pilkwang Kim](https://www.kaggle.com/pilkwang)** for the
  [baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), its twenty DINOv2 members and
  the LLM label table.
- **[Dread Development](https://www.kaggle.com/dreaddevelopment)** for the Raptor CoAtNet models and their
  training notebook.
- **[prvsiyan](https://www.kaggle.com/prvsiyan)**, **[Sofia Anjenje](https://www.kaggle.com/sofiaanjenje)**,
  **[Antoine G.](https://www.kaggle.com/antoinegg1)** and
  **[Marwan Mahmoud](https://www.kaggle.com/marwanmath)** for the RadImageNet encoder and heads.
- **[stevenleehans](https://www.kaggle.com/stevenleehans)**, **[Pilkwang Kim](https://www.kaggle.com/pilkwang)**,
  **[lixin73](https://www.kaggle.com/lixin73)** and **[laymond](https://www.kaggle.com/laymond)** for the
  public report-label tables our reader was trained on.
- The `timm` library and the ConvNeXt authors for the pretrained encoder.

If we have missed or misattributed anything, please say so in the comments and we will correct it.

<!-- Cell 2 -->
---
# Part A · The community stack (unchanged)

Everything from here to Part B is
[skarin's reproduction](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack)
of the public stack, cell for cell. It ends by writing `submission.csv`, which scores 0.943 by itself.

<!-- Cell 3 -->
## Runtime contract and artifact checks

The first cells pin checkpoint hashes, dependency versions, memory limits, and
the D4 serving contract before expensive inference begins. Any missing or
changed required asset stops the run instead of silently changing predictions.


<!-- Cell 7 -->
## Stage 1 — DINO ensemble

Each MRI study is mapped into anatomical slots. Twenty member-specific tails
share the verified frozen DINO prefix, preserving all members while avoiding
the largest repeated transformer computation.


<!-- Cell 20 -->
## Stage 2 — A5 attention pooling

Five folds pool variable-length slice features. Host preparation stays bounded
and overlaps the active GPU pass; unavailable acquisitions fall back to the
base branch, while model failures remain fatal.


<!-- Cell 26 -->
## Stage 3 — RadImageNet heads

One RadImageNet encoder feeds the E10/E13/E11 layouts. Version 7 uses the scored
0.942 routing: main alpha 0.55 and second E13 pass alpha 0.20.


<!-- Cell 28 -->
## Stages 4–5 — Raptor and complementary CoAtNet readers

Four Raptor views run first. Small cohorts overlap residual + D4 and
Global96 + Repair-v1; large cohorts run the readers one at a time. When all
four CoAt readers succeed, their probability arrays are averaged 25% each and
ranked, then blended 40% into Raptor. Missing arms use the scored upstream
fallback; outer routing is unchanged.


<!-- Cell 30 -->
## Final checks

The notebook publishes `submission.csv` only after schema, study order, finite
values, checkpoint identities, model counts, and fallback counts are recorded.


<!-- Cell 32 -->
---
# Part B · Our reader

The next three cells write our inference source to `/kaggle/working/own_src/` so that it is visible here
rather than hidden in a dataset: `preprocess.py` (DICOM series to canonical volume), `knee.py` (model and
window sampling) and `infer.py` (study loader and prediction loop). The last cell runs the reader in a
separate process on whichever GPU has more free memory, then rank-blends it into `submission.csv`.

<!-- Cell 38 -->
---
## Reusing the reader

- **On its own.** Run only Part B's three source cells and call
  `python own_src/infer.py <competition dir> test out.csv <checkpoints...>`. It needs one GPU; with five
  checkpoints our stand-alone submission was scored in under ninety minutes.
- **With your own model.** Replace the call to `infer.py` with anything that writes a CSV with the
  submission columns. Keep the fallback: a failed leg should never cost you the submission.
- **Training code.** `train.py`, `make_labels.py` and the fold/label logic are included in the
  [weights dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) for reference.

Questions and corrections are welcome in the comments. Thanks again to everyone credited at the top.

