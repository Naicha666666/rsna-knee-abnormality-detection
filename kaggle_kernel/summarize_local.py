"""Standalone screening wrap-up summarizer (no pandas/torch/sklearn needed).

Reads every experiments/<exp_id>/metrics.json and emits the SAME paper-style
comparison as src/experiments.py:summarize() -> results/results.csv and
results/RESULTS.md. Used locally where the full training stack isn't installed.
"""
import os
import json
import csv
import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(REPO, "experiments")
RESULTS_DIR = os.path.join(REPO, "results")


def _load_all():
    rows = []
    for mp in sorted(glob.glob(os.path.join(EXP_DIR, "*", "metrics.json"))):
        with open(mp) as f:
            rows.append(json.load(f))
    # stable order by exp_id, but keep screening trio first
    rows.sort(key=lambda r: r["exp_id"])
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = _load_all()
    if not rows:
        print("no metrics.json found under", EXP_DIR)
        return
    labels = list(rows[0]["per_label"].keys())

    # ---- results.csv ----
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    fields = ["exp_id", "backbone", "unfreeze_last", "max_studies", "epochs",
              "folds", "batch", "macro_auc", "mean_fold", "std_fold"]
    fields += [f"auc_{l}" for l in labels]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in
                   ["exp_id", "backbone", "unfreeze_last", "max_studies",
                    "epochs", "folds", "batch", "macro_auc", "mean_fold",
                    "std_fold"]}
            for l in labels:
                v = r["per_label"].get(l)
                row[f"auc_{l}"] = ("" if v is None else f"{v:.4f}")
            w.writerow(row)

    # ---- RESULTS.md (paper-style) ----
    md = ["# RSNA Knee — Method Comparison (paper-style)", "",
          "> Controlled experiment: the training protocol is FIXED; only the image "
          "backbone varies. CV macro-AUC is computed on the held-out GOLD portion "
          "(public labels are only 58 studies, so it is a *relative* ranking "
          "signal; the hidden test leaderboard is the final arbiter).", "",
          "## Results", "",
          "| exp_id | backbone | tune | data | ep | macro-AUC | mean±std fold |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        data = "FULL" if not r["max_studies"] else f"sub{r['max_studies']}"
        tune = "ft" if r["unfreeze_last"] > 0 else "frozen"
        md.append(
            f"| {r['exp_id']} | {r['backbone']} | {tune} | {data} | "
            f"{r['epochs']} | {r['macro_auc']:.4f} | "
            f"{r['mean_fold']:.4f}±{r['std_fold']:.4f} |")
    md += ["", "## Per-label AUC (held-out gold)", "",
           "| exp_id | " + " | ".join(labels) + " |",
           "|---" + "|".join(["---"] * len(labels)) + "|"]
    for r in rows:
        cells = []
        for l in labels:
            v = r["per_label"].get(l)
            cells.append("" if v is None else f"{v:.3f}")
        md.append(f"| {r['exp_id']} | " + " | ".join(cells) + " |")
    md += ["", "## Reproducibility", "",
           "Each `experiments/exp_<id>/` contains `config.json`, `metrics.json`, "
           "`oof.npy`, and `provenance.json` (env + source SHA-256 + fold "
           "assignment + CLI).", "",
           "## Limitations", "",
           "- Public gold labels = 58 studies; CV macro-AUC is a noisy *relative* "
           "signal, not an absolute estimate. Final ranking uses the hidden test "
           "leaderboard (public + private).",
           "- Physical-scale crop and laterality normalization rely on DICOM "
           "headers present in the Kaggle environment; they degrade gracefully "
           "(fixed crop / no flip) if stripped.",
           "- Screening runs use subset=600 studies / 5 epochs / 5-fold; they "
           "establish the *backbone* ranking, not an absolute score."]
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {csv_path} and {os.path.join(RESULTS_DIR, 'RESULTS.md')}")
    print(f"({len(rows)} experiments summarized)")


if __name__ == "__main__":
    main()
