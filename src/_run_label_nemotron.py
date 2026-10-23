"""Run Nemotron labeling on all 4,407 knee MRI reports from train.csv.

Usage:
    python src/_run_label_nemotron.py [--limit N] [--host URL]
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nemotron_labeler as NL

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "raw", "train.csv")
OUT_PARQUET = os.path.join(DATA_DIR, "labels", "labels_nemotron_soft.parquet")


def main():
    parser = argparse.ArgumentParser(description="Run Nemotron labeling on reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reports to label (smoke test)")
    parser.add_argument("--host", type=str, default=NL.DEFAULT_HOST, help="Ollama host URL")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save to parquet every N records")
    args = parser.parse_args()

    print(f"Loading reports from {TRAIN_CSV}...")
    df_raw = pd.read_csv(TRAIN_CSV)
    reports = {str(u): str(r) for u, r in zip(df_raw["StudyInstanceUID"], df_raw["Report"].fillna(""))}
    print(f"Total reports in train.csv: {len(reports)}")

    if args.limit:
        # Keep only the first `args.limit` items
        keys = list(reports.keys())[:args.limit]
        reports = {k: reports[k] for k in keys}
        print(f"Limited to first {args.limit} reports for testing.")

    print(f"Target backend: {args.host} (model: {NL.DEFAULT_MODEL})")
    print(f"Output parquet: {OUT_PARQUET}")

    df_labeled = NL.label_corpus(
        reports=reports,
        out_parquet=OUT_PARQUET,
        host=args.host,
        checkpoint_every=args.checkpoint_every,
        progress_every=10,
    )

    # If all 58 gold are present, score it
    print("\nEvaluating GAUGE 1 score on labeled results:")
    score_script = os.path.join(PROJECT_ROOT, "src", "score_label_source.py")
    os.system(f"python \"{score_script}\" \"{OUT_PARQUET}\"")


if __name__ == "__main__":
    main()
