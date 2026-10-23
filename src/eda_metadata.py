"""EDA on RSNA Knee Abnormality Detection metadata.
Analyzes train.csv / test.csv / test_series.csv to understand schema,
label distribution, missingness, and study/series structure.
"""
import os
import pandas as pd
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

def load(name):
    return pd.read_csv(os.path.join(RAW, name))

print("=" * 70)
print("TRAIN.CSV")
print("=" * 70)
train = load("train.csv")
print("shape:", train.shape)
print("columns:", list(train.columns))
print("unique StudyInstanceUID:", train["StudyInstanceUID"].nunique(),
      "| dup UIDs:", train["StudyInstanceUID"].duplicated().sum())

# Label analysis
print("\n--- Label columns: value counts (NaN/0/1) ---")
label_summary = {}
for c in LABELS:
    s = train[c]
    n_nan = s.isna().sum()
    n_1 = (s == 1).sum()
    n_0 = (s == 0).sum()
    other = len(s) - n_nan - n_1 - n_0
    pos_rate = n_1 / (n_1 + n_0) if (n_1 + n_0) > 0 else float("nan")
    label_summary[c] = dict(nan=int(n_nan), n0=int(n_0), n1=int(n_1),
                            other=int(other), pos_rate=pos_rate)
    print(f"{c:18s} nan={n_nan:6d} 0={n_0:6d} 1={n_1:6d} other={other:4d} "
          f"pos_rate={pos_rate:.4f}" if (n_1+n_0)>0 else
          f"{c:18s} nan={n_nan:6d} (NO LABELS)")

# How many studies have ANY label / ALL labels
any_label = train[LABELS].notna().any(axis=1).sum()
all_label = train[LABELS].notna().all(axis=1).sum()
print(f"\nstudies with >=1 label present: {any_label}")
print(f"studies with all 12 labels present: {all_label}")
print(f"studies with ZERO labels: {len(train) - any_label}")

# Report text stats
print("\n--- Report text ---")
rep = train["Report"].fillna("")
print("missing reports:", train["Report"].isna().sum())
print("empty reports:", (rep.str.strip() == "").sum())
lens = rep.str.len()
print(f"report char length: min={lens.min()} med={int(lens.median())} "
      f"mean={lens.mean():.0f} max={lens.max()}")
print("sample report (study 0):")
print("  ", rep.iloc[0][:300].replace("\n", " "))

print("\n" + "=" * 70)
print("TEST.CSV")
print("=" * 70)
test = load("test.csv")
print("test studies:", len(test), "| unique:", test["StudyInstanceUID"].nunique())

print("\n" + "=" * 70)
print("TEST_SERIES.CSV")
print("=" * 70)
ts = load("test_series.csv")
print("shape:", ts.shape)
print("unique studies:", ts["StudyInstanceUID"].nunique())
print("unique series:", ts["SeriesInstanceUID"].nunique())
print("\nseries-per-study distribution:")
spc = ts.groupby("StudyInstanceUID").size()
print(spc.describe())
print("\nAnatomical_Plane counts:\n", ts["Anatomical_Plane"].value_counts())
print("\nFluid_Sensitive counts:\n", ts["Fluid_Sensitive"].value_counts())
print("\nFat_Suppression counts:\n", ts["Fat_Suppression"].value_counts())
print("\nplane x fluid-sensitive crosstab:")
print(pd.crosstab(ts["Anatomical_Plane"], ts["Fluid_Sensitive"]))
