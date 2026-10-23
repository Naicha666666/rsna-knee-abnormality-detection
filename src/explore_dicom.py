"""Parse sample DICOMs and analyze train_series.csv for RSNA Knee competition."""
import os
import glob
import pydicom
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def summarize_dcm(path):
    print("\n" + "=" * 70)
    print("DICOM:", os.path.basename(path))
    print("=" * 70)
    ds = pydicom.dcmread(path)
    keys = [
        ("Modality", "0008,0060"), ("SeriesDescription", "0008,103E"),
        ("Manufacturer", "0008,0070"), ("MagneticFieldStrength", "0018,0087"),
        ("ScanningSequence", "0018,0020"), ("SequenceVariant", "0018,0021"),
        ("Rows", "0028,0010"), ("Columns", "0028,0011"),
        ("NumberOfFrames", "0028,0008"), ("SliceThickness", "0018,0050"),
        ("PixelSpacing", "0028,0030"), ("PhotometricInterpretation", "0028,0004"),
        ("BitsAllocated", "0028,0100"), ("BitsStored", "0028,0101"),
        ("HighBit", "0028,0102"), ("PixelRepresentation", "0028,0103"),
        ("RescaleIntercept", "0028,1052"), ("RescaleSlope", "0028,1053"),
        ("WindowCenter", "0028,1050"), ("WindowWidth", "0028,1051"),
        ("ImageOrientationPatient", "0020,0037"),
        ("ImagePositionPatient", "0020,0032"),
        ("StudyInstanceUID", "0020,000D"), ("SeriesInstanceUID", "0020,000E"),
    ]
    for name, tag in keys:
        try:
            val = ds[tag].value
        except Exception:
            val = "<missing>"
        print(f"  {name:28s}: {val}")
    arr = ds.pixel_array
    print(f"  --- pixel_array: shape={arr.shape} dtype={arr.dtype}")
    print(f"      raw min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
    # apply rescale if present
    try:
        slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
        inter = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
        hu = arr.astype(np.float32) * slope + inter
        print(f"      after rescale (slope={slope}, intercept={inter}): "
              f"min={hu.min():.1f} max={hu.max():.1f} mean={hu.mean():.1f}")
    except Exception as e:
        print("      rescale failed:", e)

for p in glob.glob(os.path.join(RAW, "*.dcm")):
    summarize_dcm(p)

print("\n" + "=" * 70)
print("TRAIN_SERIES.CSV")
print("=" * 70)
ts = pd.read_csv(os.path.join(RAW, "train_series.csv"))
print("shape:", ts.shape)
print("columns:", list(ts.columns))
print("unique studies:", ts["StudyInstanceUID"].nunique())
print("unique series:", ts["SeriesInstanceUID"].nunique())
spc = ts.groupby("StudyInstanceUID").size()
print(f"series-per-study: min={spc.min()} med={int(spc.median())} "
      f"mean={spc.mean():.2f} max={spc.max()}")
print("\nAnatomical_Plane:\n", ts["Anatomical_Plane"].value_counts())
print("\nFluid_Sensitive:\n", ts["Fluid_Sensitive"].value_counts())
print("\nFat_Suppression:\n", ts["Fat_Suppression"].value_counts())
print("\nplane x fluid-sensitive:\n",
      pd.crosstab(ts["Anatomical_Plane"], ts["Fluid_Sensitive"]))

# How many of the 58 labeled studies have series info here?
train = pd.read_csv(os.path.join(RAW, "train.csv"))
labeled = train[train[["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
    "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]].notna().any(axis=1)]
labeled_uids = set(labeled["StudyInstanceUID"])
ts_uids = set(ts["StudyInstanceUID"])
print(f"\nlabeled studies: {len(labeled_uids)} | in train_series.csv: "
      f"{len(labeled_uids & ts_uids)} | NOT in train_series.csv: "
      f"{len(labeled_uids - ts_uids)}")
