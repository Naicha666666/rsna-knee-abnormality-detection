"""Compile and Consolidate 1,200 3D VLM Spatial Distillation Cases into PyTorch Tensor Dataset.

Aggregates:
1. U2-Med Coronal (400 cases)
2. GPT-6 Luna Coronal (200 cases)
3. GPT-6 Luna Sagittal (200 cases)
4. GPT-6 Luna Axial (200 cases)
5. GPT-6 Luna Tri-Planar (200 cases)

Outputs PyTorch file: distillation_targets_1200.pt
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture"
]

LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
PLANE_NAMES = ["coronal", "sagittal", "axial"]
PLANE_TO_IDX = {p: i for i, p in enumerate(PLANE_NAMES)}

# Search paths (supports both R9700 and local)
SEARCH_DIRS = [Path("."), Path("scratchpad"), Path("/home/eason/rsna-knee"), Path("/home/eason/rsna-knee/scratchpad")]

def find_file(filename: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None

CACHE_FILES = [
    ("U2_Coronal", "u2_radimed_400_cache.json", "coronal"),
    ("Luna_Coronal", "gpt6_luna_200_cache.json", "coronal"),
    ("Luna_Sagittal", "gpt6_luna_sagittal_200_cache.json", "sagittal"),
    ("Luna_Axial", "gpt6_luna_axial_200_cache.json", "axial"),
    ("Luna_Triplanar", "gpt6_luna_triplanar_200_cache.json", "triplanar"),
]

def match_label(name: str) -> str | None:
    n = name.lower().strip()
    if any(k in n for k in ["baker", "popliteal", "gastrocnemius-semimembranosus", "semimembranosus-gastrocnemius"]):
        return "Baker's"
    if any(k in n for k in ["pf oa", "patellofemoral", "trochle", "patellar oa", "chondromalacia patell", "patellofemoral osteoarthritis"]):
        return "PF OA"
    if "medial" in n and "menisc" in n:
        return "Medial Meniscus"
    if "lateral" in n and "menisc" in n:
        return "Lateral Meniscus"
    if "medial" in n and any(k in n for k in ["oa", "osteoarth", "compartment wear", "compartment osteo", "joint-space", "cartilage wear", "femortibial", "tibiofemoral"]) and not any(k in n for k in ["menisc", "mcl", "collateral"]):
        return "Medial OA"
    if "lateral" in n and any(k in n for k in ["oa", "osteoarth", "compartment wear", "compartment osteo", "joint-space", "cartilage wear", "femortibial", "tibiofemoral"]) and not any(k in n for k in ["menisc", "lcl", "collateral"]):
        return "Lateral OA"
    if "menisc" in n:
        return "Medial Meniscus" if "medial" in n else ("Lateral Meniscus" if "lateral" in n else None)
    if any(k in n for k in ["acl", "anterior cruciate"]):
        return "ACL"
    if any(k in n for k in ["mcl", "medial collateral"]):
        return "MCL"
    if any(k in n for k in ["effusion", "joint fluid", "hemarthrosis", "lipohemarthrosis"]):
        return "Effusion"
    if any(k in n for k in ["synovit", "synovial proliferation", "synovial thickening", "synovial hypertrophy"]):
        return "Synovitis"
    if any(k in n for k in ["contusion", "bone bruise", "bone marrow edema", "bone marrow oedema", "trabecular microfracture"]):
        return "Contusion"
    if any(k in n for k in ["fracture", "sifk", "avulsion", "plateau fracture"]):
        return "Fracture"
    if "medial" in n and ("compartment" in n or "narrowing" in n or "osteophyte" in n):
        return "Medial OA"
    if "lateral" in n and ("compartment" in n or "narrowing" in n or "osteophyte" in n):
        return "Lateral OA"
    return None

def slice_str_to_idx(s: str) -> int | None:
    s = s.upper().strip()
    if s.startswith("S") and s[1:].isdigit():
        idx = int(s[1:]) - 1
        if 0 <= idx < 12:
            return idx
    return None

def relevance_to_distribution(rel_vec: np.ndarray, tau: float = 0.5) -> np.ndarray:
    if np.max(rel_vec) == 0:
        return np.ones(12, dtype=np.float32) / 12.0
    exp_r = np.exp(rel_vec / tau)
    return (exp_r / np.sum(exp_r)).astype(np.float32)

def main():
    spatial_targets = {}
    spatial_masks = {}
    primary_planes = {}
    bboxes_dict = {}

    stats = {
        "label_plane_supervisions": np.zeros((12, 3), dtype=int),
        "total_bboxes": 0
    }

    for src_name, filename, mode in CACHE_FILES:
        path = find_file(filename)
        if not path:
            print(f"Warning: {filename} not found in search paths!")
            continue
        print(f"Processing {src_name} ({path})...")
        data = json.load(open(path, encoding="utf-8"))

        for uid, entry in data.items():
            if uid not in spatial_targets:
                spatial_targets[uid] = np.zeros((12, 3, 12), dtype=np.float32)
                spatial_masks[uid] = np.zeros((12, 3), dtype=np.float32)
                primary_planes[uid] = np.full(12, -1, dtype=np.int64)
                bboxes_dict[uid] = []

            diag_dict = None
            if isinstance(entry, dict):
                if "annotation" in entry and isinstance(entry["annotation"], dict):
                    diag_dict = entry["annotation"].get("diagnoses", {})
                elif "diagnoses" in entry and isinstance(entry["diagnoses"], dict):
                    diag_dict = entry["diagnoses"]

            if not diag_dict:
                continue

            for diag_title, diag_info in diag_dict.items():
                if not isinstance(diag_info, dict):
                    continue
                lab = match_label(diag_title)
                if not lab:
                    continue
                lab_idx = LABEL_TO_IDX[lab]

                p_plane = diag_info.get("primary_diagnostic_plane") or diag_info.get("primary_plane")
                if p_plane and isinstance(p_plane, str) and p_plane.lower() in PLANE_TO_IDX:
                    primary_planes[uid][lab_idx] = PLANE_TO_IDX[p_plane.lower()]

                if mode == "triplanar":
                    for plane_idx, p_name in enumerate(PLANE_NAMES):
                        slices_key = f"{p_name}_slices"
                        slice_list = diag_info.get(slices_key, [])
                        if slice_list and isinstance(slice_list, list):
                            rel_vec = np.zeros(12, dtype=np.float32)
                            for item in slice_list:
                                if isinstance(item, dict):
                                    s_idx = slice_str_to_idx(item.get("slice", ""))
                                    r_val = float(item.get("relevance", 0))
                                    if s_idx is not None and r_val > 0:
                                        rel_vec[s_idx] = max(rel_vec[s_idx], r_val)
                                        if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
                                            bboxes_dict[uid].append({
                                                "label": lab,
                                                "plane": p_name,
                                                "slice_idx": s_idx,
                                                "bbox": [float(x) for x in item["bbox"]]
                                            })
                                            stats["total_bboxes"] += 1
                            if np.max(rel_vec) > 0:
                                dist = relevance_to_distribution(rel_vec)
                                spatial_targets[uid][lab_idx, plane_idx] = dist
                                spatial_masks[uid][lab_idx, plane_idx] = 1.0
                                stats["label_plane_supervisions"][lab_idx, plane_idx] += 1
                else:
                    p_idx = PLANE_TO_IDX[mode]
                    rel_vec = np.zeros(12, dtype=np.float32)
                    slice_list = diag_info.get("relevant_slices", [])
                    if slice_list and isinstance(slice_list, list):
                        for item in slice_list:
                            if isinstance(item, dict):
                                s_idx = slice_str_to_idx(item.get("slice", ""))
                                r_val = float(item.get("relevance", 0))
                                if s_idx is not None and r_val > 0:
                                    rel_vec[s_idx] = max(rel_vec[s_idx], r_val)
                    bb = diag_info.get("peak_bbox") or diag_info.get("bbox")
                    if bb and isinstance(bb, list) and len(bb) == 4:
                        peak_s = diag_info.get("peak_slice", "")
                        peak_idx = slice_str_to_idx(peak_s) if peak_s else int(np.argmax(rel_vec))
                        bboxes_dict[uid].append({
                            "label": lab,
                            "plane": mode,
                            "slice_idx": peak_idx if peak_idx is not None else 6,
                            "bbox": [float(x) for x in bb]
                        })
                        stats["total_bboxes"] += 1

                    if np.max(rel_vec) > 0:
                        dist = relevance_to_distribution(rel_vec)
                        spatial_targets[uid][lab_idx, p_idx] = dist
                        spatial_masks[uid][lab_idx, p_idx] = 1.0
                        stats["label_plane_supervisions"][lab_idx, p_idx] += 1

    print(f"\nTotal Unique Studies in Dataset: {len(spatial_targets)}")
    print(f"Total Localized BBoxes: {stats['total_bboxes']}")

    torch_spatial_targets = {u: torch.from_numpy(arr) for u, arr in spatial_targets.items()}
    torch_spatial_masks = {u: torch.from_numpy(arr) for u, arr in spatial_masks.items()}
    torch_primary_planes = {u: torch.from_numpy(arr) for u, arr in primary_planes.items()}

    dataset_bundle = {
        "uids": list(spatial_targets.keys()),
        "labels": LABELS,
        "planes": PLANE_NAMES,
        "spatial_targets": torch_spatial_targets,
        "spatial_masks": torch_spatial_masks,
        "primary_planes": torch_primary_planes,
        "bboxes": bboxes_dict,
        "stats": {
            "total_studies": len(spatial_targets),
            "total_bboxes": stats["total_bboxes"]
        }
    }

    out_file = Path("distillation_targets_1200.pt")
    if Path("data/labels").exists():
        out_file = Path("data/labels/distillation_targets_1200.pt")
    torch.save(dataset_bundle, out_file)
    print(f"Saved PyTorch dataset to {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

    print("\n" + "="*80)
    print(f"3D VLM DISTILLATION DATASET COMPILATION REPORT (1,200 CASES)")
    print("="*80)
    header = f"{'Label':<20} | {'Coronal':>8} | {'Sagittal':>8} | {'Axial':>8} | {'Total':>8}"
    print(header)
    print("-" * len(header))
    for idx, lab in enumerate(LABELS):
        c_cor = stats["label_plane_supervisions"][idx, 0]
        c_sag = stats["label_plane_supervisions"][idx, 1]
        c_ax  = stats["label_plane_supervisions"][idx, 2]
        tot   = c_cor + c_sag + c_ax
        print(f"{lab:<20} | {c_cor:8d} | {c_sag:8d} | {c_ax:8d} | {tot:8d}")
    print("="*80)

if __name__ == "__main__":
    main()
