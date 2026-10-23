"""DICOM -> per-Series cache -> ordered 2.5D windows -> Study batches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from . import dicom_geometry as geom
from .common import LABELS, UID, SLOTS, file_hash, save_json


def _series_cache(item):
    row, source_root, cache, size, fov, side, ordered, config_hash = item
    uid, sid = str(row[UID]), str(row["SeriesInstanceUID"])
    key = hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24]
    dest = Path(cache) / "series" / (key + ".npy")
    record_path = dest.with_suffix(".json")
    if record_path.exists() and dest.exists():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash or old.get("header_sha256") != ordered.header_sha256:
            raise ValueError("cache configuration/source changed; use a new cache directory")
        return old
    volume, positions, fovs, filenames, errors = [], [], [], [], []
    plane = row["Anatomical_Plane"]
    fs = int(row["Fat_Suppression"]) if pd.notna(row.get("Fat_Suppression")) else -1
    if fs not in (0, 1):
        fs = -1
    for path, pos in zip(ordered.files, ordered.positions_mm):
        try:
            payload = geom.canonicalize_payload(geom.read_pixels(path), plane)
            available = min(payload.image.shape[0] * payload.spacing[0], payload.image.shape[1] * payload.spacing[1])
            requested = min(fov, available * .98)
            image, receipt = geom.render_canonical_payload(payload, plane=plane, side=side,
                                                           crop_mm=requested, output_size=size)
            if image is None:
                raise ValueError(receipt.reason)
            volume.append(image)
            positions.append(float(pos))
            fovs.append(float(min(receipt.realized_fov_x_mm, receipt.realized_fov_y_mm)))
            filenames.append(Path(path).name)
        except Exception as exc:
            errors.append({"file": Path(path).name, "error": str(exc)[:300]})
    if not volume:
        raise ValueError(f"no decodable slices in {sid}; {errors[:2]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".npy.tmp")
    with tmp.open("wb") as f:
        np.save(f, np.stack(volume), allow_pickle=False)
    tmp.replace(dest)
    result = {UID: uid, "SeriesInstanceUID": sid, "path": str(dest.resolve()), "slot": SLOTS[(plane, fs)],
              "plane": plane, "fat_suppression": fs,
              "fluid_sensitive": int(row["Fluid_Sensitive"]) if pd.notna(row.get("Fluid_Sensitive")) else -1,
              "positions": positions, "fov_mm": fovs, "files": filenames, "errors": errors,
              "header_sha256": ordered.header_sha256, "config_hash": config_hash, "side": side}
    save_json(record_path, result)
    return result


def prepare_cache(series_csv, source_root, cache, *, study_csv=None, image_size=384,
                  fov_mm=130., workers=4, study_ids=None):
    frame = pd.read_csv(series_csv, dtype={UID: str, "SeriesInstanceUID": str})
    required = {UID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"}
    if not required.issubset(frame) or frame["SeriesInstanceUID"].duplicated().any():
        raise ValueError("invalid series metadata")
    expected = list(pd.read_csv(study_csv, dtype={UID: str})[UID]) if study_csv else list(frame[UID].unique())
    if study_ids is not None:
        selected = set(study_ids)
        expected = [s for s in expected if s in selected]
        if selected - set(expected):
            raise ValueError("requested study not in metadata")
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("empty/duplicate study list")
    cache = Path(cache)
    config = {"size": image_size, "fov_mm": fov_mm, "geometry_hash": file_hash(Path(geom.__file__)),
              "data_code_hash": file_hash(__file__), "series_csv_sha256": file_hash(series_csv),
              "source_root": str(Path(source_root).resolve())}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    studies, mapping = [], []
    grouped = {u: x.to_dict("records") for u, x in frame.groupby(UID, sort=False)}
    for step, uid in enumerate(expected):
        rows = grouped.get(uid, [])
        ordered, tags, sagittal, errors, patients = {}, set(), [], [], set()
        for row in rows:
            sid, plane = str(row["SeriesInstanceUID"]), row["Anatomical_Plane"]
            if plane not in geom.CARDINAL_NORMAL:
                errors.append({"series": sid, "error": "unsupported plane/localizer"})
                continue
            try:
                order = geom.order_series(Path(source_root) / uid / sid, plane)
                ordered[sid] = order
                tags.update(order.laterality_values)
                if plane == "Sagittal" and order.ipp_x_median is not None:
                    sagittal.append(order.ipp_x_median)
                ds = pydicom.dcmread(order.files[0], stop_before_pixels=True, specific_tags=["PatientID"])
                pid = str(getattr(ds, "PatientID", "")).strip()
                if pid:
                    patients.add(pid)
            except Exception as exc:
                errors.append({"series": sid, "error": str(exc)[:300]})
        if len(patients) > 1:
            raise ValueError(f"conflicting PatientID within study {uid}")
        try:
            side = geom.decide_study_side(uid, tags, sagittal).side
        except Exception as exc:
            side = None
            errors.append({"series": "study", "error": f"laterality: {exc}"})
        series = []
        jobs = [(r, source_root, cache, image_size, fov_mm, side, ordered[str(r["SeriesInstanceUID"])], config_hash)
                for r in rows if str(r["SeriesInstanceUID"]) in ordered]
        # One study at a time bounds memory; independent series decode concurrently.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_series_cache, j) for j in jobs]
            for job, future in zip(jobs, futures):
                try:
                    series.append(future.result())
                except Exception as exc:
                    if "cache configuration/source changed" in str(exc):
                        raise
                    errors.append({"series": str(job[0]["SeriesInstanceUID"]), "error": str(exc)[:300]})
        pid = next(iter(patients), uid)
        studies.append({UID: uid, "PatientID": pid, "series": series, "errors": errors})
        mapping.append({UID: uid, "PatientID": pid})
        if step % 25 == 0 or step + 1 == len(expected):
            print(f"prepared {step + 1}/{len(expected)} studies; current {len(series)} series", flush=True)
    result = {"config": config, "config_hash": config_hash, "studies": studies,
              "failed_studies": [s[UID] for s in studies if not s["series"]]}
    save_json(cache / "manifest.json", result)
    pd.DataFrame(mapping).to_csv(cache / "patients.csv", index=False)
    return result


class StudyDataset(Dataset):
    def __init__(self, manifest, labels=None, *, ids=None, training=False, windows_per_series=8,
                 neighbor_mm=3., seed=42, max_windows_per_series=None, cache_read_mode="mmap"):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        index = {s[UID]: s for s in self.manifest["studies"]}
        self.ids = list(index) if ids is None else list(map(str, ids))
        if set(self.ids) - set(index):
            raise ValueError("requested studies missing from cache manifest; no silent subset validation")
        self.studies = [index[u] for u in self.ids]
        self.labels = labels.set_index(UID) if labels is not None else None
        self.training, self.windows = training, windows_per_series
        self.max_windows = max_windows_per_series or windows_per_series
        if self.max_windows < self.windows:
            raise ValueError('maximum window budget below minimum')
        self.neighbor_mm, self.seed, self.epoch = neighbor_mm, seed, 0
        if cache_read_mode not in ("mmap", "eager"):
            raise ValueError("cache_read_mode must be mmap or eager")
        self.cache_read_mode = cache_read_mode
        self.size = self.manifest["config"]["size"]
        if training and any(not s["series"] for s in self.studies):
            raise ValueError("training cache contains studies with no pixels; repair cache before training")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        i,draw=i if isinstance(i,tuple) else (i,0)
        study = self.studies[i]
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + i * 97 + draw * 1009)
        images, slots, series_ids, positions, geometry, names = [], [], [], [], [], []
        for sid, series in enumerate(study["series"]):
            # On the external exFAT drive, read each Series sequentially once.
            # Keep mmap for old checkpoints and storage-specific comparisons.
            array = np.load(series["path"], mmap_mode="r" if self.cache_read_mode == "mmap" else None,
                            allow_pickle=False)
            pos = np.asarray(series["positions"], dtype=np.float64)
            if len(pos) != len(array) or len(pos) < 1 or not np.isfinite(pos).all():
                raise ValueError("corrupt series cache")
            desired = int(np.ceil(len(pos)/4)) if self.training else int(np.ceil((len(pos)+2)/3))
            n = min(len(pos), self.max_windows, max(self.windows, desired))
            if self.training:
                edges = np.linspace(0, len(pos), n + 1).astype(int)
                centers = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])])
            else:
                centers = np.unique(np.linspace(0, len(pos) - 1, n).round().astype(int))
            spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.
            names.append(series["SeriesInstanceUID"])
            for center in centers:
                left = int(np.argmin(abs(pos - (pos[center] - self.neighbor_mm)))) if self.neighbor_mm > 0 else max(0,center-1)
                right = int(np.argmin(abs(pos - (pos[center] + self.neighbor_mm)))) if self.neighbor_mm > 0 else min(len(pos)-1,center+1)
                if center > 0:
                    left = min(left, center - 1)
                if center + 1 < len(pos):
                    right = max(right, center + 1)
                images.append(np.asarray(array[[left, center, right]]).copy())
                slots.append(series["slot"])
                series_ids.append(sid)
                positions.append(pos[center])
                relative = 2 * (pos[center] - pos[0]) / max(pos[-1] - pos[0], 1e-6) - 1
                geometry.append([relative, np.log1p(abs(spacing)), series["fov_mm"][center] / 130., series.get("fluid_sensitive", -1)])
        if images:
            image_tensor = torch.from_numpy(np.stack(images))
        else:
            image_tensor = torch.zeros((0, 3, self.size, self.size), dtype=torch.uint8)
        y, w = np.zeros(12, np.float32), np.zeros(12, np.float32)
        extra = {}
        if self.labels is not None:
            row = self.labels.loc[study[UID]]
            y = row[LABELS].to_numpy(np.float32)
            w = row[[x + "__weight" for x in LABELS]].to_numpy(np.float32)
            y = np.nan_to_num(y)
            from .supervision import targets_from_row
            extra.update(targets_from_row(row))
        return {"images": image_tensor, "slots": torch.tensor(slots, dtype=torch.long),
                "series": torch.tensor(series_ids, dtype=torch.long),
                "position": torch.tensor(positions, dtype=torch.float32),
                "geometry": torch.tensor(geometry, dtype=torch.float32).reshape(-1, 4),
                "target": torch.from_numpy(y), "weight": torch.from_numpy(w), UID: study[UID],
                "series_names": names, "fallback": not bool(images), **extra}


def collate_studies(items):
    from .input_provenance import PROVENANCE_KEYS
    tensor_provenance_keys=(*PROVENANCE_KEYS,'channel_evidence_id')
    source_metadata_keys=('source_series','source_series_aliases','duplicate_resolution')
    b, k = len(items), max(1, max(len(x["slots"]) for x in items))
    h, w = items[0]["images"].shape[-2:]
    out = {"images": torch.zeros(b, k, 3, h, w, dtype=items[0]['images'].dtype),
           "slots": torch.zeros(b, k, dtype=torch.long), "series": torch.full((b, k), -1, dtype=torch.long),
           "position": torch.zeros(b, k), "geometry": torch.zeros(b, k, 4),
           "target": torch.stack([x["target"] for x in items]), "weight": torch.stack([x["weight"] for x in items]),
           UID: [x[UID] for x in items], "series_names": [x["series_names"] for x in items],
           "fallback": [x["fallback"] for x in items]}
    for key in (*tensor_provenance_keys,*source_metadata_keys):
        present=[key in item for item in items]
        if any(present) and not all(present):raise ValueError('mixed/missing input provenance: '+key)
        if not all(present):continue
        if key in source_metadata_keys:out[key]=[item[key] for item in items]
        else:
            fill=-1 if key in ('channel_native_index','channel_evidence_id') else 0
            out[key]=torch.full((b,k,*items[0][key].shape[1:]),fill,dtype=items[0][key].dtype)
    for key in ('view_ids','original_series'):
        if key in items[0]:out[key]=torch.full((b,k),-1,dtype=torch.long)
    for key in ('presence_target','presence_weight','teacher_target','teacher_weight','factors_target','factors_weight',
                'meniscus_routes_target','meniscus_routes_weight','local_oa_target','local_oa_weight'):
        if key in items[0]:out[key]=torch.stack([x[key] for x in items])
    for key in ('local_evidence_regions','local_evidence_tissues','local_evidence_tissue_quality'):
        present=[key in item for item in items]
        if any(present) and not all(present):raise ValueError('mixed/missing local evidence: '+key)
    for key in ('encoded_values','encoded_global','source_depth','spatial_valid','anatomy_regions','region_quality','tissue_targets','tissue_quality','tissue_pixel_weight','crop_center','localized',
                'local_evidence_regions','local_evidence_tissues','local_evidence_tissue_quality'):
        if key in items[0]:out[key]=torch.zeros((b,k,*items[0][key].shape[1:]),dtype=items[0][key].dtype)
    for i, item in enumerate(items):
        n = len(item["slots"])
        for key in ("images", "slots", "series", "position", "geometry"):
            out[key][i, :n] = item[key]
        for key in ('view_ids','original_series'):
            if key in out:out[key][i,:n]=item[key]
        for key in ('encoded_values','encoded_global','source_depth','spatial_valid','anatomy_regions','region_quality','tissue_targets','tissue_quality','tissue_pixel_weight','crop_center','localized',
                    'local_evidence_regions','local_evidence_tissues','local_evidence_tissue_quality'):
            if key in out:out[key][i,:n]=item[key]
        for key in tensor_provenance_keys:
            if key in out:out[key][i,:n]=item[key]
    return out


def augment_series(batch,*,intensity_policy='legacy'):
    """Same gentle affine and intensity transform for every window in a Series."""
    from torch.nn import functional as F
    from .input_provenance import update_augmented_provenance
    from .intensity_mapping import augment_intensity
    images = batch["images"].float()
    if batch['images'].dtype==torch.uint8:images=images/255.
    for b in range(len(images)):
        groups=batch.get('original_series',batch['series'])
        for sid in groups[b, batch["slots"][b] > 0].unique():
            mask = groups[b] == sid
            x = images[b, mask]
            angle = (torch.rand((), device=x.device) - .5) * (10 * np.pi / 180)
            scale = 1 + (torch.rand((), device=x.device) - .5) * .1
            shift = (torch.rand(2, device=x.device) - .5) * .08
            c, s = angle.cos() / scale, angle.sin() / scale
            theta = torch.stack([torch.stack([c, -s, shift[0]]), torch.stack([s, c, shift[1]])])
            grid = F.affine_grid(theta[None].expand(len(x), -1, -1), x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
            update_augmented_provenance(batch,b,mask,theta)
            if ('tissue_targets' in batch)!=('tissue_pixel_weight' in batch):
                raise ValueError('pixel target augmentation requires its explicit support')
            if 'tissue_targets' in batch:
                from .masked_supervision import grid_sample_masked_targets
                targets=batch['tissue_targets'][b,mask];weights=batch['tissue_pixel_weight'][b,mask]
                mask_grid=F.affine_grid(theta[None].expand(len(targets),-1,-1),targets.shape,align_corners=False)
                targets,weights=grid_sample_masked_targets(targets,weights,mask_grid)
                batch['tissue_targets'][b,mask]=targets;batch['tissue_pixel_weight'][b,mask]=weights
            for key in ('anatomy_regions','local_evidence_regions','local_evidence_tissues'):
                if key in batch:
                    masks=batch[key][b,mask]
                    original_shape=masks.shape
                    flat=masks.reshape(len(masks),-1,*masks.shape[-2:])
                    mask_grid=F.affine_grid(theta[None].expand(len(flat),-1,-1),flat.shape,align_corners=False)
                    batch[key][b,mask]=F.grid_sample(flat,mask_grid,mode='bilinear',padding_mode='zeros',align_corners=False).reshape(original_shape)
            gain = 1 + (torch.rand((), device=x.device) - .5) * .2
            if 'intensity_gain' in batch:
                batch['intensity_gain'][b,mask]=batch['intensity_gain'][b,mask]*gain
            images[b, mask] = augment_intensity(x,gain,intensity_policy)
    return {**batch, "images": images,'intensity_gain_mapping':intensity_policy}
