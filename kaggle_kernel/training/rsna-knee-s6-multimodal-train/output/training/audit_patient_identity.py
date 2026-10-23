"""Read patient identity and acquisition metadata without decoding MRI pixels.

The archive is already verified as a whole. We read the first and last ZIP
member of each Series, not all slices. This checks all Series identities and
detects endpoint disagreement; the native builder still checks every decoded
slice. The result is separate from the actively written native cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pydicom

from .archive_cache import open_archive
from .common import UID, file_hash, save_json

SID = "SeriesInstanceUID"
VERSION = "series-endpoint-patient-identity-v1"
TAGS = [
    UID, SID, "PatientID", "IssuerOfPatientID", "PatientSex", "PatientAge",
    "Laterality", "ImageLaterality", "Rows", "Columns", "PixelSpacing",
    "SliceThickness", "SpacingBetweenSlices", "ImageOrientationPatient",
    "ImagePositionPatient", "ImageType", "FrameOfReferenceUID", "Modality",
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SeriesDescription", "ProtocolName", "SequenceName", "EchoTime",
    "RepetitionTime", "FlipAngle", "SOPInstanceUID", "InstanceNumber",
]


def clean(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value).strip()


def read_header(archive, entry):
    with archive.open(entry) as stream:
        ds = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=TAGS)
        prefix_bytes = stream.tell()
    return {
        "source_file": entry.filename,
        "zip_header_offset": entry.header_offset,
        "zip_crc32": entry.CRC,
        "dicom_prefix_bytes_consumed": prefix_bytes,
        "tags": {tag: clean(getattr(ds, tag, None)) for tag in TAGS},
    }


def summarize_series(uid, sid, entries, archive, source_row):
    endpoints = [entries[0]] if len(entries) == 1 else [entries[0], entries[-1]]
    headers = []
    errors = []
    for entry in endpoints:
        try:
            headers.append(read_header(archive, entry))
        except Exception as exc:
            errors.append({"source_file": entry.filename, "error": repr(exc)[:400]})
    mismatches = []
    for h in headers:
        for key, wanted in ((UID, uid), (SID, sid)):
            if h["tags"][key] != wanted:
                mismatches.append({"source_file": h["source_file"], "field": key,
                                   "expected": wanted, "observed": h["tags"][key]})
    patients = sorted({h["tags"]["PatientID"] for h in headers if h["tags"]["PatientID"]})
    issuers = sorted({h["tags"]["IssuerOfPatientID"] for h in headers if h["tags"]["IssuerOfPatientID"]})
    return {UID: uid, SID: sid, "source_metadata": source_row,
            "dicom_entries": len(entries), "endpoint_headers": headers,
            "PatientIDs": patients, "IssuerOfPatientIDs": issuers,
            "missing_patient_headers": sum(not h["tags"]["PatientID"] for h in headers),
            "uid_mismatches": mismatches, "errors": errors}


def aggregate(expected, records, out, *, cache=None):
    by_study = defaultdict(list)
    for record in records:
        by_study[record[UID]].append(record)
    rows = []
    for uid in expected:
        series = by_study.get(uid, [])
        pids = sorted({p for r in series for p in r["PatientIDs"]})
        tags = [h["tags"] for r in series for h in r["endpoint_headers"]]
        sides = sorted({t.get(k) for t in tags for k in ("Laterality", "ImageLaterality")
                        if t.get(k) in ("L", "R")})
        issues = []
        if not series:
            issues.append("no_series")
        if len(pids) != 1:
            issues.append("missing_patient" if not pids else "conflicting_patient")
        if any(r["errors"] for r in series):
            issues.append("header_read_error")
        if any(r["uid_mismatches"] for r in series):
            issues.append("path_dicom_uid_mismatch")
        if any(r["missing_patient_headers"] for r in series):
            issues.append("partly_missing_patient")
        rows.append({UID: uid, "PatientID": pids[0] if len(pids) == 1 else "",
                     "all_patient_ids": "|".join(pids), "series_count": len(series),
                     "headers_read": len(tags), "explicit_sides": "|".join(sides),
                     "patient_sex": "|".join(sorted({t["PatientSex"] for t in tags if t["PatientSex"]})),
                     "patient_ages": "|".join(sorted({t["PatientAge"] for t in tags if t["PatientAge"]})),
                     "issuers": "|".join(sorted({p for r in series for p in r["IssuerOfPatientIDs"]})),
                     "identity_status": "ok" if not issues else "|".join(issues)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "study_identity.csv", index=False)
    frame[[UID, "PatientID"]].to_csv(out / "patients.csv", index=False)
    valid = frame.loc[frame.PatientID.ne("")]
    sizes = valid.groupby("PatientID").size()
    repeated = valid.loc[valid.PatientID.isin(sizes[sizes > 1].index)].sort_values(["PatientID", UID])
    repeated.to_csv(out / "repeated_patient_studies.csv", index=False)
    group_records = []
    for pid, group in valid.groupby("PatientID", sort=False):
        sides = set("|".join(group.explicit_sides).split("|")) - {""}
        sexes = set("|".join(group.patient_sex).split("|")) - {""}
        issuers = set("|".join(group.issuers).split("|")) - {""}
        group_records.append({"PatientID": pid, "studies": len(group),
                              "both_laterality_tags": sides == {"L", "R"},
                              "sex_disagreement": len(sexes) > 1,
                              "multiple_issuers": len(issuers) > 1})
    groups = pd.DataFrame(group_records)
    groups.to_csv(out / "patient_groups.csv", index=False)
    mixed_side_studies = set(frame.loc[frame.explicit_sides.eq("L|R"), UID])
    laterality_rows = []
    for record in records:
        if record[UID] not in mixed_side_studies:
            continue
        for header in record["endpoint_headers"]:
            tags = header["tags"]
            laterality_rows.append({UID: record[UID], SID: record[SID],
                                   "source_file": header["source_file"],
                                   "plane": record["source_metadata"].get("Anatomical_Plane", ""),
                                   "Laterality": tags.get("Laterality"),
                                   "ImageLaterality": tags.get("ImageLaterality"),
                                   "SeriesDescription": tags.get("SeriesDescription"),
                                   "ImagePositionPatient": json.dumps(tags.get("ImagePositionPatient")),
                                   "FrameOfReferenceUID": tags.get("FrameOfReferenceUID")})
    pd.DataFrame(laterality_rows).to_csv(out / "mixed_laterality_headers.csv", index=False)
    acquisition_rows = []
    for record in records:
        if not record["endpoint_headers"]:
            continue
        tags = record["endpoint_headers"][0]["tags"]
        spacing = tags.get("PixelSpacing") or []
        orientation = tags.get("ImageOrientationPatient") or []
        normal = np.cross(orientation[:3], orientation[3:]) if len(orientation) == 6 else None
        if normal is not None and np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = None
        plane = record["source_metadata"].get("Anatomical_Plane", "")
        axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}.get(plane)
        angle = float(np.rad2deg(np.arccos(np.clip(abs(normal[axis]), 0, 1)))) \
            if normal is not None and axis is not None else None
        row = {UID: record[UID], SID: record[SID], **record["source_metadata"],
               "dicom_entries": record["dicom_entries"],
               "row_spacing_mm": spacing[0] if len(spacing) == 2 else None,
               "column_spacing_mm": spacing[1] if len(spacing) == 2 else None,
               "normal_angle_to_csv_plane_deg": angle}
        row.update({k: tags.get(k) for k in ["Rows", "Columns", "SliceThickness", "SpacingBetweenSlices",
                   "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SeriesDescription",
                   "Laterality", "ImageLaterality", "FrameOfReferenceUID", "EchoTime", "RepetitionTime", "FlipAngle"]})
        row.update({k: json.dumps(tags.get(k), ensure_ascii=False) for k in ["ImageType", "ImageOrientationPatient"]})
        acquisition_rows.append(row)
    pd.DataFrame(acquisition_rows).to_csv(out / "series_acquisition.csv", index=False)
    links = {key: defaultdict(set) for key in ("FrameOfReferenceUID", "SOPInstanceUID")}
    for record in records:
        for header in record["endpoint_headers"]:
            for key, values in links.items():
                value = header["tags"].get(key)
                if value:
                    values[value].add(record[UID])
    pid_map = frame.set_index(UID).PatientID.to_dict()
    shared = []
    for kind, values in links.items():
        for value, studies in values.items():
            if len(studies) > 1:
                shared.append({"link_kind": kind, "dicom_uid": value, "study_count": len(studies),
                               "patient_id_count": len({pid_map[uid] for uid in studies}),
                               "studies": "|".join(sorted(studies))})
    pd.DataFrame(shared, columns=["link_kind", "dicom_uid", "study_count", "patient_id_count", "studies"]).to_csv(
        out / "cross_study_uid_links.csv", index=False)
    cache_checks = []
    if cache:
        for path in sorted((Path(cache) / "studies").glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            selected = frame.loc[frame[UID].eq(item[UID])]
            if selected.empty:
                continue
            cache_checks.append({UID: item[UID], "cache_patient": item.get("PatientID", ""),
                                 "header_patient": selected.iloc[0].PatientID,
                                 "same": item.get("PatientID", "") == selected.iloc[0].PatientID})
        pd.DataFrame(cache_checks).to_csv(out / "cache_identity_comparison.csv", index=False)
    placeholders = {"anonymous", "anonymized", "anonymised", "unknown", "none", "null", "patient", "0"}
    summary = {
        "version": VERSION, "studies": len(frame), "series": len(records),
        "headers_read": sum(r["headers_read"] for r in rows),
        "unique_nonempty_patient_ids": len(sizes),
        "missing_or_conflicting_patient_studies": int(frame.PatientID.eq("").sum()),
        "studies_with_any_identity_issue": int(frame.identity_status.ne("ok").sum()),
        "repeated_patient_groups": int((sizes > 1).sum()), "studies_in_repeated_groups": len(repeated),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "group_size_histogram": {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "patient_id_equals_study_uid": int(frame.PatientID.eq(frame[UID]).sum()),
        "placeholder_patient_studies": int(frame.PatientID.str.lower().isin(placeholders).sum()),
        "patient_groups_with_both_laterality_tags": int(groups.both_laterality_tags.sum()) if len(groups) else 0,
        "multi_study_groups_with_both_laterality_tags": int((groups.both_laterality_tags & groups.studies.gt(1)).sum()) if len(groups) else 0,
        "studies_with_mixed_laterality_tags": len(mixed_side_studies),
        "studies_with_explicit_laterality": int(frame.explicit_sides.ne("").sum()),
        "sex_disagreement_groups": int(groups.sex_disagreement.sum()) if len(groups) else 0,
        "multiple_issuer_groups": int(groups.multiple_issuers.sum()) if len(groups) else 0,
        "shared_frame_uid_groups": sum(r["link_kind"] == "FrameOfReferenceUID" for r in shared),
        "shared_sampled_sop_uid_groups": sum(r["link_kind"] == "SOPInstanceUID" for r in shared),
        "cache_studies_compared": len(cache_checks),
        "cache_identity_disagreements": sum(not row["same"] for row in cache_checks),
        "patient_map_sha256": file_hash(out / "patients.csv"),
        "scope": "All Study and Series identities; first/last archive member headers per Series, not every slice.",
        "identity_limit": "Uses the released pseudonymous DICOM PatientID. It cannot establish whether a person was assigned multiple pseudonyms.",
        "training_dispatch": False,
    }
    summary["ready_for_released_patient_grouping"] = not any([
        summary["studies_with_any_identity_issue"], summary["placeholder_patient_studies"],
        summary["cache_identity_disagreements"],
    ]) and summary["unique_nonempty_patient_ids"] > 1
    save_json(out / "summary.json", summary)
    return summary


def run(root, output, *, split="train", cache=None):
    root = Path(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    archive, reader, metadata = open_archive(root)
    started = time.perf_counter()
    try:
        status = json.loads((root / "download/status.json").read_text(encoding="utf-8"))
        if status.get("phase") != "complete" or status.get("md5_verified") is not True:
            raise RuntimeError("The header audit requires the completed, MD5-verified archive")
        expected = pd.read_csv(io.BytesIO(archive.read(f"{split}.csv")), dtype=str)[UID].tolist()
        source = pd.read_csv(io.BytesIO(archive.read(f"{split}_series.csv")), dtype=str).fillna("")
        if source[SID].duplicated().any() or len(expected) != len(set(expected)):
            raise ValueError("Duplicate source Study or Series identity")
        source_rows = {(r[UID], r[SID]): r for r in source.to_dict("records")}
        entries = defaultdict(list)
        for entry in archive.infolist():
            parts = entry.filename.split("/")
            if len(parts) == 4 and parts[0] == f"{split}_series" and parts[-1].endswith(".dcm"):
                entries[(parts[1], parts[2])].append(entry)
        if set(entries) != set(source_rows):
            raise ValueError("Series CSV and DICOM archive Series sets disagree")
        for values in entries.values():
            values.sort(key=lambda item: item.header_offset)
        provenance = {"version": VERSION, "split": split, "archive": metadata,
                      "source_series_csv_sha256": hashlib.sha256(archive.read(f"{split}_series.csv")).hexdigest(),
                      "source_train_csv_sha256": hashlib.sha256(archive.read(f"{split}.csv")).hexdigest(),
                      "sample": "first and last ZIP members of every Series", "pixels_decoded": False,
                      "archive_md5_verified": True}
        source_file = out / "source.json"
        if source_file.exists() and json.loads(source_file.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Existing audit is from a different archive or method; use a new output directory")
        save_json(source_file, provenance)
        journal = out / "series_headers.jsonl"
        records = []
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        done = {(r[UID], r[SID]) for r in records}
        if len(done) != len(records) or not done <= set(entries):
            raise ValueError("Duplicate or unexpected Series in audit journal")
        ordered = sorted(entries, key=lambda key: entries[key][0].header_offset)
        with journal.open("a", encoding="utf-8", buffering=1) as handle:
            for key in ordered:
                if key in done:
                    continue
                record = summarize_series(*key, entries[key], archive, source_rows[key])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                if len(records) % 250 == 0 or len(records) == len(ordered):
                    progress = {"phase": "reading_series_endpoint_headers", "series": len(records),
                                "expected_series": len(ordered), "seconds": time.perf_counter() - started,
                                "pixels_decoded": 0, "training_dispatch": False}
                    save_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        summary = aggregate(expected, records, out, cache=cache)
        summary["seconds"] = time.perf_counter() - started
        summary["source_sha256"] = file_hash(source_file)
        save_json(out / "summary.json", summary)
        save_json(out / "progress.json", {"phase": "complete", **summary})
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        archive.close()
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="F:/rsnaknee")
    parser.add_argument("--output", default="F:/rsnaknee/analysis/patient_identity")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--cache", default="F:/rsnaknee/cache_native")
    args = parser.parse_args()
    run(args.root, args.output, split=args.split, cache=args.cache)
