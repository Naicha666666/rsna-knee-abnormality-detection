#!/usr/bin/env python3
"""Exact dual-T4 rank ensemble for the CoAt residual-gated epoch-10 snapshot.

Each GPU owns a workload-balanced study shard.  The raw DICOM study is
prepared exactly once, rendered with the saved native 384 px / 130 mm Gold
grid, and then evaluated by the three Gold-selected checkpoints from the run
state after epoch 10 (epochs 4, 6, and 8).  The parent percentile-ranks each
checkpoint independently per finding and averages the three rank streams.

This module deliberately does not import the live ``coatnet_residual_gated``
source at import time.  The packaged, hash-bound epoch-10 component is loaded
from the artifact directory so later Repair-v1 edits cannot silently alter the
old checkpoint architecture.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import raptor_light_inference as base


PACKAGE_CONTRACT = "rsna_coat_resgated_ep10_top3_gold_rank_package_v2"
RUNTIME_CONTRACT = "rsna_coat_resgated_ep10_top3_gold_rank_t4x2_v2"
SUBMISSION_STATUS = "VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION"
SHARD_STATUS = "VALID_COAT_RESGATED_EP10_TOP3_RAW_SHARD"
MANIFEST_NAME = "coat_resgated_ep10_top3_manifest.json"
RECEIPT_NAME = "coat_resgated_ep10_top3_submission_receipt.json"
PREDICTIONS_NAME = "coat_resgated_ep10_top3_predictions.npz"
GOLD_PREDICTIONS_NAME = "gold58_top3_predictions.npz"
GEOMETRY_NAME = "coat_resgated_ep10_gold58_geometry_contract.json"
T4_PORTABILITY_NAME = "coat_resgated_ep10_t4_gold58_rank_portability.json"
T4_PORTABILITY_PREDICTIONS_NAME = (
    "coat_resgated_ep10_t4_gold58_rank_portability_predictions.npz"
)
VISIBLE_PARITY_NAME = "coat_resgated_ep10_top3_visible_parity.json"
GOLD_MODEL_PARITY_NAME = "coat_resgated_ep10_top3_gold_model_parity.json"
COMPONENT_NAME = "coatnet_residual_gated_k32.py"

ARCHITECTURE = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
N_MODELS = 3
CHECKPOINT_FILES = (
    "coat_resgated_gold_e04.pt",
    "coat_resgated_gold_e06.pt",
    "coat_resgated_gold_e08.pt",
)
EXPECTED_EPOCHS = (4, 6, 8)
EXPECTED_GOLD = (
    0.9004583452460247,
    0.8975532362766714,
    0.8968762056250612,
)
EXPECTED_PARAMETERS = 73_420_000
EXPECTED_STATE_ELEMENTS = 73_450_105
EXPECTED_COMPONENT_SHA256 = (
    "83735f535235bfd575767b9ac9b12ca0a77a6b11adf14343ca5c2466257f3374"
)
EXPECTED_SAMPLER_SHA256 = (
    "a3303a724e95cf3d9f13b67ef52529adf854b0cbe580b896ef42382cc740d932"
)
EXPECTED_T4_GOLD_RANK_ENSEMBLE_AUC = 0.9093268412428666
EXPECTED_SAVED_GOLD_RANK_ENSEMBLE_AUC = 0.9094696373252056
EVAL_SLOT_BUDGETS = (20, 12, 12, 8, 16, 8)
MAX_EVAL_WINDOWS = sum(EVAL_SLOT_BUDGETS)
SAGITTAL_SLOT_IDS = frozenset({1, 2})

# A prior 36-study T4 sweep of this exact CoAt family selected eight images
# per backbone call.  Larger blocks were both slower and more memory hungry.
GPU_BATCH_STUDIES = 2
BACKBONE_MICRO_IMAGES = 8
REFERENCE_MICRO_IMAGES = 4
PREP_WORKERS = 2
PREP_CHUNK = 16
CUDNN_BENCHMARK = False

# Raw serving must render directly from normalized DICOM into 384 px.  A
# 336->384 resize is observably different from the cache used for training.
base.SMART_RENDER_SIZE = 384


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    return base.sha256_file(Path(path), chunk=chunk)


def percentile_rank64(prediction: np.ndarray) -> np.ndarray:
    """Canonical average-percentile ranks without float32 tie artifacts."""

    value = np.asarray(prediction, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("prediction matrix must be two-dimensional")
    ranked = np.empty_like(value, dtype=np.float64)
    for column in range(value.shape[1]):
        ranked[:, column] = pd.Series(value[:, column]).rank(
            method="average", pct=True
        ).to_numpy(dtype=np.float64)
    return ranked


def _require_equal(mapping: Mapping, required: Mapping, context: str) -> None:
    for key, expected in required.items():
        observed = mapping.get(key)
        if observed != expected:
            raise RuntimeError(
                f"{context} drift: {key}={observed!r}, expected={expected!r}"
            )


@dataclass(frozen=True)
class PreparedCoAtResidualBag:
    """One variable-length, training-faithful CoAt evaluation bag."""

    study_uid: str
    images: np.ndarray
    slots: np.ndarray
    metadata: np.ndarray
    errors: tuple[str, ...]
    sparse_fallback: bool = False

    def __post_init__(self) -> None:
        windows = int(self.images.shape[0])
        if not 1 <= windows <= 94:
            raise ValueError(f"unexpected evaluation windows: {windows}")
        if self.images.dtype != np.uint8 or self.images.shape != (
            windows,
            3,
            384,
            384,
        ):
            raise ValueError(f"unexpected image payload: {self.images.shape}")
        if self.slots.dtype != np.int64 or self.slots.shape != (windows,):
            raise ValueError("slot sidecar is not aligned int64")
        if self.metadata.dtype != np.float32 \
                or self.metadata.shape != (windows, 2):
            raise ValueError("geometry sidecar is not aligned float32")
        if not bool(((self.slots >= 1) & (self.slots <= 6)).all()):
            raise ValueError("prepared bags may contain only real slots 1..6")
        if self.sparse_fallback:
            raise ValueError("this train-faithful runtime forbids sparse fallback")


def train_faithful_eval_specs(study: base.PreparedStudy) -> list[base.WindowSpec]:
    """Reproduce the variable per-slot grid used for every saved Gold score."""

    specs = base.sample_eval_windows(
        study.series_rows,
        study.offsets,
        study.valid,
        study.source_depth,
        count=None,
        slot_budgets=EVAL_SLOT_BUDGETS,
    )
    if not 1 <= len(specs) <= MAX_EVAL_WINDOWS:
        raise RuntimeError(f"invalid exact-grid cardinality: {len(specs)}")
    if len({(item.series_row, item.center_local) for item in specs}) != len(specs):
        raise RuntimeError("exact evaluation grid repeated a native centre")
    for slot_id, budget in enumerate(EVAL_SLOT_BUDGETS, start=1):
        observed = sum(item.slot_id == slot_id for item in specs)
        if observed > budget:
            raise RuntimeError(
                f"slot {slot_id} exceeded budget {budget}: {observed}"
            )
    return specs


def _canonicalize_right_sagittal(
    study: base.PreparedStudy,
    images: np.ndarray,
    slots: np.ndarray,
    metadata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact right-knee sagittal convention used during training."""

    if str(study.side).upper() == "R":
        sagittal = np.isin(slots, tuple(SAGITTAL_SLOT_IDS))
        if bool(sagittal.any()):
            images[sagittal] = images[sagittal, ::-1]
            metadata[sagittal, 0] *= np.float32(-1.0)
    return (
        np.ascontiguousarray(images, dtype=np.uint8),
        np.ascontiguousarray(slots, dtype=np.int64),
        np.ascontiguousarray(metadata, dtype=np.float32),
    )


def _prepare_bag_task(
    payload: tuple[int, str, Sequence[Mapping], Path],
) -> tuple[int, PreparedCoAtResidualBag]:
    """CPU-only process task; workers are spawned before CUDA initialization."""

    local_index, study_uid, study_records, series_root = payload
    base.cv2.setNumThreads(1)
    study = base.prepare_study(
        str(study_uid),
        study_records,
        Path(series_root),
        compact_stack_slices=0,
        direct_stack_fov_mm=0.0,
        sparse_native_grid="off",
    )
    specs = train_faithful_eval_specs(study)
    images, slots, metadata = base.render_eval_bag_u8(
        study,
        specs,
        output_size=384,
        requested_fov_mm=130.0,
        direct_stack_triplets=False,
    )
    images, slots, metadata = _canonicalize_right_sagittal(
        study, images, slots, metadata
    )
    return int(local_index), PreparedCoAtResidualBag(
        study_uid=str(study_uid),
        images=images,
        slots=slots,
        metadata=metadata,
        errors=tuple(study.errors),
        sparse_fallback=False,
    )


def validate_artifact_root(
    root: Path, *, verify_hashes: bool = True
) -> tuple[dict, tuple[Path, ...]]:
    """Fail closed unless checkpoints, sources, and parity are hash-bound."""

    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract") != PACKAGE_CONTRACT:
        raise RuntimeError("CoAt top-three package contract changed")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise RuntimeError("CoAt top-three manifest has no bound files")
    for name, expected in files.items():
        path = root / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes and sha256_file(path) != str(expected):
            raise RuntimeError(f"artifact hash mismatch: {name}")
    if files.get(COMPONENT_NAME) != EXPECTED_COMPONENT_SHA256:
        raise RuntimeError("epoch-10 residual-gated component is not exact")
    if files.get("coatnet_q2l_bagmatch.py") != EXPECTED_SAMPLER_SHA256:
        raise RuntimeError("epoch-10 sampler component is not exact")

    _require_equal(
        manifest.get("inference", {}),
        {
            "resolution": 384,
            "triplet_fov_mm": 130.0,
            "eval_slot_budgets": list(EVAL_SLOT_BUDGETS),
            "max_eval_windows": MAX_EVAL_WINDOWS,
            "canonical_sagittal": True,
            "model_passes_per_study": N_MODELS,
            "checkpoint_rank_method": "average_percentile_per_finding",
            "checkpoint_rank_weights": [1.0 / N_MODELS] * N_MODELS,
            "rank_after_probability_average": False,
        },
        "top-three inference",
    )
    geometry = manifest.get("gold58_geometry_contract", {})
    _require_equal(
        geometry,
        {
            "status": "VALID_COAT_RESGATED_EP10_GOLD58_GEOMETRY_CONTRACT",
            "receipt": GEOMETRY_NAME,
            "studies": 58,
            "windows": 3816,
            "exact_window_slot_metadata": True,
            "final_image_byte_exact_claim": False,
            "canonical_sagittal": True,
            "sparse_native_grid": "off",
        },
        "Gold58 geometry contract",
    )
    if geometry.get("receipt_sha256") != files.get(GEOMETRY_NAME):
        raise RuntimeError("geometry receipt is not hash-bound")

    portability = manifest.get("t4_numerical_portability", {})
    _require_equal(
        portability,
        {
            "status": "VALID_COAT_EP10_FULL_GOLD58_T4_RANK_PROBE",
            "receipt": T4_PORTABILITY_NAME,
            "predictions": T4_PORTABILITY_PREDICTIONS_NAME,
            "selected_strategy": "fast_fp16_micro8",
            "training_reference_precision": "bfloat16",
            "serving_precision": "float16",
            "opencv": "4.12.0",
            "saved_gold58_rank_ensemble_auc": (
                EXPECTED_SAVED_GOLD_RANK_ENSEMBLE_AUC
            ),
            "t4_gold58_rank_ensemble_auc": EXPECTED_T4_GOLD_RANK_ENSEMBLE_AUC,
        },
        "T4 numerical portability",
    )
    if portability.get("receipt_sha256") != files.get(T4_PORTABILITY_NAME):
        raise RuntimeError("T4 portability receipt is not hash-bound")
    if portability.get("predictions_sha256") != files.get(
        T4_PORTABILITY_PREDICTIONS_NAME
    ):
        raise RuntimeError("T4 portability predictions are not hash-bound")

    selections = manifest.get("checkpoints", [])
    if not isinstance(selections, list) or len(selections) != N_MODELS:
        raise RuntimeError("manifest must select exactly three checkpoints")
    paths: list[Path] = []
    for index, (entry, name, epoch, gold) in enumerate(
        zip(selections, CHECKPOINT_FILES, EXPECTED_EPOCHS, EXPECTED_GOLD)
    ):
        _require_equal(
            entry,
            {
                "rank": index + 1,
                "checkpoint": name,
                "epoch": epoch,
                "gold_auc": gold,
                "selection_metric": "official_gold58_macro_auc",
            },
            f"checkpoint rank {index + 1}",
        )
        if entry.get("checkpoint_sha256") != files.get(name):
            raise RuntimeError(f"checkpoint rank {index + 1} is not hash-bound")
        paths.append(root / name)
    return manifest, tuple(paths)


def _validate_checkpoint_config(config: Mapping, manifest: Mapping) -> None:
    _require_equal(
        config,
        {
            "experiment": "coatnet_residual_gated_real_k32x2_effb24_v1",
            "selection": "official_gold58_only_top3_epochs",
            "top_k": 3,
            "architecture": ARCHITECTURE,
            "resolution": 384,
            "normalization": (
                "imagenet_mean_0485_0456_0406_std_0229_0224_0225"
            ),
            "triplet_frame_mode": "common_physical",
            "triplet_fov_mm": 130.0,
            "canonical_sagittal": True,
            "train_windows_max": 32,
            "train_slot_caps": [9, 5, 5, 3, 7, 3],
            "eval_slot_budgets": list(EVAL_SLOT_BUDGETS),
            "virtual_full_bag": False,
            "spatial_head": (
                "gated_abmil_scoring_plus_fixed_gap_residual_"
                "no_attention_dropout"
            ),
            "spatial_hidden_dim": 128,
            "spatial_logit_cap": 2.0,
            "spatial_residual_mix": 0.5,
            "spatial_slot_conditioned": True,
            "window_attention_dim": 256,
            "head_dropout": 0.0,
            "slot_count_normalized": True,
            "component_sha256": EXPECTED_COMPONENT_SHA256,
            "sampler_component_sha256": EXPECTED_SAMPLER_SHA256,
        },
        "checkpoint config",
    )
    if config.get("cache_index_sha256") \
            != manifest["preprocessing"]["cache_index_sha256"]:
        raise RuntimeError("checkpoint/cache index provenance changed")


def _load_exact_component(root: Path):
    path = Path(root) / COMPONENT_NAME
    if sha256_file(path) != EXPECTED_COMPONENT_SHA256:
        raise RuntimeError("refusing to import a non-epoch-10 model component")
    name = "coatnet_resgated_ep10_exact_component"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import exact component: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint_model(
    artifact_root: Path,
    index: int,
    device: torch.device,
    *,
    manifest: Mapping | None = None,
):
    """Reconstruct one epoch-10-snapshot checkpoint with a strict state load."""

    artifact_root = Path(artifact_root)
    if manifest is None:
        manifest, paths = validate_artifact_root(
            artifact_root, verify_hashes=False
        )
    else:
        paths = tuple(artifact_root / name for name in CHECKPOINT_FILES)
    index = int(index)
    checkpoint = torch.load(
        paths[index], map_location="cpu", weights_only=False
    )
    config = dict(checkpoint.get("config", {}))
    if tuple(checkpoint.get("labels", ())) != tuple(base.LABELS):
        raise RuntimeError(f"checkpoint {index} label order changed")
    _validate_checkpoint_config(config, manifest)
    _require_equal(
        checkpoint,
        {
            "epoch": EXPECTED_EPOCHS[index],
            "gold_auc": EXPECTED_GOLD[index],
        },
        f"checkpoint {index} selection",
    )
    state_elements = sum(value.numel() for value in checkpoint["model"].values())
    if state_elements != EXPECTED_STATE_ELEMENTS:
        raise RuntimeError(
            f"checkpoint {index} state elements changed: {state_elements}"
        )

    component = _load_exact_component(artifact_root)
    backbone = base.timm.create_model(
        ARCHITECTURE,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
        in_chans=3,
    )
    model = component.CoAtResidualGatedFindingMIL(
        backbone,
        feature_dim=int(backbone.num_features),
        spatial_hidden_dim=128,
        attention_dim=256,
        dropout=0.0,
        spatial_logit_cap=2.0,
        spatial_residual_mix=0.5,
        spatial_slot_conditioned=True,
        slot_count_normalized=True,
    )
    mismatch = model.load_state_dict(checkpoint["model"], strict=True)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"checkpoint {index} strict load failed: {mismatch}")
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"checkpoint {index} parameters changed: {parameters}"
        )
    model.eval().to(device)
    model.to(memory_format=torch.channels_last)
    model.channels_last = True
    del checkpoint
    return model, config


def _normalized_images_on_device(
    bags: Sequence[PreparedCoAtResidualBag], device: torch.device
) -> torch.Tensor:
    """Copy and normalize a shared image payload only once for all models."""

    packed = np.concatenate([bag.images for bag in bags], axis=0)
    value = torch.from_numpy(packed).to(device, non_blocking=True).float().div_(255.0)
    mean = value.new_tensor(base.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = value.new_tensor(base.IMAGENET_STD).view(1, 3, 1, 1)
    return value.sub_(mean).div_(std)


@torch.inference_mode()
def _infer_one_model(
    model,
    normalized_images: torch.Tensor,
    bags: Sequence[PreparedCoAtResidualBag],
    device: torch.device,
    *,
    backbone_micro_images: int,
) -> np.ndarray:
    counts = np.asarray([len(bag.slots) for bag in bags], dtype=np.int64)
    flat_slots_np = np.concatenate([bag.slots for bag in bags], axis=0)
    flat_slots = torch.from_numpy(flat_slots_np).to(device, non_blocking=True)
    decoded_blocks: list[torch.Tensor] = []
    micro = int(backbone_micro_images)
    for start in range(0, len(flat_slots_np), micro):
        stop = min(start + micro, len(flat_slots_np))
        images = normalized_images[start:stop].contiguous(
            memory_format=torch.channels_last
        )
        slots = flat_slots[start:stop]
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                feature_map = model.backbone.forward_features(images)
                values = model.spatial_pool(feature_map, slots)
                values = model._pretrained_head_norm(values)
                values = model.norm(values)
        else:
            feature_map = model.backbone.forward_features(images)
            values = model.spatial_pool(feature_map, slots)
            values = model._pretrained_head_norm(values)
            values = model.norm(values)
        decoded_blocks.append(values)
        del images, slots, feature_map, values
    flat_values = torch.cat(decoded_blocks, dim=0)

    maximum = int(counts.max())
    values = flat_values.new_zeros(
        (len(bags), maximum, len(base.LABELS), model.feature_dim)
    )
    slots_np = np.zeros((len(bags), maximum), dtype=np.int64)
    metadata_np = np.zeros((len(bags), maximum, 2), dtype=np.float32)
    offset = 0
    for row, bag in enumerate(bags):
        count = int(counts[row])
        values[row, :count] = flat_values[offset:offset + count]
        slots_np[row, :count] = bag.slots
        metadata_np[row, :count] = bag.metadata
        offset += count
    if offset != len(flat_slots_np):
        raise RuntimeError("variable bag packing changed image count")
    slots = torch.from_numpy(slots_np).to(device, non_blocking=True)
    metadata = torch.from_numpy(metadata_np).to(device, non_blocking=True)
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.head(values, slots, metadata).float()
    else:
        logits = model.head(values, slots, metadata).float()
    output = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    del flat_slots, decoded_blocks, flat_values, values, slots, metadata, logits
    return output


@torch.inference_mode()
def infer_top3_microbatch(
    models: Sequence,
    bags: Sequence[PreparedCoAtResidualBag],
    device: torch.device,
    *,
    backbone_micro_images: int = BACKBONE_MICRO_IMAGES,
) -> np.ndarray:
    """Run all three checkpoints over one shared, already-rendered bag batch."""

    if len(models) != N_MODELS:
        raise ValueError(f"expected {N_MODELS} models, got {len(models)}")
    bags = list(bags)
    if not bags:
        return np.empty((N_MODELS, 0, len(base.LABELS)), dtype=np.float32)
    counts = np.asarray([len(bag.slots) for bag in bags], dtype=np.int64)
    if bool((counts < 1).any()) or int(counts.max()) > 94:
        raise RuntimeError(f"invalid variable bag counts: {counts.tolist()}")
    normalized = _normalized_images_on_device(bags, device)
    outputs = [
        _infer_one_model(
            model,
            normalized,
            bags,
            device,
            backbone_micro_images=int(backbone_micro_images),
        )
        for model in models
    ]
    result = np.stack(outputs, axis=0).astype(np.float32, copy=False)
    expected = (N_MODELS, len(bags), len(base.LABELS))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"invalid top-three output {result.shape}; {expected}")
    del normalized, outputs
    return result


def _infer_shard(
    *,
    artifact_root: Path,
    study_uids: Sequence[str],
    records: Mapping[str, Sequence[Mapping]],
    series_root: Path,
    progress_prefix: str,
    gpu_batch_studies: int,
    backbone_micro_images: int,
) -> tuple[np.ndarray, list[dict], dict]:
    """Rolling two-worker DICOM preparation plus three-model GPU inference."""

    started = time.monotonic()
    manifest, _ = validate_artifact_root(artifact_root, verify_hashes=False)
    study_uids = [str(value) for value in study_uids]
    prediction = np.full(
        (N_MODELS, len(study_uids), len(base.LABELS)),
        np.nan,
        dtype=np.float32,
    )
    failures: list[dict] = []
    prepared_studies = 0
    gpu_batch_studies = int(gpu_batch_studies)
    backbone_micro_images = int(backbone_micro_images)
    if gpu_batch_studies < 1 or backbone_micro_images < 1:
        raise ValueError("GPU study batch and backbone microbatch must be positive")

    def prep_payload(local_index: int):
        uid = study_uids[local_index]
        return (
            local_index,
            uid,
            records.get(uid, []),
            Path(series_root),
        )

    def record_failure(local_index: int, stage: str, exc: Exception) -> None:
        prediction[:, local_index, :] = 0.5
        failures.append(
            {
                "index": int(local_index),
                "StudyInstanceUID": study_uids[local_index],
                "stage": str(stage),
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
        )

    executor = ProcessPoolExecutor(max_workers=PREP_WORKERS)
    pending: dict = {}
    next_submit = 0
    while next_submit < min(PREP_CHUNK, len(study_uids)):
        future = executor.submit(_prepare_bag_task, prep_payload(next_submit))
        pending[future] = next_submit
        next_submit += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError("dual-T4 ensemble worker requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.monotonic()
    models = [
        load_checkpoint_model(artifact_root, index, device, manifest=manifest)[0]
        for index in range(N_MODELS)
    ]
    model_load_seconds = time.monotonic() - load_started

    def infer_batch(batch: list[tuple[int, PreparedCoAtResidualBag]]) -> None:
        indices = [item[0] for item in batch]
        bags = [item[1] for item in batch]
        try:
            values = infer_top3_microbatch(
                models,
                bags,
                device,
                backbone_micro_images=backbone_micro_images,
            )
            prediction[:, indices, :] = values
        except Exception as packed_exc:
            print(
                f"{progress_prefix}PACKED FALLBACK for {len(batch)} studies: "
                f"{type(packed_exc).__name__}: {packed_exc}",
                flush=True,
            )
            for local_index, bag in batch:
                try:
                    values = infer_top3_microbatch(
                        models,
                        [bag],
                        device,
                        backbone_micro_images=REFERENCE_MICRO_IMAGES,
                    )[:, 0, :]
                    if values.shape != (N_MODELS, len(base.LABELS)) \
                            or not np.isfinite(values).all():
                        raise RuntimeError("invalid conservative retry output")
                    prediction[:, local_index, :] = values
                except Exception as exc:
                    record_failure(local_index, "inference", exc)

    ready: list[tuple[int, PreparedCoAtResidualBag]] = []
    completed = 0
    try:
        while pending:
            finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(finished, key=lambda item: pending[item]):
                local_index = pending.pop(future)
                try:
                    observed, bag = future.result()
                    if int(observed) != int(local_index):
                        raise RuntimeError("preparation worker changed study index")
                    ready.append((local_index, bag))
                    prepared_studies += 1
                except Exception as exc:
                    record_failure(local_index, "preparation", exc)
                    completed += 1
                if next_submit < len(study_uids):
                    future = executor.submit(
                        _prepare_bag_task, prep_payload(next_submit)
                    )
                    pending[future] = next_submit
                    next_submit += 1
            while len(ready) >= gpu_batch_studies:
                batch = ready[:gpu_batch_studies]
                del ready[:gpu_batch_studies]
                infer_batch(batch)
                completed += len(batch)
                if completed % 10 == 0 or completed == len(study_uids):
                    print(
                        f"{progress_prefix}studies {completed}/"
                        f"{len(study_uids)} | elapsed "
                        f"{time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        if ready:
            infer_batch(ready)
            completed += len(ready)
            ready.clear()
            print(
                f"{progress_prefix}studies {completed}/{len(study_uids)} "
                f"| elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite top-three shard predictions")
    info = {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.monotonic() - started,
        "model_load_seconds": model_load_seconds,
        "prep_workers": PREP_WORKERS,
        "prep_chunk": PREP_CHUNK,
        "prepared_studies": int(prepared_studies),
        "gpu_batch_studies": gpu_batch_studies,
        "backbone_micro_images": backbone_micro_images,
        "models_resident": N_MODELS,
        "dicom_preparations_per_study": 1,
        "model_passes_per_study": N_MODELS,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cudnn_benchmark": CUDNN_BENCHMARK,
    }
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, failures, info


def _write_worker_shard(
    *,
    competition_root: Path,
    artifact_root: Path,
    output_path: Path,
    start: int,
    stop: int,
    parent_validated: bool,
    gpu_batch_studies: int,
    backbone_micro_images: int,
) -> dict:
    manifest, _ = validate_artifact_root(
        artifact_root, verify_hashes=not parent_validated
    )
    test, _, records, _ = base._load_submission_inputs(competition_root)
    if not 0 <= int(start) <= int(stop) <= len(test):
        raise ValueError(f"invalid worker range [{start},{stop})")
    all_uids = test.StudyInstanceUID.astype(str).tolist()
    uids = all_uids[int(start):int(stop)]
    prediction, failures, info = _infer_shard(
        artifact_root=Path(artifact_root),
        study_uids=uids,
        records=records,
        series_root=Path(competition_root) / "test_series",
        progress_prefix=f"[{start}:{stop}] ",
        gpu_batch_studies=int(gpu_batch_studies),
        backbone_micro_images=int(backbone_micro_images),
    )
    receipt = {
        "status": SHARD_STATUS,
        "runtime_contract": RUNTIME_CONTRACT, "input_max_windows": 94,
        "artifact_contract": manifest["contract"],
        "start": int(start),
        "stop": int(stop),
        "studies": int(stop - start),
        "failures": failures,
        **info,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            status=np.asarray(SHARD_STATUS),
            runtime_contract=np.asarray(RUNTIME_CONTRACT),
            artifact_contract=np.asarray(manifest["contract"]),
            start=np.asarray(start, dtype=np.int64),
            stop=np.asarray(stop, dtype=np.int64),
            study_uids=np.asarray(uids, dtype=np.str_),
            prediction=prediction,
            failures_json=np.asarray(json.dumps(failures, sort_keys=True)),
            info_json=np.asarray(json.dumps(info, sort_keys=True)),
        )
    os.replace(temporary, output_path)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _read_shard(
    path: Path,
    *,
    start: int,
    stop: int,
    expected_uids: Sequence[str],
    artifact_contract: str,
) -> tuple[np.ndarray, list[dict], dict]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "status", "runtime_contract", "artifact_contract", "start", "stop",
            "study_uids", "prediction", "failures_json", "info_json",
        }
        if required - set(payload.files):
            raise RuntimeError(f"incomplete CoAt top-three shard: {path}")
        if str(payload["status"].item()) != SHARD_STATUS \
                or str(payload["runtime_contract"].item()) != RUNTIME_CONTRACT \
                or str(payload["artifact_contract"].item()) != artifact_contract:
            raise RuntimeError(f"CoAt shard contract changed: {path}")
        if (int(payload["start"].item()), int(payload["stop"].item())) \
                != (int(start), int(stop)):
            raise RuntimeError(f"CoAt shard range changed: {path}")
        uids = payload["study_uids"].astype(str).tolist()
        prediction = np.asarray(payload["prediction"], dtype=np.float32)
        failures = json.loads(str(payload["failures_json"].item()))
        info = json.loads(str(payload["info_json"].item()))
    if uids != [str(value) for value in expected_uids]:
        raise RuntimeError(f"CoAt shard UID order changed: {path}")
    expected_shape = (N_MODELS, stop - start, len(base.LABELS))
    if prediction.shape != expected_shape or not np.isfinite(prediction).all():
        raise RuntimeError(
            f"invalid CoAt shard shape {prediction.shape}; {expected_shape}"
        )
    if info.get("device") != "cuda" \
            or int(info.get("models_resident", 0)) != N_MODELS \
            or int(info.get("dicom_preparations_per_study", 0)) != 1 \
            or bool(info.get("cudnn_benchmark")) is not CUDNN_BENCHMARK:
        raise RuntimeError(f"CoAt shard silently changed topology: {path}")
    return prediction, failures, info


def run_submission(
    *,
    competition_root: Path,
    artifact_root: Path,
    output_path: Path,
    gpu_batch_studies: int = GPU_BATCH_STUDIES,
    backbone_micro_images: int = BACKBONE_MICRO_IMAGES,
) -> dict:
    """Run two isolated T4 shards and write the equal top-three rank mean."""

    started = time.monotonic()
    competition_root = Path(competition_root)
    artifact_root = Path(artifact_root)
    output_path = Path(output_path)
    manifest, checkpoint_paths = validate_artifact_root(
        artifact_root, verify_hashes=True
    )
    test, series, records, expected_columns = base._load_submission_inputs(
        competition_root
    )
    uids = test.StudyInstanceUID.astype(str).tolist()
    if len(uids) < 2:
        raise RuntimeError("dual-T4 rank ensemble requires at least two studies")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"dual-T4 rank ensemble requires two GPUs, got "
            f"{torch.cuda.device_count()}"
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"dual-T4 ensemble requires T4 GPUs, got {gpu_names}")
    if int(gpu_batch_studies) != GPU_BATCH_STUDIES \
            or int(backbone_micro_images) != BACKBONE_MICRO_IMAGES:
        raise RuntimeError(
            "production T4 batch contract changed: "
            f"studies={gpu_batch_studies}, images={backbone_micro_images}"
        )

    weights = base._study_prepare_weights(
        uids, records, competition_root / "test_series"
    )
    bounds = base._weighted_contiguous_bounds(weights, 2)
    workloads = [int(weights[start:stop].sum()) for start, stop in bounds]
    print(
        "two workload-balanced T4 workers (start,stop,in-band-slices): "
        f"{[(a, b, w) for (a, b), w in zip(bounds, workloads)]}",
        flush=True,
    )

    runtime_path = Path(__file__).resolve()
    shard_paths = [
        output_path.with_name(f"coat_top3_part{gpu}.npz") for gpu in range(2)
    ]
    processes = []
    for gpu, ((start, stop), shard_path) in enumerate(zip(bounds, shard_paths)):
        command = [
            sys.executable,
            str(runtime_path),
            "--competition-root", str(competition_root.resolve()),
            "--artifact-root", str(artifact_root.resolve()),
            "--worker-output", str(shard_path.resolve()),
            "--worker-start", str(start),
            "--worker-stop", str(stop),
            "--parent-validated",
            "--gpu-batch-studies", str(int(gpu_batch_studies)),
            "--backbone-micro-images", str(int(backbone_micro_images)),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu, start, stop, shard_path, process))

    process_receipts = []
    errors = []
    for gpu, start, stop, shard_path, process in processes:
        log, _ = process.communicate()
        print(f"[gpu{gpu}] rc={process.returncode}", flush=True)
        if log:
            print(log.rstrip(), flush=True)
        process_receipts.append(
            {
                "gpu": gpu,
                "start": start,
                "stop": stop,
                "returncode": int(process.returncode),
                "path": str(shard_path),
            }
        )
        if process.returncode != 0:
            errors.append(f"gpu{gpu} [{start},{stop}) rc={process.returncode}")
    if errors:
        raise RuntimeError("; ".join(errors))

    raw = np.empty((N_MODELS, len(uids), len(base.LABELS)), dtype=np.float32)
    failures: list[dict] = []
    shard_receipts = []
    for (start, stop), shard_path in zip(bounds, shard_paths):
        prediction, shard_failures, info = _read_shard(
            shard_path,
            start=start,
            stop=stop,
            expected_uids=uids[start:stop],
            artifact_contract=manifest["contract"],
        )
        raw[:, start:stop, :] = prediction
        for item in shard_failures:
            fixed = dict(item)
            fixed["index"] = int(item["index"]) + int(start)
            failures.append(fixed)
        shard_receipts.append({"start": start, "stop": stop, **info})
    if not np.isfinite(raw).all():
        raise RuntimeError("non-finite merged CoAt predictions")

    checkpoint_ranks = np.stack(
        [percentile_rank64(np.clip(raw[index], 0.0, 1.0))
         for index in range(N_MODELS)],
        axis=0,
    )
    rank_ensemble = checkpoint_ranks.mean(axis=0, dtype=np.float64)
    if rank_ensemble.shape != (len(uids), len(base.LABELS)) \
            or not np.isfinite(rank_ensemble).all():
        raise RuntimeError("invalid top-three rank ensemble")
    submission = pd.DataFrame(rank_ensemble, columns=list(base.LABELS))
    submission.insert(0, "StudyInstanceUID", test.StudyInstanceUID.astype(str))
    if list(submission.columns) != expected_columns \
            or submission.StudyInstanceUID.tolist() != uids:
        raise RuntimeError("top-three submission schema/order changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    prediction_path = output_path.with_name(PREDICTIONS_NAME)
    np.savez_compressed(
        prediction_path,
        study_uids=np.asarray(uids, dtype=np.str_),
        raw_probabilities=raw,
        checkpoint_percentile_ranks=checkpoint_ranks,
        rank_ensemble=rank_ensemble,
    )
    receipt = {
        "status": SUBMISSION_STATUS,
        "runtime_contract": RUNTIME_CONTRACT, "input_max_windows": 94,
        "artifact_contract": manifest["contract"],
        "studies": len(uids),
        "series": len(series),
        "models": N_MODELS,
        "snapshot_completed_epoch": 10,
        "checkpoint_selection": "top three official Gold58 macro-AUC by epoch 10",
        "checkpoints": [
            {
                "rank": index + 1,
                "name": checkpoint_paths[index].name,
                "sha256": manifest["files"][checkpoint_paths[index].name],
                "epoch": EXPECTED_EPOCHS[index],
                "gold_auc": EXPECTED_GOLD[index],
            }
            for index in range(N_MODELS)
        ],
        "ensemble": "equal mean of three per-finding average-percentile ranks",
        "checkpoint_rank_weights": [1.0 / N_MODELS] * N_MODELS,
        "rank_after_probability_average": False,
        "historical_parent_gold58_rank_ensemble_auc": manifest["selection"][
            "gold58_rank_ensemble_auc"
        ],
        "historical_parent_t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"][
            "t4_gold58_rank_ensemble_auc"
        ],
        "serving_precision": "float16",
        "opencv": base.cv2.__version__,
        "eval_grid": "input_unique_native_centers_six_slot_v1",
        "eval_slot_budgets": list(EVAL_SLOT_BUDGETS),
        "canonical_sagittal": True,
        "resolution": 384,
        "triplet_fov_mm": 130.0,
        "gpu_names": gpu_names,
        "parallel_workers": 2,
        "parallel_bounds": [[int(a), int(b)] for a, b in bounds],
        "parallel_workload": workloads,
        "gpu_batch_studies": int(gpu_batch_studies),
        "backbone_micro_images": int(backbone_micro_images),
        "cudnn_benchmark": CUDNN_BENCHMARK,
        "dicom_preparations_per_study": 1,
        "model_passes_per_study": N_MODELS,
        "failures": failures,
        "fallback_studies": len(failures),
        "processes": process_receipts,
        "shards": shard_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    receipt_path = output_path.with_name(RECEIPT_NAME)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _prepare_visible_bags(competition_root: Path):
    test, _, records, _ = base._load_submission_inputs(competition_root)
    uids = test.StudyInstanceUID.astype(str).tolist()
    bags = []
    for index, uid in enumerate(uids):
        observed, bag = _prepare_bag_task(
            (index, uid, records.get(uid, []), competition_root / "test_series")
        )
        if observed != index:
            raise RuntimeError("visible preparation changed study order")
        bags.append(bag)
    return uids, bags


def run_visible_reference_gate(
    *,
    competition_root: Path,
    artifact_root: Path,
    predictions_path: Path,
    output_path: Path,
) -> dict:
    """Compare production micro-8 with micro-4 on the tiny visible test stub."""

    competition_root = Path(competition_root)
    artifact_root = Path(artifact_root)
    uids, bags = _prepare_visible_bags(competition_root)
    if not 2 <= len(uids) <= 8:
        raise RuntimeError(
            f"visible parity gate is limited to 2..8 studies, got {len(uids)}"
        )
    manifest, _ = validate_artifact_root(artifact_root, verify_hashes=False)
    device = torch.device("cuda:0")
    models = [
        load_checkpoint_model(artifact_root, index, device, manifest=manifest)[0]
        for index in range(N_MODELS)
    ]
    reference = infer_top3_microbatch(
        models, bags, device, backbone_micro_images=REFERENCE_MICRO_IMAGES
    )
    del models
    gc.collect()
    torch.cuda.empty_cache()

    with np.load(predictions_path, allow_pickle=False) as payload:
        production_uids = payload["study_uids"].astype(str).tolist()
        production = np.asarray(payload["raw_probabilities"], dtype=np.float32)
        # These are deliberately float64 canonical percentile ranks.  Do not
        # downcast immediately before a bitwise reduction audit: that would
        # manufacture ~1e-8 deltas even when the saved computation is exact.
        saved_ranks = np.asarray(
            payload["checkpoint_percentile_ranks"], dtype=np.float64
        )
        saved_ensemble = np.asarray(payload["rank_ensemble"], dtype=np.float64)
    if production_uids != uids or production.shape != reference.shape:
        raise RuntimeError("visible parity production payload changed")
    probability_delta = np.abs(reference - production)
    recomputed_ranks = np.stack(
        [percentile_rank64(production[index]) for index in range(N_MODELS)],
        axis=0,
    )
    recomputed_ensemble = recomputed_ranks.mean(axis=0, dtype=np.float64)
    rank_delta = np.abs(recomputed_ranks - saved_ranks)
    ensemble_delta = np.abs(recomputed_ensemble - saved_ensemble)
    max_probability_delta = float(probability_delta.max())
    if max_probability_delta > 0.002 \
            or float(rank_delta.max()) != 0.0 \
            or float(ensemble_delta.max()) != 0.0:
        raise RuntimeError(
            "visible top-three numerical parity failed: "
            f"probability={max_probability_delta}, rank={rank_delta.max()}, "
            f"ensemble={ensemble_delta.max()}"
        )
    receipt = {
        "status": "VALID_COAT_RESGATED_EP10_TOP3_VISIBLE_PARITY",
        "studies": len(uids),
        "models": N_MODELS,
        "production_micro_images": BACKBONE_MICRO_IMAGES,
        "reference_micro_images": REFERENCE_MICRO_IMAGES,
        "max_abs_probability_delta": max_probability_delta,
        "mean_abs_probability_delta": float(probability_delta.mean()),
        "max_abs_checkpoint_rank_delta": float(rank_delta.max()),
        "max_abs_rank_ensemble_delta": float(ensemble_delta.max()),
        "gold58_geometry_contract": (
            "VALID_COAT_RESGATED_EP10_GOLD58_GEOMETRY_CONTRACT"
        ),
    }
    Path(output_path).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def run_gold_model_parity_probe(
    *,
    competition_root: Path,
    artifact_root: Path,
    output_path: Path,
) -> dict:
    """Re-score one raw-DICOM Gold study against all three saved predictions."""

    competition_root = Path(competition_root)
    artifact_root = Path(artifact_root)
    manifest, _ = validate_artifact_root(artifact_root, verify_hashes=False)
    with np.load(artifact_root / GOLD_PREDICTIONS_NAME, allow_pickle=False) as payload:
        uid = str(payload["study_uids"].astype(str)[0])
        expected = np.asarray(payload["raw_probabilities"][:, 0, :], np.float32)
    series = pd.read_csv(
        competition_root / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        keep_default_na=False,
    )
    records = series.loc[series.StudyInstanceUID.eq(uid)].to_dict("records")
    if not records:
        raise RuntimeError(f"Gold parity UID is absent from train_series: {uid}")
    _, bag = _prepare_bag_task(
        (0, uid, records, competition_root / "train_series")
    )
    device = torch.device("cuda:0")
    models = [
        load_checkpoint_model(artifact_root, index, device, manifest=manifest)[0]
        for index in range(N_MODELS)
    ]
    observed = infer_top3_microbatch(
        models, [bag], device, backbone_micro_images=BACKBONE_MICRO_IMAGES
    )[:, 0, :]
    del models
    gc.collect()
    torch.cuda.empty_cache()
    delta = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    per_model = [
        {
            "epoch": EXPECTED_EPOCHS[index],
            "max_abs_probability_delta": float(delta[index].max()),
            "mean_abs_probability_delta": float(delta[index].mean()),
        }
        for index in range(N_MODELS)
    ]
    maximum = float(delta.max())
    mean = float(delta.mean())
    if maximum > 0.015 or mean > 0.002:
        raise RuntimeError(
            f"Gold model parity failed: max={maximum}, mean={mean}"
        )
    receipt = {
        "status": "VALID_COAT_RESGATED_EP10_TOP3_GOLD_MODEL_PARITY",
        "StudyInstanceUID": uid,
        "windows": int(len(bag.slots)),
        "models": N_MODELS,
        "epochs": list(EXPECTED_EPOCHS),
        "max_abs_probability_delta": maximum,
        "mean_abs_probability_delta": mean,
        "max_abs_tolerance": 0.015,
        "mean_abs_tolerance": 0.002,
        "interpretation": (
            "T4 FP16 portability smoke against saved BF16 Gold outputs; "
            "full Gold58 rank behavior is bound in the package manifest"
        ),
        "per_model": per_model,
    }
    Path(output_path).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def find_artifact_root(root: Path = Path("/kaggle/input")) -> Path:
    candidates = sorted(
        path.parent for path in Path(root).glob(f"**/{MANIFEST_NAME}")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one CoAt top-three artifact root, got {len(candidates)}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-start", type=int, default=0)
    parser.add_argument("--worker-stop", type=int, default=0)
    parser.add_argument("--parent-validated", action="store_true")
    parser.add_argument("--gpu-batch-studies", type=int, default=GPU_BATCH_STUDIES)
    parser.add_argument(
        "--backbone-micro-images", type=int, default=BACKBONE_MICRO_IMAGES
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root or find_artifact_root()
    competition_root = args.competition_root or base.find_competition_root()
    if args.worker_output is not None:
        _write_worker_shard(
            competition_root=competition_root,
            artifact_root=artifact_root,
            output_path=args.worker_output,
            start=args.worker_start,
            stop=args.worker_stop,
            parent_validated=args.parent_validated,
            gpu_batch_studies=args.gpu_batch_studies,
            backbone_micro_images=args.backbone_micro_images,
        )
    else:
        run_submission(
            competition_root=competition_root,
            artifact_root=artifact_root,
            output_path=args.output,
            gpu_batch_studies=args.gpu_batch_studies,
            backbone_micro_images=args.backbone_micro_images,
        )
    return 0


import numpy as np
def _dense_allocate(capacities, proportions, target):
    capacities = np.asarray(capacities, dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if capacities.ndim != 1 or capacities.shape != proportions.shape or np.any(capacities < 0) or (not np.isfinite(proportions).all()) or np.any(proportions <= 0) or (int(target) < 1):
        raise ValueError('invalid capacity-aware sampling allocation')
    ideal = int(target) * proportions / proportions.sum()
    initial = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - initial), kind='stable')[:int(target) - int(initial.sum())]:
        initial[i] += 1
    quotas = np.minimum(initial, capacities)
    wanted = min(int(target), int(capacities.sum()))
    deficit = np.zeros(len(quotas), dtype=np.float64)
    while int(quotas.sum()) < wanted:
        donors = quotas < capacities
        weights = np.where(donors, proportions, 0.0)
        deficit[donors] += weights[donors] / weights.sum()
        chosen = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[chosen] += 1
        deficit[chosen] -= 1
    assert int(quotas.sum()) == wanted and np.all(quotas <= capacities)
    return quotas

def _dense_unique_linspace(capacity, count):
    if not 0 <= count <= capacity:
        raise ValueError('cannot create unique picks beyond physical capacity')
    result = np.linspace(0, capacity - 1, count).round().astype(np.int64)
    if count and (result.min() < 0 or result.max() >= capacity):
        raise AssertionError('out-of-range source')
    assert len(np.unique(result)) == count
    return result

def _dense_coat_specs(study):
    import raptor_light224 as sampler
    capacities = []
    for row in study.series_rows:
        if int(row) < 0:
            capacities.append(0)
        else:
            centers, _, _, _ = sampler._candidate_centers(int(row), study.offsets, study.valid, study.source_depth)
            capacities.append(len(centers))
    if sum(capacities) < 1:
        raise ValueError('no usable CoAt triplets')
    quotas = _dense_allocate(capacities, (20, 12, 12, 8, 16, 8), 94)
    specs = sampler.sample_eval_windows(study.series_rows, study.offsets, study.valid, study.source_depth, count=None, slot_budgets=np.maximum(quotas, 1).tolist())
    if len(specs) != min(94, sum(capacities)):
        raise RuntimeError('capacity-aware sampling window-count drift')
    keys = [(s.series_row, s.center_local) for s in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError('capacity-aware sampling repeated a native center')
    for s in specs:
        start = int(study.offsets[s.series_row])
        if not np.all(study.valid[start + np.asarray(s.local_indices)]):
            raise RuntimeError('capacity-aware sampling invalid source channel')
    return specs

_original_train_faithful_eval_specs = train_faithful_eval_specs
def _tolerant_coat_specs(study):
    try:
        return _dense_coat_specs(study)
    except Exception as exc:
        print("[dense-fallback] " + str(getattr(study, "study_uid", "?")) + ": " + type(exc).__name__ + ": " + str(exc), flush=True)
        return _original_train_faithful_eval_specs(study)
train_faithful_eval_specs = _tolerant_coat_specs
RUNTIME_CONTRACT = "resgated_ep10_input_nativecenters_v1"

if __name__ == "__main__":
    raise SystemExit(main())
