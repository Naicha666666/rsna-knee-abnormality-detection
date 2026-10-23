"""Training, validation, checkpoint resume and one-model inference."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import LABELS, UID, file_hash, save_json, seed_all, to_device
from .data import StudyDataset, augment_series, collate_studies
from .metrics import auc_report
from .model import KneeModel, ModelConfig
from .native_data import dataset_for
from .supervision import supervision_loss,channel_columns
HOLD_PATH=Path(__file__).with_name('TRAINING_HOLD.json')


def autocast(device, *, training=False):
    if device.type!='cuda':return nullcontext()
    if torch.cuda.get_device_capability(device)[0]>=8:
        return torch.autocast('cuda',dtype=torch.bfloat16)
    # BF16 training was validated on the 5090. Older inference GPUs use FP16;
    # older training GPUs use FP32 rather than unscaled FP16 gradients.
    return nullcontext() if training else torch.autocast('cuda',dtype=torch.float16)


def initialize_worker(worker_id):
    import cv2
    cv2.setNumThreads(1)


def loader(dataset, batch_size, workers, shuffle=False, seed=42, sampler=None):
    initialize_worker(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,num_workers=workers,
                      collate_fn=collate_studies, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=initialize_worker,
                      persistent_workers=False, generator=torch.Generator().manual_seed(seed),
                      prefetch_factor=2 if workers else None)


def bce_numerator(logits, target, weight):
    return (F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none") * weight).sum(0)


def normalize_loss(numerator, denominator):
    present = denominator > 0
    return (numerator / denominator.clamp_min(1e-8))[present].sum() / present.sum().clamp_min(1)


def rank_loss(logits,target,weight):
    positive=(target>=.6)&(weight>=.5);negative=(target<=.4)&(weight>=.5)
    mask=positive[:,None,:]&negative[None,:,:]
    mass=mask*(weight[:,None,:]*weight[None,:,:])
    count=mass.sum((0,1));active=count>0
    delta=logits.float()[:,None,:]-logits.float()[None,:,:]
    per_class=(F.softplus(-delta)*mass).sum((0,1))/count.clamp_min(1e-8)
    return per_class[active].sum()/active.sum().clamp_min(1)


def stopped_for_plateau(history,config):
    patience=int(config.get('early_stop_patience',0))
    if not patience or len(history)<config.get('early_stop_min_epochs',4):return False
    significant=-math.inf;last=0;delta=float(config.get('early_stop_min_delta',.001))
    for row in history:
        report=row['validation'];score=report['macro_auc'] if report['macro_auc'] is not None else -report['bce']
        if score>significant+delta:significant=score;last=row['epoch']
    return history[-1]['epoch']-last>=patience


class EMA:
    def __init__(self, model):
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay):
        source = dict(model.named_parameters())
        for name, dest in self.model.named_parameters():
            dest.lerp_(source[name].detach(), 1 - decay)
        source = dict(model.named_buffers())
        for name, dest in self.model.named_buffers():
            dest.copy_(source[name])


def initialize_shared_heads(model,seed):
    """Keep unchanged heads identical when a context/auxiliary branch changes."""
    devices=list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed)+4123)
        model.spatial.value.reset_parameters();model.spatial.gate.reset_parameters()
        torch.nn.init.zeros_(model.spatial.query);torch.nn.init.zeros_(model.spatial.slot_query.weight)
    if model.config.aggregation=='source':return  # original author initialization is retained
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed)+8123)
        for pool in (model.within,model.across):
            pool.project.reset_parameters();torch.nn.init.normal_(pool.query,std=.02)
        torch.nn.init.normal_(model.classifier,std=.02);torch.nn.init.zeros_(model.bias)
        if model.config.auxiliary_head:
            torch.manual_seed(int(seed)+9123);model.auxiliary.reset_parameters()
        if model.config.presence_head:
            torch.manual_seed(int(seed)+9124);model.presence.reset_parameters()
        if model.config.clinical_head:
            torch.manual_seed(int(seed)+9125);torch.nn.init.normal_(model.factor_classifier,std=.02);torch.nn.init.zeros_(model.factor_bias)
        if model.config.mcl_joint:model.mcl_joint_head.reset_parameters()


@torch.no_grad()
def evaluate(model, data_loader, device, *, soft, attention=False):
    model.eval()
    ids, preds, targets, weights, fallback, explanations = [], [], [], [], [], []
    num, den = torch.zeros(12, device=device), torch.zeros(12, device=device)
    start = time.perf_counter()
    for cpu in data_loader:
        batch = to_device(cpu, device)
        with autocast(device):
            out = model(batch, return_attention=attention)
        if attention:
            logits, detail = out
            for i, d in enumerate(detail):
                item = {UID: cpu[UID][i], "fallback": d["fallback"]}
                if not d["fallback"]:
                    names = cpu["series_names"][i]
                    item['series_attention']=[]
                    for sid,a in zip(d['series'],d['series_attention']):
                        entry={'SeriesInstanceUID':names[int(sid)],'weights':dict(zip(LABELS,a.tolist()))}
                        if 'view_ids' in cpu:
                            first=torch.where(cpu['series'][i].eq(int(sid)))[0][0]
                            entry['view_id']=int(cpu['view_ids'][i,first])
                            entry['field_extent_mm']=float(cpu['geometry'][i,first,2]*130)
                        item['series_attention'].append(entry)
                explanations.append(item)
        else:
            logits = out
        num += bce_numerator(logits, batch["target"], batch["weight"])
        den += batch["weight"].sum(0)
        ids.extend(cpu[UID]); fallback.extend(cpu["fallback"])
        preds.append(logits.float().sigmoid().cpu().numpy())
        targets.append(cpu["target"].numpy()); weights.append(cpu["weight"].numpy())
    if not ids:
        raise ValueError("empty evaluation set")
    prediction, target, weight = np.concatenate(preds), np.concatenate(targets), np.concatenate(weights)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation studies")
    report = auc_report(target, prediction, weight, soft=soft)
    report.update(bce=float(normalize_loss(num, den)), seconds=time.perf_counter() - start,
                  studies=len(ids), fallback_studies=sum(fallback),
                  fallback_ids=[u for u, f in zip(ids, fallback) if f])
    frame = pd.DataFrame(prediction, columns=LABELS)
    frame.insert(0, UID, ids)
    from .ranking_data import evaluation_diagnostics
    report['ranking_diagnostics'] = evaluation_diagnostics(data_loader.dataset, ids, prediction,
        target, weight, soft=soft, engine_report=report)
    return report, frame, explanations


def checkpoint_payload(model, *, config, epoch, provenance,optimizer_steps=0):
    frozen=config.get('frozen_encoder_reference')
    payload={"model": {k: v.detach().cpu() for k, v in model.state_dict().items()
                       if not frozen or not k.startswith(('backbone.','spatial.','norm.'))},
            "model_config": model.export_config(), "train_config": config,
            "labels": LABELS, "epoch": epoch, "provenance": provenance,'optimizer_steps':int(optimizer_steps)}
    if frozen:payload['encoder_reference']=frozen
    return payload


def load_checkpoint_model(model,state):
    weights=dict(state['model'])
    if state.get('encoder_reference'):
        source=state['encoder_reference']
        if file_hash(source['path'])!=source['sha256']:raise ValueError('checkpoint frozen encoder changed')
        weights={**torch.load(source['path'],map_location='cpu',weights_only=True),**weights}
    model.load_state_dict(weights)


def atomic_checkpoint(path, value):
    path = Path(path); tmp = path.with_suffix(".tmp")
    with tmp.open("wb", buffering=8 * 2**20) as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def recovery_checkpoint(output):
    paths=[Path(output)/name for name in ('last.pt','pending.pt','step.pt') if (Path(output)/name).exists()]
    if not paths:raise FileNotFoundError('No recoverable training checkpoint')
    return max(paths,key=lambda p:p.stat().st_mtime_ns)


def training_ids(frame, manifest, fold,*,reserve_gold=True):
    if frame[UID].duplicated().any():
        raise ValueError("duplicate labels")
    from .unknown_review_isolation import reserved_mask
    frame = frame.loc[~reserved_mask(frame)].copy()
    if fold == "full":
        train = frame[frame.fold >= 0]
        valid = frame[frame.is_gold.astype(bool)] if reserve_gold else frame.iloc[:0]
    else:
        f = int(fold)
        train = frame[(frame.fold >= 0) & frame.fold.ne(f)]
        valid = frame[frame.fold.eq(f)]
    if train.empty or (valid.empty and fold!='full'):
        raise ValueError("empty train or validation partition")
    if set(train.group_id) & set(valid.group_id):
        raise ValueError("group leakage in label split")
    cache = {r[UID]: r for r in json.loads(Path(manifest).read_text(encoding="utf-8"))["studies"]}
    required = set(train[UID]) | set(valid[UID])
    if required - set(cache):
        raise ValueError(f"{len(required - set(cache))} required studies absent from image cache")
    # Re-check raw DICOM identities even if labels were split before download.
    tr_groups = {cache[u].get("PatientID") or u for u in train[UID]}
    va_groups = {cache[u].get("PatientID") or u for u in valid[UID]}
    gold = frame[frame.is_gold.astype(bool)] if reserve_gold else frame.iloc[:0]
    gold_groups = {cache[u].get("PatientID") or u for u in gold[UID] if u in cache}
    if tr_groups & (va_groups | gold_groups):
        raise ValueError("patient overlap: regenerate labels with cache/patients.csv")
    return train, valid


def train(config, manifest, labels_path, output, *, fold="0", resume=False, pretrained=True,
          smoke=False, evaluation_labels_path=None, feature_manifest=None, init_checkpoint=None, stop_after_epochs=None,
          execution_options=None,resume_from=None,stop_after_updates=None):
    hold=HOLD_PATH
    if hold.exists() and json.loads(hold.read_text(encoding='utf-8')).get('active'):
        raise RuntimeError('Training is held at the user request. Complete construction review before obtaining a new training instruction.')
    from .label_source_policy import validate_label_source,POLICY
    own_label_contract=validate_label_source(labels_path)
    config=deepcopy(config)
    from .execution_runtime import validate_options,OrderedPrefetch,OrderedStudyPrefetch,prune_unused
    runtime=validate_options(execution_options)
    if fold=='full':config['early_stop_patience']=0  # Gold58 is reporting-only in fixed-budget full training.
    if feature_manifest:
        bank=json.loads(Path(feature_manifest).read_text(encoding='utf-8'))
        if bank['contract']['manifest_sha256']!=file_hash(manifest):raise ValueError('feature bank and image manifest differ')
        config['frozen_encoder_reference']={'path':bank['encoder'],'sha256':bank['encoder_sha256']}
    for name in ("epochs", "batch_size", "accumulate", "train_windows_per_series", "eval_windows_per_series"):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('last.pt','pending.pt','step.pt')) and not resume:
        raise FileExistsError("run already exists; use --resume or a new --output")
    seed = int(config.get("seed", 42)); seed_all(seed)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_config = ModelConfig(**config["model"])
    if runtime.get('encoder_chunk'):model_config.encode_chunk=int(runtime['encoder_chunk'])
    if bool(model_config.partial_joint)!=bool(config.get('partial_joint_weight',0)>0):
        raise ValueError('Partial clinical readout and its loss must be enabled together')
    if model_config.partial_joint and (feature_manifest or config.get('rank_weight',0) or config.get('aux_weight',0)):
        raise ValueError('Partial clinical readout cannot be bypassed by cached or legacy auxiliary heads')
    if bool(model_config.mcl_joint)!=bool(config.get('mcl_joint_weight',0)>0):
        raise ValueError('MCL joint readout and partial-supervision loss must be enabled together')
    if model_config.mcl_joint and (feature_manifest or config.get('rank_weight',0) or config.get('aux_weight',0)):
        raise ValueError('MCL joint requires image training; cached rank or auxiliary main heads bypass its readout')
    frame = pd.read_csv(labels_path, dtype={UID: str, "group_id": str})
    weights_array = frame[[x + "__weight" for x in LABELS]].to_numpy(float)
    targets_array = frame[LABELS].to_numpy(float)
    if not np.isfinite(weights_array).all() or (weights_array < 0).any():
        raise ValueError("label weights must be finite and nonnegative")
    valid_target = np.isfinite(targets_array) & (targets_array >= 0) & (targets_array <= 1)
    if ((weights_array > 0) & ~valid_target).any():
        raise ValueError("observed targets must be finite probabilities")
    reserve_gold=config.get('reserve_gold',True)
    train_frame, valid_frame = training_ids(frame, manifest, fold,reserve_gold=reserve_gold)
    factor_support=None
    if config.get('factor_weight',0):
        from .supervision import supported_factors
        from .partial_joint import JOINT_LABELS
        joint_labels=tuple(JOINT_LABELS if model_config.partial_joint else ())+ (('MCL',) if model_config.mcl_joint else ())
        frame,factor_support=supported_factors(frame,train_frame[UID],retain_for_joint=joint_labels)
        train_frame=frame[frame[UID].isin(train_frame[UID])];valid_frame=frame[frame[UID].isin(valid_frame[UID])]
    local_evidence_support=None
    if model_config.local_oa_partial:
        from .local_oa_supervision import attach as attach_local_oa
        source_labels=Path(config.get('labels_dir',Path(labels_path).parent))
        if config.get('input_package_path'):
            source_labels=Path(json.loads(Path(config['input_package_path']).read_text(encoding='utf-8'))['labels_source']).parent
        if config.get('local_oa_evidence_sha256') and file_hash(source_labels/'evidence.jsonl')!=config['local_oa_evidence_sha256']:
            raise ValueError('Frozen local observation source changed')
        frame,local_oa_support=attach_local_oa(frame,source_labels/'evidence.jsonl')
        save_json(output/'local_oa_support.json',local_oa_support)
        train_frame=frame[frame[UID].isin(train_frame[UID])];valid_frame=frame[frame[UID].isin(valid_frame[UID])]
    if config['model'].get('local_evidence'):
        from .local_evidence_supervision import attach_local_evidence,ROUTE_COLUMNS
        sidecar=config.get('local_evidence_sidecar')
        if not sidecar:raise ValueError('local evidence supervision sidecar is missing')
        label_release_hash=file_hash(labels_path)
        if config.get('input_package_path'):
            label_release_hash=json.loads(Path(config['input_package_path']).read_text(encoding='utf-8'))['labels_source_sha256']
        frame,local_evidence_support=attach_local_evidence(frame,sidecar,training_ids=train_frame[UID],
            expected_labels_sha256=label_release_hash)
        train_frame=frame[frame[UID].isin(train_frame[UID])];valid_frame=frame[frame[UID].isin(valid_frame[UID])]
        save_json(output/'local_evidence_support.json',local_evidence_support)
    active_columns=[c+'__weight' for c in LABELS]
    if model_config.partial_joint:
        from .partial_joint import audit as partial_audit,from_frame as partial_from_frame,mass_vector as partial_mass
        from .clinical_factors import FACTOR_COLUMNS
        active_columns += [c+'__presence_weight' for c in LABELS]+[c+'_weight' for c in FACTOR_COLUMNS]
        save_json(output/'partial_joint_support.json',partial_audit(train_frame))
    if model_config.local_oa_partial:
        from .local_oa_supervision import COLUMNS as LOCAL_OA_COLUMNS
        active_columns += [c+'_weight' for c in LOCAL_OA_COLUMNS]
    if model_config.mcl_joint:
        from .mcl_joint import PARTIAL_COLUMNS,audit_frame,evidence_from_frame
        active_columns += [c+'_weight' for c in PARTIAL_COLUMNS]
        # Audits only this fold's fitting patients, after factor support masks.
        save_json(output/'mcl_joint_support.json',audit_frame(train_frame))
    if config['model'].get('local_evidence'):active_columns += [c+'_weight' for c in ROUTE_COLUMNS]
    for channel,coefficient in [('presence',config.get('presence_weight',0)),('teacher',config.get('distill_weight',0)),('factors',config.get('factor_weight',0))]:
        if not coefficient:continue
        targets=channel_columns(channel);columns=[c+'_weight' for c in targets]
        if not set(columns+targets).issubset(frame):raise ValueError(channel+' labels missing')
        cw=frame[columns].to_numpy(float);cy=frame[targets].to_numpy(float)
        if not np.isfinite(cw).all() or (cw<0).any() or ((cw>0)&(~np.isfinite(cy)|(cy<0)|(cy>1))).any():
            raise ValueError(channel+' target or weight invalid')
        active_columns+=columns
    eligible=train_frame[active_columns].sum(1)>0
    train_frame=train_frame[eligible].copy()
    if train_frame.empty:raise ValueError('no observed training supervision')
    if config.get('distill_weight',0):
        lineage_path=Path(labels_path).with_name('lineage.json')
        if not lineage_path.is_file():raise ValueError('distillation lineage missing')
        lineage=json.loads(lineage_path.read_text())
        if fold=='full' or lineage['outer_fold']!=int(fold):raise ValueError('distillation targets belong to another outer fold')
        if lineage.get('compiled_labels_sha256')!=file_hash(labels_path):raise ValueError('distillation labels were changed after lineage verification')
        if train_frame[[c+'__teacher_weight' for c in LABELS]].to_numpy().sum()<=0:
            raise ValueError('the training partition has no qualified teacher targets')
    reference = frame
    if evaluation_labels_path:
        reference = pd.read_csv(evaluation_labels_path,dtype={UID:str,'group_id':str})
        if reference[UID].duplicated().any() or set(reference[UID]) != set(frame[UID]):
            raise ValueError('evaluation reference IDs differ from training label table')
        aligned = reference.set_index(UID).loc[frame[UID]]
        for key in ('group_id','fold','is_gold'):
            if not np.array_equal(aligned[key].to_numpy(),frame[key].to_numpy()):
                raise ValueError('evaluation reference patient groups or split changed')
        reference_y = reference[LABELS].to_numpy(float)
        reference_w = reference[[x+'__weight' for x in LABELS]].to_numpy(float)
        if (not np.isfinite(reference_w).all() or (reference_w<0).any()
            or ((reference_w>0) & ~np.isin(reference_y,[0.,1.])).any()):
            raise ValueError('evaluation reference must have binary observed labels and finite nonnegative masks')
    if not smoke and config.get('supervision_semantics')=='competition_criterion':
        problems=[]
        partitions=[(train_frame,'training')]
        if len(valid_frame):partitions.append((reference[reference[UID].isin(valid_frame[UID])],'validation'))
        for part,name in partitions:
            for c in LABELS:
                observed=part.loc[part[c+'__weight']>0,c]
                if observed.nunique()!=2:problems.append(name+': '+c)
        if problems:raise ValueError('criterion supervision cannot support the 12-label experiment: '+', '.join(problems))
    if (train_frame[active_columns].sum(axis=1) <= 0).any():
        raise ValueError("unlabelled studies cannot be counted as supervised training updates")
    if feature_manifest:
        from .feature_bank import FeatureStudyDataset
        if config.get('augment',True):raise ValueError('frozen feature training must explicitly disable image augmentation')
        train_data=FeatureStudyDataset(feature_manifest,frame,ids=train_frame[UID])
        valid_data=FeatureStudyDataset(feature_manifest,reference,ids=valid_frame[UID])
        if any(s.get('fallback',False) for s in train_data.studies):
            raise ValueError('training feature bank contains studies without MRI; repair those training inputs')
    else:
        train_data=dataset_for(manifest,frame,config,ids=train_frame[UID],training=True)
        valid_data=dataset_for(manifest,reference,config,ids=valid_frame[UID])
    if train_data.size != model_config.image_size:
        raise ValueError("cache size differs from model input size")
    gold_frame = frame[frame.is_gold.astype(bool)]
    if not reserve_gold:gold_frame=gold_frame[gold_frame[UID].isin(valid_frame[UID])]
    cached = {s[UID] for s in train_data.manifest["studies"]}
    gold_loader = None
    if reserve_gold and fold != "full" and len(gold_frame) and set(gold_frame[UID]).issubset(cached):
        gold_data = FeatureStudyDataset(feature_manifest,frame,ids=gold_frame[UID]) if feature_manifest else dataset_for(manifest,frame,config,ids=gold_frame[UID])
        gold_loader = loader(gold_data, 1, config["workers"])
    valid_loader = loader(valid_data, 1, config["workers"]) if len(valid_data) else None
    from .ranking_data import bind_evaluation_source
    ranking_binding = None
    if valid_loader is not None:
        ranking_binding = bind_evaluation_source(valid_data, reference, output, name='evaluation',
            source_path=evaluation_labels_path or labels_path, image_manifest=manifest,
            train_ids=train_frame[UID], scope='validation' if fold != 'full' else 'development_reference')
    if gold_loader is not None:
        bind_evaluation_source(gold_data, frame, output, name='gold_development',
            source_path=labels_path, image_manifest=manifest, train_ids=train_frame[UID], scope='development_reference')
    model = KneeModel(model_config, pretrained=pretrained and not resume and not feature_manifest and not init_checkpoint)
    external_source=None;external_validation=None
    external_weight=float(config.get('external_mcl_weight',0.))
    if not math.isfinite(external_weight) or external_weight<0:raise ValueError('invalid external MCL loss weight')
    if external_weight:
        if not model_config.external_mcl or feature_manifest:
            raise ValueError('external expert supervision must update shared image features')
        from .external_mcl import ExternalMCLSource
        external_source=ExternalMCLSource(config['external_mcl_manifest'],
            source_fold=int(config.get('external_mcl_source_fold',0)),image_size=model_config.image_size)
        external_validation=ExternalMCLSource(config['external_mcl_manifest'],
            source_fold=int(config.get('external_mcl_source_fold',0)),image_size=model_config.image_size,role='evaluation')
    if config.get('controlled_head_initialization',False):initialize_shared_heads(model,seed)
    model=model.to(device)
    if feature_manifest:
        from .feature_bank import attach_encoder
        attach_encoder(model,feature_manifest,config)
    y = train_frame[LABELS].to_numpy(float)
    w = train_frame[[x + "__weight" for x in LABELS]].to_numpy(float)
    prior = (np.nan_to_num(y) * w).sum(0) / w.sum(0).clip(1)
    prior = np.clip(prior, .001, .999)
    with torch.no_grad():
        model.fallback_logits.copy_(torch.tensor(np.log(prior / (1 - prior)), device=device))
        model.bias.copy_(model.fallback_logits)
    if init_checkpoint:
        initial=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
        validate_label_source(labels_path,checkpoint=initial)
        if set(initial['provenance']['train_ids']) & (set(valid_frame[UID])|set(gold_frame[UID])):
            raise ValueError('initial knee checkpoint has seen this validation/development partition')
        load_checkpoint_model(model,initial)
    backbone_params, spatial_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:continue
        (backbone_params if name.startswith(('backbone.','late_stage.')) else spatial_params if name.startswith('spatial.') else other_params).append(p)
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": config["backbone_lr"]},
                                   {"params": spatial_params,"lr":config.get('spatial_lr',config['head_lr'])},
                                   {"params": other_params, "lr": config["head_lr"]}],
                                  weight_decay=config["weight_decay"])
    draws=int(config.get('train_draws_per_study',1))
    micro_batches = math.ceil(len(train_data)*draws / config["batch_size"])
    updates = math.ceil(micro_batches / config["accumulate"])
    total = updates * config["epochs"]
    warmup = max(1, int(total * .1))
    def rate(step):
        if step < warmup:
            return .1 + .9 * step / warmup
        return .01 + .99 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    ema = EMA(model)
    provenance = {"labels_sha256": file_hash(labels_path), "manifest_sha256": file_hash(manifest),
                  "label_source_policy":POLICY,"label_source_contract_sha256":file_hash(Path(labels_path).with_name('contract.json')),
                  "fold": fold, "train_ids": train_frame[UID].tolist(), "validation_ids": valid_frame[UID].tolist(),
                  "gold_reserved": gold_frame[UID].tolist(), "initialization": "imagenet_pretrained" if pretrained else "random",
                  "smoke": smoke, "target_score": .975,
                  "selection": "last_epoch_ema_fixed_budget" if fold == "full" else "weak_fold_macro_auc",
                  "target_score_is_not_measured": True}
    if ranking_binding is not None:provenance['evaluation_reference_binding'] = ranking_binding
    if external_source is not None:
        provenance['external_expert_auxiliary']=external_source.binding()
    if pretrained and config.get('pretraining_provenance') and not init_checkpoint:
        source_path=Path(config['pretraining_provenance'])
        source=json.loads(source_path.read_text(encoding='utf-8'))
        if source['encoder_sha256']!=file_hash(model_config.pretrained_file):
            raise ValueError('medical encoder and pretraining source record differ')
        if source['input_normalization'].startswith('unit interval') and model_config.input_normalization!='unit':
            raise ValueError('medical encoder normalization differs from its source contract')
        if config.get('source_input_convention','canonical')!=source.get('source_input_convention','canonical'):
            raise ValueError('medical encoder source input convention differs from its source contract')
        if source.get('kind')=='external_ligament_encoder':
            if source.get('optimizer_steps',0)<=0:raise ValueError('External anatomy pretraining has no fitted updates')
            if source.get('competition_study_exposure')!=[]:raise ValueError('External anatomy source includes target-study exposure')
            if source['backbone']!=model_config.backbone or source['input_normalization']!=model_config.input_normalization:
                raise ValueError('External anatomy encoder architecture/normalization mismatch')
            provenance['initialization']='external_ligament_encoder'
        elif source.get('kind') in ('released_general_vision_encoder','released_orthopedic_encoder'):
            if source.get('backbone')!=model_config.backbone:raise ValueError('Vision encoder architecture differs from source record')
            if source.get('required_backbone_position_mode',model_config.backbone_position_mode)!=model_config.backbone_position_mode:
                raise ValueError('Encoder position handling differs from its verified public forward')
            if source.get('required_backbone_freeze_attention_bias',model_config.backbone_freeze_attention_bias)!=model_config.backbone_freeze_attention_bias:
                raise ValueError('Encoder attention-bias policy differs from its source masks')
            provenance['initialization']=source['kind']
        else:provenance['initialization']='released_knee_tissue_encoder'
        provenance['pretraining_source']=source
        provenance['pretraining_manifest_sha256']=file_hash(source_path)
    provenance.update(selection_ids=[] if fold=='full' else valid_frame[UID].tolist(),
        supervision_development_ids=config.get('supervision_development_ids',[]),parent_exposure_ids=[],
        supervision_semantics=config.get('supervision_semantics','legacy_report_labels'),
        groups={r[UID]:str(r.group_id) for _,r in frame.iterrows()})
    if 'supervision_development_ids' not in config:provenance['supervision_lineage_incomplete']=True
    if factor_support is not None:provenance['clinical_factor_support']=factor_support
    if init_checkpoint:
        ancestors=initial['provenance']
        if all(k in ancestors for k in ('train_ids','selection_ids','supervision_development_ids','parent_exposure_ids')):
            provenance['parent_exposure_ids']=sorted(set().union(*(set(ancestors[k]) for k in ('train_ids','selection_ids','supervision_development_ids','parent_exposure_ids'))))
        else:provenance['parent_lineage_incomplete']=True
        if ancestors.get('parent_lineage_incomplete') or ancestors.get('supervision_lineage_incomplete'):
            provenance['parent_lineage_incomplete']=True
    if evaluation_labels_path:
        provenance['evaluation_labels_sha256']=file_hash(evaluation_labels_path)
        provenance['selection']='last_epoch_ema_fixed_budget' if fold=='full' else 'frozen_weak_reference_macro_auc'
    if feature_manifest:provenance['feature_manifest_sha256']=file_hash(feature_manifest)
    if init_checkpoint:provenance['init_checkpoint_sha256']=file_hash(init_checkpoint)
    if pretrained and model_config.pretrained_file and Path(model_config.pretrained_file).is_file():
        provenance['pretrained_sha256']=file_hash(model_config.pretrained_file)
    first_epoch, best, step_number = 0, -math.inf, 0
    pending_training=None
    partial_epoch=None
    history = []
    if resume:
        recovery=Path(resume_from) if resume_from is not None else recovery_checkpoint(output)
        state = torch.load(recovery, map_location=device, weights_only=False)  # own trusted training checkpoint
        validate_label_source(labels_path,checkpoint=state)
        if 'pretrained_sha256' in state['provenance']:
            provenance['pretrained_sha256']=state['provenance']['pretrained_sha256']
        comparable_provenance = dict(provenance)
        if 'evaluation_reference_binding' not in state['provenance']:
            # Old checkpoints still verify their original label/reference hashes;
            # adding an immutable diagnostic snapshot is not a new experiment.
            comparable_provenance.pop('evaluation_reference_binding', None)
        recovered_provenance=deepcopy(state['provenance'])
        if resume_from is not None and 'evaluation_reference_binding' in recovered_provenance:
            old=recovered_provenance['evaluation_reference_binding'];new=comparable_provenance['evaluation_reference_binding']
            for key,hash_key in [('reference_path','reference_sha256'),('protocol_metadata_path','protocol_metadata_sha256')]:
                if old[hash_key]!=new[hash_key] or file_hash(new[key])!=old[hash_key]:raise ValueError('Relocated evaluation bytes differ')
                old[key]=new[key]
        if state["train_config"] != config or recovered_provenance != comparable_provenance:
            raise ValueError("resume data/config/init differs; this is a new experiment")
        load_checkpoint_model(model,state);load_checkpoint_model(ema.model,{**state,'model':state['ema']})
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        pending_training=state.get('completed_training') if recovery.name=='pending.pt' else None
        # A verified checkpoint copy may have a different filename. The
        # partial-epoch payload, not its basename, defines the next update.
        partial_epoch=state.get('partial_epoch')
        first_epoch, best, step_number = state["epoch"] + (0 if pending_training or partial_epoch else 1), state["best"], state["step"]
        torch.set_rng_state(state["rng_torch"].cpu())
        np.random.set_state(state["rng_numpy"]); random.setstate(state["rng_python"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        history = [r for r in json.loads((output / "history.json").read_text(encoding="utf-8")) if r['epoch']<first_epoch] if (output/'history.json').exists() else []
        for row in history:
            metric=row['validation'];best=max(best,metric['macro_auc'] if metric['macro_auc'] is not None else (-metric['bce'] if metric['bce'] is not None else -math.inf))
        del state
    save_json(output / "config.json", config)
    save_json(output / "provenance.json", provenance)
    save_json(output/'execution_runtime.json',{'options':runtime,'resume_source':str(resume_from) if resume_from else None,
        'starting_optimizer_step':step_number,'device':str(device),'unchanged_train_config':True})
    if pending_training is None and stopped_for_plateau(history,config):return history
    until=config['epochs'] if stop_after_epochs is None else min(config['epochs'],int(stop_after_epochs))
    for epoch in range(first_epoch, until):
        epoch_started=time.perf_counter()
        epoch_started_unix=time.time()
        model.train(); train_data.epoch = epoch
        sampler=None
        if draws!=1 or config.get('balanced_batches',False):
            from .sampling import StudyExposureSampler
            sampler=StudyExposureSampler(train_data.ids,frame,draws=draws,
                effective_batch=config['batch_size']*config['accumulate'],balanced=config.get('balanced_batches',False),seed=seed+epoch,
                hard_negatives=config.get('hard_negatives',False),paired_contrasts=config.get('paired_contrasts',False),
                factor_contrasts=config.get('factor_contrasts',False))
        schedule=None
        if config.get('stream_accumulation',False):
            from .sampling import StudyExposureSampler
            if sampler is None:sampler=StudyExposureSampler(train_data.ids,frame,draws=draws,
                effective_batch=config['batch_size']*config['accumulate'],seed=seed+epoch)
            schedule=list(sampler);sampler=schedule
        first_update=int(partial_epoch['updates_completed']) if partial_epoch else 0
        if first_update:
            if schedule is None:raise ValueError('Mid-epoch resume requires the deterministic streamed schedule')
            sampler=schedule[first_update*config['accumulate']*config['batch_size']:]
        train_loader = loader(train_data, config["batch_size"], config["workers"], True, seed + epoch,sampler=sampler)
        iterator = iter(train_loader) if pending_training is None else None
        if iterator is not None and runtime.get('prefetch_studies')==2:
            if schedule is None or config['batch_size']!=1:raise ValueError('Two-producer preparation requires the fixed single-Study streamed schedule')
            iterator=OrderedStudyPrefetch(train_data,schedule[first_update*config['accumulate']:])
        elif iterator is not None and runtime.get('prefetch_studies'):iterator=OrderedPrefetch(iterator)
        start = time.perf_counter(); loss_sum = 0.; samples = 0; data_seconds = 0.
        exposure_positive=np.zeros(12,int);exposure_negative=np.zeros(12,int);rank_sum=0.;aux_sum=0.;extra_loss_sums={}
        if partial_epoch:
            loss_sum=partial_epoch['loss_sum'];samples=partial_epoch['samples'];data_seconds=partial_epoch['data_seconds']
            exposure_positive=np.asarray(partial_epoch['positive']);exposure_negative=np.asarray(partial_epoch['negative'])
            rank_sum=partial_epoch['rank_sum'];aux_sum=partial_epoch['aux_sum'];extra_loss_sums=partial_epoch['extra_loss_sums']
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        save_json(output/'progress.json',{'phase':'training','epoch':epoch+1,'epochs':config['epochs'],
                    'update':first_update,'updates':updates,'smoke':smoke})
        for update in range(first_update,updates if pending_training is None else 0):
            update_start=time.perf_counter();update_data_start=data_seconds;update_loss_start=loss_sum
            t = time.perf_counter(); group = []
            if schedule is None:
                for _ in range(config["accumulate"]):
                    try:group.append(next(iterator))
                    except StopIteration:break
                data_seconds += time.perf_counter() - t
                denominator=sum((b['weight'].sum(0) for b in group),torch.zeros(12)).to(device)
                group_batches=len(group)
                extra_den={'studies':sum(len(b[UID]) for b in group)}
                for channel in ('presence','teacher','factors'):
                    if channel+'_weight' in group[0]:extra_den[channel]=sum((b[channel+'_weight'].sum(0) for b in group),torch.zeros(len(channel_columns(channel)))).to(device)
                if model_config.mcl_joint:
                    from .mcl_joint import evidence_from_batch
                    extra_den['mcl_joint']=sum((evidence_from_batch(b)['mass'].sum() for b in group),torch.tensor(0.)).to(device)
                if model_config.partial_joint:
                    from .partial_joint import from_batch as partial_from_batch
                    extra_den['partial_joint']=sum((partial_mass(partial_from_batch(b)) for b in group),torch.zeros(11)).to(device)
            else:
                offset=update*config['accumulate']*config['batch_size']
                indices=schedule[offset:offset+config['accumulate']*config['batch_size']]
                ids=[train_data.ids[i] for i,draw in indices]
                weights=torch.from_numpy(train_data.labels.loc[ids,[c+'__weight' for c in LABELS]].to_numpy(np.float32))
                denominator=sum((w.sum(0) for w in weights.split(config['batch_size'])),torch.zeros(12)).to(device)
                group_batches=math.ceil(len(indices)/config['batch_size'])
                extra_den={'studies':len(ids)}
                for channel in ('presence','teacher','factors'):
                    columns=[c+'_weight' for c in channel_columns(channel)]
                    if set(columns).issubset(train_data.labels):
                        extra_den[channel]=torch.from_numpy(train_data.labels.loc[ids,columns].to_numpy(np.float32).sum(0)).to(device)
                if model_config.mcl_joint:
                    extra_den['mcl_joint']=evidence_from_frame(train_data.labels.loc[ids])['mass'].sum().to(device)
                if model_config.partial_joint:
                    extra_den['partial_joint']=partial_mass(partial_from_frame(train_data.labels.loc[ids])).to(device)
            optimizer.zero_grad(set_to_none=True)
            rank_blocks=[];group_samples=0
            for micro in range(group_batches):
                if schedule is None:cpu=group[micro]
                else:
                    t=time.perf_counter();cpu=next(iterator);data_seconds+=time.perf_counter()-t
                group_samples+=len(cpu[UID])
                batch = to_device(prune_unused(cpu,model_config,runtime), device)
                if config.get("augment", True):
                    batch = augment_series(batch, intensity_policy=config.get('intensity_augmentation_policy', 'legacy'))
                with autocast(device,training=True):
                    out = model(batch,return_training=True)
                    logits=out['logits'];num=bce_numerator(logits,batch['target'],batch['weight'])
                    loss = num.sum()/denominator.sum().clamp_min(1e-8) if config.get('loss_normalization')=='global' else normalize_loss(num,denominator)
                    if config.get('aux_weight',0):
                        if out['auxiliary'] is None:raise ValueError('auxiliary Study head missing')
                        aux=normalize_loss(bce_numerator(out['auxiliary'],batch['target'],batch['weight']),denominator)
                        loss=loss+config['aux_weight']*aux;aux_sum+=float(aux.detach())
                    extra_den['criterion']=denominator
                    extra_loss,extra_parts=supervision_loss(out,batch,extra_den,config)
                    loss=loss+extra_loss
                    for name,value in extra_parts.items():extra_loss_sums[name]=extra_loss_sums.get(name,0.)+float(value)
                if config.get('rank_weight',0):
                    rank_blocks.append((out['values'].detach(),out['global'].detach(),
                        {k:v for k,v in batch.items() if k not in ('images','encoded_values','encoded_global')}))
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                loss_sum += float(loss.detach()); samples += len(cpu[UID])
                observed=cpu['weight'].numpy()>=.5;targets=cpu['target'].numpy()
                exposure_positive+=((targets>=.6)&observed).sum(0);exposure_negative+=((targets<=.4)&observed).sum(0)
            if external_source is not None:
                t=time.perf_counter()
                from .external_mcl import auxiliary_step
                with autocast(device,training=True):
                    external_loss,external_parts=auxiliary_step(model,external_source,step_number,seed,device,external_weight)
                if not torch.isfinite(external_loss):raise RuntimeError('nonfinite external MCL supervision')
                external_loss.backward();loss_sum+=float(external_loss.detach())
                for name in ('source_box_attention','source_grade','source_presence'):
                    extra_loss_sums[name]=extra_loss_sums.get(name,0.)+float(external_parts[name])
                extra_loss_sums['source_seconds']=extra_loss_sums.get('source_seconds',0.)+time.perf_counter()-t
            if rank_blocks:
                with autocast(device,training=True):
                    rank_logits=torch.cat([model.aggregate(v,g,b) for v,g,b in rank_blocks])
                    ranking=rank_loss(rank_logits,torch.cat([b['target'] for _,_,b in rank_blocks]),
                                      torch.cat([b['weight'] for _,_,b in rank_blocks]))
                (config['rank_weight']*ranking).backward()
                rank_sum+=float(ranking.detach());loss_sum+=config['rank_weight']*float(ranking.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradient")
            optimizer.step(); scheduler.step(); step_number += 1
            # One epoch of examples sets the EMA timescale, independent of accumulation.
            decay = min(config["ema_decay"], math.exp(-group_samples / len(train_data)))
            ema.update(model, decay)
            interval=int(config.get('checkpoint_every_updates',0))
            stop_requested=(output/'STOP_AFTER_UPDATE').exists() or (stop_after_updates is not None and step_number>=stop_after_updates)
            if (interval and ((update+1)%interval==0 or (step_number==1 and config.get('checkpoint_first_update')))) or stop_requested:
                state=checkpoint_payload(model,config=config,epoch=epoch,provenance=provenance,optimizer_steps=step_number)
                state.update(ema=checkpoint_payload(ema.model,config=config,epoch=epoch,provenance=provenance,optimizer_steps=step_number)['model'],
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,step=step_number,
                    rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),
                    rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    partial_epoch={'updates_completed':update+1,'loss_sum':loss_sum,'samples':samples,'data_seconds':data_seconds,
                        'positive':exposure_positive.tolist(),'negative':exposure_negative.tolist(),'rank_sum':rank_sum,'aux_sum':aux_sum,'extra_loss_sums':extra_loss_sums})
                state['execution_options']=runtime
                atomic_checkpoint(output/'step.pt',state);del state
                save_json(output/'step_checkpoint.json',{'epoch':epoch+1,'update':update+1,'optimizer_steps':step_number,
                    'samples':samples,'path':str(output/'step.pt'),'partial_epoch':True,'saved_unix':time.time()})
            if runtime:
                record={'optimizer_step':step_number,'epoch':epoch+1,'update':update+1,'studies':group_samples,
                    'seconds':time.perf_counter()-update_start,'input_wait_seconds':data_seconds-update_data_start,
                    'loss':loss_sum-update_loss_start,
                    'max_prefetched_batch_bytes':getattr(iterator,'max_batch_bytes',0),'execution_options':runtime}
                with (output/'execution_updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                save_json(output/'execution_progress.json',record)
            if stop_requested:
                save_json(output/'progress.json',{'phase':'paused_after_update','epoch':epoch+1,'update':update+1,'optimizer_steps':step_number})
                if hasattr(iterator,'close'):iterator.close()
                return history
            if update % 10 == 0:
                print(f"epoch {epoch + 1}/{config['epochs']} update {update + 1}/{updates} loss={loss_sum / (update + 1):.4f}", flush=True)
                save_json(output/'progress.json',{'phase':'training','epoch':epoch+1,'epochs':config['epochs'],
                          'update':update+1,'updates':updates,'studies_seen_this_epoch':samples,
                          'elapsed_epoch_seconds':None if first_update else time.perf_counter()-epoch_started,
                          'elapsed_this_process_epoch_seconds':time.perf_counter()-epoch_started,
                          'resumed_at_update':first_update,'new_updates_this_process':update+1-first_update,'smoke':smoke})
        if hasattr(iterator,'close'):iterator.close()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - start
        if pending_training:
            training_seconds=pending_training['training_seconds'];data_seconds=pending_training['data_seconds']
            loss_sum=pending_training['loss_sum'];samples=pending_training['samples']
            exposure_positive=np.asarray(pending_training['positive']);exposure_negative=np.asarray(pending_training['negative'])
            rank_sum=pending_training['rank_sum'];aux_sum=pending_training['aux_sum']
            extra_loss_sums=pending_training.get('extra_loss_sums',{})
            peak_training=pending_training['peak_gib'];epoch_started_unix=pending_training['epoch_started_unix']
        else:
            peak_training=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None
            save_json(output/'progress.json',{'phase':'saving_training_checkpoint','epoch':epoch+1,'epochs':config['epochs'],'smoke':smoke})
            state=checkpoint_payload(model,config=config,epoch=epoch,provenance=provenance,optimizer_steps=step_number)
            state.update(ema=checkpoint_payload(ema.model,config=config,epoch=epoch,provenance=provenance,optimizer_steps=step_number)['model'],
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,step=step_number,
                rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),
                rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                completed_training={'training_seconds':training_seconds,'data_seconds':data_seconds,'loss_sum':loss_sum,
                    'samples':samples,'positive':exposure_positive.tolist(),'negative':exposure_negative.tolist(),
                    'rank_sum':rank_sum,'aux_sum':aux_sum,'extra_loss_sums':extra_loss_sums,'peak_gib':peak_training,'epoch_started_unix':epoch_started_unix})
            atomic_checkpoint(output/'pending.pt',state)
            del state
        save_json(output/'progress.json',{'phase':'validating','epoch':epoch+1,'epochs':config['epochs'],'smoke':smoke})
        if valid_loader is not None:
            report, prediction, _ = evaluate(ema.model, valid_loader, device,
                                            soft=fold != "full" and evaluation_labels_path is None)
        else:
            report={'macro_auc':None,'bce':None,'studies':0,'kind':'full_fixed_budget_no_validation'}
            prediction=pd.DataFrame(columns=[UID,*LABELS])
        if evaluation_labels_path and fold != 'full':
            report['kind']='frozen_weak_label_reference'
        if report["macro_auc"] is None and not smoke and fold != "full":
            save_json(output / "invalid_validation.json", report)
            raise ValueError("validation cannot define all 12 AUCs; inspect invalid_validation.json")
        score = report["macro_auc"] if report["macro_auc"] is not None else (-report['bce'] if report['bce'] is not None else -math.inf)
        record = {"epoch": epoch, "train_loss": loss_sum / updates, "train_studies": samples,
                  "training_seconds": training_seconds, "data_wait_seconds": data_seconds,
                  "validation": report, "selection_scope": provenance["selection"],
                  "peak_allocated_gib": max(peak_training,torch.cuda.max_memory_allocated(device)/2**30) if device.type == "cuda" else None,
                  'recovered_training_before_validation':pending_training is not None}
        record.update(rank_loss=rank_sum/updates,aux_loss=aux_sum/updates,
            independent_supervision_losses={name:value/updates for name,value in extra_loss_sums.items()},
            rank_gradient_scope='context_and_Study_aggregation_only; detached_visual_features',
            unique_training_studies=len(train_data),draws_per_study=draws,
            supervised_exposure={c:{'positive':int(exposure_positive[j]),'negative':int(exposure_negative[j])} for j,c in enumerate(LABELS)})
        if external_validation is not None:
            from .external_mcl import evaluate_source
            record['external_mcl_source_diagnostic']=evaluate_source(ema.model,external_validation,device)
        if gold_loader is not None:
            gold_report, gold_prediction, _ = evaluate(ema.model, gold_loader, device, soft=False)
            record["gold58_development_only"] = gold_report
            gold_prediction.to_csv(output / "gold_development_predictions.csv", index=False)
        elif not reserve_gold and len(gold_frame):
            gold_prediction=prediction.set_index(UID).loc[gold_frame[UID],LABELS]
            record['expert_fold_development_only']=auc_report(gold_frame[LABELS].to_numpy(float),gold_prediction.to_numpy(float),
                gold_frame[[c+'__weight' for c in LABELS]].to_numpy(float))
            gold_prediction.reset_index().to_csv(output/'expert_fold_predictions.csv',index=False)
        improved = score > best
        best = max(best, score)
        record['selection_metric'] = 'macro_auc' if report['macro_auc'] is not None else ('negative_bce_engineering_pilot_only' if valid_loader else 'fixed_budget_no_validation')
        save_json(output/'progress.json',{'phase':'saving_checkpoint','epoch':epoch+1,'epochs':config['epochs'],'smoke':smoke})
        if improved or fold == "full":
            payload = checkpoint_payload(ema.model, config=config, epoch=epoch, provenance=provenance,optimizer_steps=step_number)
            atomic_checkpoint(output / "best.pt", payload)
            prediction.to_csv(output / "validation_predictions.csv", index=False)
            if 'ranking_diagnostics' in report:
                report['ranking_diagnostics'].update(prediction_producer_epoch=int(epoch),
                    prediction_producer_optimizer_steps=int(step_number),
                    checkpoint_path=str((output / 'best.pt').resolve()),
                    saved_predictions_sha256=file_hash(output / 'validation_predictions.csv'))
            save_json(output / "validation_metrics.json", report)
            if 'ranking_diagnostics' in report:
                save_json(output / "validation_ranking.json", report['ranking_diagnostics'])
        history.append(record); save_json(output / "history.json", history)
        (output/'pending.pt').replace(output/'last.pt')
        pending_training=None
        partial_epoch=None
        (output/'step.pt').unlink(missing_ok=True)
        record['epoch_wall_seconds_including_validation_and_checkpoint']=time.perf_counter()-epoch_started
        record['wall_seconds_including_recovery_downtime']=time.time()-epoch_started_unix
        save_json(output/'history.json',history)
        save_json(output/'progress.json',{'phase':'epoch_complete','epoch':epoch+1,'epochs':config['epochs'],
                  'validation_auc':report['macro_auc'],'epoch_wall_seconds':record['epoch_wall_seconds_including_validation_and_checkpoint'],
                  'smoke':smoke})
        print(f"epoch {epoch + 1}: validation AUC={report['macro_auc']} BCE={report['bce']}, train {training_seconds:.1f}s", flush=True)
        if stopped_for_plateau(history,config):
            save_json(output/'progress.json',{'phase':'early_stopped','epoch':epoch+1,'epochs':config['epochs'],
                      'reason':'fixed-reference development metric plateau','smoke':smoke})
            break
    return history


def predict(checkpoint_path, manifest, output, *, device=None, workers=None, study_csv=None,input_overrides=None):
    if workers is None:workers=0 if os.name=='nt' else 2
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(int(state['train_config'].get('cpu_threads',4)))
    if state.get("labels") != LABELS:
        raise ValueError("checkpoint label order differs")
    config = ModelConfig(**state["model_config"])
    ids = pd.read_csv(study_csv, dtype={UID: str})[UID].tolist() if study_csv else None
    inference_config={**state['train_config'],**(input_overrides or {})}
    data = dataset_for(manifest,None,inference_config,ids=ids)
    if data.size != config.image_size:
        raise ValueError("cache/checkpoint input sizes differ")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = KneeModel(config, pretrained=False).to(device)
    load_checkpoint_model(model,state)
    from .four_gpu.precision import checkpoint_policy,apply_model_policy,inference_policy
    numerical_policy=checkpoint_policy(state);apply_model_policy(model,numerical_policy)
    with inference_policy(numerical_policy):
        report, frame, attention = evaluate(model, loader(data, 1, workers), device, soft=False, attention=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "submission.csv", index=False)
    save_json(output / "evidence.json", {"note": "attention is model evidence weighting, not verified lesion localization", "studies": attention})
    save_json(output / "inference.json", {"studies": report["studies"], "seconds": report["seconds"],
              "fallback_studies": report["fallback_studies"], "fallback_ids": report["fallback_ids"],
              "checkpoint_sha256": file_hash(checkpoint_path), "source_smoke": state["provenance"].get("smoke", False),"precision_policy":numerical_policy})
    return frame
