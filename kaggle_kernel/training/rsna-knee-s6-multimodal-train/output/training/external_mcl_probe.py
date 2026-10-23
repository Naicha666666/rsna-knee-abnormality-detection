"""Actual expert MRI -> source losses -> shared CoAtNet gradient; no optimizer."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch

from .common import LABELS,UID,save_json,file_hash
from .model import KneeModel,ModelConfig
from .external_mcl import ExternalMCLSource,external_objective
from .mcl_joint import audit_frame


def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(root='F:/rsnaknee',device='cuda'):
    root=Path(root);out=root/'analysis/external_mcl_supervision';out.mkdir(parents=True,exist_ok=True)
    manifest=root/'external_mcl/manifest.json';source=ExternalMCLSource(manifest,image_size=384)
    model=KneeModel(ModelConfig(pretrained_file=str(root/'pretrained/model.safetensors'),
        mcl_joint=True,external_mcl=True,context_type='none'),pretrained=True).to(device).eval()
    before=digest(model);rows=[]
    for grade in [0,1]:
        record=next(r for r in source.rows if r['source_grade']==grade);z=record['marked_slices'][len(record['marked_slices'])//2]
        b=source.sample(record,z);batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            loss,parts=external_objective(model,batch)
        loss.backward()
        names={'backbone':sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None),
            'MCL_spatial_query':float(model.spatial.query.grad[1].abs().sum()),
            'source_grade':float(model.external_mcl_grade.weight.grad.abs().sum()),
            'shared_MCL_classifier':float(model.classifier.grad[1].abs().sum()),
            'MCL_state_head':float(model.mcl_joint_head.negative_states.weight.grad.abs().sum())}
        if not all(np.isfinite(v) and v>0 for v in names.values()):raise RuntimeError('external evidence is not consumed by shared modules')
        rows.append({'file':record['file'],'source_grade':grade,'slice':z,'views':b['view_ids'],
            'retention':b['box_area_retention'],'loss':float(loss.detach()),'gradient_l1':names,
            'loss_parts':{k:float(parts[k]) for k in ('source_box_attention','source_grade','source_presence')},
            'acute_known':False,'competition_main_target':None})
        print(json.dumps(rows[-1]),flush=True)
    after=digest(model);assert before==after
    current=json.loads((root/'supervision_prepare_status.json').read_text());labels=Path(current['labels'])
    f=pd.read_csv(labels);known=f['MCL__weight']>0
    stats={'positive':int(((f.MCL==1)&known).sum()),'negative':int(((f.MCL==0)&known).sum()),'unknown':int((~known).sum()),
        'partial':audit_frame(f),'conditions':{}}
    for c in ('MCL__presence','MCL__factor_high_grade','MCL__factor_acute'):
        w=f[c+'_weight']>0;stats['conditions'][c]={'positive':int((f[c].eq(1)&w).sum()),'negative':int((f[c].eq(0)&w).sum()),'unknown':int((~w).sum())}
    result={'source_manifest':str(manifest),'source_manifest_sha256':file_hash(manifest),
        'source_volumes':125,'source_patients':102,'source_high_grade_volumes':4,'source_high_grade_patients':3,
        'source_low_mod_grade_volumes':121,'source_boxes':296,'current_labels':str(labels),'current_labels_sha256':file_hash(labels),
        'target_supervision':stats,'actual_source_loss_consumption':rows,'parameters_and_buffers_unchanged':before==after,
        'optimizer_updates':0,'source_training_started':False,'new_target_main_labels':0,
        'mechanism':'box envelope trains actual spatial attention; source grade updates shared representation; known presence constrains deployed MCL state head; unknown acuity is marginalized',
        'scope':'real pretrained384 CoAtNet backward consumer only, not fitted prediction or target-positive recovery',
        'train_recipe':'mcl_external','control_recipe':'mcl_external_control'}
    save_json(out/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--device',default='cuda');a=p.parse_args();run(a.root,a.device)
