"""External expert-reference evaluation using complete source volumes.

No optimizer, target relabeling, expert-box crop, or hidden competition-score
claim. Historical checkpoints may be diagnosed but never promoted to training
initializers here. Source endpoints retain their original clinical definitions.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .common import LABELS,file_hash,save_json


def prepare(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    references=root/'reference_anatomy';kroot=references/'KneeMRI'
    meta=pd.read_csv(kroot/'metadata.csv');records=[]
    for r in meta.to_dict('records'):
        records.append({'source':'KneeMRI','id':r['volumeFilename'],'group':'kneemri_exam_'+str(r['examId']),
            'path':str(kroot/'extracted/volumetric_data'/r['volumeFilename']),
            'source_diagnosis':int(r['aclDiagnosis']),'source_grade_labels':[],
            'slot':1,'fluid_sensitive':1,'expected_source_roi':{k:r[k] for k in r if k.startswith('roi')},
            'roi_used_as_model_input':False})
    fast=references/'fastmri_plus';annotations=pd.read_csv(fast/'knee.csv')
    source=json.loads((root/'external_mcl/manifest.json').read_text())
    for row in source['records']:
        local=annotations[annotations.file.eq(row['file'])]
        records.append({'source':'fastMRI+','id':row['file'],'group':'fastmri_patient_'+row['source_patient'],
            'path':row['path'],'source_diagnosis':None,'source_grade_labels':sorted(local.label.unique().tolist()),
            'source_fold':row['source_fold'],'image_sha256':row['image_sha256'],
            'roi_used_as_model_input':False})
    for record in records:
        if not Path(record['path']).is_file():raise FileNotFoundError(record['path'])
    manifest={'records':records,'sources':{'KneeMRI_metadata':str(kroot/'metadata.csv'),
        'KneeMRI_metadata_sha256':file_hash(kroot/'metadata.csv'),'fastMRI_annotations':str(fast/'knee.csv'),
        'fastMRI_annotations_sha256':file_hash(fast/'knee.csv')},
        'source_selection':'All 917 local KneeMRI volumes and all 125 already downloaded MCL-selected fastMRI+ volumes.',
        'semantics':'Whole source-volume diagnostic, not complete multi-sequence competition Study evaluation.',
        'evaluation_exposure':'Some external images have informed preprocessing development; no untouched-test claim.',
        'negative_policy':'Absent fastMRI disease annotations are unknown, never disease-negative.',
        'unknown_geometry':'KneeMRI has no verified spacing/affine. Full field preserves source pixel aspect; geometry features are masked to zero.',
        'training_dispatch':False,'cohort_frozen_before_predictions':True}
    save_json(output/'manifest.json',manifest)
    return manifest


def source_volume(record,root):
    if record['source']=='KneeMRI':
        from .kneemri_clinical_reference import KneeMRIClinicalReference
        obj=KneeMRIClinicalReference(Path(root)/'reference_anatomy/KneeMRI',[record['id']])[0]
        # The reference adapter checks the source archive CRC and array shape.
        return obj['image'],record['slot'],1.,None
    import h5py
    path=Path(record['path'])
    if file_hash(path)!=record['image_sha256']:raise ValueError('External MRI changed')
    with h5py.File(path,'r') as f:
        volume=np.asarray(f['reconstruction_rss'],np.float32)
        acquisition=f.attrs.get('acquisition','')
        if isinstance(acquisition,bytes):acquisition=acquisition.decode()
        from .fastmri_reference import reference_geometry
        geom=reference_geometry(f['ismrmrd_header'][()] if 'ismrmrd_header' in f else None,volume.shape)
    slot=3 if 'FS' in str(acquisition) else 4 if 'PD' in str(acquisition) else 8
    return volume,slot,1. if slot==3 else -1.,geom['spacing_mm']


def full_volume_batch(volume,slot,fluid,spacing,size,device):
    """Triplets cover every native slice; labels and expert boxes are not read."""
    from .native_data import render_view
    if volume.ndim!=3 or not np.isfinite(volume).all():raise ValueError('Invalid volume')
    d,h,w=volume.shape;centres=list(range(1,d,3)) if d>1 else [0]
    if not centres or centres[-1]+1<d-1:centres.append(d-1)
    low,high=np.percentile(volume,[1,99]);high=max(float(high),float(low)+1e-8)
    images=[];covered=set()
    for z in centres:
        indices=np.clip([z-1,z,z+1],0,d-1);covered.update(map(int,indices))
        image,_,_=render_view(volume[indices],spacing or [1.,1.],size,window=(low,high),return_geometry=True)
        images.append(image)
    if len(covered)!=d:raise ValueError('Source slices dropped')
    n=len(images);geometry=np.zeros((n,4),np.float32)
    geometry[:,0]=np.asarray(centres)/max(d-1,1);geometry[:,3]=fluid
    batch={'images':torch.from_numpy(np.stack(images))[None],
        'slots':torch.full((1,n),slot,dtype=torch.long),'series':torch.zeros((1,n),dtype=torch.long),
        'original_series':torch.zeros((1,n),dtype=torch.long),'view_ids':torch.zeros((1,n),dtype=torch.long),
        'position':torch.tensor(centres,dtype=torch.float32)[None],
        'geometry':torch.from_numpy(geometry)[None]}
    return {k:v.to(device) for k,v in batch.items()},{'native_slices':d,'windows':n,
        'unique_slices_used':len(covered),'full_field':True,'physical_slice_positions_known':False,
        'in_plane_spacing_known':spacing is not None,'expert_roi_input':False}


def endpoint_metrics(records,predictions):
    by_id={(r['source'],r['id']):r for r in records}
    rows=[{**by_id[(r['source'],r['id'])],**r} for r in predictions]
    specifications=[
        ('ACL_complete_vs_normal','KneeMRI','ACL',2,0),
        ('ACL_any_injury_vs_normal','KneeMRI','ACL',(1,2),0),
        ('ACL_complete_vs_partial','KneeMRI','ACL',2,1),
        ('ACL_source_high_vs_low','fastMRI+','ACL','Ligament - ACL High Grade Sprain','Ligament - ACL Low Grade sprain'),
        ('MCL_source_high_vs_lowmoderate','fastMRI+','MCL','Ligament - MCL High Grade sprain','Ligament - MCL Low-Mod Grade Sprain')]
    results={}
    for name,source,label,positive,negative in specifications:
        grouped=defaultdict(list);ambiguous=set();not_in_endpoint=0
        for r in rows:
            if r['source']!=source:continue
            if source=='KneeMRI':
                pos=r['source_diagnosis'] in (positive if isinstance(positive,tuple) else (positive,));neg=r['source_diagnosis']==negative
            else:pos=positive in r['source_grade_labels'];neg=negative in r['source_grade_labels']
            if pos and neg:ambiguous.add(r['group']);continue
            if not(pos or neg):not_in_endpoint+=1;continue
            grouped[r['group']].append((int(pos),float(r['probabilities'][LABELS.index(label)])))
        truth=[];scores=[]
        for group,items in grouped.items():
            labels={x[0] for x in items}
            if group in ambiguous or len(labels)!=1:ambiguous.add(group);continue
            truth.append(items[0][0]);scores.append(float(np.mean([x[1] for x in items])))
        auc=float(roc_auc_score(truth,scores)) if len(set(truth))==2 else None
        results[name]={'auc':auc,'positive_groups':sum(truth),'negative_groups':len(truth)-sum(truth),
            'ambiguous_groups_excluded':len(ambiguous),'volumes_outside_endpoint':not_in_endpoint,
            'score_readout':label,'source_target':name,'competition_macro_auc':None,
            'scope':'Historical checkpoint on external source endpoint; source selection and prior development exposure remain.'}
    return results


def evaluate(root,checkpoint,output,limit=None):
    output=Path(output);mp=output/'manifest.json'
    manifest=json.loads(mp.read_text()) if mp.exists() else prepare(root,output)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from .model import KneeModel,ModelConfig
    config=ModelConfig(**state['model_config']);config.encode_chunk=4
    if config.context_type!='none':raise ValueError('This adapter cannot invent physical Z positions for a contextual model')
    if any((config.spatial_3d,config.anatomy_pool,config.anatomy_auxiliary,config.local_evidence,config.oa_binding!='none')):
        raise ValueError('External geometry/anatomy input for this checkpoint is not available')
    model=KneeModel(config,pretrained=False);model.load_state_dict(state['model'],strict=True)
    provenance=state.get('provenance',{});del state;gc.collect();model.to(device).eval()
    binding={'checkpoint_sha256':file_hash(checkpoint),'cohort_sha256':file_hash(mp),'adapter_sha256':file_hash(__file__),
        'model_config':model.export_config(),'checkpoint_label_policy':provenance.get('label_source_policy','legacy_unverified'),
        'use':'Historical diagnostic only; cannot seed own-label training.',
        'preprocessing':'Source full field, [1,99] percentile, all native slices in triplets; no expert ROI; not claimed identical to target multi-sequence input.'}
    bp=output/'prediction_binding.json'
    if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('Predictions belong to different input/model; choose a new output directory')
    save_json(bp,binding);path=output/'predictions.jsonl'
    predictions=[json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
    done={(r['source'],r['id']) for r in predictions};errors=[];start=time.perf_counter()
    selected=manifest['records'][:limit] if limit else manifest['records']
    for record in selected:
        if (record['source'],record['id']) in done:continue
        try:
            volume,slot,fluid,spacing=source_volume(record,root)
            batch,receipt=full_volume_batch(volume,slot,fluid,spacing,config.image_size,device)
            with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=model(batch);probability=logits.float().sigmoid()[0].cpu().tolist()
            if len(probability)!=12 or not np.isfinite(probability).all():raise ValueError('Invalid output')
            row={'source':record['source'],'id':record['id'],'probabilities':probability,'input':receipt}
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            predictions.append(row);del batch,volume
        except Exception as e:
            errors.append({'source':record['source'],'id':record['id'],'error':str(e)})
            if len(errors)>=3:break
        if len(predictions)%25==0:
            progress={'completed':len(predictions),'expected':len(selected),'errors':len(errors),'seconds':round(time.perf_counter()-start,2)}
            save_json(output/'status.json',progress);print(json.dumps(progress),flush=True)
    summary={'completed':len(predictions),'expected_full_cohort':len(manifest['records']),
        'complete':len(predictions)==len(manifest['records']),'errors':errors,
        'endpoints':endpoint_metrics(manifest['records'],predictions),'optimizer_steps':0,
        'current_process_seconds':round(time.perf_counter()-start,2),'binding':binding}
    save_json(output/'summary.json',summary);save_json(output/'status.json',{k:v for k,v in summary.items() if k not in ('endpoints','binding')})
    print(json.dumps({k:v for k,v in summary.items() if k!='binding'}),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate']);p.add_argument('--root',default='F:/rsnaknee')
    p.add_argument('--checkpoint');p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='prepare':print({'volumes':len(prepare(a.root,a.output)['records'])})
    else:evaluate(a.root,a.checkpoint,a.output,a.limit)
