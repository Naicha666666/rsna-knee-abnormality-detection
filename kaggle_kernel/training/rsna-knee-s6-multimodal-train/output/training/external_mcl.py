"""External expert boxes -> shared MCL visual features and partial presence.

Source low/moderate-vs-high grades have their own training-only head: neither
their threshold nor acuity is equated to the competition criterion. Boxes
locate positive evidence; unmarked pixels, slices and other diseases are unknown.
This module never updates target labels or runs an optimizer by itself.
"""
from collections import OrderedDict
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .common import file_hash,save_json,UID
from .fastmri_reference import mcl_annotations,load_reference_volume
from .fastmri_mcl_bilateral_diagnostic import anonymous_patient_key
from .coronal_views import coronal_bilateral_views
from .native_data import render_view
from .mcl_joint import MCL,evidence,conditional_loss

VERSION='external-mcl-box-presence-source-grade-v1'


def prepare(root='F:/rsnaknee'):
    root=Path(root);source=root/'reference_anatomy/fastmri_plus'
    annotations=mcl_annotations(source/'knee.csv');records=[]
    for name,rows in annotations.groupby('file',sort=True):
        path=source/'images'/f'{name}.h5';receipt=path.with_suffix('.json')
        if not path.exists() or not receipt.exists():raise ValueError('complete expert reference images required')
        r=json.loads(receipt.read_text());group=anonymous_patient_key(path)
        if not group:raise ValueError('external source patient identity missing')
        labels=rows.label.unique()
        if len(labels)!=1:raise ValueError('mixed source grades need local instance targets')
        records.append({'file':name,'path':str(path),'source_patient':group,
            'source_grade':int('High Grade' in labels[0]),'source_label':labels[0],
            'image_sha256':r['image_sha256'],'receipt_sha256':file_hash(receipt),
            'marked_slices':sorted(map(int,rows['slice'].unique())),'boxes':len(rows),
            'competition_main_target':None,'acute_target':None})
    grouped={}
    for row in records:
        old=grouped.setdefault(row['source_patient'],row['source_grade'])
        if old!=row['source_grade']:raise ValueError('one source patient has conflicting grades')
    rng=np.random.default_rng(20261001);folds={}
    for grade in [0,1]:
        ids=sorted(g for g,v in grouped.items() if v==grade)
        if len(ids)<3:raise ValueError('need three source patient groups for each source grade')
        for i,g in enumerate(rng.permutation(ids)):folds[str(g)]=i%3
    for row in records:row['source_fold']=folds[row['source_patient']]
    # One patient is a sampling unit; repeated acquisitions/boxes are not new patients.
    summary={}
    for fold in range(3):
        summary[str(fold)]={role:{'volumes':sum((r['source_fold']==fold)==(role=='evaluation') for r in records),
            'patients':len({r['source_patient'] for r in records if (r['source_fold']==fold)==(role=='evaluation')}),
            'high_grade_patients':len({r['source_patient'] for r in records if r['source_grade']==1 and (r['source_fold']==fold)==(role=='evaluation')})}
            for role in ('training','evaluation')}
    manifest={'version':VERSION,'annotations':str(source/'knee.csv'),'annotation_sha256':file_hash(source/'knee.csv'),
        'records':records,'source_patient_folds':folds,'fold_summary':summary,
        'source_grade_semantics':'original fastMRI+ low/moderate versus high; not a verified competition-grade threshold',
        'main_labels_added':0,'acute_labels_added':0,'outside_box_is_negative':False,
        'source_split_limit':'only three high-grade patients: one evaluation patient per split, no robust high-grade AUC claim',
        'adapter_sha256':file_hash(__file__),'training_dispatch':False}
    path=root/'external_mcl/manifest.json';save_json(path,manifest)
    return path,manifest


def box_mask(boxes,shape):
    """Fractional pixel support, a localization envelope rather than tissue truth."""
    h,w=shape;result=np.zeros((h,w),np.float32)
    yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]
    for x0,y0,x1,y1 in boxes:
        area=np.clip(np.minimum(xx+1,x1)-np.maximum(xx,x0),0,1)*np.clip(np.minimum(yy+1,y1)-np.maximum(yy,y0),0,1)
        result=np.maximum(result,area.astype(np.float32))
    return result


class ExternalMCLSource:
    def __init__(self,manifest,*,source_fold=0,image_size=384,role='training'):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        if self.manifest['version']!=VERSION or self.manifest['adapter_sha256']!=file_hash(__file__):
            raise ValueError('external MCL adapter changed; prepare source contract again')
        if role not in ('training','evaluation') or source_fold not in (0,1,2):raise ValueError('invalid source split')
        if file_hash(self.manifest['annotations'])!=self.manifest['annotation_sha256']:raise ValueError('expert annotations changed')
        self.rows=[r for r in self.manifest['records'] if (r['source_fold']==source_fold)==(role=='evaluation')]
        self.annotations=mcl_annotations(self.manifest['annotations']);self.image_size=image_size
        self.groups={g:[r for r in self.rows if r['source_patient']==g] for g in sorted({r['source_patient'] for r in self.rows})}
        self.patients=list(self.groups);self.cache=OrderedDict();self.role=role;self.source_fold=source_fold
        if not self.rows:raise ValueError('empty external source split')

    def _volume(self,record):
        path=Path(record['path']);stat=path.stat();key=(record['file'],stat.st_size,stat.st_mtime_ns)
        if key not in self.cache:
            rows=self.annotations[self.annotations.file.eq(record['file'])]
            value=load_reference_volume(path,rows)
            if value['image_sha256']!=record['image_sha256'] or file_hash(path.with_suffix('.json'))!=record['receipt_sha256']:
                raise ValueError('external source image or receipt changed')
            self.cache[key]=value
            while len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def draw(self,step,seed=42):
        # A permutation per cycle gives every patient equal exposure; neither
        # high-grade patients nor repeated slices are replicated as new cases.
        cycle,offset=divmod(int(step),len(self.patients));rng=np.random.default_rng(seed+cycle*1000003)
        group=self.patients[int(rng.permutation(len(self.patients))[offset])]
        records=self.groups[group];record=records[int(rng.integers(len(records)))];z=int(rng.choice(record['marked_slices']))
        return self.sample(record,z)

    def sample(self,record,z):
        if record not in self.rows:raise ValueError('external sample outside current source patient split')
        data=self._volume(record);volume=data['native_volume'];spacing=data['geometry']['spacing_mm']
        if spacing is None:raise ValueError('known physical in-plane spacing required')
        boxes=data['native_boxes_xyxy'][data['box_slice_indices']==z]
        if not len(boxes):raise ValueError('unmarked slices are unknown, not negatives')
        original_mask=box_mask(boxes,volume.shape[-2:]);mass=float(original_mask.sum())
        indices=np.clip([z-1,z,z+1],0,len(volume)-1)
        scale=max(float(np.percentile(volume,99.9)),1e-12);triplet=np.clip(volume[indices]/scale,0,1)
        acquisition=str(data['source_attrs'].get('acquisition',''));slot=3 if 'FS' in acquisition else 4 if 'PD' in acquisition else 8
        views=coronal_bilateral_views(volume.shape[-2:],spacing,self.image_size)
        images=[];masks=[];usable=[];retention=[];positions=[]
        for view,crop,center in views:
            image,_,receipt=render_view(triplet,spacing,self.image_size,crop_mm=crop,center=center,window=(0.,1.),return_geometry=True)
            mask,_,_=render_view(np.stack([original_mask]*3),spacing,self.image_size,crop_mm=crop,center=center,window=(0.,1.),return_geometry=True)
            x0,y0,x1,y1=receipt['native_crop_xyxy'];fraction=float(original_mask[y0:y1,x0:x1].sum()/mass)
            images.append(image);masks.append(mask[0]);usable.append(fraction>=1-1e-6);retention.append(fraction);positions.append(view)
        return {'images':torch.from_numpy(np.stack(images)), 'box_support':torch.from_numpy(np.stack(masks)),
            'slots':torch.full((len(images),),slot,dtype=torch.long),'observed_view':torch.tensor(usable),
            'source_grade':record['source_grade'],'file':record['file'],'source_patient':record['source_patient'],
            'slice':z,'view_ids':positions,'box_area_retention':retention,'competition_main_target':None,'acute_target':None}

    def binding(self):
        return {'manifest':str(self.path),'manifest_sha256':file_hash(self.path),'source_fold':self.source_fold,
            'source_training_patients':sorted(self.groups),'source_training_volumes':len(self.rows),
            'source_labels_are_competition_labels':False,'role':self.role}


def external_objective(model,batch):
    """Losses use the actual classifier's backbone, spatial gate and MCL readout.

    Expert boxes enter only the training loss. The source grade classifier is
    separate because source grades lack a verified acute/threshold equivalence.
    """
    if not model.config.external_mcl:raise ValueError('external MCL source head disabled')
    x=batch['images'].float()
    if model.config.input_normalization=='imagenet':x=(x-model.mean)/model.std
    fmap=model.backbone.forward_features(x.contiguous(memory_format=torch.channels_last))
    support=F.adaptive_avg_pool2d(batch['box_support'][:,None].float(),fmap.shape[-2:]).flatten(1)
    valid=batch['observed_view']&support.sum(1).gt(0)
    if not valid.any():raise ValueError('no fully observed expert box in source views')
    attention=model.spatial.attention_weights(fmap,batch['slots'])[:,:,MCL]
    mass=(attention*support).sum(1).clamp_min(1e-8)
    localization=-mass[valid].log().mean()
    # Only source observations supply local ROI pooling. No ROI is requested
    # at inference and no mask pixels are labelled as ligament or background.
    tokens=fmap.flatten(2).transpose(1,2)
    local=(tokens*support[:,:,None]).sum(1)/support.sum(1).clamp_min(1e-8)[:,None]
    old_norm=getattr(getattr(model.backbone,'head',None),'norm',None)
    if old_norm is not None and getattr(old_norm,'weight',None) is not None:
        local=F.layer_norm(local,(model.dim,),old_norm.weight,old_norm.bias,old_norm.eps)
    local=model.norm(local)
    logits=model.external_mcl_grade(local)[valid]
    grade=F.cross_entropy(logits,torch.full((len(logits),),int(batch['source_grade']),device=x.device,dtype=torch.long))
    positive=local@model.classifier[MCL]+model.bias[MCL]
    joint=model.mcl_joint_head(local,positive)
    n=len(local);partial=x.new_zeros(n,3);weights=x.new_zeros(n,3)
    partial[:,0]=1;weights[:,0]=valid.float()  # existence known; grade/acuity not borrowed
    ev=evidence(x.new_zeros(n),x.new_zeros(n),partial,weights)
    presence=conditional_loss(joint['state_logits'],ev,ev['mass'].sum())
    loss=localization+grade+presence
    return loss,{'source_box_attention':localization.detach(),'source_grade':grade.detach(),
                 'source_presence':presence.detach(),'source_grade_logits':logits.detach(),
                 'observed_views':int(valid.sum()),'main_labels_added':0,'acute_labels_added':0}


def auxiliary_step(model,source,step,seed,device,coefficient):
    """One source patient per target effective update, independent of accumulation."""
    batch=source.draw(step,seed)
    batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    loss,detail=external_objective(model,batch)
    return float(coefficient)*loss,detail


@torch.no_grad()
def evaluate_source(model,source,device):
    """Grouped, marked-region source diagnostics; never selects a target checkpoint."""
    if source.role!='evaluation':raise ValueError('source evaluation must use held-out patient groups')
    was_training=model.training;model.eval();groups={};total=0.;count=0
    try:
        for record in source.rows:
            predictions=[]
            for z in record['marked_slices']:
                batch=source.sample(record,z)
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                loss,parts=external_objective(model,batch)
                predictions.append(float(parts['source_grade_logits'].softmax(-1)[:,1].mean()))
                total+=float(loss);count+=1
            group=groups.setdefault(record['source_patient'],{'target':record['source_grade'],'scores':[]})
            group['scores'].append(float(np.mean(predictions)))
    finally:model.train(was_training)
    truth=np.asarray([g['target'] for g in groups.values()]);scores=np.asarray([np.mean(g['scores']) for g in groups.values()])
    from sklearn.metrics import roc_auc_score
    return {'source_fold':source.source_fold,'patient_groups':len(groups),'volumes':len(source.rows),
        'high_grade_patients':int(truth.sum()),'annotated_centers':count,'auxiliary_loss':total/max(count,1),
        'source_grade_patient_auc':float(roc_auc_score(truth,scores)) if len(np.unique(truth))==2 else None,
        'checkpoint_selection':False,'competition_auc':None,
        'scope':'source-grade discrimination on expert-localized held-out regions; only one high-grade patient per fold; not reliable clinical or target AUC'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='F:/rsnaknee');a=p.parse_args()
    path,data=prepare(a.root);print(json.dumps({'manifest':str(path),'volumes':len(data['records']),
        'patients':len(data['source_patient_folds']),'folds':data['fold_summary'],'training_dispatch':False}))
