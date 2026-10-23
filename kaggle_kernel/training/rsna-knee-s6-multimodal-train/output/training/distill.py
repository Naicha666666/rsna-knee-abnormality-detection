"""Cross-fitted teacher export with patient and model-selection isolation.

Teacher probabilities are an independent loss channel, never replacements for
official or explicit criterion targets. This module does not train teachers.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,file_hash,save_json


def prepare_partitions(labels_path,outer_fold,output,root='F:/rsnaknee'):
    """Three disjoint roles: teacher fit, teacher selection, student targets."""
    from .labels import group_folds
    frame=pd.read_csv(labels_path,dtype={UID:str,'group_id':str});out=Path(output);out.mkdir(parents=True,exist_ok=True)
    contract_path=Path(labels_path).with_name('contract.json')
    contract=json.loads(contract_path.read_text(encoding='utf-8')) if contract_path.exists() else {}
    dev_ids=contract.get('supervision_development_ids',[])
    development=frame.get('development_only',frame.fold.eq(5)).astype(bool)
    from .unknown_review_isolation import reserved_mask
    image_review_reserved=reserved_mask(frame)
    if (development & image_review_reserved).any():raise ValueError('image-review reserve overlaps development')
    if dev_ids and not set(dev_ids).issubset(set(frame.loc[development,UID])):
        raise ValueError('supervision development patients are not declared training-only')
    allowed=frame.fold.ge(0)&frame.fold.ne(int(outer_fold))&~development&~image_review_reserved;part=frame[allowed]
    inner=group_folds(np.nan_to_num(part[LABELS].to_numpy(float)),part[[c+'__weight' for c in LABELS]].to_numpy(float),part.group_id.to_numpy(),3,42)
    jobs=[]
    for i in range(3):
        target_ids=part.loc[inner==i,UID].tolist();selection_ids=part.loc[inner==(i+1)%3,UID].tolist()
        train_ids=part.loc[inner==(i+2)%3,UID].tolist()+frame.loc[development,UID].tolist();teacher=frame.copy();teacher['fold']=-2
        teacher.loc[teacher[UID].isin(train_ids),'fold']=1;teacher.loc[teacher[UID].isin(selection_ids),'fold']=0
        path=out/f'inner_{i}';path.mkdir(exist_ok=True);teacher.to_csv(path/'labels.csv',index=False)
        if contract.get('label_source_policy'):
            save_json(path/'contract.json',{**contract,'parent_label_release_sha256':file_hash(labels_path),
                'labels_sha256':file_hash(path/'labels.csv'),'partition_only_change':True})
        ref=teacher.copy()
        for c in LABELS:ref[c+'__weight']=(ref[c+'__weight']>0).astype(float)
        ref.to_csv(path/'evaluation_reference.csv',index=False)
        from .recipes import configuration
        config=configuration('clinical',root,stage='finetune')
        config['supervision_development_ids']=dev_ids
        config['supervision_source_contract_sha256']=file_hash(contract_path) if contract_path.exists() else None
        config['labels_dir']=str(path.resolve())
        save_json(path/'config.json',config)
        groups=[set(frame.loc[frame[UID].isin(ids),'group_id']) for ids in (train_ids,selection_ids,target_ids)]
        if any(groups[a]&groups[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('inner patient overlap')
        jobs.append({'checkpoint':str((path/'run/best.pt').resolve()),'labels':str((path/'labels.csv').resolve()),
            'target_ids':target_ids,'quality_source_ids':selection_ids,'fit_ids':train_ids,
            'config':str((path/'config.json').resolve()),
            'train_arguments':['-m','training.run','train','--cache',str(Path(root)/'cache_native'),'--labels',str(path/'labels.csv'),
                '--eval-labels',str(path/'evaluation_reference.csv'),'--config',str(path/'config.json'),'--output',str(path/'run'),'--fold','0']})
    save_json(out/'teacher_jobs.json',jobs)
    save_json(out/'split_evidence.json',{'outer_fold':int(outer_fold),'outer_excluded_ids':frame.loc[frame.fold.eq(int(outer_fold)),UID].tolist(),
              'source_labels_sha256':file_hash(labels_path),'roles_disjoint_by_patient':True,'teacher_training_started':False})
    return jobs


def validate_teacher(state,labels,outer_fold,predicted_ids):
    p=state['provenance'];index=labels.set_index(UID)
    if p.get('smoke',True) or state.get('optimizer_steps',0)<=0:
        raise ValueError('teacher is untrained or only an engineering diagnostic')
    if p.get('parent_lineage_incomplete') or p.get('supervision_lineage_incomplete'):
        raise ValueError('teacher has incomplete parent or supervision development lineage')
    required=('train_ids','selection_ids','supervision_development_ids','parent_exposure_ids')
    if any(k not in p for k in required):raise ValueError('teacher lacks complete training/selection/parent/evidence lineage')
    consumed=set().union(*(set(p[k]) for k in required))
    if consumed-set(index.index):raise ValueError('teacher exposure contains unknown patient identities')
    declared=p.get('groups')
    if not declared:raise ValueError('teacher patient identity lineage is missing')
    for uid in consumed|set(predicted_ids):
        if uid not in declared or str(declared[uid])!=str(index.loc[uid,'group_id']):
            raise ValueError('patient grouping changed since teacher fitting')
    groups=set(index.loc[list(consumed),'group_id']) if consumed else set()
    reserved=labels.supervision_role.eq('reserved_expert') if 'supervision_role' in labels else labels.is_gold.astype(bool)
    from .unknown_review_isolation import reserved_mask
    outside=labels[labels.fold.eq(int(outer_fold))|reserved|reserved_mask(labels)]
    if groups&set(outside.group_id):raise ValueError('teacher has seen outer-validation or reserved expert patients')
    target=index.loc[list(predicted_ids)]
    if groups&set(target.group_id):raise ValueError('teacher targets are not out-of-patient predictions')
    if target.fold.lt(0).any() or target.fold.eq(int(outer_fold)).any() or reserved_mask(target).any():raise ValueError('teacher targets must be in the outer training partition')
    if p.get('supervision_semantics')!='competition_criterion':raise ValueError('teacher does not predict the intended criterion')
    return {'consumed_ids':sorted(consumed),'predicted_ids':list(predicted_ids),'outer_fold':int(outer_fold)}


def teacher_quality_gate(truth,prediction,weights,groups,minimum=5):
    """Patient resampling, conditional on the selected teacher. Not an external test."""
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(42);groups=np.asarray(groups);unique=np.unique(groups);quality=np.full(12,np.nan);details={}
    for j,c in enumerate(LABELS):
        known=weights[:,j]>0;y=truth[known,j];p=prediction[known,j]
        pos=int((y==1).sum());neg=int((y==0).sum())
        if min(pos,neg)<minimum:
            details[c]={'accepted':False,'positive':pos,'negative':neg,'reason':'too few known inner-selection cases'};continue
        estimates=[]
        for _ in range(100):
            selected=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
            ix=selected[weights[selected,j]>0];yy=truth[ix,j]
            if len(np.unique(yy))==2:estimates.append(roc_auc_score(yy,prediction[ix,j]))
        lower=float(np.quantile(estimates,.05)) if len(estimates)>=80 else 0.
        score=float(roc_auc_score(y,p));accepted=lower>.5
        if accepted:quality[j]=score
        details[c]={'accepted':accepted,'positive':pos,'negative':neg,'auc':score,'conditional_lower_90pct':lower,
                    'scope':'inner selection data; repeated model selection bias is not removed'}
    return quality,details


def validate_quality_sources(state,labels,outer_fold,predicted_ids,quality_ids):
    """Selection is held out from all fitting and supervision-development exposure.

    An ancestor's training or model selection still exposes its descendants to
    those patients.  Checking only the current teacher's fit set would allow an
    inherited teacher to qualify on patients its parent already used.
    """
    p=state['provenance'];index=labels.set_index(UID)
    ids=list(map(str,quality_ids));targets=list(map(str,predicted_ids))
    if not ids:raise ValueError('inner validation identity evidence required')
    if len(ids)!=len(set(ids)):raise ValueError('duplicate teacher quality identities')
    if set(ids)-set(index.index):raise ValueError('teacher quality contains unknown patient identities')
    if not set(ids).issubset(p.get('selection_ids',[])):
        raise ValueError('quality cases missing from teacher selection lineage')
    exposure_fields=('train_ids','parent_exposure_ids','supervision_development_ids')
    if any(k not in p for k in exposure_fields):raise ValueError('teacher quality lacks complete fitting/evidence lineage')
    exposed=set().union(*(set(p[k]) for k in exposure_fields))
    if exposed-set(index.index):raise ValueError('teacher exposure contains unknown patient identities')
    exposure_groups=set(index.loc[list(exposed),'group_id']) if exposed else set()
    reserved=labels.supervision_role.eq('reserved_expert') if 'supervision_role' in labels else labels.is_gold.astype(bool)
    development=labels.development_only.astype(bool) if 'development_only' in labels else labels.fold.eq(5)
    from .unknown_review_isolation import reserved_mask
    protected=labels[labels.fold.eq(int(outer_fold))|reserved|development|labels[UID].isin(targets)|reserved_mask(labels)]
    quality_groups=set(index.loc[ids,'group_id'])
    if quality_groups&(exposure_groups|set(protected.group_id)):
        raise ValueError('teacher quality assessment overlaps fitting, parent, development, targets or outer-validation patients')
    return {'quality_source_ids':ids,'quality_patients':len(quality_groups),
            'excluded_exposure_fields':list(exposure_fields),'patient_isolated':True}


def positive_teacher_evidence_conflicts(row,label):
    """Only known absence/false necessary conditions can block a positive teacher.

    A present finding does not imply that it passes the clinical criterion; one
    true factor does not imply the remaining factors. Unknowns therefore never
    become negatives, and a teacher negative is not blocked by mere presence.
    """
    from .clinical_factors import REQUIRED
    reasons=[]
    def known_zero(column,weight_column):
        value=row.get(column,np.nan);weight=row.get(weight_column,0.)
        return bool(pd.notna(value) and pd.notna(weight) and np.isfinite(float(weight))
                    and float(weight)>=.5 and float(value)==0.)
    if known_zero(label+'__presence',label+'__presence_weight'):
        reasons.append('explicit_finding_absence')
    for factor in REQUIRED[label]:
        column=label+'__factor_'+factor
        if known_zero(column,column+'_weight'):reasons.append('required_predicate_false:'+factor)
    return reasons


def priority_weights(row,prob,quality):
    """Mask confirmed evidence; unknown/conflicting criteria are never relabelled."""
    result=np.zeros(12,np.float32)
    for j,c in enumerate(LABELS):
        if row.get(c+'__weight',0)>0:continue
        state=str(row.get(c+'__criterion_state','unknown'))
        if state in ('conflict','official'):continue
        # Per-label teacher quality must be measured on inner held-out patients.
        q=float(quality[j]);p=float(prob[j])
        if not np.isfinite(q) or not .5<q<=1:continue
        if not np.isfinite(p) or not 0<=p<=1:continue
        if p>.5 and positive_teacher_evidence_conflicts(row,c):continue
        if abs(p-.5)<.25:continue
        result[j]=min(.25,(q-.5)*.5)*abs(2*p-1)
    return result


def build(labels_path,teachers,outer_fold,output=None,*,manifest,workers=0,root='F:/rsnaknee'):
    """teachers JSON: [{checkpoint, target_ids, quality_source_ids}].
    Both quality and target predictions are recomputed from actual weights.
    The default destination is the matching outer-fold student recipe input.
    """
    from .engine import predict
    from .label_source_policy import validate_label_source
    source_contract=validate_label_source(labels_path)
    labels=pd.read_csv(labels_path,dtype={UID:str,'group_id':str});index=labels.set_index(UID)
    from .recipes import distilled_labels_dir
    default_output=distilled_labels_dir(root,outer_fold)
    out=Path(output) if output is not None else default_output
    lineage_path=out/'lineage.json'
    if lineage_path.exists() and json.loads(lineage_path.read_text(encoding='utf-8'))['outer_fold']!=int(outer_fold):
        raise ValueError('output already contains targets for another outer fold')
    out.mkdir(parents=True,exist_ok=True);result=labels.copy();receipts=[];seen=set()
    for c in LABELS:result[c+'__teacher']=np.nan;result[c+'__teacher_weight']=0.
    for number,entry in enumerate(teachers):
        ckpt=Path(entry['checkpoint']);sha=file_hash(ckpt)
        state=torch.load(ckpt,map_location='cpu',weights_only=True);ids=list(map(str,entry['target_ids']))
        validate_label_source(labels_path,checkpoint=state)
        if len(ids)!=len(set(ids)) or seen&set(ids):raise ValueError('duplicate teacher target identities')
        receipt=validate_teacher(state,labels,outer_fold,ids)
        qids=list(map(str,entry.get('quality_source_ids',[])))
        quality_isolation=validate_quality_sources(state,labels,outer_fold,ids,qids)
        quality_csv=out/f'teacher_{number}_quality_ids.csv';pd.DataFrame({UID:qids}).to_csv(quality_csv,index=False)
        qp=predict(ckpt,manifest,out/f'teacher_{number}_quality',workers=workers,study_csv=quality_csv).set_index(UID).loc[qids]
        from .metrics import auc_report
        ref=index.loc[qids]
        report=auc_report(ref[LABELS].to_numpy(float),qp[LABELS].to_numpy(float),ref[[c+'__weight' for c in LABELS]].to_numpy(float))
        quality,quality_details=teacher_quality_gate(ref[LABELS].to_numpy(float),qp[LABELS].to_numpy(float),
            ref[[c+'__weight' for c in LABELS]].to_numpy(float),ref.group_id.to_numpy())
        study_csv=out/f'teacher_{number}_ids.csv';pd.DataFrame({UID:ids}).to_csv(study_csv,index=False)
        pred=predict(ckpt,manifest,out/f'teacher_{number}',workers=workers,study_csv=study_csv).set_index(UID)
        for uid in ids:
            p=pred.loc[uid,LABELS].to_numpy(float);row=index.loc[uid];w=priority_weights(row,p,quality)
            mask=result[UID].eq(uid)
            for j,c in enumerate(LABELS):result.loc[mask,c+'__teacher']=p[j];result.loc[mask,c+'__teacher_weight']=w[j]
        seen.update(ids);receipts.append({**receipt,'checkpoint':str(ckpt),'checkpoint_sha256':sha,'quality_source_ids':qids,
            'quality_isolation':quality_isolation,'inner_quality':report,'quality_acceptance':quality_details})
    taught=int((result[[c+'__teacher_weight' for c in LABELS]].to_numpy()>0).sum())
    if taught==0:
        save_json(out/'qualification_failed.json',{'outer_fold':int(outer_fold),'teachers':receipts,'reason':'no target passed evidence priority and teacher quality requirements'})
        raise ValueError('no qualified distillation targets; do not run an unchanged student as a distillation experiment')
    result.to_csv(out/'labels.csv',index=False)
    save_json(out/'contract.json',{**source_contract,'parent_label_release_sha256':file_hash(labels_path),
        'labels_sha256':file_hash(out/'labels.csv'),'teacher_source_policy_verified':True})
    save_json(out/'lineage.json',{'outer_fold':int(outer_fold),'labels_sha256':file_hash(labels_path),'compiled_labels_sha256':file_hash(out/'labels.csv'),'teachers':receipts,
        'distilled_cells':taught,'scope':'isolated teacher soft-target channel; no student fit or improvement claim'})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--teachers')
    p.add_argument('--outer-fold',type=int,choices=range(5),required=True);p.add_argument('--manifest')
    p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--output',help='default for export: ROOT/labels_distilled/outer_FOLD')
    p.add_argument('--prepare-partitions',action='store_true')
    a=p.parse_args()
    if a.prepare_partitions:
        if not a.output:p.error('--output is required with --prepare-partitions')
        prepare_partitions(a.labels,a.outer_fold,a.output,root=a.root)
    else:
        if not a.teachers or not a.manifest:p.error('--teachers and --manifest are required for actual teacher inference')
        build(a.labels,json.loads(Path(a.teachers).read_text(encoding='utf-8')),a.outer_fold,a.output,manifest=a.manifest,root=a.root)
