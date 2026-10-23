"""Construct real train-fold contrast schedules and verify their actual consumers.

No optimizer steps. Pairing changes exposure order, never creates new labels or
increases a Study's exposure budget. The image probe is engineering evidence.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,file_hash,save_json
from .clinical_factors import FACTOR_COLUMNS
from .criteria import labels_directory
from .sampling import StudyExposureSampler
from .supervision import supported_factors,channel_loss,supervision_loss
from .recipes import configuration


def digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def audit(root='F:/rsnaknee'):
    root=Path(root);out=root/'analysis/clinical_contrasts';out.mkdir(parents=True,exist_ok=True)
    source=labels_directory(root);frame=pd.read_csv(source/'labels.csv',dtype={UID:str,'group_id':str})
    config=configuration('clinical_contrasts',root,stage='finetune');folds=[];probe_pair=None
    hard_counts={c:int(frame[c+'__hard_negative'].sum()) for c in LABELS}
    pos_counts={c:int(((frame[c]==1)&(frame[c+'__weight']>=.5)).sum()) for c in LABELS}
    def cached(uid):return root/'cache_native/studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')
    for fold in range(5):
        start=time.perf_counter();allowed=frame.fold.ge(0)&frame.fold.ne(fold)
        ids=frame.loc[allowed,UID].tolist();f,support=supported_factors(frame,ids)
        active=[c+'__weight' for c in LABELS]+[c+'__presence_weight' for c in LABELS]+[c+'_weight' for c in FACTOR_COLUMNS]
        ids=f.loc[allowed & f[active].sum(axis=1).gt(0),UID].tolist()
        sampler=StudyExposureSampler(ids,f,draws=config['train_draws_per_study'],effective_batch=config['batch_size']*config['accumulate'],
            balanced=config['balanced_batches'],hard_negatives=config['hard_negatives'],paired_contrasts=config['paired_contrasts'],
            factor_contrasts=config['factor_contrasts'],seed=config['seed'])
        schedule=list(sampler);index=f.set_index(UID);events=sampler.pair_schedule
        counts=Counter(i for i,draw in schedule)
        assert counts=={i:config['train_draws_per_study'] for i in range(len(ids))}
        assert len(set(schedule))==len(schedule)
        assert set(index.loc[ids,'group_id']).isdisjoint(set(f.loc[f.fold.eq(fold),'group_id']))
        for e in events:
            block=schedule[e['effective_batch']*sampler.batch:(e['effective_batch']+1)*sampler.batch]
            observed={ids[i] for i,draw in block};p,n=index.loc[e['positive_uid']],index.loc[e['negative_uid']]
            assert e['positive_uid'] in observed and e['negative_uid'] in observed and p.group_id!=n.group_id
            col=e['label'] if e['kind']=='criterion' else e['label']+'__factor_'+e['factor']
            weight=col+'__weight' if e['kind']=='criterion' else col+'_weight'
            assert p[col]==1 and n[col]==0 and min(p[weight],n[weight])>=.5
            if e['kind']=='criterion':assert n[e['label']+'__hard_negative']==1
            else:assert p[e['label']+'__presence']==n[e['label']+'__presence']==1
            if fold==0 and probe_pair is None and e['kind']=='criterion' and all(cached(u).exists() for u in (e['positive_uid'],e['negative_uid'])):
                probe_pair=e.copy()
        record={'fold':fold,'training_studies':len(ids),'exposures':len(schedule),'exposure_per_study':config['train_draws_per_study'],
            'effective_batch':sampler.batch,'pairs':len(events),'criterion_pairs':sum(e['kind']=='criterion' for e in events),
            'factor_pairs':sum(e['kind']=='factor' for e in events),'pair_counts':dict(Counter(e['kind']+':'+e['label']+(':'+e['factor'] if e['factor'] else '') for e in events)),
            'available_main_labels':sorted({p[1] for p in sampler.contrasts if p[0]=='criterion'}),
            'available_factor_conditions':sorted(p[1]+':'+p[2] for p in sampler.contrasts if p[0]=='factor'),
            'different_patients_verified':True,'heldout_patients_excluded':True,'seconds':time.perf_counter()-start}
        folds.append(record);save_json(out/f'fold{fold}_pairs.json',events);print('FOLD',fold,'pairs',len(events),'seconds',round(record['seconds'],2),flush=True)
    if probe_pair is None:raise RuntimeError('No paired source images are cached; cohort schedules exist but image consumer remains unverified')
    # Read genuine cached studies, through the trainer's dataset and DataLoader.
    from .native_data import NativeStudyDataset
    from .engine import loader
    from .model import KneeModel,ModelConfig
    ids=[probe_pair['positive_uid'],probe_pair['negative_uid']]
    original=json.loads((root/'cache_native/manifest.json').read_text(encoding='utf-8'))
    manifest={'config':original['config'],'scope':'explicit_subset','studies':[json.loads(cached(u).read_text(encoding='utf-8')) for u in ids]}
    mp=out/'image_probe_manifest.json';save_json(mp,manifest)
    dataset=NativeStudyDataset(mp,frame,ids=ids,training=False,image_size=64,views='global_local')
    sampler=StudyExposureSampler(ids,frame,draws=1,effective_batch=2,balanced=True,hard_negatives=True,paired_contrasts=True)
    batches=list(loader(dataset,1,0,sampler=sampler));assert {b[UID][0] for b in batches}==set(ids)
    torch.set_num_threads(2);torch.manual_seed(42)
    model=KneeModel(ModelConfig(backbone='resnet18',image_size=64,context_type='none',hidden=16,spatial_hidden=8,
        presence_head=True,clinical_head=True,encode_chunk=16),pretrained=False).eval()
    before=digest(model);den={'studies':2,'presence':sum(b['presence_weight'].sum(0) for b in batches),
        'factors':sum(b['factors_weight'].sum(0) for b in batches)}
    main_den=sum(b['weight'].sum(0) for b in batches);loss=0;outputs=[]
    for b in batches:
        pred=model(b,return_training=True);pred['logits'].retain_grad();outputs.append(pred)
        extra,_=supervision_loss(pred,b,den,config)
        loss=loss+channel_loss(pred['logits'],b['target'],b['weight'],main_den)+extra
    loss.backward();j=LABELS.index(probe_pair['label']);gradients={b[UID][0]:float(p['logits'].grad[0,j]) for b,p in zip(batches,outputs)}
    assert gradients[probe_pair['positive_uid']]<0<gradients[probe_pair['negative_uid']]
    visual_gradient=float(model.backbone.conv1.weight.grad.abs().sum());assert visual_gradient>0 and digest(model)==before
    consumer={'pair':probe_pair,'input_windows':[int(b['images'].shape[1]) for b in batches],'source':'complete cached Studies; all available Series and global/local views',
        'diagnostic_resolution':64,'random_cpu_model':True,'logit_gradients':gradients,'visual_gradient_l1':visual_gradient,
        'parameters_unchanged':True,'optimizer_steps':0,'loss':float(loss.detach()),'scope':'actual image/dataset/sampler/model/loss path, not learned clinical ability or AUC'}
    save_json(out/'actual_image_consumer.json',consumer)
    # Expose the remaining missing evidence without converting it to positives.
    unresolved=frame[(frame.MCL__weight==0)&(frame.MCL__factor_high_grade==1)&(frame.MCL__factor_high_grade_weight>=.5)
        &(frame.MCL__factor_acute_weight==0)]
    unresolved[[UID,'group_id','fold','MCL__criterion_state','MCL__factor_high_grade','MCL__factor_high_grade_weight','MCL__factor_acute_weight']].to_csv(out/'mcl_missing_acute_evidence.csv',index=False)
    result={'labels':str(source),'labels_sha256':file_hash(source/'labels.csv'),'positive_counts':pos_counts,'hard_negative_counts':hard_counts,
        'folds':folds,'image_consumer':str(out/'actual_image_consumer.json'),'MCL_high_grade_without_acute_evidence':len(unresolved),
        'training_started':False,'optimizer_steps':0,'scope':'supervision and bounded paired consumption constructed; predictive benefit not measured'}
    save_json(out/'summary.json',result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='F:/rsnaknee');args=ap.parse_args()
    x=audit(args.root);print(json.dumps({'positive_counts':x['positive_counts'],'hard_negative_counts':x['hard_negative_counts'],
        'MCL_high_grade_without_acute_evidence':x['MCL_high_grade_without_acute_evidence'],'optimizer_steps':0}))
