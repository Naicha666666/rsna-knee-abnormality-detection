"""Check source-language repairs against published targets and reviewed reports.

No optimization or image adjudication. The full-corpus delta is not a precision
estimate; the reviewed reports are explicitly development cases.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,save_json,file_hash
from .criteria import extract
from .supervision import channel_loss,targets_from_row,supported_factors


def audit(root='F:/rsnaknee',before='labels_clinical_v5_1',after='labels_clinical_v5_2',reference=None):
    root=Path(root);out=root/'analysis'/after.replace('labels_clinical_','supervision_');out.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(root/before/'labels.csv').set_index(UID)
    new=pd.read_csv(root/after/'labels.csv').set_index(UID).reindex(old.index)
    raw=pd.read_csv(root/'raw/train.csv').set_index(UID).reindex(old.index)
    reference=Path(reference) if reference else Path(__file__).parent/'data/report_language_recovery_v5_2.json'
    reviewed=json.loads(reference.read_text(encoding='utf-8'))
    if len(new)!=4407 or new.index.has_duplicates or new.group_id.isna().any():raise ValueError('Study coverage changed')
    if not old.group_id.equals(new.group_id):raise ValueError('Patient identity changed')
    if (new.groupby('group_id').fold.nunique()>1).any():raise ValueError('A patient crosses folds')
    gold=raw[LABELS].notna().any(axis=1)
    if not old.loc[gold,'fold'].equals(new.loc[gold,'fold']):raise ValueError('Expert validation patients changed')
    for c in LABELS:
        known=raw[c].notna()
        if not new.loc[known,c].eq(raw.loc[known,c]).all() or not new.loc[known,c+'__weight'].eq(1).all():
            raise ValueError('Official target changed: '+c)
    checks=[]
    for item in reviewed['cases']:
        uid=item[UID];c=item['label'];text=raw.loc[uid,'Report'];fact=extract(text)
        if item['quote'] not in text:raise ValueError('Evidence quotation changed')
        f=fact[c];passed=(f.presence,f.criterion)==(item['expected_presence'],item['expected_criterion'])
        for key,target in item.get('expected_factors',{}).items():passed=passed and f.attributes.get(key,{}).get('target')==target
        if item.get('expected_certainty'):passed=passed and f.certainty==item['expected_certainty']
        postcheck=item.get('role')=='postimplementation_report_check_only'
        if postcheck:pass  # Text-only check after implementation; official labels have precedence.
        elif item['expected_criterion'] is not None:
            passed=passed and new.loc[uid,c]==item['expected_criterion'] and new.loc[uid,c+'__weight']>0
        else:passed=passed and new.loc[uid,c+'__weight']==0
        if not postcheck and new.loc[uid,'fold']!=5:raise ValueError('Reviewed patient was not excluded from independent validation')
        checks.append({**item,'passed':bool(passed),'actual_presence':f.presence,'actual_criterion':f.criterion,'reason':f.reason})
    if not all(r['passed'] for r in checks):raise ValueError('Reviewed semantics regression')
    evidence={}
    for line in (root/after/'evidence.jsonl').open(encoding='utf-8'):
        r=json.loads(line);evidence[(r[UID],r['label'])]=r
    changes=[];presence=[];recovered={};hard={};factor_support={}
    def value(x):return None if pd.isna(x) else float(x)
    for c in LABELS:
        a=old[c].where(old[c+'__weight']>0);b=new[c].where(new[c+'__weight']>0)
        changed=~(a.eq(b)|(a.isna()&b.isna()))
        recovered[c]={'positive':int((a.isna()&b.eq(1)).sum()),'negative':int((a.isna()&b.eq(0)).sum())}
        for uid in new.index[changed]:
            e=evidence[(uid,c)]
            changes.append({UID:uid,'label':c,'before':value(a[uid]),'after':value(b[uid]),
                'before_weight':float(old.loc[uid,c+'__weight']),'after_weight':float(new.loc[uid,c+'__weight']),
                'official':bool(raw.loc[uid,c]==raw.loc[uid,c]),'reason':e['reason'],'evidence':e['evidence']})
        a=old[c+'__presence'].where(old[c+'__presence_weight']>0)
        b=new[c+'__presence'].where(new[c+'__presence_weight']>0)
        for uid in new.index[~(a.eq(b)|(a.isna()&b.isna()))]:
            presence.append({UID:uid,'label':c,'before':value(a[uid]),'after':value(b[uid]),
                'before_weight':float(old.loc[uid,c+'__presence_weight']),'after_weight':float(new.loc[uid,c+'__presence_weight'])})
        hard[c]=int(new[c+'__hard_negative'].sum())
    frame=new.copy().reset_index()
    for fold in range(5):
        ids=frame.loc[(frame.fold>=0)&frame.fold.ne(fold),UID].tolist()
        _,factor_support[str(fold)]=supported_factors(frame,ids)
    # Consume corrected targets with the trainer's actual loss implementation.
    gradient=[]
    examples=[]
    for r in reviewed['cases']:
        uid=r[UID];c=r['label']
        if r.get('role')=='postimplementation_report_check_only' or new.loc[uid,c+'__weight']<=0:continue
        changed=(old.loc[uid,c+'__weight']==0 or old.loc[uid,c]!=new.loc[uid,c]
                 or old.loc[uid,c+'__presence']!=new.loc[uid,c+'__presence'])
        if changed:examples.append(r)
    # Real corrected targets, including both positive and negative examples.
    examples=examples[:6]+[r for r in examples[6:] if r['expected_criterion']==1][:2]
    for r in examples:
        uid=r[UID];c=r['label'];one={'case':r['case_id'],'label':c}
        for name,frame in [('before',old),('after',new)]:
            row=frame.loc[uid];extra=targets_from_row(row)
            target=torch.tensor(np.nan_to_num(row[LABELS].to_numpy(np.float32)))[None]
            weight=torch.tensor(row[[c+'__weight' for c in LABELS]].to_numpy(np.float32))[None]
            logits=torch.zeros_like(target,requires_grad=True);p=torch.zeros_like(target,requires_grad=True)
            loss=channel_loss(logits,target,weight,weight.sum(0))
            loss=loss+channel_loss(p,extra['presence_target'][None],extra['presence_weight'][None],extra['presence_weight'])
            loss.backward();j=LABELS.index(c)
            one[name]={'criterion_gradient':float(logits.grad[0,j]),'presence_gradient':float(p.grad[0,j])}
        if one['after']['criterion_gradient']*(1 if new.loc[uid,c]==1 else -1)>=0:raise ValueError('Incorrect main gradient direction')
        if old.loc[uid,c+'__presence']==1 and new.loc[uid,c+'__presence']==0 and not (one['before']['presence_gradient']<0<one['after']['presence_gradient']):
            raise ValueError('Existence supervision did not reverse its erroneous direction')
        gradient.append(one)
    summary={'scope':'Reviewed development reports and deterministic compiler/consumer checks; NOT expert image truth, independent precision, or AUC evidence',
        'cases':len(checks),'studies':len({x[UID] for x in checks}),'passed':sum(x['passed'] for x in checks),'failed':0,
        'whole_corpus_studies':len(new),'changed_main_targets':len(changes),'changed_official_targets':sum(r['official'] for r in changes),
        'recovered_main_targets':sum(sum(v.values()) for v in recovered.values()),'recovered_by_label':recovered,
        'withdrawn_to_unknown':sum(r['before'] is not None and r['after'] is None for r in changes),
        'changed_known_class':sum(r['before'] is not None and r['after'] is not None for r in changes),
        'per_label_changes':{c:sum(r['label']==c for r in changes) for c in LABELS},'hard_negatives':hard,
        'presence_reversals_1_to_0':{c:sum(r['label']==c and r['before']==1 and r['after']==0 for r in presence) for c in LABELS},
        'enabled_factors_by_fold':{f:sum(x['enabled'] for x in v.values()) for f,v in factor_support.items()},
        'patient_groups_unchanged':True,'expert_validation_folds_unchanged':True,'development_only_studies':int(new.development_only.sum()),
        'new_development_studies':int((new.development_only&~old.development_only).sum()),
        'reference_sha256':file_hash(reference),'labels_contract':str(root/after/'contract.json'),
        'labels_sha256':file_hash(root/after/'labels.csv'),'no_auc_claim':True,'independent_accuracy_estimate':False,'optimizer_steps':0}
    summary['remaining_unknown_targets']=int(new[[c+'__weight' for c in LABELS]].eq(0).sum().sum())
    summary['remaining_unknown_by_label']={c:int(new[c+'__weight'].eq(0).sum()) for c in LABELS}
    summary['before_directory']=before;summary['after_directory']=after
    summary['postimplementation_report_checks']=[r['case_id'] for r in reviewed['cases'] if r.get('role')=='postimplementation_report_check_only']
    if summary['changed_official_targets']:raise ValueError('Official references changed')
    pd.DataFrame(changes).to_csv(out/'target_changes.csv',index=False)
    pd.DataFrame(presence).to_csv(out/'presence_changes.csv',index=False)
    save_json(out/'report_semantics_checks.json',checks);save_json(out/'factor_support_by_fold.json',factor_support)
    save_json(out/'actual_loss_gradients.json',gradient);save_json(out/'summary.json',summary)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='F:/rsnaknee')
    ap.add_argument('--before',default='labels_clinical_v5_1');ap.add_argument('--after',default='labels_clinical_v5_2');ap.add_argument('--reference')
    args=ap.parse_args();print(json.dumps(audit(args.root,args.before,args.after,args.reference),ensure_ascii=False,indent=2))
