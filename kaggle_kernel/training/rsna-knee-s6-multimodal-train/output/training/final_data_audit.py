"""Compare released supervision and close the original condition question ledger."""
import json
from pathlib import Path
import pandas as pd
from .common import UID,LABELS,file_hash,save_json
from .clinical_factors import FACTOR_COLUMNS
from .partial_joint import audit as partial_audit
from .mcl_joint import audit_frame
from .local_oa_supervision import attach


def run(root='F:/rsnaknee',output='research/orthofoundation_mainline_20261001'):
    root,out=Path(root),Path(output);out.mkdir(parents=True,exist_ok=True)
    old_path=root/'labels_clinical_v7_3/labels.csv';new_path=root/'labels_clinical_v7_4/labels.csv'
    old=pd.read_csv(old_path).set_index(UID);new=pd.read_csv(new_path).set_index(UID).loc[old.index]
    if not old[['fold','group_id','is_gold','development_only']].equals(new[['fold','group_id','is_gold','development_only']]):raise ValueError('Final label review changed frozen patient roles')
    official=new.is_gold.astype(bool)
    if not old.loc[official,LABELS].equals(new.loc[official,LABELS]):raise ValueError('Official labels changed')
    columns=[(c,c+'__weight','main') for c in LABELS]+[(c+'__presence',c+'__presence_weight','presence') for c in LABELS]+[(c,c+'_weight','condition') for c in FACTOR_COLUMNS]
    changes=[]
    for column,weight,kind in columns:
        for uid in new.index:
            a,b=old.loc[uid],new.loc[uid];ka=a[weight]>0;kb=b[weight]>0
            if ka==kb and (not ka or a[column]==b[column]):continue
            changes.append({UID:uid,'column':column,'kind':kind,'before':float(a[column]) if ka else None,
                'after':float(b[column]) if kb else None,'transition':'added_known' if not ka and kb else 'withdrawn' if ka and not kb else 'changed',
                'development_only':bool(b.development_only)})
    pd.DataFrame(changes).to_csv(out/'label_changes.csv',index=False)
    questions=json.loads(Path('training/output/final_condition_review_20261001/queue_status_before.json').read_text(encoding='utf-8'));rows=[]
    for q in questions:
        r=new.loc[q[UID]];c=q['label']+'__factor_'+q['missing_factor']
        if r[c+'_weight']>0:status='condition_observed'
        elif not(r[q['label']+'__presence']==1 and r[q['label']+'__presence_weight']>=.5):status='abnormality_premise_unconfirmed_or_retracted'
        else:status='report_does_not_resolve_condition_use_partial_supervision'
        rows.append({**q,'final_status':status,'condition_target':float(r[c]) if r[c+'_weight']>0 else None,
            'main_target':float(r[q['label']]) if r[q['label']+'__weight']>0 else None})
    pd.DataFrame(rows).to_csv(out/'condition_queue_result.csv',index=False)
    frame=new.reset_index();partial=partial_audit(frame);mcl=audit_frame(frame);_,local=attach(frame,new_path.with_name('evidence.jsonl'))
    coverage=[]
    for c in LABELS:
        w=new[c+'__weight']>0
        coverage.append({'label':c,'positive':int((w&new[c].eq(1)).sum()),'negative':int((w&new[c].eq(0)).sum()),'unknown':int((~w).sum())})
    pd.DataFrame(coverage).to_csv(out/'main_label_coverage.csv',index=False)
    summary={'old_labels_sha256':file_hash(old_path),'labels_sha256':file_hash(new_path),'studies':len(new),
        'official_labels_unchanged':True,'patient_roles_unchanged':True,'fold_counts':{str(k):int(v) for k,v in new.fold.value_counts().sort_index().items()},
        'changes':pd.DataFrame(changes).groupby(['kind','transition']).size().to_dict() if changes else {},
        'condition_queue':pd.DataFrame(rows).final_status.value_counts().to_dict(),'partial_supervision':partial,'mcl_partial_supervision':mcl,
        'local_oa_records':len(local['local_records_consumed']),'local_oa_coverage':local['coverage'],
        'remaining_limits':['One repaired source interpretation remains rejected; two repaired targets have no agreed known fact. Existing independently supported facts are retained.',
            'Missing acuity or same-lesion extent cannot be reconstructed from report silence.',
            'Report evidence remains weak supervision; no expert image re-adjudication or learned AUC is implied.'],
        'training_started':False}
    summary['changes']={'/'.join(k):int(v) for k,v in summary['changes'].items()}
    save_json(out/'data_acceptance.json',summary);return summary


if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('partial_supervision','mcl_partial_supervision','local_oa_coverage')},ensure_ascii=False))
