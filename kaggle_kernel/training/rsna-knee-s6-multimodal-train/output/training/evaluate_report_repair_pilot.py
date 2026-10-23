"""Freeze the pre-run reference and compare a completed re-extraction pilot.

Counts are evidence availability / model review outcomes, not clinical accuracy
or MRI AUC. No candidate is automatically published into production labels.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import pandas as pd
from .common import LABELS,UID,file_hash,save_json
from .semantic_review_contract import validate_batch
from .semantic_review_verify_corpus import aggregate
from .criteria import Fact
from .semantic_review_overlay import apply_semantic_review


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def corpus_rows(root,selected=None):
    root=Path(root);m=read(root/'manifest.json');rows={};reports={}
    for b in m['batches']:
        if selected is not None and not set(b['report_ids'])&selected:continue
        records=read(root/b['path'])['reports']
        answer=root/'reviews'/(b['batch_id']+'.json');receipt=root/'receipts'/(b['batch_id']+'.json')
        rec=read(receipt)
        if rec['status']!='completed' or rec['input_sha256']!=file_hash(root/b['path']) or rec['output_sha256']!=file_hash(answer):
            raise ValueError('Incomplete or changed model result: '+b['batch_id'])
        for r in records:
            if selected is None or r['report_id'] in selected:reports[r['report_id']]=r
        for r in validate_batch(read(answer),records):
            if selected is None or r['report_id'] in selected:rows[(r['report_id'],r['label'])]=r
    return rows,reports


def freeze(pilot,labels):
    pilot,labels=Path(pilot),Path(labels);target=pilot/'baseline_snapshot.json'
    if target.exists():return read(target)
    m=read(pilot/'manifest.json');parent=Path(m['parent_corpus']);chosen={rid for b in m['batches'] for rid in b['report_ids']}
    rows,reports=corpus_rows(parent,chosen)
    decisions=read(parent.parent/'semantic_review_verification_20261001/final_decisions.json')['decisions']
    second=[d for d in decisions if d['report_id'] in chosen]
    mapping=defaultdict(set)
    with (parent/'cell_mapping.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['report_id'] in chosen:mapping[r['report_id']].add(r[UID])
    wanted=set().union(*mapping.values());frame=pd.read_csv(labels)
    picked=frame[frame[UID].isin(wanted)]
    if len(picked)!=len(wanted):raise ValueError('Baseline labels missing selected study')
    rejected=read(m['rejected_snapshot'])
    result={'frozen_utc':datetime.now(timezone.utc).isoformat(),'labels_path':str(labels.resolve()),
        'labels_sha256':file_hash(labels),'report_to_studies':{k:sorted(v) for k,v in mapping.items()},
        'prior_primary_rows':list(rows.values()),'prior_secondary_decisions':second,
        'original_repair_targets':[r for r in rejected if r['report_id'] in chosen],
        'active_label_rows':json.loads(picked.to_json(orient='records')),
        'pilot_manifest_sha256':file_hash(pilot/'manifest.json'),'reports':len(chosen),'studies':len(wanted),
        'scope':'Frozen pre-pilot reference; report/model agreement is not image ground truth.'}
    save_json(target,result);return result


def components(row):
    return {'presence':row['presence'],'criterion':row['criterion'],**{a['name']:a['target'] for a in row['attributes']}}


def consumption_probe(row,decision,report):
    """Replay the real overlay against an empty fact; never publish a label.

    This isolates which candidate components the study-level consumer permits.
    It is not a merge into the active label dataset or a quality judgment.
    """
    entry={**row,'StudyInstanceUID':row['report_id'],'review_completed':True,'is_image_truth':False,
           'secondary_review_complete':True,
           'secondary_status':decision['status'] if decision else 'not_required',
           'secondary_decision':decision}
    fact,flags=apply_semantic_review(Fact(label=row['label']),report,entry)
    values={'presence':fact.presence,'criterion':fact.criterion,
            **{a['name']:fact.attributes.get(a['name'],{}).get('target') for a in row['attributes']}}
    local={k:v for k,v in fact.attributes.items() if v.get('local_target') is not None}
    return {'study_components':values,'local_records':local,'flags':flags,
            'scope':'Empty-fact overlay probe, not an active-label merge or expert truth.'}


def compare(pilot,secondary):
    pilot,secondary=Path(pilot),Path(secondary);baseline=read(pilot/'baseline_snapshot.json')
    if baseline['pilot_manifest_sha256']!=file_hash(pilot/'manifest.json'):raise ValueError('Pilot inputs changed')
    new,reports=corpus_rows(pilot)
    second_path=secondary/'final_decisions.json';aggregate(secondary,second_path)
    decisions={(r['report_id'],r['label']):r for r in read(second_path)['decisions']}
    old={(r['report_id'],r['label']):r for r in baseline['prior_primary_rows']}
    old_dec={(r['report_id'],r['label']):r for r in baseline['prior_secondary_decisions']}
    focus={(r['report_id'],r['label']) for r in baseline['original_repair_targets']}
    labels={r[UID]:r for r in baseline['active_label_rows']}
    stats=Counter();details=[]
    for key,row in new.items():
        rid,label=key;current=components(row);prior=components(old[key]) if key in old else {}
        dec=decisions.get(key);accepted=bool(dec and dec['status']=='accepted')
        has_known=any(v is not None for v in current.values())
        stats['target_rows']+=1;stats['new_status_'+row['status']]+=1
        stats['second_'+(dec['status'] if dec else 'not_required')]+=1
        if has_known:stats['rows_with_known_evidence']+=1
        changes=[{'component':k,'before':prior.get(k),'after':v,'previously_requested':key in old}
                 for k,v in current.items() if prior.get(k)!=v]
        if changes:stats['changed_rows']+=1
        if key in focus:
            stats['originally_rejected_targets']+=1
            stats['repair_target_'+('accepted' if accepted else 'rejected' if dec else 'all_unknown')]+=1
        retained_old=bool(old_dec.get(key,{}).get('status')=='accepted')
        if retained_old and not accepted and has_known:stats['old_accepted_new_rejected']+=1
        production=[]
        probe=consumption_probe(row,dec,reports[rid]['report_text'])
        component_comparison=[]
        for uid in baseline['report_to_studies'][rid]:
            oldmain=labels[uid].get(label);oldweight=labels[uid].get(label+'__weight',0) or 0
            candidate=row['criterion'] if accepted else None
            kind=('not_offered' if candidate is None else 'new_main_evidence' if not oldweight else
                  'same_main_value' if oldmain==candidate else 'main_disagreement')
            production.append({'StudyInstanceUID':uid,'baseline_value':oldmain,'baseline_weight':oldweight,
                               'accepted_candidate':candidate,'relation':kind})
            stats['study_'+kind]+=1
            for name,value in probe['study_components'].items():
                column=label if name=='criterion' else label+'__presence' if name=='presence' else label+'__factor_'+name
                weight_column=label+'__weight' if name=='criterion' else column+'_weight'
                oldvalue=labels[uid].get(column);weight=labels[uid].get(weight_column,0) or 0
                relation=('not_offered' if value is None else 'new_component' if not weight else
                          'same_value' if value==oldvalue else 'disagreement')
                component_comparison.append({'StudyInstanceUID':uid,'component':name,
                    'baseline_value':oldvalue,'baseline_weight':weight,'consumable_candidate':value,'relation':relation})
                if relation!='not_offered':stats['consumable_'+name+'_'+relation]+=1
        stats['local_only_records']+=len(probe['local_records'])
        details.append({'report_id':rid,'label':label,'repair_target':key in focus,
            'previous_primary':old.get(key),'previous_secondary':old_dec.get(key),
            'new_primary':row,'new_secondary':dec,'changes':changes,
            'baseline_production_comparison':production,'consumption_probe':probe,
            'component_production_comparison':component_comparison,'full_report':reports[rid]['report_text']})
    assert len(new)==len(reports)*12
    summary={'reports':len(reports),'studies':baseline['studies'],'baseline_labels':baseline['labels_path'],
             **dict(stats),'clinical_accuracy_measured':False,'MRI_AUC_measured':False,
             'production_labels_changed':False,'manual_source_review_pending':True}
    save_json(pilot/'comparison_summary.json',summary);save_json(pilot/'comparison_details.json',details)
    save_json(pilot/'repair_target_details.json',[d for d in details if d['repair_target']])
    print(json.dumps(summary,ensure_ascii=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','compare']);p.add_argument('--pilot',required=True)
    p.add_argument('--labels',default='F:/rsnaknee/labels_clinical_v7_1/labels.csv');p.add_argument('--secondary')
    a=p.parse_args()
    if a.command=='freeze':
        result=freeze(a.pilot,a.labels);print({k:result[k] for k in ('reports','studies','labels_path','labels_sha256')})
    else:compare(a.pilot,a.secondary)
