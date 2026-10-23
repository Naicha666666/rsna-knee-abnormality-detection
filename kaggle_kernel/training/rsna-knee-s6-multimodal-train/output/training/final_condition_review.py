"""Close the existing targeted data queue against full reports, without training.

Preparation records already-resolved/retracted premises separately. Only
remaining questions on existing development cases are sent for model review.
Completion/acceptance does not turn missing report facts into image truth.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from .common import UID,LABELS,save_json,file_hash


def prepare(root,queue,output,batch_size=8):
    root,queue,output=Path(root),Path(queue),Path(output)
    if (output/'manifest.json').exists():return json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    from .criteria import labels_directory
    labels=labels_directory(root)/'labels.csv';frame=pd.read_csv(labels).set_index(UID)
    raw=pd.read_csv(root/'raw/train.csv').set_index(UID);questions=pd.read_csv(queue)
    records={};status=[]
    for q in questions.to_dict('records'):
        uid=q[UID];label=q['label'];factor=q['missing_factor'];r=frame.loc[uid];column=label+'__factor_'+factor
        if bool(r.is_gold) or not bool(r.development_only) or r.fold<0:raise ValueError('Review queue escaped existing development-only non-gold scope')
        if r[column+'_weight']>0:state='condition_already_observed'
        elif not(r[label+'__presence']==1 and r[label+'__presence_weight']>=.5):state='premise_retracted_in_current_release'
        else:state='full_source_review_requested'
        status.append({**q,'current_status':state,'current_label_value':None if pd.isna(r[label]) else float(r[label])})
        if state!='full_source_review_requested':continue
        text=str(raw.loc[uid,'Report']);rid='F'+hashlib.sha256(uid.encode()).hexdigest()[:12]
        item=records.setdefault(rid,{'report_id':rid,'report_text':text,'report_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'labels':[],'questions':[],'study_id':uid})
        if label not in item['labels']:item['labels'].append(label)
        item['questions'].append({'label':label,'issue':q['question']})
    output.mkdir(parents=True,exist_ok=True);(output/'batches').mkdir(exist_ok=True)
    rows=sorted(records.values(),key=lambda x:x['report_id']);context={'questions_by_report':{r['report_id']:r['questions'] for r in rows},
        'reference_contrasts':[
            'Before finalizing, reread every sentence concerning each requested structure. Explicit abnormality must not disappear merely because grade, acuity or extent is unknown.',
            'MCL high grade and acute status must describe the same injury. Some remaining intact fibers can coexist with a high-grade partial tear.',
            'An OA lesion may be deep yet have unknown extent. Bone edema extent, cyst size and a separate shallow lesion cannot supply its qualifying extent.',
            'A local negative remains local. Do not negate an entire compartment based on one surface or one small lesion.',
            'Injury absence, below-threshold injury, and missing information are different. No acute fracture does not exclude an old fracture.',
            'Any explicit statements in Greek, Bulgarian, Croatian or other languages are evidence. Preserve their literal original quotes.',
            'A missing condition is a valid result. Do not invent it to close a task. Preserve all independently supported components.'
        ]}
    save_json(output/'review_context.json',context);batches=[];mapping={}
    for start in range(0,len(rows),batch_size):
        bid=f'B{start//batch_size+1:04}';items=[]
        for r in rows[start:start+batch_size]:
            mapping[r['report_id']]=r['study_id'];items.append({k:r[k] for k in ['report_id','report_text','report_sha256','labels']})
        path=output/'batches'/f'{bid}.json';save_json(path,{'batch_id':bid,'reports':items})
        batches.append({'batch_id':bid,'path':'batches/'+path.name,'sha256':file_hash(path),'reports':len(items),
            'report_ids':[r['report_id'] for r in items],'requested_report_label_pairs':sum(len(r['labels']) for r in items)})
    save_json(output/'queue_status_before.json',status);save_json(output/'report_to_study.json',mapping)
    manifest={'purpose':'Final targeted condition review, original full reports and no official answers',
        'batches':batches,'batch_count':len(batches),'unique_full_reports':len(rows),
        'requested_report_label_pairs':sum(b['requested_report_label_pairs'] for b in batches),
        'review_context':{'path':'review_context.json','sha256':file_hash(output/'review_context.json')},
        'label_source':str(labels),'label_sha256':file_hash(labels),'queue_source':str(queue),'queue_sha256':file_hash(queue),
        'original_questions':len(status),'resolved_before_review':sum(s['current_status']!='full_source_review_requested' for s in status),
        'outputs_are_candidates_not_image_truth':True,'official_labels_in_prompts':False}
    save_json(output/'manifest.json',manifest);return manifest


def prepare_rejections(primary,decisions,output):
    primary,decisions,output=map(Path,(primary,decisions,output))
    if (output/'manifest.json').exists():raise ValueError('Repair inputs are frozen')
    source=json.loads((primary/'manifest.json').read_text(encoding='utf-8'))
    records={r['report_id']:r for b in source['batches'] for r in json.loads((primary/b['path']).read_text(encoding='utf-8'))['reports']}
    rejected=[r for r in json.loads(decisions.read_text(encoding='utf-8'))['decisions'] if r['status']=='rejected']
    questions={}
    for r in rejected:questions.setdefault(r['report_id'],[]).append({'label':r['label'],'issue':r['reason']})
    output.mkdir(parents=True,exist_ok=True)
    context=json.loads((primary/'review_context.json').read_text(encoding='utf-8'))
    context['questions_by_report']=questions
    context['reference_contrasts'].append('The second opinion may itself be wrong. Reread the source and definitions; retain uncontested components separately. A disputed depth or extent must not erase explicit abnormality.')
    save_json(output/'review_context.json',context);batches=[];selected=sorted(questions)
    for start in range(0,len(selected),4):
        bid=f'B{start//4+1:04}';rows=[{**records[rid],'labels':list(dict.fromkeys(q['label'] for q in questions[rid]))} for rid in selected[start:start+4]]
        path=output/'batches'/f'{bid}.json';save_json(path,{'batch_id':bid,'reports':rows})
        batches.append({'batch_id':bid,'path':'batches/'+path.name,'sha256':file_hash(path),'reports':len(rows),
            'report_ids':[r['report_id'] for r in rows],'requested_report_label_pairs':sum(len(r['labels']) for r in rows)})
    mapping=json.loads((primary/'report_to_study.json').read_text(encoding='utf-8'));save_json(output/'report_to_study.json',{k:mapping[k] for k in selected})
    manifest={'purpose':'Preserve independently explicit components of rejected targeted proposals','batches':batches,
        'batch_count':len(batches),'unique_full_reports':len(selected),'requested_report_label_pairs':len(rejected),
        'parent_corpus':str(primary.resolve()),'decisions_sha256':file_hash(decisions),
        'review_context':{'path':'review_context.json','sha256':file_hash(output/'review_context.json')},'official_labels_in_prompts':False}
    save_json(output/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee')
    p.add_argument('--queue',default='research/criterion_boundaries_20261001_v7_2/development_review_queue.csv')
    p.add_argument('--output',default='training/output/final_condition_review_20261001');a=p.parse_args()
    m=prepare(a.root,a.queue,a.output);print({k:v for k,v in m.items() if k not in ['batches']})
