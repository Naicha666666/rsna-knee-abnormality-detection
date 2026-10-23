"""Read completed primary/secondary outputs and replay the real label overlay.

No model calls, training, corpus edits, or label activation. A partial running
corpus is explicitly a snapshot, never a completed quality certificate.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

from .common import file_hash,save_json
from .criteria import extract
from .semantic_review_contract import validate_batch,REVIEW_KIND
from .semantic_review_verify import validate_verification
from .semantic_review_overlay import apply_semantic_review,requires_secondary_review


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def language_bucket(text):
    if any('\u0370'<=c<='\u03ff' for c in text):return 'Greek_script'
    if any('\u0400'<=c<='\u04ff' for c in text):return 'Cyrillic_script'
    if any(c in text for c in 'čćđšžČĆĐŠŽ'):return 'South_Slavic_Latin_candidate'
    return 'other'


def audit(primary,secondary,output):
    primary,secondary,output=map(Path,(primary,secondary,output));output.mkdir(parents=True,exist_ok=True)
    pm=read(primary/'manifest.json');sm=read(secondary/'manifest.json')
    decisions={};secondary_errors=[];secondary_batches=0
    for b in sm['batches']:
        rp=secondary/'receipts'/(b['batch_id']+'.json');ap=secondary/'reviews'/rp.name;bp=secondary/b['path']
        if not rp.exists() or not ap.exists():continue
        try:
            rec=read(rp)
            if rec.get('status')!='completed':continue
            if rec['output_sha256']!=file_hash(ap) or rec['input_sha256']!=file_hash(bp):raise ValueError('receipt hash mismatch')
            rows=validate_verification(read(ap),read(bp)['reports'])
            for row in rows:decisions[row['report_id'],row['label']]=row
            secondary_batches+=1
        except (ValueError,KeyError,json.JSONDecodeError) as e:secondary_errors.append({'batch':b['batch_id'],'error':str(e)})
    counts=Counter();languages={};samples={};losses=[];rejected=[];primary_errors=[];snapshots=[]
    for b in pm['batches']:
        rp=primary/'receipts'/(b['batch_id']+'.json');ap=primary/'reviews'/rp.name
        if not rp.exists() or not ap.exists():continue
        try:
            rec=read(rp)
            if rec.get('status')!='completed':continue
            if rec['output_sha256']!=file_hash(ap) or rec['input_sha256']!=b['sha256']:raise ValueError('receipt hash mismatch')
            inputs=read(primary/b['path'])['reports'];records={r['report_id']:r for r in inputs}
            rows=validate_batch(read(ap),inputs)
        except (ValueError,KeyError,json.JSONDecodeError) as e:
            primary_errors.append({'batch':b['batch_id'],'error':str(e)});continue
        counts['completed_primary_batches']+=1;counts['full_reports']+=len(inputs)
        base={rid:extract(r['report_text']) for rid,r in records.items()}
        for row in rows:
            rid,label=row['report_id'],row['label'];report=records[rid]['report_text'];old=base[rid][label]
            bucket=language_bucket(report);lc=languages.setdefault(bucket,Counter())
            counts['report_label_pairs']+=1;counts['primary_status_'+row['status']]+=1;lc['reviewed_pairs']+=1
            known=requires_secondary_review(row)
            if known:counts['primary_known_evidence']+=1;lc['known_evidence']+=1
            dec=decisions.get((rid,label))
            if known and dec is None:counts['awaiting_secondary']+=1;continue
            if dec and dec['status']=='rejected':
                counts['secondary_rejected']+=1
                rejected.append({'batch_id':b['batch_id'],'report_id':rid,'label':label,'proposal':row,'decision':dec,'report':report})
                continue
            if not known:continue
            entry={**row,'StudyInstanceUID':rid,'report_sha256':records[rid]['report_sha256'],
                   'review_kind':REVIEW_KIND,'review_completed':True,'is_image_truth':False,
                   'secondary_review_complete':True,'secondary_status':'accepted','secondary_decision':dec}
            final,flags=apply_semantic_review(old,report,entry)
            counts['secondary_accepted']+=1;lc['secondary_accepted']+=1
            if old.criterion is None and final.criterion is not None:
                counts['main_target_recovered']+=1;lc['main_target_recovered']+=1
            if old.presence is None and final.presence is not None:
                counts['presence_recovered']+=1;lc['presence_recovered']+=1
            for a in row['attributes']:
                before=old.attributes.get(a['name'],{}).get('target');after=final.attributes.get(a['name'],{}).get('target')
                if before is None and after is not None:counts['conditions_recovered']+=1;lc['conditions_recovered']+=1
            suppressed=[]
            if row['criterion'] is not None and final.criterion!=row['criterion']:suppressed.append('criterion')
            if row['presence'] is not None and final.presence!=row['presence']:suppressed.append('presence')
            for a in row['attributes']:
                if a['target'] is not None and final.attributes.get(a['name'],{}).get('target')!=a['target']:suppressed.append(a['name'])
            detail={'batch_id':b['batch_id'],'report_id':rid,'label':label,'language_bucket':bucket,
                    'before':{'presence':old.presence,'criterion':old.criterion,'attributes':old.attributes,'reason':old.reason},
                    'proposal':row,'after':{'presence':final.presence,'criterion':final.criterion,'attributes':final.attributes},
                    'flags':flags,'report':report,'decision':dec}
            if suppressed:
                counts['accepted_evidence_suppressed_during_overlay']+=1
                losses.append({**detail,'suppressed_components':suppressed})
            elif bucket!='other' and len(samples.setdefault(bucket,[]))<3 and (old.criterion!=final.criterion or old.presence!=final.presence):
                samples[bucket].append(detail)
        snapshots.append({'batch':b['batch_id'],'output_sha256':file_hash(ap),'receipt_sha256':file_hash(rp)})
    summary={'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'snapshot of completed report semantics and real overlay; not expert image accuracy',
             'primary_total_batches':pm['batch_count'],'completed_secondary_batches':secondary_batches,
             **dict(counts),'language_buckets':{k:dict(v) for k,v in languages.items()},
             'primary_validation_errors':primary_errors,'secondary_validation_errors':secondary_errors,
             'labels_activated':False,'optimizer_steps':0}
    save_json(output/'summary.json',summary);save_json(output/'accepted_but_suppressed.json',losses)
    save_json(output/'secondary_rejected.json',rejected);save_json(output/'multilingual_examples.json',samples)
    save_json(output/'snapshot_receipts.json',snapshots)
    print(json.dumps(summary,ensure_ascii=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',default='training/output/semantic_review_full_20261001')
    p.add_argument('--secondary',default='training/output/semantic_review_verification_20261001');p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.primary,a.secondary,a.output)
