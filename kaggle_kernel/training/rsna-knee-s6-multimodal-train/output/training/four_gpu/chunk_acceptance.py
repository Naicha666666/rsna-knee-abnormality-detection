"""Bounded three-update adoption gate; does not claim long-run scaling."""
import argparse,json,math
from pathlib import Path
from ..common import save_json,file_hash
from .acceptance import compare_audit,assess_trajectory


def assess(source,reference,candidate,ready,chunk,output):
    root=Path(output).parent
    gradient=compare_audit(reference,candidate,root/'gradient_acceptance.json')
    trajectory=assess_trajectory(source,reference,candidate,root/'trajectory_acceptance.json')
    paths=[Path(reference),Path(candidate)];meta=[json.loads((p/'run.json').read_text()) for p in paths]
    runs=[[json.loads(s) for s in (p/'updates.jsonl').read_text().splitlines() if s.strip()] for p in paths]
    errors=[]
    for key in ('source_sha256','ready_sha256','world_size','config','device','deterministic_cuda','hostname','precision_policy'):
        if meta[0].get(key)!=meta[1].get(key):errors.append('Different '+key)
    if len(runs[0])!=3 or len(runs[1])!=3:errors.append('Need three complete updates in both runs')
    for a,b in zip(*runs):
        if a['source_exposures']!=b['source_exposures'] or a['optimizer_step']!=b['optimizer_step'] or a['studies']!=b['studies']:
            errors.append('Different complete update')
        if sum(q['windows'] for q in a['ranks'])!=sum(q['windows'] for q in b['ranks']):errors.append('Different image workload')
    times=[sum(r['seconds'] for r in run[1:]) for run in runs]
    if not all(math.isfinite(t) and t>0 for t in times):raise ValueError('Invalid timing')
    speedup=times[0]/times[1]
    if speedup<1.10:errors.append('Less than 10% complete-update improvement')
    if not gradient['passed']:errors.append('Gradient gate failed')
    if not trajectory['passed']:errors.append('Trajectory gate failed')
    if any(q['peak_allocated_gib']>.85*q['device_total_gib'] or q['peak_reserved_gib']>.94*q['device_total_gib'] for row in runs[1] for q in row['ranks']):errors.append('Memory reserve failed')
    result={'state':'passed' if not errors else 'failed','errors':errors,'encoder_chunk':chunk,
        'ready_sha256':file_hash(ready),'source_sha256':file_hash(source),'device':meta[1]['device'],
        'deterministic_cuda':meta[1]['deterministic_cuda'],'same_world_execution_speedup':speedup,'world_size':meta[1]['world_size'],
        'encoder_graph':bool((meta[1].get('execution_profile') or {}).get('encoder_graph')),
        'precision_policy':meta[1].get('precision_policy','legacy'),
        'mean_seconds_reference':times[0]/2,'mean_seconds_candidate':times[1]/2,
        'compared_updates':3,'timed_updates_excluding_first_audit':2,
        'candidate_step':runs[1][-1]['optimizer_step'],
        'scope':'Short adoption gate, first audit update excluded from speed. Requires continued live throughput observation; not full-epoch performance or AUC.'}
    save_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','reference','candidate','ready','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--chunk',type=int,required=True);a=p.parse_args()
    result=assess(a.source,a.reference,a.candidate,a.ready,a.chunk,a.output)
    print(json.dumps(result));raise SystemExit(0 if result['state']=='passed' else 2)
