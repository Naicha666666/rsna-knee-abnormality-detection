"""Compare real same-host 1/2/4-GPU records. Missing evidence is a failure."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from ..common import save_json


def compare_audit(one,four,output):
    a=torch.load(Path(one)/'first_gradient.pt',map_location='cpu',weights_only=True)
    b=torch.load(Path(four)/'first_gradient.pt',map_location='cpu',weights_only=True)
    if set(a)!=set(b):raise ValueError('Gradient parameter sets differ')
    delta=base=0.;unused=[]
    for key in a:
        if (a[key] is None)!=(b[key] is None):unused.append(key);continue
        if a[key] is not None:
            delta+=float((a[key].double()-b[key].double()).square().sum());base+=float(a[key].double().square().sum())
    f=[json.loads((Path(d)/'first_forward.json').read_text(encoding='utf-8')) for d in (one,four)]
    alignment=len(f[0])==len(f[1]) and all({k:r[k] for k in ('ordinal','uid','draw','windows')}=={k:s[k] for k in ('ordinal','uid','draw','windows')} for r,s in zip(*f))
    logit_error=max((float(np.abs(np.asarray(r['logits'])-s['logits']).max()) for r,s in zip(*f)),default=math.inf)
    relative=math.sqrt(delta/max(base,1e-30))
    runs=[json.loads((Path(d)/'run.json').read_text(encoding='utf-8')) for d in (one,four)]
    source_match=runs[0]['source_sha256'] is not None and runs[0]['source_sha256']==runs[1]['source_sha256'] and runs[0]['ready_sha256']==runs[1]['ready_sha256'] and runs[0].get('deterministic_cuda')==runs[1].get('deterministic_cuda')
    source_match=source_match and runs[0].get('precision_policy','legacy')==runs[1].get('precision_policy','legacy')
    result={'passed':bool(source_match and alignment and not unused and relative<=.005 and logit_error<=1e-4),
        'source_match':source_match,'exposure_alignment':alignment,'unused_gradient_mismatch':unused,
        'gradient_relative_l2':relative,'max_logit_absolute_error':logit_error,
        'thresholds':{'gradient_relative_l2':.005,'logit_absolute_error':1e-4},
        'scope':'One real global update only; no proof of identical long-run trajectory or AUC.'}
    save_json(output,result);return result


def read_run(path,warmup):
    path=Path(path);meta=json.loads((path/'run.json').read_text(encoding='utf-8'))
    if meta.get('audit_first_update'):raise ValueError('Gradient-audit runs cannot be throughput measurements')
    rows=[json.loads(line) for line in (path/'updates.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)<warmup+12:raise ValueError('At least 12 measured updates after warmup are required')
    if len({r['optimizer_step'] for r in rows})!=len(rows):raise ValueError('Use a new output directory for each acceptance run')
    if any(not math.isfinite(r['seconds']) or r['seconds']<=0 or r['studies']<=0 for r in rows):
        raise ValueError('Timing and complete-Study counts must be positive and finite')
    return meta,rows[warmup:]


def assess(one,two,four,output,warmup=2):
    runs=[read_run(p,warmup) for p in (one,two,four)];base=runs[0][0]
    failures=[]
    if not base['source_sha256']:failures.append('Acceptance needs the same full recovery checkpoint')
    for (meta,rows),world in zip(runs,(1,2,4)):
        if meta['world_size']!=world:failures.append('Wrong world size')
        for key in ('source_sha256','ready_sha256','hostname','config','torch','cuda','device'):
            if meta[key]!=base[key]:failures.append('Run mismatch: '+key)
        if meta.get('execution_profile_sha256')!=base.get('execution_profile_sha256'):
            failures.append('Different execution profiles cannot establish GPU-count scaling')
        if meta.get('deterministic_cuda')!=base.get('deterministic_cuda'):
            failures.append('Different CUDA numerical policies cannot establish GPU-count scaling')
        if [r['source_exposures'] for r in rows]!=[r['source_exposures'] for r in runs[0][1]]:
            failures.append('Different global batches; cannot report scaling')
        if [sum(v['windows'] for v in r['ranks']) for r in rows]!=[sum(v['windows'] for v in r['ranks']) for r in runs[0][1]]:
            failures.append('Different MRI window counts')
    summary=[]
    for meta,rows in runs:
        seconds=sum(r['seconds'] for r in rows);studies=sum(r['studies'] for r in rows)
        summary.append({'world':meta['world_size'],'studies_per_second':studies/seconds,
            'mean_update_seconds':seconds/len(rows),'p95_update_seconds':float(np.quantile([r['seconds'] for r in rows],.95)),
            'input_wait_fraction':sum(max(x['input_wait'] for x in r['ranks']) for r in rows)/seconds,
            'reduce_wait_fraction':sum(max(x['reduce_wait_seconds'] for x in r['ranks']) for r in rows)/seconds,
            'p95_local_max_mean':float(np.quantile([max(x['local_seconds'] for x in r['ranks'])/np.mean([x['local_seconds'] for x in r['ranks']]) for r in rows],.95)),
            'updates':len(rows)})
    speedup=summary[2]['studies_per_second']/summary[0]['studies_per_second']
    if speedup<3:failures.append('Four-GPU speedup below proposed 3.0x target')
    if summary[2]['input_wait_fraction']>.1:failures.append('Input wait exceeds 10%')
    if summary[2]['p95_local_max_mean']>1.15:failures.append('p95 rank imbalance exceeds 15%')
    if any(x['peak_allocated_gib']/x['device_total_gib']>.9 for r in runs[2][1] for x in r['ranks']):
        failures.append('Less than 10% GPU allocation headroom')
    result={'passed':not failures,'failures':failures,'four_gpu_speedup':speedup,'efficiency':speedup/4,
        'runs':summary,'scope':'Real fixed-global-batch updates, includes rank-0 snapshot stalls; excludes first-stage startup and full-epoch validation.',
        'long_run_gate':'Also require passed GPU gradient audit, full validation/OOF identity check, checkpoint resume and host memory reserve.'}
    save_json(output,result);return result


def assess_execution(reference,candidate,output,warmup=2):
    a,ar=read_run(reference,warmup);b,br=read_run(candidate,warmup);errors=[]
    if not a.get('source_sha256'):errors.append('Execution comparison requires a full recovery checkpoint')
    if a.get('execution_profile_sha256') or not b.get('execution_profile_sha256'):
        errors.append('Use the original execution as reference and the calibrated profile as candidate')
    if a.get('deterministic_cuda')!=b.get('deterministic_cuda'):errors.append('Different CUDA numerical policies')
    for key in ('source_sha256','ready_sha256','hostname','config','torch','cuda','device','world_size'):
        if a[key]!=b[key]:errors.append('Execution comparison differs: '+key)
    if [r['source_exposures'] for r in ar]!=[r['source_exposures'] for r in br]:errors.append('Different exposure schedule')
    if [sum(v['windows'] for v in r['ranks']) for r in ar]!=[sum(v['windows'] for v in r['ranks']) for r in br]:errors.append('Different image workload')
    if [r['studies'] for r in ar]!=[r['studies'] for r in br]:errors.append('Different complete-Study counts')
    ta=sum(r['seconds'] for r in ar);tb=sum(r['seconds'] for r in br);speedup=ta/tb
    if speedup<1.10:errors.append('Less than the proposed 10% full-update throughput improvement')
    if any(v['peak_allocated_gib']>.85*v['device_total_gib'] for r in br for v in r['ranks']):errors.append('PRO memory reserve exceeded')
    if any(v['peak_reserved_gib']>.94*v['device_total_gib'] for r in br for v in r['ranks']):errors.append('PRO allocator reservation leaves too little headroom')
    if any(not math.isfinite(v[k]) or v[k]<=0 for r in br for v in r['ranks']
           for k in ('peak_allocated_gib','peak_reserved_gib','device_total_gib')):errors.append('Invalid GPU memory measurements')
    result={'passed':not errors,'errors':errors,'same_gpu_count_execution_speedup':speedup,
        'reference_seconds':ta,'candidate_seconds':tb,'measured_updates':len(ar),
        'scope':'Identical complete global updates on the same host and GPU count; this is execution speedup, not card-count scaling or AUC.'}
    save_json(output,result);return result


def assess_trajectory(source,reference,candidate,output):
    """Compare already-produced full states; no additional training runs."""
    states=[torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            for p in (source,Path(reference)/'step.pt',Path(candidate)/'step.pt')]
    initial,a,b=states;errors=[];metrics={}
    if a['step']!=b['step'] or a['step']<=initial['step']:errors.append('Recovery steps differ or no updates occurred')
    if a['scheduler']!=b['scheduler']:errors.append('Scheduler states differ')
    for branch in ('model','ema'):
        if not (set(initial[branch])==set(a[branch])==set(b[branch])):
            errors.append('Parameter names differ: '+branch);continue
        error=change=0.;maximum=0.
        for name,start in initial[branch].items():
            left,right=a[branch][name],b[branch][name]
            if start.shape!=left.shape or left.shape!=right.shape:
                errors.append('Shape differs: '+branch+'/'+name);continue
            if not left.is_floating_point():
                if not torch.equal(left,right):errors.append('Nonfloat state differs: '+branch+'/'+name)
                continue
            x,y,z=start.double(),left.double(),right.double()
            if not all(torch.isfinite(v).all() for v in (x,y,z)):
                errors.append('Nonfinite state: '+branch+'/'+name);continue
            delta=y-z
            error+=float(delta.square().sum());change+=float((y-x).square().sum())
            if delta.numel():maximum=max(maximum,float(delta.abs().max()))
        relative=math.sqrt(error/max(change,1e-30))
        metrics[branch]={'difference_relative_to_reference_update':relative,'max_absolute_error':maximum}
        if relative>.03:errors.append('More than 3% relative update divergence: '+branch)
    if set(a['optimizer']['state'])!=set(b['optimizer']['state']):errors.append('Optimizer state entries differ')
    else:
        for key,left in a['optimizer']['state'].items():
            right=b['optimizer']['state'][key]
            if set(left)!=set(right):errors.append('Optimizer fields differ');continue
            for field,value in left.items():
                other=right[field]
                if isinstance(value,torch.Tensor):
                    if value.shape!=other.shape or not torch.isfinite(other).all():errors.append('Invalid optimizer state')
                    if field=='step' and not torch.equal(value,other):errors.append('Optimizer step differs')
    result={'passed':not errors,'errors':errors,'steps':{'source':initial['step'],'reference':a['step'],'candidate':b['step']},
        'metrics':metrics,'threshold':.03,
        'scope':'Short trajectory from existing timing runs, including EMA; not proof of bitwise equivalence or unchanged AUC.'}
    save_json(output,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--one',required=True);p.add_argument('--four',required=True)
    p.add_argument('--two');p.add_argument('--output',required=True);p.add_argument('--gradient-audit',action='store_true')
    a=p.parse_args()
    if a.gradient_audit:result=compare_audit(a.one,a.four,a.output)
    else:
        if not a.two:p.error('--two is required for scaling acceptance')
        result=assess(a.one,a.two,a.four,a.output)
    print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
