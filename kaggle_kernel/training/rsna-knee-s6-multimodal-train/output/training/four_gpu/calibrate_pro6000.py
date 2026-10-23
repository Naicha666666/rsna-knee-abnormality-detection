"""Memory/recompute calibration on full training MRIs, with zero optimizer steps.

Requires the actual 96-GB target GPU. The result is only a candidate until the
normal full-update gradient, resume and throughput commissioning passes.
"""
import argparse,gc,json,math,os,time
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from ..common import UID,file_hash,save_json,to_device
from ..model import ModelConfig
from ..data import augment_series,collate_studies
from ..engine import EMA,autocast,bce_numerator,normalize_loss
from ..native_data import dataset_for
from .train import validate_ready,validate_resume,check_stochastic,prepared_dataset
from .input_identity import verify_receipts,digest_memo
from .recipe import frames,denominators,optimizer_for,descriptor
from .activation_policy import ActivationPolicy,fit_policy,content_binding,implementation_binding,GIB
from .retained_model import RetainedKneeModel,REFERENCE_ENCODER_SHA256
from .evidence_cache import EvidenceBank,cached_supervision_loss
from .precision import training_autocast


def hardware(device):
    p=torch.cuda.get_device_properties(device)
    return {'name':p.name,'total_memory_bytes':p.total_memory,'capability':list(torch.cuda.get_device_capability(device)),
            'torch':torch.__version__,'cuda':torch.version.cuda}


def gradient_snapshot(model):
    return {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}


def compare(model,out,loss,reference):
    delta=den=0.;unused=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        old=reference['gradients'][name]
        if (old is None)!=(p.grad is None):unused.append(name);continue
        if old is None:continue
        value=p.grad.detach().cpu().double();r=old.double()
        if not torch.isfinite(value).all() or not torch.isfinite(r).all():
            return {'passed':False,'reason':'nonfinite gradient','parameter':name}
        delta+=float((value-r).square().sum());den+=float(r.square().sum())
    error=float((out['logits'].detach().float().cpu()-reference['logits']).abs().max())
    relative=math.sqrt(delta/max(den,1e-30));loss_error=abs(float(loss.detach())-reference['loss'])
    return {'passed':not unused and error<=1e-4 and relative<=.005 and loss_error<=.005*max(abs(reference['loss']),1e-6),
        'max_logit_error':error,'gradient_relative_l2':relative,'loss_error':loss_error,'unused_mismatch':unused}


def trial(model,cpu,bank,cfg,policy,device,seed,reference=None,capture=False,precision_policy='legacy'):
    model.activation_policy=policy;model.measure_saved_tensors=True;model.train()
    torch.cuda.synchronize(device);model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    # Include the gradient buffers that will already exist after the first
    # Study in a global accumulated update. No optimizer step is executed.
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.zeros_like(p)
    torch.cuda.manual_seed(seed)
    b=out=loss=extra=None
    try:
        torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
        b=to_device(cpu,device)
        if cfg.get('augment',True):b=augment_series(b,intensity_policy=cfg['intensity_augmentation_policy'])
        uid=cpu[UID][0];den=denominators(bank.frame,[uid],device,bank)
        with training_autocast(device,precision_policy):
            out=model(b,return_training=True)
            extra,_=cached_supervision_loss(out,b,den,cfg,bank.case(uid))
            loss=normalize_loss(bce_numerator(out['logits'],b['target'],b['weight']),den['criterion'])+extra
        loss.backward();torch.cuda.synchronize(device);elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite diagnostic loss')
        stats={k:v for k,v in model.execution_stats.items() if k!='retained_starts'}
        result={'state':'ok','uid':uid,'windows':int(cpu['slots'].gt(0).sum()),'seconds':elapsed,
            'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'policy':policy.export(),'execution':stats}
        if reference is not None:result['parity']=compare(model,out,loss,reference)
        snapshot={'logits':out['logits'].detach().float().cpu().clone(),'loss':float(loss.detach()),
                  'gradients':gradient_snapshot(model)} if capture else None
        return result,snapshot
    except torch.cuda.OutOfMemoryError:
        return {'state':'oom','uid':cpu[UID][0],'policy':policy.export()},None
    finally:
        del b,out,loss,extra
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('Calibration requires the target GPU')
    torch.cuda.set_device(args.device);device=torch.device('cuda',args.device);hw=hardware(device)
    if 'PRO 6000' not in hw['name'].upper() or hw['total_memory_bytes']<88*GIB:
        raise RuntimeError('Use a full 96-GB RTX PRO 6000, not a MIG slice or a different GPU')
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'profile.json').exists():raise FileExistsError('Use a fresh calibration directory')
    ready=validate_ready(args.ready);runtime=Path(args.ready).parent;entry=ready['entries'][0];cfg=entry['config']
    frame,_=frames(entry,ready['evidence']);plans=json.loads((runtime/'duplicate_plans.json').read_text())
    receipts=json.loads((runtime/'verified_files.json').read_text());verify_receipts(receipts)
    data=prepared_dataset(entry,frame,entry['training_ids'],runtime,plans,receipts,training=True,shared_digest_memo=digest_memo(receipts))
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    validate_resume(state,ready['original_entries'][0],ready);data.epoch=state['epoch']
    costs=[descriptor(s,plans[s[UID]]['representative_indices'],cfg,i,0,data.epoch)['windows'] for i,s in enumerate(data.studies)]
    order=sorted(range(len(data)),key=lambda i:(costs[i],i))
    selected=list(dict.fromkeys(order[int(q*(len(order)-1))] for q in (.25,.70,1.)))
    weights=np.bincount([min(range(len(selected)),key=lambda j:abs(c-costs[selected[j]])) for c in costs],minlength=len(selected))/len(costs)
    reference_policy=ActivationPolicy(chunk=2,max_windows=max(costs),memory_limit_bytes=0)
    model=RetainedKneeModel(ModelConfig(**cfg['model']),pretrained=False,activation_policy=reference_policy).to(device)
    model.load_state_dict(state['model']);check_stochastic(model)
    optimizer,scheduler=optimizer_for(model,cfg,len(data));optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    ema=EMA(model);ema.model.load_state_dict(state['ema']) # Real persistent footprint, never updated.
    source_step=state['step'];del state;gc.collect()
    bank=EvidenceBank.from_frame(frame,entry['training_ids'],device);bank.frame=data.labels
    batches=[collate_studies([data[(i,0)]]) for i in selected]
    report={'format':'pro6000_activation_calibration_v1','hardware':hw,'source_step':source_step,
        'optimizer_steps_performed':0,'cases':[{'uid':data.ids[i],'windows':costs[i],'weight':float(w)} for i,w in zip(selected,weights)],
        'rows':[],'candidates':[],'note':'Complete MRI forward/loss/backward; measured with loaded Adam and EMA, no optimizer steps.'}
    def record(row):
        report['rows'].append(row);save_json(outdir/'calibration.json',report)
        print(json.dumps({k:row[k] for k in ('state','windows','seconds','peak_allocated_bytes','execution','parity') if k in row}),flush=True)
    # Warm kernels outside ranking. Subsequent reference/candidate input bytes,
    # augmentation seed, weights and numerical precision are identical.
    warm,_=trial(model,batches[0],bank,cfg,reference_policy,device,9123);record({**warm,'phase':'warmup'})
    references=[];reference_rows=[]
    for j,cpu in enumerate(batches):
        row,ref=trial(model,cpu,bank,cfg,reference_policy,device,9123+j,capture=True)
        if row['state']!='ok':raise RuntimeError('The original complete-Study execution does not fit this host')
        record({**row,'phase':'reference'});references.append(ref);reference_rows.append(row)
    for chunk in args.chunks:
        baseline=ActivationPolicy(chunk=chunk,max_windows=max(costs),memory_limit_bytes=0)
        warmed,_=trial(model,batches[0],bank,cfg,baseline,device,9123);record({**warmed,'phase':'chunk_warmup'})
        if warmed['state']!='ok':continue
        base_rows=[];valid=True
        for j,cpu in enumerate(batches):
            row,_=trial(model,cpu,bank,cfg,baseline,device,9123+j,references[j]);record({**row,'phase':'chunk_baseline'})
            if row['state']!='ok' or not row['parity']['passed']:valid=False;break
            base_rows.append(row)
        if not valid:continue
        # Probe at least 32 windows so fixed autocast weight storage is not
        # incorrectly charged once per tiny two-window chunk. Still use the
        # complete largest Study, never a cropped/smaller substitute workload.
        probe=ActivationPolicy(chunk=chunk,max_windows=max(costs),memory_limit_bytes=hw['total_memory_bytes'],retained_window_cap=max(32,chunk))
        row,_=trial(model,batches[-1],bank,cfg,probe,device,9123+len(batches)-1,references[-1]);record({**row,'phase':'activation_cost'})
        if row['state']!='ok' or not row['parity']['passed'] or row['execution']['retained_windows']==0:continue
        cost=row['execution']['saved_activation_bytes']/row['execution']['retained_windows']
        policy=fit_policy(base_rows,cost,hw['total_memory_bytes'],chunk)
        # At most three bounded reductions after an OOM/resource rejection.
        for attempt in range(3):
            measured=[];okay=True
            for j,cpu in enumerate(batches):
                row,_=trial(model,cpu,bank,cfg,policy,device,9123+j,references[j]);record({**row,'phase':'retained_candidate','attempt':attempt})
                if row['state']!='ok' or not row['parity']['passed'] or row['peak_allocated_bytes']>policy.memory_limit_bytes or row['peak_reserved_bytes']>hw['total_memory_bytes']*.94:
                    okay=False;break
                measured.append(row)
            if okay:
                seconds=sum(float(w)*r['seconds'] for w,r in zip(weights,measured))
                report['candidates'].append({'policy':policy.export(),'weighted_seconds':seconds,
                    'retained_windows':[r['execution']['retained_windows'] for r in measured],
                    'cost_observations':[{'windows':r['windows'],'checkpointed_windows':r['execution']['checkpointed_windows'],'seconds':r['seconds']} for r in [*base_rows,*measured]]})
                break
            if row['state']=='ok' and not row['parity']['passed']:break # Never relax numerical gates.
            current=max(policy.choose(int(cpu['slots'].gt(0).sum()))['retained_windows'] for cpu in batches)
            policy=ActivationPolicy(**{**policy.export(),'retained_window_cap':current//2})
    baseline_seconds=sum(float(w)*r['seconds'] for w,r in zip(weights,reference_rows))
    viable=[c for c in report['candidates'] if any(c['retained_windows']) and c['weighted_seconds']<baseline_seconds]
    if not viable:
        report['status']='no_accepted_faster_candidate';save_json(outdir/'calibration.json',report)
        raise RuntimeError('No safe faster retention candidate; long training remains blocked')
    chosen=min(viable,key=lambda c:c['weighted_seconds'])
    from scipy.optimize import nnls
    observations=chosen['cost_observations']
    x=np.array([[1.,r['windows'],r['checkpointed_windows']] for r in observations]);y=np.array([r['seconds'] for r in observations])
    coefficients,residual=nnls(x,y)
    profile={'format':'pro6000_activation_profile_v1','calibration_passed':True,'hardware':hw,'policy':chosen['policy'],
        'implementation_binding':implementation_binding(),
        'scheduling_cost':{'coefficients':coefficients.tolist(),'fit_residual':float(residual),
            'fallback_seconds_per_window':max(r['seconds']/r['windows'] for r in reference_rows),
            'units':'seconds ~ overhead + windows + recomputed windows; heuristic for whole-Study rank balancing'},
        'input_binding':content_binding(ready),'source_checkpoint_sha256':file_hash(args.checkpoint),'source_step':source_step,
        'reference_encoder_sha256':REFERENCE_ENCODER_SHA256,'calibration_report_sha256':None,
        'requires_full_update_commissioning':True,'backward_model_batch':'complete Study, global 24 exposures unchanged',
        'numerical_thresholds':{'logit_max':1e-4,'gradient_relative_l2':.005,'relative_loss':.005}}
    report.update(status='candidate_selected',reference_weighted_seconds=baseline_seconds,chosen=chosen,
                  full_update_speedup_measured=False,optimizer_steps_performed=0)
    save_json(outdir/'calibration.json',report);profile['calibration_report_sha256']=file_hash(outdir/'calibration.json')
    save_json(outdir/'profile.json',profile)
    print('Candidate saved; full-update commissioning is still required.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ready',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--chunks',nargs='+',type=int,default=[2,8,16,32]);run(p.parse_args())
