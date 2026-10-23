"""Real complete-Study forward/gradient diagnostics, WITHOUT optimizer updates."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,save_json,to_device,seed_all
from .recipes import configuration
from .model import KneeModel,ModelConfig
from .native_data import dataset_for
from .data import collate_studies
from .engine import autocast,bce_numerator,normalize_loss,initialize_shared_heads
from .supervision import supervision_loss


def parameter_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(root,labels_path,output,limit=3):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(42);torch.set_num_threads(4);device=torch.device('cuda');started=time.perf_counter()
    config=configuration('constructed',root,stage='finetune');config['workers']=0
    meta=json.loads((root/'cache_native/manifest.json').read_text(encoding='utf-8'));ids=[s[UID] for s in meta['studies'][:limit]]
    from .native_data import covering_centers
    largest=max(meta['studies'],key=lambda s:sum(len(covering_centers(len(x['positions']))) for x in s['series']))[UID]
    if largest not in ids:ids.append(largest)
    study_index={s[UID]:s for s in meta['studies']}
    labels=pd.read_csv(labels_path,dtype={UID:str})
    from .supervision import supported_factors
    labels,factor_support=supported_factors(labels,labels.loc[labels.fold.ge(0)&labels.fold.ne(0),UID])
    data=dataset_for(root/'cache_native/manifest.json',labels,config,ids=ids)
    model=KneeModel(ModelConfig(**config['model']),pretrained=True)
    initialize_shared_heads(model,42);model=model.to(device);before=parameter_hash(model);model.eval();results=[];predictions=[]
    with torch.no_grad():
        for n in range(len(data)):
            batch=to_device(collate_studies([data[n]]),device)
            with autocast(device):
                full=model(batch);zero={**batch,'region_quality':torch.zeros_like(batch['region_quality'])}
                no_anatomy=model(zero)
                model.config.spatial_3d=False
                no_spatial=model(batch)
                model.config.spatial_3d=True
                model.config.protocol_prior=False;no_prior=model(batch);model.config.protocol_prior=True
            p=full.sigmoid()[0].float().cpu().numpy();predictions.append({UID:ids[n],**dict(zip(LABELS,p))})
            results.append({UID:ids[n],'windows':int(batch['slots'].gt(0).sum()),
                'real_series':len(study_index[ids[n]]['series']),
                'anatomy_logit_max_delta':float((full-no_anatomy).abs().max()),
                'spatial_logit_max_delta':float((full-no_spatial).abs().max()),
                'protocol_prior_logit_max_delta':float((full-no_prior).abs().max()),
                'nonzero_region_weights':int(batch['region_quality'].gt(0).sum()),'all_outputs_finite':bool(torch.isfinite(full).all())})
            results[-1]['localized_windows']=int(batch['localized'].sum())
            results[-1]['max_crop_center_shift']=float((batch['crop_center']-.5).abs().max())
            print(json.dumps(results[-1]),flush=True)
    # Full study backpropagation reaches the pretrained visual encoder. No
    # optimizer is even constructed. Eval predictions above use every slice.
    model.train();model.zero_grad(set_to_none=True)
    batch=to_device(collate_studies([data[ids.index(largest)]]),device);torch.cuda.reset_peak_memory_stats()
    with autocast(device,training=True):
        result=model(batch,return_training=True)
        main=normalize_loss(bce_numerator(result['logits'],batch['target'],batch['weight']),batch['weight'].sum(0))
        denominators={'presence':batch['presence_weight'].sum(0),'factors':batch['factors_weight'].sum(0),'studies':1,
                      'criterion':batch['weight'].sum(0)}
        if config['model'].get('mcl_joint'):
            from .mcl_joint import evidence_from_batch
            denominators['mcl_joint']=evidence_from_batch(batch)['mass'].sum()
        extra,parts=supervision_loss(result,batch,denominators,config)
        loss=main+extra
    stem=next(p for n,p in model.named_parameters() if n.startswith('backbone.stem') and p.requires_grad)
    specific={}
    for name,only in [('presence',{'presence_weight':.2}),('clinical_factors',{'factor_weight':.15}),('tissues',{'anatomy_aux_weight':.05})]:
        separate,_=supervision_loss(result,batch,denominators,only)
        gradient=torch.autograd.grad(separate,stem,retain_graph=True,allow_unused=True)[0]
        specific[name]=float(gradient.abs().sum()) if gradient is not None else 0.
    loss.backward()
    gradients={}
    for prefix in ('backbone.stem','map_project','map_readout','spatial_context','tissue_decoder','presence','factor_classifier','within','across','classifier'):
        gradients[prefix]=sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
    after=parameter_hash(model)
    if before!=after:raise RuntimeError('diagnostic unexpectedly modified model state')
    evidence={'studies':results,'gradient_l1':gradients,'parameters_unchanged':before==after,
        'largest_cached_study':largest,'gradient_probe_windows':int(batch['slots'].gt(0).sum()),
        'loss_specific_backbone_gradient_l1':specific,
        'factor_support':factor_support,
        'parameter_sha256':before,'optimizer_steps':0,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'seconds':time.perf_counter()-started,'main_loss':float(main.detach()),
        'independent_losses':{k:float(v) for k,v in parts.items()},
        'scope':'complete real-image consumer and causal/gradient check; new branches untrained; no AUC or accuracy claim'}
    save_json(out/'evidence.json',evidence);pd.DataFrame(predictions).to_csv(out/'untrained_diagnostic_predictions.csv',index=False)
    # Diagnostic checkpoint for offline raw-DICOM deployment verification only.
    from .engine import checkpoint_payload,atomic_checkpoint
    atomic_checkpoint(out/'diagnostic.pt',checkpoint_payload(model,config=config,epoch=-1,
        provenance={'smoke':True,'train_ids':[],'validation_ids':[],'optimizer_steps':0,'scope':'untrained construction diagnostic'}))
    return evidence


if __name__=='__main__':
    from .criteria import labels_directory
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--labels',default=str(labels_directory('F:/rsnaknee')/'labels.csv'))
    p.add_argument('--output',default='F:/rsnaknee/construction_probe');p.add_argument('--limit',type=int,default=3)
    a=p.parse_args();run(a.root,a.labels,a.output,a.limit)
