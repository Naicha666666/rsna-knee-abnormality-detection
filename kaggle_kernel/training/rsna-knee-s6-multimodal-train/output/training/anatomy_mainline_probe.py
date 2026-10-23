"""Show real anatomical input changing the final readout, with exact zero-quality fallback."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,save_json,to_device,seed_all,file_hash
from .recipes import configuration
from .model import KneeModel,ModelConfig
from .native_data import dataset_for
from .data import collate_studies
from .engine import autocast,initialize_shared_heads,checkpoint_payload,atomic_checkpoint,bce_numerator,normalize_loss
from .partial_joint import from_batch,mass_vector
from .mcl_joint import evidence_from_batch
from .local_oa_supervision import attach
from .supervision import supervision_loss
from .construction_probe import parameter_hash


def run(root='F:/rsnaknee'):
    root=Path(root);out=root/'analysis/orthofoundation_anatomy_20261001';out.mkdir(parents=True,exist_ok=True)
    cfg=configuration('orthofoundation_anatomy',root,stage='finetune');cfg['model']['encode_chunk']=2
    weights=Path(cfg['anatomy_weights']);assets=weights.parent.parent/'kneexnet'
    cfg['anatomy_source_binding']={'weights_sha256':file_hash(weights),
        'asset_sha256':{n:file_hash(assets/n) for n in ('localizer.pt','best_model_256_noise_notebook_5_0.15.pth')},
        'atlas_sha256':file_hash(Path(__file__).with_name('data')/'ligament_atlas.json')}
    current=json.loads((root/'input_ready/current.json').read_text(encoding='utf-8'))
    package=json.loads(Path(current['package_path']).read_text(encoding='utf-8'));variant=package['variants']['anatomy']
    frame=pd.read_csv(variant['labels']);frame,support=attach(frame,Path(package['labels_source']).with_name('evidence.jsonl'))
    manifest=json.loads(Path(variant['manifest']).read_text(encoding='utf-8'));studies={s[UID]:s for s in manifest['studies']}
    rows=json.loads((root/'anatomy_v3/manifest.json').read_text(encoding='utf-8'))['series'];quality={u:np.zeros(12) for u in studies}
    for r in rows:quality[r[UID]]=np.maximum(quality[r[UID]],r['region_quality'])
    dev=set(frame.loc[frame.development_only,UID]);ordered=sorted(dev,key=lambda u:(sum(len(s['positions']) for s in studies[u]['series']),u))
    uid=next(u for u in ordered if quality[u].max()>.1);empty=next(u for u in ordered if quality[u].max()==0)
    coverage={'studies':len(studies),'with_any_region':sum(q.max()>0 for q in quality.values()),
        'per_label':{c:sum(q[j]>0 for q in quality.values()) for j,c in enumerate(LABELS)},
        'scope':'Nonzero producer quality, not independently adjudicated region accuracy.'}
    coverage={k:int(v) if isinstance(v,np.integer) else v for k,v in coverage.items()}
    coverage['per_label']={k:int(v) for k,v in coverage['per_label'].items()};save_json(out/'region_coverage.json',coverage)
    seed_all(42);torch.set_num_threads(3);device=torch.device('cuda');start=time.perf_counter()
    dataset=dataset_for(variant['manifest'],frame,cfg,ids=[uid,empty],training=False)
    b=to_device(collate_studies([dataset[0]]),device)
    b['anatomy_regions'].requires_grad_(True)
    model=KneeModel(ModelConfig(**cfg['model']),pretrained=True);initialize_shared_heads(model,42);model=model.to(device).eval()
    before=parameter_hash(model);zero={**b,'region_quality':torch.zeros_like(b['region_quality'])}
    with torch.no_grad(),autocast(device):
        actual=model(b);masked=model(zero)
        model.config.anatomy_pool=False;reference=model(zero);model.config.anatomy_pool=True
    change=float((actual-masked).abs().max());fallback=float((masked-reference).abs().max())
    if change<=1e-6:raise ValueError('Real regions did not affect final scores')
    if fallback!=0:raise ValueError('Zero-quality regions changed the image-only fallback')
    model.train();model.zero_grad(set_to_none=True)
    den={'studies':1,'criterion':b['weight'].sum(0),'presence':b['presence_weight'].sum(0),'factors':b['factors_weight'].sum(0),
        'partial_joint':mass_vector(from_batch(b)),'mcl_joint':evidence_from_batch(b)['mass'].sum()}
    with autocast(device,training=True):
        output=model(b,return_training=True);loss,parts=supervision_loss(output,b,den,cfg)
        loss=loss+normalize_loss(bce_numerator(output['logits'],b['target'],b['weight']),den['criterion'])
    loss.backward();region_gradient=float(b['anatomy_regions'].grad.abs().sum());readout_gradient=float(model.map_readout.weight.grad.abs().sum())
    if not(region_gradient>0 and readout_gradient>0):raise ValueError('Anatomy did not reach the supervised model')
    model.zero_grad(set_to_none=True);del output,loss,b;model.eval()
    eb=to_device(collate_studies([dataset[1]]),device)
    with torch.no_grad(),autocast(device):ep=model(eb)
    if not torch.isfinite(ep).all() or eb['region_quality'].count_nonzero():raise ValueError('Real zero-region Study was not retained correctly')
    after=parameter_hash(model)
    if after!=before:raise ValueError('Diagnostic changed model parameters')
    checkpoint=out/'untrained.pt';atomic_checkpoint(checkpoint,checkpoint_payload(model,config=cfg,epoch=-1,
        provenance={'fold':'diagnostic','smoke':True,'train_ids':[],'validation_ids':[]},optimizer_steps=0))
    record={UID:uid,'actual_quality':quality[uid].tolist(),'final_logit_change_from_regions':change,
        'zero_quality_fallback_max_error':fallback,'region_input_gradient_l1':region_gradient,'region_readout_gradient_l1':readout_gradient,
        'real_no_region_study':empty,'real_no_region_finite_output':True,'parameter_hash_before':before,'parameter_hash_after':after,
        'optimizer_steps':0,'checkpoint':str(checkpoint),'seconds':time.perf_counter()-start,
        'scope':'Input consumption and gradient proof on real development MRIs. Does not demonstrate AUC improvement.'}
    save_json(out/'consumer_results.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':run()
