"""Bounded-disk, offline inference for the selected whole-Study mainline."""
import argparse,gc,json,shutil,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import UID,LABELS,file_hash,save_json,to_device
from .recipes import input_contract


def mean_predictions(frames,ids):
    from .ensemble import probabilities
    if not frames or len(ids)!=len(set(ids)):raise ValueError('Invalid prediction inputs')
    arrays=[]
    for frame in frames:
        probabilities(frame)
        if set(frame[UID])!=set(ids):raise ValueError('Fold prediction coverage differs')
        arrays.append(frame.set_index(UID).loc[ids,LABELS].to_numpy(float))
    result=pd.DataFrame(np.mean(arrays,axis=0),columns=LABELS);result.insert(0,UID,ids)
    return result


def inspect_checkpoints(paths,diagnostic=False):
    if len(paths)!=(1 if diagnostic else 5):raise ValueError('Delivery requires five fitted folds; diagnostics require one explicit unfitted checkpoint')
    entries=[];folds=[];contracts=[];architectures=[]
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if state['labels']!=LABELS:raise ValueError('Checkpoint labels differ')
        model=state['model_config'];cfg=state['train_config'];prov=state['provenance']
        if model.get('backbone_position_mode')!='orthofoundation_public_no_rope' or not model.get('partial_joint') or not model.get('mcl_joint'):
            raise ValueError('Checkpoint is not the selected OrthoFoundation mainline')
        if any(model.get(k) for k in ('anatomy_auxiliary','spatial_3d','late_3d','local_evidence')):
            raise ValueError('This delivery entrypoint supports global inputs and quality-weighted region pooling')
        if model.get('anatomy_pool') and not cfg.get('anatomy_source_binding'):raise ValueError('Anatomy assets must be hash-bound before deployment')
        if not diagnostic and (state.get('optimizer_steps',0)<=0 or prov.get('smoke')):raise ValueError('Unfitted/smoke models cannot produce a competition submission')
        if diagnostic and state.get('optimizer_steps',0)!=0:raise ValueError('Diagnostic scope must be explicit')
        folds.append(str(prov.get('fold')));contracts.append(input_contract(cfg))
        architectures.append({k:v for k,v in model.items() if k not in ('pretrained_file','encode_chunk')})
        entries.append({'path':str(Path(path).resolve()),'sha256':file_hash(path),'fold':folds[-1],
                        'optimizer_steps':state.get('optimizer_steps',0),'train_config':cfg,'anatomy_pool':model.get('anatomy_pool',False)})
        del state
    if not diagnostic and set(folds)!=set('01234'):raise ValueError('Five distinct outer folds are required')
    if any(c!=contracts[0] for c in contracts) or any(m!=architectures[0] for m in architectures):raise ValueError('Fold input/model contracts differ')
    return entries


def run(data,checkpoints,output,chunk_studies=8,diagnostic=False,anatomy_weights=None):
    from .native_cache import prepare
    from .native_data import dataset_for
    from .data import collate_studies
    from .model import KneeModel,ModelConfig
    from .engine import load_checkpoint_model,autocast
    if chunk_studies<1:raise ValueError('chunk_studies must be positive')
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    entries=inspect_checkpoints(checkpoints,diagnostic);ids=pd.read_csv(data/'test.csv',dtype={UID:str})[UID].tolist()
    if len(ids)!=len(set(ids)):raise ValueError('Test Study IDs are not unique')
    scratch=(output/('decode_'+uuid.uuid4().hex)).resolve();scratch.mkdir()
    # Delete only this invocation's new directory, after verifying containment.
    if scratch.parent!=output.resolve():raise ValueError('Scratch outside output workspace')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(3)
    started=time.perf_counter();parts=[];failures=[];fallback=[];chunks=[]
    use_anatomy=entries[0]['anatomy_pool'];weights=None
    if use_anatomy:
        cfg=entries[0]['train_config'];weights=Path(anatomy_weights or cfg['anatomy_weights'])
        binding=cfg['anatomy_source_binding'];assets=weights.parent.parent/'kneexnet' if weights.parent.name=='skm_tea' else weights.parent/'kneexnet'
        if file_hash(weights)!=binding['weights_sha256']:raise ValueError('Deployment anatomy weights differ from training')
        for name,sha in binding['asset_sha256'].items():
            if file_hash(assets/name)!=sha:raise ValueError('Deployment anatomy asset differs: '+name)
        if file_hash(Path(__file__).with_name('data')/'ligament_atlas.json')!=binding['atlas_sha256']:raise ValueError('Anatomy atlas changed')
        if any(e['train_config']['anatomy_source_binding']!=binding for e in entries):raise ValueError('Fold anatomy source bindings differ')
    try:
        for start in range(0,len(ids),chunk_studies):
            selected=ids[start:start+chunk_studies];cache=scratch/f'chunk{start}'
            manifest=prepare(data,cache,split='test',study_ids=selected,source_directory=True)
            failures.extend({UID:s[UID],'errors':s['errors']} for s in manifest['studies'] if s['errors'])
            fallback.extend(s[UID] for s in manifest['studies'] if not s['series'])
            overrides={}
            if use_anatomy:
                from .geometry_bank import build as geometry_build
                from .anatomy import build as anatomy_build
                geometry_build(data,cache/'manifest.json',cache/'geometry',split='test',source_directory=True)
                anatomy_build(cache/'manifest.json',cache/'geometry',weights,cache/'anatomy',device=str(device))
                overrides={'geometry_dir':str(cache/'geometry'),'anatomy_dir':str(cache/'anatomy'),'anatomy_weights':str(weights)}
            frames=[]
            for entry in entries:
                state=torch.load(entry['path'],map_location='cpu',weights_only=True,mmap=True)
                cfg={**state['train_config'],**overrides,'load_pixel_supervision':False}
                dataset=dataset_for(cache/'manifest.json',None,cfg,ids=selected,training=False)
                model=KneeModel(ModelConfig(**state['model_config']),pretrained=False)
                load_checkpoint_model(model,state);del state
                model=model.to(device).eval();rows=[]
                with torch.inference_mode():
                    for i in range(len(dataset)):
                        batch=to_device(collate_studies([dataset[i]]),device)
                        with autocast(device):prob=model(batch).float().sigmoid().cpu().numpy()[0]
                        rows.append({UID:batch[UID][0],**dict(zip(LABELS,prob))})
                        del batch
                frames.append(pd.DataFrame(rows));del model,dataset;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
            parts.append(mean_predictions(frames,selected));chunks.append({'start':start,'studies':len(selected)})
            if cache.resolve().parent!=scratch:raise ValueError('Unexpected decode path')
            shutil.rmtree(cache)
        result=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=[UID,*LABELS])
        result=result.set_index(UID).loc[ids].reset_index()
        if (data/'sample_submission.csv').exists():
            sample=pd.read_csv(data/'sample_submission.csv',dtype={UID:str})
            if list(sample.columns)!=[UID,*LABELS] or set(sample[UID])!=set(ids):raise ValueError('Official sample submission schema differs')
            result=result.set_index(UID).loc[sample[UID]].reset_index()
        name='untrained_format_check.csv' if diagnostic else 'submission.csv'
        result.to_csv(output/name,index=False)
        save_json(output/'delivery_receipt.json',{'diagnostic':diagnostic,'optimizer_steps_per_checkpoint':[e['optimizer_steps'] for e in entries],
            'checkpoints':[{k:v for k,v in e.items() if k!='train_config'} for e in entries],
            'studies':len(result),'chunks':chunks,'fallback_ids':fallback,'decode_errors':failures,'anatomy_regions_used':use_anatomy,
            'input_contract':input_contract(entries[0]['train_config']),'output':str(output/name),'sha256':file_hash(output/name),
            'seconds':time.perf_counter()-started,'network_required':False,
            'scope':'Local input/output verification; diagnostic scores are unfitted. Kaggle runtime not measured.' if diagnostic else 'Fitted fold ensemble; Kaggle runtime must be measured on the actual platform.'})
        return result
    finally:
        if scratch.exists() and scratch.parent==output.resolve():shutil.rmtree(scratch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--chunk-studies',type=int,default=8);p.add_argument('--diagnostic-untrained',action='store_true');p.add_argument('--anatomy-weights')
    a=p.parse_args();run(a.data,a.checkpoints,a.output,a.chunk_studies,a.diagnostic_untrained,a.anatomy_weights)
