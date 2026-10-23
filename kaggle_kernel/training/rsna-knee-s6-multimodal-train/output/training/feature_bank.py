"""Encode once; compare trainable Study heads; keep deployable image checkpoints."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import UID,LABELS,file_hash,save_json,seed_all,to_device
from .native_data import dataset_for
from .data import collate_studies
from .model import KneeModel,ModelConfig
from .supervision import targets_from_row
from .recipes import input_contract,input_code_hashes

ENCODER_PREFIXES=('backbone.','spatial.','norm.')


def build(manifest,config,output):
    if config.get('views')=='global_anatomic':raise ValueError('anatomical crop comparisons require the image-level input and proposal contract')
    if any(config['model'].get(k) for k in ('spatial_3d','late_3d','anatomy_pool','anatomy_auxiliary','presence_head','auxiliary_head','clinical_head','local_evidence')):
        raise ValueError('pooled feature banks cannot implement image-level spatial or auxiliary supervision')
    from .engine import loader,autocast,atomic_checkpoint
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_all(config.get('seed',42));torch.set_num_threads(config.get('cpu_threads',4))
    contract={'format':'input_bound_feature_bank_v2','manifest_sha256':file_hash(manifest),'model':config['model'],
              **input_contract(config),'input_code_hashes':input_code_hashes(),
              'weights_sha256':file_hash(config['model']['pretrained_file']),
              'encoder_code_sha256':file_hash(Path(__file__).with_name('model.py')),
              'input_code_sha256':file_hash(Path(__file__).with_name('native_data.py'))}
    existing=output/'manifest.json'
    if existing.exists():
        old=json.loads(existing.read_text(encoding='utf-8'))
        # Older banks only supported the fixed per-Series percentile window.
        # Preserve that known default while rejecting reuse for another policy.
        old_contract={**old['contract'],'intensity_window_policy':old['contract'].get('intensity_window_policy','series_percentile')}
        if old_contract!=contract:raise ValueError('feature producer changed; choose a new bank directory')
        if all(Path(s['path']).is_file() for s in old['studies']):return old
    device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    model=KneeModel(ModelConfig(**config['model']),pretrained=True).to(device).eval()
    encoder={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(ENCODER_PREFIXES)}
    atomic_checkpoint(output/'encoder.pt',encoder)
    data=dataset_for(manifest,None,config);rows=[];started=time.perf_counter()
    with torch.no_grad():
        for n,cpu in enumerate(loader(data,1,config.get('workers',2))):
            batch=to_device(cpu,device)
            with autocast(device):values,glob=model.encode(batch['images'],batch['slots'])
            path=output/(f'{n:06d}.npz');tmp=path.with_suffix('.tmp')
            payload={key:cpu[key][0].numpy() for key in ('slots','series','position','geometry','view_ids','original_series','source_depth') if key in cpu}
            storage=np.float16 if values.dtype in (torch.float16,torch.bfloat16) else np.float32
            payload.update(values=values[0].float().cpu().numpy().astype(storage),
                           global_features=glob[0].float().cpu().numpy().astype(storage))
            with tmp.open('wb',buffering=8*2**20) as handle:np.savez_compressed(handle,**payload)
            tmp.replace(path)
            rows.append({UID:cpu[UID][0],'path':str(path.resolve()),'series_names':cpu['series_names'][0],
                         'fallback':cpu['fallback'][0]})
            if n%10==0 or n+1==len(data):
                status={'phase':'encoding','processed':n+1,'expected':len(data),'seconds':time.perf_counter()-started}
                save_json(output/'progress.json',status);print(json.dumps(status),flush=True)
    result={'contract':contract,'config':{'size':config['model']['image_size']},'studies':rows,
            'feature_dtype':str(values.dtype),
            'encoder':str((output/'encoder.pt').resolve()),'encoder_sha256':file_hash(output/'encoder.pt'),
            'source_image_manifest':str(Path(manifest).resolve()),
            'representation_training':'ImageNet initialization only; no knee labels or fitted knee encoder',
            'seconds':time.perf_counter()-started}
    save_json(existing,result);save_json(output/'progress.json',{'phase':'complete','studies':len(rows),'seconds':result['seconds']})
    return result


class FeatureStudyDataset(Dataset):
    def __init__(self,manifest,labels,*,ids=None):
        self.manifest_path=Path(manifest);self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        index={s[UID]:s for s in self.manifest['studies']};self.ids=list(index) if ids is None else list(map(str,ids))
        self.studies=[index[u] for u in self.ids];self.size=self.manifest['config']['size'];self.epoch=0
        self.labels=labels.set_index(UID)

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        i=index[0] if isinstance(index,tuple) else index;s=self.studies[i]
        with np.load(s['path'],allow_pickle=False) as arrays:
            payload={k:torch.from_numpy(arrays[k].copy()) for k in ('slots','series','position','geometry','view_ids','original_series','source_depth') if k in arrays}
            payload['encoded_values']=torch.from_numpy(arrays['values'].astype(np.float32))
            payload['encoded_global']=torch.from_numpy(arrays['global_features'].astype(np.float32))
        if self.manifest.get('feature_dtype')=='torch.bfloat16':
            payload['encoded_values']=payload['encoded_values'].to(torch.bfloat16)
            payload['encoded_global']=payload['encoded_global'].to(torch.bfloat16)
        row=self.labels.loc[s[UID]];n=len(payload['slots'])
        payload.update(targets_from_row(row))
        payload.update(images=torch.zeros(n,3,1,1),target=torch.from_numpy(np.nan_to_num(row[LABELS].to_numpy(np.float32))),
                       weight=torch.from_numpy(row[[x+'__weight' for x in LABELS]].to_numpy(np.float32)),
                       series_names=s['series_names'],fallback=s['fallback'])
        payload[UID]=s[UID];return payload


def attach_encoder(model,manifest,config):
    if any(config['model'].get(k) for k in ('spatial_3d','anatomy_pool','anatomy_auxiliary','presence_head','auxiliary_head','clinical_head','local_evidence')):
        raise ValueError('this mechanism must update image features; pooled bank experiment is invalid')
    bank=json.loads(Path(manifest).read_text(encoding='utf-8'));saved=bank['contract']['model']
    for key in ('backbone','image_size','spatial_hidden','spatial_pool','input_normalization'):
        defaults={'spatial_pool':'gated','input_normalization':'imagenet'}
        if saved.get(key,defaults.get(key))!=config['model'].get(key,defaults.get(key)):
            raise ValueError('frozen encoder architecture/pooling/normalization differs')
    for key,default in [('views','global_local'),('sampling','cover'),('local_mm',100.)]:
        if config.get(key,default)!=bank['contract'][key]:raise ValueError('feature/image input recipe differs')
    if config.get('source_input_convention','canonical')!=bank['contract'].get('source_input_convention','canonical'):
        raise ValueError('feature/image source input convention differs')
    if config.get('intensity_window_policy','series_percentile')!=bank['contract'].get('intensity_window_policy','series_percentile'):
        raise ValueError('feature/image intensity window policy differs')
    if config.get('exact_duplicate_policy','keep')!=bank['contract'].get('exact_duplicate_policy','keep'):
        raise ValueError('feature/image exact duplicate policy differs')
    if config.get('coronal_bilateral_context',False)!=bank['contract'].get('coronal_bilateral_context',False):
        raise ValueError('feature/image coronal peripheral views differ')
    if bank['contract'].get('input_code_hashes')!=input_code_hashes():
        raise ValueError('feature/image input implementation differs; regenerate a versioned bank')
    if bank['contract'].get('encoder_code_sha256')!=file_hash(Path(__file__).with_name('model.py')):
        raise ValueError('feature/image encoder implementation differs; regenerate a versioned bank')
    if file_hash(bank['encoder'])!=bank['encoder_sha256']:raise ValueError('encoder checksum differs')
    state=torch.load(bank['encoder'],map_location='cpu',weights_only=True)
    incompatible=model.load_state_dict(state,strict=False)
    if incompatible.unexpected_keys or any(k.startswith(ENCODER_PREFIXES) for k in incompatible.missing_keys):
        raise ValueError('incomplete frozen encoder state')
    for name,param in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):param.requires_grad_(False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.manifest,json.loads(Path(a.config).read_text(encoding='utf-8')),a.output)
