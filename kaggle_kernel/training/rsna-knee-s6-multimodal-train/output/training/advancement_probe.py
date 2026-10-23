"""Exercise new consumers on real cached Studies without fitting any model."""
from dataclasses import replace
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from .common import UID,save_json
from .model import KneeModel,ModelConfig
from .feature_bank import FeatureStudyDataset
from .native_data import NativeStudyDataset
from .data import collate_studies
from .engine import initialize_shared_heads


def parameter_hash(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(root,output):
    root=Path(root);output=Path(output);torch.set_num_threads(4)
    bank=root/'features/pilot_global_local_100_gated_384/manifest.json'
    metadata=json.loads(bank.read_text());labels=pd.read_csv(root/'labels_clinical_v4_pretrain/labels.csv',dtype={UID:str})
    dataset=FeatureStudyDataset(bank,labels,ids=[metadata['studies'][0][UID]])
    batch=collate_studies([dataset[0]])
    results=[]
    for name,changes in [('lstm',{'context_type':'lstm'}),('score_lstm',{'context_type':'score_lstm'}),
                         ('score_summary',{'context_type':'none','score_summary_head':True}),
                         ('pairwise_series',{'context_type':'none','pairwise_series':True})]:
        cfg={**metadata['contract']['model'],**changes};torch.manual_seed(42)
        model=KneeModel(ModelConfig(**cfg),pretrained=False).eval();initialize_shared_heads(model,42)
        before=parameter_hash(model);start=time.perf_counter();logits=model(batch);logits.square().mean().backward()
        module=model.context if 'lstm' in name else model.score_summary if name=='score_summary' else model.pairwise
        grads=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        assert grads>0 and torch.isfinite(logits).all() and parameter_hash(model)==before
        results.append({'candidate':name,'real_study':batch[UID][0],'windows':int(batch['slots'].gt(0).sum()),
                        'gradient_abs_sum':grads,'parameter_hash_unchanged':True,'seconds':time.perf_counter()-start})
        del model
    manifest=root/'cache_native/manifest.json';native=json.loads(manifest.read_text())
    uid=native['studies'][0][UID]
    data=NativeStudyDataset(manifest,ids=[uid],training=True,image_size=64,context_curriculum_epochs=4)
    early=data[0];data.epoch=3;late=data[0];data.training=False;inference=data[0]
    assert len(early['slots'])<len(late['slots']) and len(late['slots'])==len(inference['slots'])
    dropped=NativeStudyDataset(manifest,ids=[uid],training=True,image_size=64,series_dropout=.95)[0]
    plane=lambda slot:0 if slot in (1,2,7) else 1 if slot in (3,4,8) else 2
    assert {plane(int(x)) for x in dropped['slots']}=={plane(int(x)) for x in inference['slots']}
    results.append({'candidate':'context_curriculum','early_windows':len(early['slots']),'final_windows':len(late['slots']),
                    'inference_windows':len(inference['slots']),'all_series_preserved':True})
    results.append({'candidate':'series_dropout','original_acquisitions':len(inference['original_series'].unique()),
                    'retained_acquisitions':len(dropped['original_series'].unique()),'all_planes_preserved':True})
    # Actual released medical encoder, as opposed to a random spatial backbone.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=NativeStudyDataset(manifest,ids=[uid],image_size=128,geometry_dir=root/'geometry')
    batch=collate_studies([data[0]]);batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    cfg=ModelConfig(backbone='resnet34',image_size=128,context_type='none',input_normalization='unit',
                    pretrained_file=str(root/'pretrained/kneexnet_resnet34_encoder.safetensors'),late_3d=True,encode_chunk=8)
    model=KneeModel(cfg,pretrained=True).to(device).eval();before=parameter_hash(model)
    with torch.no_grad():logits=model(batch)
    assert logits.shape==(1,12) and torch.isfinite(logits).all() and before==parameter_hash(model)
    results.append({'candidate':'medical_late3d','real_study':uid,'windows':int(batch['slots'].gt(0).sum()),
                    'pretrained_encoder_loaded':True,'parameter_hash_unchanged':True,'output_shape':list(logits.shape)})
    record={'optimizer_steps':0,'new_models_trained':False,'results':results,
            'scope':'real-input consumer/gradient checks only; new heads untrained; no AUC or localization-accuracy claim'}
    save_json(output/'evidence.json',record);print(json.dumps(record,indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--output',default='F:/rsnaknee/analysis/advancement_probe')
    a=p.parse_args();run(a.root,a.output)
