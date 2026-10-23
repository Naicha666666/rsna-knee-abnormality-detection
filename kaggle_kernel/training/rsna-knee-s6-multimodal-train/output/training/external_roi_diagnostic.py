"""KneeMRI expert-ROI conditional diagnosis; explicitly uses privileged boxes.

Compare with external_capability full-field scores to diagnose missing location
or scale support. This is neither a deployable pipeline nor an absolute upper
bound. It also changes context/resolution; do not attribute every delta to ROI.
"""
import argparse,json,time
from pathlib import Path
import gc
import numpy as np
import torch

from .common import save_json,file_hash
from .external_capability import full_volume_batch,endpoint_metrics
from .kneemri_clinical_reference import KneeMRIClinicalReference
from .model import KneeModel,ModelConfig


def run(root,checkpoint,full_result,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    reference=json.loads((Path(full_result)/'manifest.json').read_text(encoding='utf-8'))
    records=[r for r in reference['records'] if r['source']=='KneeMRI']
    torch.set_num_threads(4);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ModelConfig(**state['model_config']);cfg.encode_chunk=4
    model=KneeModel(cfg,pretrained=False);model.load_state_dict(state['model']);del state;gc.collect()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=model.to(device).eval()
    binding={'checkpoint_sha256':file_hash(checkpoint),'adapter_sha256':file_hash(__file__),
        'full_field_reference':str(Path(full_result).resolve()),'expert_ROI_used':True,
        'scope':'Privileged source-ROI diagnostic; not deployment or competition score.'}
    bp=output/'binding.json'
    if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('Existing binding mismatch')
    save_json(bp,binding);path=output/'predictions.jsonl'
    predictions=[json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []
    done={r['id'] for r in predictions};errors=[];start=time.perf_counter()
    for r in records:
        if r['id'] in done:continue
        try:
            item=KneeMRIClinicalReference(root/'reference_anatomy/KneeMRI',[r['id']])[0]
            box=r['expected_source_roi'];z,y,x=int(box['roiZ']),int(box['roiY']),int(box['roiX'])
            crop=item['image'][z:z+int(box['roiDepth']),y:y+int(box['roiHeight']),x:x+int(box['roiWidth'])]
            batch,receipt=full_volume_batch(crop,1,1.,None,cfg.image_size,device)
            with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                pred=model(batch).float().sigmoid()[0].cpu().tolist()
            row={'source':'KneeMRI','id':r['id'],'probabilities':pred,'expert_ROI_used':True,'roi_shape':list(crop.shape)}
            with path.open('a') as f:f.write(json.dumps(row)+'\n')
            predictions.append(row);del batch,crop,item
        except Exception as e:
            errors.append({'id':r['id'],'error':str(e)})
            if len(errors)>=3:break
        if len(predictions)%100==0:print({'completed':len(predictions),'expected':len(records)},flush=True)
    result={'complete':len(predictions)==len(records),'completed':len(predictions),'expected':len(records),
        'errors':errors,'endpoints':endpoint_metrics(records,predictions),'expert_ROI_used':True,
        'comparison_limit':'ROI supplies expert location and changes field/resolution/intensity scaling; not an isolated localization ablation.',
        'seconds':time.perf_counter()-start,'optimizer_steps':0}
    save_json(output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--checkpoint',required=True)
    p.add_argument('--full-result',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.root,a.checkpoint,a.full_result,a.output)
