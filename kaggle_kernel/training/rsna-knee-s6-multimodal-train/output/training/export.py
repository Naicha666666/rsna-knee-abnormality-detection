"""Materialize a portable image checkpoint from a compact frozen-feature run."""
from pathlib import Path
import argparse
import torch
import shutil
from .common import file_hash
from .engine import atomic_checkpoint


def export_checkpoint(source,output):
    state=torch.load(source,map_location='cpu',weights_only=True)
    reference=state.pop('encoder_reference',None)
    if reference:
        if file_hash(reference['path'])!=reference['sha256']:raise ValueError('encoder checksum changed')
        state['model']={**torch.load(reference['path'],map_location='cpu',weights_only=True),**state['model']}
        state['exported_encoder_sha256']=reference['sha256']
        state['train_config'].pop('frozen_encoder_reference',None)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    if state['model_config'].get('anatomy_pool') or state['model_config'].get('local_evidence') or state['train_config'].get('views')=='global_anatomic':
        anatomy=Path(state['train_config']['anatomy_weights'])
        target=Path(output).with_name('anatomy.pt')
        if anatomy.resolve()!=target.resolve():shutil.copyfile(anatomy,target)
        state['anatomy_sha256']=file_hash(target)
        assets=anatomy.parent.parent/'kneexnet';destination=Path(output).parent/'kneexnet';destination.mkdir(exist_ok=True)
        state['anatomy_asset_sha256']={}
        for name in ('localizer.pt','best_model_256_noise_notebook_5_0.15.pth'):
            if (assets/name).resolve()!=(destination/name).resolve():shutil.copyfile(assets/name,destination/name)
            state['anatomy_asset_sha256'][name]=file_hash(destination/name)
        state['ligament_atlas_sha256']=file_hash(Path(__file__).parent/'data/ligament_atlas.json')
    atomic_checkpoint(output,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export_checkpoint(a.checkpoint,a.output)
