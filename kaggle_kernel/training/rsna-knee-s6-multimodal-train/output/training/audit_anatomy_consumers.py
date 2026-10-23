"""Real cached-input replay of anatomical supervision and physical consumers.

No network fitting or producer source edits. The legacy loss calculation is
kept locally solely to quantify the observed-only normalization correction.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .common import UID,save_json,file_hash
from .native_data import NativeStudyDataset
from .data import collate_studies
from .spatial import tissue_loss,physical_segments


def legacy_tissue_loss(logits,targets,quality,flat_index,b,k,pixel_weight):
    targets=F.interpolate(targets.float(),size=logits.shape[-2:],mode='bilinear',align_corners=False).clamp(0,1)
    positive=(targets>=.8).float();negative=(targets<=.2).float()
    valid=F.interpolate(pixel_weight.float(),size=logits.shape[-2:],mode='bilinear',align_corners=False).ge(.9)
    positive*=valid;negative*=valid
    loss=F.binary_cross_entropy_with_logits(logits.float(),targets,reduction='none')
    pc=positive.sum((-1,-2));nc=negative.sum((-1,-2))
    balanced=((loss*positive).sum((-1,-2))/pc.clamp_min(1)+(loss*negative).sum((-1,-2))/nc.clamp_min(1))
    balanced/=(pc.gt(0).float()+nc.gt(0).float()).clamp_min(1)
    per=(balanced*quality).sum(1)/quality.sum(1).clamp_min(1)
    weights=quality.gt(0).any(1).float();study=flat_index//k
    total=per.new_zeros(b).index_add(0,study,per*weights)
    count=per.new_zeros(b).index_add(0,study,weights)
    return total/count.clamp_min(1)


def run(root='F:/rsnaknee'):
    torch.set_num_threads(4);root=Path(root);out=root/'analysis/anatomy_consumer_audit';out.mkdir(parents=True,exist_ok=True)
    native=json.loads((root/'cache_native/manifest.json').read_text())
    selected={
        '1.2.826.0.1.3680043.8.498.81863400543736189183553734054370704788',
        '1.2.826.0.1.3680043.8.498.11660423496805057440817786799305499257',
        '1.2.826.0.1.3680043.8.498.80126107048478277295386095863259178495'}
    studies=[{**study,'series':[s for s in study['series'] if s['SeriesInstanceUID'] in selected]} for study in native['studies']]
    studies=[study for study in studies if study['series']]
    save_json(out/'manifest.json',{**native,'studies':studies,'scope':'already-used development cases; frozen consumer audit subset'})
    dataset=NativeStudyDataset(out/'manifest.json',training=False,image_size=384,views='global_local',
        geometry_dir=root/'geometry',anatomy_dir=root/'anatomy_v3',exact_duplicate_policy='keep')
    rows=[];physical_rows=[]
    for item in dataset:
        target=item['tissue_targets'];support=item['tissue_pixel_weight'];quality=item['tissue_quality'];n=len(target)
        observed=((support>=.9)&((target>=.8)|(target<=.2))).flatten(2).any(2)&quality.gt(0)
        active=observed.any(1);indices=torch.arange(n)
        logits=torch.zeros_like(target,requires_grad=True)
        old=legacy_tissue_loss(logits,target,quality,indices,1,n,support)
        old_without_unknown=legacy_tissue_loss(logits[active],target[active],quality[active],indices[active],1,n,support[active])
        new=tissue_loss(logits,target,quality,indices,1,n,support)
        new_without_unknown=tissue_loss(logits[active],target[active],quality[active],indices[active],1,n,support[active])
        gradient=torch.autograd.grad(new.sum(),logits,retain_graph=True)[0]
        selected_gradient=torch.autograd.grad(new_without_unknown.sum(),logits)[0]
        assert torch.allclose(new,new_without_unknown) and torch.allclose(gradient,selected_gradient)
        assert gradient[~active].eq(0).all()
        series=[]
        for sid,name in enumerate(item['series_names']):
            pick=item['series'].eq(sid)
            series.append({'SeriesInstanceUID':name,'view_id':int(item['view_ids'][pick][0]),
                'windows':int(pick.sum()),'windows_with_quality':int((quality.gt(0).any(1)&pick).sum()),
                'windows_with_actual_observed_targets':int((active&pick).sum()),
                'observed_channel_windows':observed[pick].sum(0).tolist()})
        rows.append({UID:item[UID],'windows':n,'observed_windows':int(active.sum()),'legacy_loss':float(old.detach().sum()),
            'legacy_loss_with_unknown_windows_removed':float(old_without_unknown.detach().sum()),
            'corrected_loss':float(new.detach().sum()),'corrected_loss_with_unknown_windows_removed':float(new_without_unknown.detach().sum()),
            'gradient_change_when_unknown_windows_removed':float((gradient-selected_gradient).abs().max()),
            'unknown_window_gradient_max':float(gradient[~active].abs().max()) if (~active).any() else 0.,
            'series':series})
        batch=collate_studies([item]);segments=list(physical_segments(batch,indices,torch.device('cpu')))
        covered=0;maximum_position_error=0.;group_rows=[]
        for ids,positions in segments:
            assert len(item['series'][ids].unique())==len(item['view_ids'][ids].unique())==len(item['original_series'][ids].unique())==1
            actual=item['channel_position_mm'][ids,1]
            if len(ids)>1:
                error=(positions.diff().abs().double()-actual.diff().abs()).abs().max().item()
                maximum_position_error=max(maximum_position_error,error)
            covered+=len(ids)
            group_rows.append({'windows':len(ids),'view_id':int(item['view_ids'][ids[0]]),'real_series_index':int(item['original_series'][ids[0]]),
                'gap_mm_min':float(positions.diff().min()) if len(ids)>1 else None,'gap_mm_max':float(positions.diff().max()) if len(ids)>1 else None})
        physical_rows.append({UID:item[UID],'windows':n,'covered_by_physical_segments':covered,'max_gap_error_vs_original_slice_positions_mm':maximum_position_error,
            'cross_series_or_view_connections':0,'segments':group_rows})
        print(item[UID],'windows',n,'observed',int(active.sum()),'old/new',float(old.detach()),float(new.detach()),flush=True)
    result={'loss_replays':rows,'physical_segment_replays':physical_rows,'training_dispatch':False,'optimizer_steps':0,
        'consumer_source_sha256':file_hash(Path(__file__).with_name('spatial.py')),
        'scope':'real NativeStudyDataset targets and explicit supports; zero logits isolate loss/gradient normalization, not predictive accuracy',
        'change':'Only channels/windows with a valid unambiguous supervised pixel enter denominators; original continuous quality discount preserved'}
    save_json(out/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');run(p.parse_args().root)
