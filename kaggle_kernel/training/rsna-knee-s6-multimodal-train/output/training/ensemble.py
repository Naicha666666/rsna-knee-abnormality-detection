"""Aligned prediction fusion, isolated blend fitting, and budgeted second inference."""
from pathlib import Path
import argparse,json,math,time
import hashlib,platform
import numpy as np
import pandas as pd
from .common import LABELS,UID,file_hash,save_json


def probabilities(frame):
    if frame[UID].isna().any() or frame[UID].duplicated().any():raise ValueError('missing or duplicate Study IDs')
    p=frame[LABELS].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('probabilities must be finite in [0,1]')
    return p


def blend(first,second,second_weight=.5,*,subset=False):
    a=probabilities(first);probabilities(second)
    ia=first.set_index(UID);ib=second.set_index(UID)
    if set(ib.index)-set(ia.index) or (not subset and set(ib.index)!=set(ia.index)):
        raise ValueError('prediction Study coverage differs')
    w=np.asarray(second_weight,float)
    if w.shape not in ((),(12,)) or not np.isfinite(w).all() or (w<0).any() or (w>1).any():
        raise ValueError('one bounded weight per label is required')
    # Inference emits float32; pandas rejects assigning rounded float64 blends
    # into those columns. Preserve full fusion precision explicitly.
    result=ia.astype({c:'float64' for c in LABELS});ids=ib.index
    result.loc[ids,LABELS]=ia.loc[ids,LABELS].to_numpy()*(1-w)+ib[LABELS].to_numpy()*w
    return result.reset_index().loc[:,[UID,*LABELS]]


def route_studies(first,fraction):
    p=probabilities(first)
    if not 0<=fraction<=1:raise ValueError('route budget must lie in [0,1]')
    # Fixed label-union ambiguity policy. Calibration and cross-route ranking
    # must be evaluated on complete held-out Studies, not the selected subset.
    ambiguity=(1-2*abs(p-.5)).max(1)
    order=np.lexsort((first[UID].to_numpy(str),-ambiguity))
    return first.iloc[order[:math.ceil(len(first)*fraction)]][UID].tolist()


def fit_pair_weights(first,second,labels,checkpoint_states,outer_fold,fit_ids,output):
    hold=Path(__file__).with_name('TRAINING_HOLD.json')
    if hold.exists() and json.loads(hold.read_text(encoding='utf-8')).get('active'):
        raise RuntimeError('Training is held; blend-weight fitting is also fitting and is not authorized.')
    from .distill import validate_teacher
    from sklearn.metrics import roc_auc_score
    if len(checkpoint_states)!=2:raise ValueError('two source checkpoint lineages required')
    for state in checkpoint_states:validate_teacher(state,labels,outer_fold,fit_ids)
    if set(first[UID])!=set(fit_ids) or set(second[UID])!=set(fit_ids):raise ValueError('fit predictions must exactly cover the inner calibration set')
    probabilities(first);probabilities(second)
    y=labels.set_index(UID).loc[fit_ids];a=first.set_index(UID).loc[fit_ids,LABELS].to_numpy()
    b=second.set_index(UID).loc[fit_ids,LABELS].to_numpy();weights=[];details={}
    # Prefer equal averaging when improvements tie; the grid is fixed before
    # looking at the outer validation set.
    grid=sorted(np.linspace(0,1,21),key=lambda w:abs(w-.5))
    for j,c in enumerate(LABELS):
        known=y[c+'__weight'].to_numpy()>0;truth=y[c].to_numpy(float)[known]
        if set(np.unique(truth))!={0.,1.}:
            weights.append(.5);details[c]={'fitted':False,'reason':'insufficient binary calibration evidence'};continue
        scores=[roc_auc_score(truth,(a[known,j]*(1-w)+b[known,j]*w)) for w in grid]
        w=float(grid[int(np.argmax(scores))]);weights.append(w)
        details[c]={'fitted':True,'second_weight':w,'inner_calibration_auc':float(max(scores))}
    record={'labels':LABELS,'second_weight':weights,'outer_fold':int(outer_fold),'fit_ids':fit_ids,
            'details':details,'scope':'inner calibration fit; no outer/generalization score claimed'}
    save_json(output,record);return record


def run_routed(data,primary,secondary,output,*,fraction=.25,second_weight=.5,workers=0,shared_native_cache=None):
    from .infer import run_inference
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    first=run_inference(data,primary,output/'primary',workers=workers,shared_native_cache=shared_native_cache)
    selected=route_studies(first,fraction)
    if selected:
        second=run_inference(data,secondary,output/'secondary',workers=workers,study_ids=selected,
                             shared_native_cache=output/'primary/native_test_cache')
        if set(second[UID])!=set(selected):raise ValueError('secondary inference did not cover the route')
        result=blend(first,second,second_weight,subset=True)
    else:result=first
    result.to_csv(output/'submission.csv',index=False)
    primary_hash=file_hash(primary);secondary_hash=file_hash(secondary)
    source_scopes={}
    for name in ('primary','secondary'):
        path=output/name/'inference.json'
        source_scopes[name]=json.loads(path.read_text())['source_smoke'] if path.exists() else None
    record={'studies':len(first),'secondary_studies':len(selected),'selected_ids':selected,
            'route_fraction':fraction,'second_weight':np.asarray(second_weight).tolist(),
            'seconds_complete_pipeline':time.perf_counter()-started,
            'reused_primary_native_cache':bool(shared_native_cache),
            'primary_sha256':primary_hash,'secondary_sha256':secondary_hash,
            'study_ids_sha256':hashlib.sha256(json.dumps(sorted(first[UID].tolist())).encode()).hexdigest(),
            'timing_boundary':'raw-data preparation through complete submission CSV',
            'runtime_environment':platform.platform(),
            'source_engineering_diagnostics':source_scopes,
            'scope':'complete local routed inference; speed/accuracy superiority and platform timing not established'}
    save_json(output/'routed_inference.json',record);return result,record


def efficiency_comparison(rows,reference_gap=None):
    """Measured AUC/time Pareto comparison; optional explicitly assumed M-B."""
    if reference_gap is not None and reference_gap<=0:raise ValueError('M-B must be positive')
    compared=[r for r in rows if r.get('macro_auc') is not None]
    keys=('evaluation_reference_sha256','study_ids_sha256','timing_boundary','runtime_environment','native_cache_state')
    if compared and any(not r.get(k) for r in compared for k in keys):
        raise ValueError('measured score/time rows need evaluation, Study, environment and cache identities')
    if compared and any(len({r[k] for r in compared})!=1 for k in keys):
        raise ValueError('cannot compare different evaluation data, hardware, timing boundaries or cache states')
    result=[]
    for row in rows:
        if row.get('macro_auc') is None or not np.isfinite(row['macro_auc']):
            result.append({**row,'pareto':None,'efficiency_objective':None});continue
        a=float(row['macro_auc']);seconds=float(row['seconds'])
        if not 0<=a<=1 or not np.isfinite(seconds) or seconds<0:raise ValueError('invalid measured AUC/time')
        dominated=any(other.get('macro_auc') is not None and other['macro_auc']>=a and other['seconds']<=seconds
                      and (other['macro_auc']>a or other['seconds']<seconds) for other in rows)
        result.append({**row,'pareto':not dominated,'efficiency_objective':None if reference_gap is None else a-reference_gap*seconds/32400})
    return {'rows':result,'assumed_M_minus_B':reference_gap,'scope':'same evaluation set and timing boundary required; not an official efficiency score'}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('blend');q.add_argument('--first',required=True);q.add_argument('--second',required=True);q.add_argument('--weight',type=float,default=.5);q.add_argument('--output',required=True)
    q=sub.add_parser('route');q.add_argument('--data',required=True);q.add_argument('--primary',required=True);q.add_argument('--secondary',required=True);q.add_argument('--fraction',type=float,default=.25);q.add_argument('--output',required=True);q.add_argument('--shared-native-cache')
    q=sub.add_parser('pareto');q.add_argument('--measurements',required=True);q.add_argument('--reference-gap',type=float);q.add_argument('--output',required=True)
    q=sub.add_parser('fit');q.add_argument('--labels',required=True);q.add_argument('--first',required=True);q.add_argument('--second',required=True);q.add_argument('--primary-checkpoint',required=True);q.add_argument('--secondary-checkpoint',required=True);q.add_argument('--outer-fold',type=int,required=True);q.add_argument('--fit-ids',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='blend':
        result=blend(pd.read_csv(a.first,dtype={UID:str}),pd.read_csv(a.second,dtype={UID:str}),a.weight)
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    elif a.command=='route':run_routed(a.data,a.primary,a.secondary,a.output,fraction=a.fraction,shared_native_cache=a.shared_native_cache)
    elif a.command=='fit':
        import torch
        states=[torch.load(p,map_location='cpu',weights_only=True) for p in (a.primary_checkpoint,a.secondary_checkpoint)]
        fit_pair_weights(pd.read_csv(a.first,dtype={UID:str}),pd.read_csv(a.second,dtype={UID:str}),
            pd.read_csv(a.labels,dtype={UID:str,'group_id':str}),states,a.outer_fold,pd.read_csv(a.fit_ids,dtype={UID:str})[UID].tolist(),a.output)
    else:save_json(a.output,efficiency_comparison(json.loads(Path(a.measurements).read_text()),a.reference_gap))


if __name__=='__main__':main()
