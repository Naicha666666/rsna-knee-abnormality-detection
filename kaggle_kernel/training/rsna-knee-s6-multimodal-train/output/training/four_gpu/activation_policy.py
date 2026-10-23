"""Shape-only activation retention under a measured complete-Study budget."""
from dataclasses import dataclass,asdict
import hashlib,json,math
from pathlib import Path

GIB=2**30


def implementation_binding():
    root=Path(__file__).resolve().parent
    paths=[root/name for name in ('activation_policy.py','retained_model.py','evidence_cache.py',
           'cuda_prefetch.py','calibrate_pro6000.py','train.py','recipe.py','core.py','acceptance.py')]+[
           root.parent/'model.py',root.parent/'spatial.py',root.parent/'oa_binding.py',
           root.parent/'supervision.py',root.parent/'partial_joint.py',root.parent/'mcl_joint.py']
    return {str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def content_binding(ready):
    # Excludes paths, file timestamps and the user-controlled pause flag.
    data=[{'fold':e['fold'],'config':e['config'],'inputs':e['inputs']['hashes'],
           'train_ids':e['training_ids'],'validation_ids':e['validation_ids']}
          for e in ready['original_entries']]
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ActivationPolicy:
    chunk: int = 2
    baseline_intercept_bytes: int = 0
    baseline_bytes_per_window: float = 0.
    saved_bytes_per_window: float = 1.
    memory_limit_bytes: int = 0
    max_windows: int = 0
    retained_window_cap: int = 0
    fallback_chunk: int = 2

    def __post_init__(self):
        if self.chunk not in (2,4,8,16,24,32) or self.fallback_chunk!=2:raise ValueError('Unsupported encoder chunk')
        if self.max_windows<0 or self.retained_window_cap<0:raise ValueError('Negative activation limit')
        if not math.isfinite(self.saved_bytes_per_window) or self.saved_bytes_per_window<=0:raise ValueError('Invalid activation cost')
        if not math.isfinite(self.baseline_bytes_per_window) or self.baseline_bytes_per_window<0:raise ValueError('Invalid baseline cost')

    def choose(self,windows):
        if windows>self.max_windows:
            return {'chunk':self.fallback_chunk,'retained_starts':[], 'retained_windows':0,
                    'checkpointed_windows':windows,'fallback':True,'reason':'larger than calibrated complete Study'}
        baseline=self.baseline_intercept_bytes+self.baseline_bytes_per_window*windows
        available=max(0,self.memory_limit_bytes-baseline)
        count=min(windows,self.retained_window_cap,int(available/self.saved_bytes_per_window))
        remaining=count;starts=[];kept=0
        for start in range(0,windows,self.chunk):
            size=min(self.chunk,windows-start)
            if size<=remaining:starts.append(start);kept+=size;remaining-=size
        return {'chunk':self.chunk,'retained_starts':starts,'retained_windows':kept,
                'checkpointed_windows':windows-kept,'fallback':False,'predicted_peak_bytes':baseline+kept*self.saved_bytes_per_window}

    def export(self):return asdict(self)


def fit_policy(baselines,saved_bytes_per_window,total_bytes,chunk,max_retained=None):
    if not baselines:raise ValueError('Full-Study memory observations required')
    slopes=[(b['peak_allocated_bytes']-a['peak_allocated_bytes'])/(b['windows']-a['windows'])
        for a in baselines for b in baselines if b['windows']>a['windows']]
    slope=max([0.,*slopes])*1.25
    intercept=max(r['peak_allocated_bytes']-slope*r['windows'] for r in baselines)+2*GIB
    # Reserve at least 12 GiB, including reduction scratch and a possible next
    # input buffer. No artificial VRAM-filling tensors are allocated.
    limit=min(int(total_bytes*.85),total_bytes-12*GIB)
    maximum=max(r['windows'] for r in baselines)
    return ActivationPolicy(chunk=chunk,baseline_intercept_bytes=math.ceil(intercept),
        baseline_bytes_per_window=slope,saved_bytes_per_window=max(1.,saved_bytes_per_window*1.25),
        memory_limit_bytes=limit,max_windows=maximum,retained_window_cap=maximum if max_retained is None else max_retained)


def scheduling_cost(windows,policy,profile):
    plan=policy.choose(windows);fit=profile.get('scheduling_cost',{})
    if plan['fallback']:return windows*max(float(fit.get('fallback_seconds_per_window',1.)),1e-6)
    coefficients=fit.get('coefficients')
    if coefficients is None:return float(windows)
    return max(1e-6,coefficients[0]+coefficients[1]*windows+coefficients[2]*plan['checkpointed_windows'])


def load_profile(path,ready,device_info):
    path=Path(path);obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('format')!='pro6000_activation_profile_v1' or not obj.get('calibration_passed'):
        raise ValueError('Use a completed target-GPU activation calibration')
    if obj['input_binding']!=content_binding(ready):raise ValueError('Profile belongs to another training construction')
    if obj['implementation_binding']!=implementation_binding():raise ValueError('Execution implementation changed; recalibrate')
    for key in ('name','total_memory_bytes'):
        if obj['hardware'][key]!=device_info[key]:raise ValueError('Recalibrate on this GPU hardware')
    report_path=path.parent/'calibration.json'
    raw=report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=obj['calibration_report_sha256']:raise ValueError('Calibration receipt changed')
    report=json.loads(raw)
    if report['status']!='candidate_selected' or report['chosen']['policy']!=obj['policy']:raise ValueError('Profile is not the selected measured policy')
    return obj,ActivationPolicy(**obj['policy'])
