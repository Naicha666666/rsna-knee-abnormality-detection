"""Small, measured input audit: actual series lengths and native-slice coverage."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from .common import LABELS,UID,save_json


def audit(root='F:/rsnaknee',cache=None):
    root=Path(root);cache=Path(cache or root/'cache384')
    studies=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((cache/'studies').glob('*.json'))]
    records=[]
    for study in studies:
        for series in study['series']:
            pos=np.asarray(series['positions']);length=len(pos)
            def coverage(n,physical):
                seen=set()
                for c in np.unique(np.linspace(0,length-1,min(n,length)).round().astype(int)):
                    if physical:
                        left=int(np.argmin(abs(pos-(pos[c]-3))))
                        right=int(np.argmin(abs(pos-(pos[c]+3))))
                        if c>0:left=min(left,c-1)
                        if c+1<length:right=max(right,c+1)
                    else:left,right=max(c-1,0),min(c+1,length-1)
                    seen.update([left,c,right])
                return len(seen)/length
            train_n=min(length,32,max(8,int(np.ceil(length/4))))
            eval_n=min(length,128,max(16,int(np.ceil((length+2)/3))))
            records.append({UID:study[UID],'SeriesInstanceUID':series['SeriesInstanceUID'],
                'plane':series['plane'],'slices':length,'span_mm':float(np.ptp(pos)),
                'old_fixed8_coverage':coverage(8,True),'old_fixed16_coverage':coverage(16,True),
                'train_windows':train_n,'eval_windows':eval_n,
                'new_train_uniform_proxy_coverage':coverage(train_n,False),
                'new_eval_coverage':coverage(eval_n,False)})
    frame=pd.DataFrame(records)
    if frame.empty:raise RuntimeError('no prepared studies to audit')
    output=root/'analysis';output.mkdir(exist_ok=True)
    frame.to_csv(output/'slice_coverage.csv',index=False)
    summary={'scope':'prepared study subset; training coverage is uniform-center diagnostic, actual training is jittered',
        'studies':len(studies),'series':len(frame),'slice_count_quantiles':{str(q):float(frame.slices.quantile(q)) for q in [0,.5,.9,1]},
        'old_fixed16_less_than_half':int((frame.old_fixed16_coverage<.5).sum()),
        'new_eval_less_than_half':int((frame.new_eval_coverage<.5).sum()),
        'median_new_eval_coverage':float(frame.new_eval_coverage.median()),
        'train_windows_total':int(frame.train_windows.sum()),'old_train_windows_total':int(np.minimum(frame.slices,8).sum()),
        'eval_windows_total':int(frame.eval_windows.sum()),'old_eval_windows_total':int(np.minimum(frame.slices,16).sum()),
        'expected_auc_gain':None}
    save_json(output/'feature_findings.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');a=p.parse_args();audit(a.root)
