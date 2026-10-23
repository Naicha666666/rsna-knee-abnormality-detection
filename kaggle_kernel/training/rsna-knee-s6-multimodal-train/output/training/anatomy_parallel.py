"""Bounded, independent complete-Study anatomy workers with one publisher.

Every worker keeps its frozen models and registration cache. Its outputs are
private until the parent promotes a complete Study in the original order.
No worker writes the shared anatomy manifest. Model batches and arithmetic are
unchanged; this schedules independent inference, never training updates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

_RUNTIME=None


def _init_worker(weights,device,assets_dir,compression_level,registration_bytes):
    global _RUNTIME
    import torch,cv2
    from .anatomy_runtime import AnatomyRuntime
    from .registration_runtime import default_cache
    torch.set_num_threads(4);cv2.setNumThreads(1)
    default_cache().ram_budget_bytes=int(registration_bytes)
    _RUNTIME=AnatomyRuntime(weights,device,assets_dir,compression_level=compression_level)


def _process_study(manifest,geometry_dir,output):
    import torch,psutil
    from .anatomy import build
    began=time.perf_counter()
    build(manifest,geometry_dir,_RUNTIME.weights,output,device=_RUNTIME.device,
          assets_dir=_RUNTIME.assets,append=False,runtime=_RUNTIME)
    data=json.loads((Path(output)/'manifest.json').read_text(encoding='utf-8'))
    rss=psutil.Process().memory_info()
    return {'manifest':data,'worker_pid':os.getpid(),'seconds':time.perf_counter()-began,
            'rss_bytes':rss.rss,'peak_rss_bytes':getattr(rss,'peak_wset',rss.rss),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            'resources':_RUNTIME.snapshot()}


def _replace_with_retry(source,destination):
    for attempt in range(7):
        try:
            source.replace(destination);return
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,.8))


def promote_study(result,study,staging,output):
    """Prepare an immutable Study revision, then atomically switch its manifest.

    Any failed validation/copy/receipt/manifest write leaves all previously
    published payloads untouched. Staging survives failure for a CPU-only retry;
    an unreferenced revision can be reclaimed separately after inspection.
    """
    from .common import UID,save_json
    from .anatomy import key_for,publish_manifest
    staging,output=Path(staging).resolve(),Path(output).resolve()
    rows=result['manifest']['series']
    validate_study_identities({'studies':[study]})
    expected={s['SeriesInstanceUID'] for s in study['series']}
    if len(rows)!=len(expected) or {r['SeriesInstanceUID'] for r in rows}!=expected:
        raise ValueError('Worker did not complete the exact Study Series set')
    contract=result['manifest']['contract'];producer=contract.get('producer_signature')
    if not producer:raise ValueError('Worker manifest has no producer identity')
    checked=[]
    for row in rows:
        if row[UID]!=study[UID]:raise ValueError('Worker crossed Study identity')
        source=Path(row['path']).resolve()
        if staging not in source.parents:raise ValueError('Worker payload escaped its staging directory')
        key=key_for(study[UID],row['SeriesInstanceUID'])
        if source.name!=key+'.npz':raise ValueError('Unexpected worker payload name')
        if not source.is_file():raise ValueError('Worker payload is missing')
        receipt=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        if (row.get('producer_signature')!=producer or receipt.get('producer_signature')!=producer
            or receipt.get('contract',{}).get('producer_signature')!=producer):
            raise ValueError('Worker receipt/manifest producer mismatch')
        if receipt.get(UID)!=study[UID] or receipt.get('SeriesInstanceUID')!=row['SeriesInstanceUID']:
            raise ValueError('Worker receipt crossed Study/Series identity')
        if Path(receipt.get('path','')).resolve()!=source or any(receipt.get(k)!=v for k,v in row.items()):
            raise ValueError('Worker receipt/manifest payload metadata mismatch')
        # Check all archive directories before creating any publication files.
        # Arrays were already written atomically by the worker; this adds no
        # duplicate full-volume decompression to the GPU preprocessing path.
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if not names or len(set(names))!=len(names) or any(not n.endswith('.npy') or Path(n).name!=n for n in names):
                raise ValueError('Worker payload is not a complete ndarray archive')
            if '_base_payload.npy' in names:raise ValueError('Worker revision must contain self-contained arrays')
        checked.append((source,receipt,row,key))
    revision=output/'revisions'/hashlib.sha256(study[UID].encode()).hexdigest()[:24]/uuid.uuid4().hex
    revision.mkdir(parents=True,exist_ok=False)
    promoted=[]
    for source,receipt,row,key in checked:
        destination=revision/(key+'.npz')
        shutil.copyfile(source,destination)
        updated={**row,'path':str(destination)}
        save_json(destination.with_suffix('.json'),{**receipt,**updated})
        promoted.append(updated)
    previous_path=output/'manifest.json'
    previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {'series':[]}
    # Replace the complete Study membership, retaining every other Study.
    combined=[row for row in previous['series'] if row[UID]!=study[UID]]+promoted
    publish_manifest(output,contract,combined,append=False)
    # Only a successful commit makes staged copies dispensable. Cleanup failure
    # must not turn a committed Study into a failed publication.
    for source,_,_,_ in checked:
        try:source.unlink()
        except OSError:pass
    return promoted


def validate_study_identities(data):
    """Reject shared worker output directories before submitting any task."""
    from .common import UID
    studies=set();series=set()
    for study in data['studies']:
        uid=study[UID]
        if not isinstance(uid,str) or not uid or uid in studies:raise ValueError('Duplicate or invalid Study UID')
        studies.add(uid)
        if not study['series']:raise ValueError('A complete Study must contain native Series')
        for row in study['series']:
            sid=row['SeriesInstanceUID']
            if not isinstance(sid,str) or not sid or sid in series:raise ValueError('Duplicate or invalid Series UID')
            series.add(sid)


class AnatomyWorkerPool:
    def __init__(self,weights,output,staging,*,workers=2,device='cuda',assets_dir=None,
                 compression_level=1,registration_bytes=512*2**20):
        if workers<1 or workers>4:raise ValueError('Only the bounded 1..4 worker range is supported')
        self.weights=str(Path(weights).resolve());self.output=Path(output).resolve()
        self.staging=Path(staging).resolve();self.staging.mkdir(parents=True,exist_ok=True)
        self.workers=int(workers);self.last_batch=None
        # Windows spawn imports numeric libraries before the initializer.
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            os.environ[name]='4'
        self.executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,initargs=(self.weights,device,str(assets_dir) if assets_dir else None,
                                              compression_level,registration_bytes))

    def build(self,manifest,geometry_dir):
        from .common import UID,save_json
        data=json.loads(Path(manifest).read_text(encoding='utf-8'))
        validate_study_identities(data)
        started=time.perf_counter();jobs=[];stats=[]
        for study in data['studies']:
            key=hashlib.sha256(study[UID].encode()).hexdigest()[:24]
            folder=self.staging/key;folder.mkdir(parents=True,exist_ok=True)
            path=folder/'native_manifest.json'
            save_json(path,{'config':data['config'],'studies':[study],
                            'scope':'complete independent Study from a frozen preprocessing batch'})
            future=self.executor.submit(_process_study,str(path),str(geometry_dir),str(folder/'output'))
            jobs.append((study,folder,future))
        # Compute overlaps, but commits keep the original complete-Study order.
        for study,folder,future in jobs:
            result=future.result()
            promote_study(result,study,folder,self.output)
            stats.append({k:v for k,v in result.items() if k!='manifest'})
        self.last_batch={'workers':self.workers,'studies':len(jobs),
            'series':sum(len(st['series']) for st in data['studies']),
            'wall_seconds':time.perf_counter()-started,'worker_results':stats,
            'boundary':'enqueue complete Studies through all inference/registration/output and ordered canonical manifest publication',
            'training_dispatch':False}
        return self.last_batch

    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


if __name__=='__main__':
    import argparse
    from .common import file_hash,save_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--geometry',required=True)
    parser.add_argument('--weights',default='F:/rsnaknee/pretrained/skm_tea/model-raw.ckpt')
    parser.add_argument('--output',required=True);parser.add_argument('--staging',required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--result',required=True)
    args=parser.parse_args()
    with AnatomyWorkerPool(args.weights,args.output,args.staging,workers=args.workers) as pool:
        result=pool.build(args.manifest,args.geometry)
    result.update(input_manifest_sha256=file_hash(args.manifest),
                  output_manifest=str(Path(args.output)/'manifest.json'))
    save_json(args.result,result)
    print(json.dumps({k:result[k] for k in ('workers','studies','series','wall_seconds')},ensure_ascii=False),flush=True)
