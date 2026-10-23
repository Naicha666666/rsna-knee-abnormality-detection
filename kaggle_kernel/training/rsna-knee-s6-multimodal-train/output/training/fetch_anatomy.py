"""Pinned SKM-TEA anatomy initialization, physically direct network only."""
from pathlib import Path
import hashlib
import json
from .direct_network import DirectNetwork
from .common import save_json,file_hash

REPO='arjundd/skm-tea-models'
SUBDIR='neurips2021/segmentation-models/V-Net_E1'


def fetch(root='F:/rsnaknee/pretrained/skm_tea'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    net=DirectNetwork('192.168.1.31',11);host='https://hf-mirror.com'
    with net.session() as session:
        response=session.get(f'{host}/api/models/{REPO}',timeout=(15,45));response.raise_for_status()
        metadata=response.json();revision=metadata['sha']
        response=session.get(f'{host}/api/models/{REPO}/tree/{revision}/{SUBDIR}',timeout=(15,45));response.raise_for_status()
        entries={Path(x['path']).name:x for x in response.json()}
        receipts=[]
        for name in ('config.yaml','model-raw.ckpt'):
            entry=entries[name];target=root/name;expected=entry.get('lfs',{}).get('oid')
            if name.endswith('.ckpt') and not expected:raise RuntimeError('missing published weight hash')
            if not target.exists():
                response=session.get(f'{host}/{REPO}/resolve/{revision}/{SUBDIR}/{name}',stream=True,timeout=(15,45));response.raise_for_status()
                tmp=target.with_suffix('.download');digest=hashlib.sha256();size=0
                with tmp.open('wb',buffering=8*2**20) as f:
                    for block in response.iter_content(2**20):f.write(block);digest.update(block);size+=len(block)
                if size!=entry['size'] or (expected and digest.hexdigest()!=expected):raise RuntimeError('anatomy artifact checksum mismatch')
                tmp.replace(target)
            if target.stat().st_size!=entry['size'] or (expected and file_hash(target)!=expected):raise RuntimeError('existing anatomy artifact differs')
            receipts.append({'file':name,'sha256':file_hash(target),'bytes':target.stat().st_size})
    save_json(root/'source.json',{'repository':REPO,'revision':revision,'subdirectory':SUBDIR,
        'official_model_zoo':'https://github.com/StanfordMIMI/skm-tea/blob/main/MODEL_ZOO.md',
        'model_card_license':metadata.get('cardData',{}).get('license'),'retrieval_host':host,
        'hash_verification':'against mirrored Git-LFS metadata','files':receipts,
        'network':'DIRECT_ETHERNET','classes':['patellar_cartilage','femoral_cartilage','tibial_cartilage','meniscus'],
        'domain':'sagittal qDESS echo1; target MRI transfer not yet established'})
    save_json(root/'connections.json',net.receipt())
    print(json.dumps({'root':str(root),'files':receipts,'network':'DIRECT_ETHERNET'}),flush=True)


if __name__=='__main__':fetch()
