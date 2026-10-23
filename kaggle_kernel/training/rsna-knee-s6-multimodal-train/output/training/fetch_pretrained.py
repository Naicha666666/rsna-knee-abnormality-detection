"""Fetch the fixed public timm initialization over the physical Ethernet NIC."""
from pathlib import Path
import hashlib
import json

from .common import save_json
from .direct_network import DirectNetwork

REPO = "timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


def fetch(destination="F:/rsnaknee/pretrained", source_ip="192.168.1.31", interface=11):
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    network=DirectNetwork(source_ip,interface)
    # The official host was not reachable on this broadband route. This public
    # HF mirror is reached directly too; no account credential is sent to it.
    base="https://hf-mirror.com"
    with network.session() as session:
        info=session.get(f"{base}/api/models/{REPO}",timeout=(15,30)); info.raise_for_status()
        revision=info.json()["sha"]
        tree=session.get(f"{base}/api/models/{REPO}/tree/{revision}",timeout=(15,30)); tree.raise_for_status()
        entry=next(x for x in tree.json() if x['path']=='model.safetensors')
        expected=entry.get('lfs',{}).get('oid')
        size=entry['size']
        if not expected or len(expected)!=64:
            raise RuntimeError('no published LFS SHA256 for pretrained weights')
        path=root/'model.safetensors'
        if path.exists():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
            if path.stat().st_size==size and h.hexdigest()==expected:
                return str(path)
            raise RuntimeError('existing pretrained file differs; preserve and inspect it')
        url=f"{base}/{REPO}/resolve/{revision}/model.safetensors"
        response=session.get(url,stream=True,timeout=(15,60)); response.raise_for_status()
        h=hashlib.sha256();count=0
        tmp=path.with_suffix('.download')
        with tmp.open('wb') as f:
            for block in response.iter_content(4*2**20):
                f.write(block);h.update(block);count+=len(block)
        if count!=size or h.hexdigest()!=expected:
            raise RuntimeError('pretrained size/SHA256 mismatch')
        tmp.replace(path)
    save_json(root/'source.json',{'repo':REPO,'revision':revision,'file':'model.safetensors',
              'sha256':expected,'bytes':size,'retrieval_host':base,'verification':'SHA256 against mirrored Git-LFS metadata',
              'license':info.json().get('cardData',{}).get('license'),'network':'DIRECT_ETHERNET'})
    save_json(root/'connections.json',network.receipt())
    print(json.dumps({'file':str(path),'bytes':size,'sha256':expected,'network':'DIRECT_ETHERNET'}),flush=True)
    return str(path)


if __name__=='__main__':fetch()
