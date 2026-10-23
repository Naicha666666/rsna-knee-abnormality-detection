"""Resumable official fastMRI XZ block transport over the physical interface.

Signed access URLs stay in a local file outside the project and are never put
in receipts or errors. This cache downloads requested compressed blocks, not
unneeded k-space archives. The consumer verifies each XZ block's CRC on decode.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import requests

from .common import save_json
from .direct_network import DirectNetwork


class TransferStopped(RuntimeError):pass
class SourceChanged(RuntimeError):pass


class BlockCache:
    """root contains downloads/<archive>/metadata.json and the private access file."""
    def __init__(self,root,archive,*,workers=4,source_ip='192.168.1.31',interface_index=11):
        if not re.fullmatch(r'knee_singlecoil_(train|val)\.tar\.xz',archive):
            raise ValueError('only the needed single-coil train/val reference sources are enabled')
        self.root=Path(root);self.archive=archive;self.directory=self.root/'downloads'/archive
        self.meta=json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        self.blocks=self.meta['blocks'];self.prefix=(self.directory/'prefix.bin').read_bytes()
        self.network=DirectNetwork(source_ip,interface_index);self.local=threading.local()
        self.pool=ThreadPoolExecutor(max_workers=workers);self.sessions=[]
        self.lock=threading.RLock();self.block_locks={};self.started=time.perf_counter()
        self.received=0;self.retries=0;self.used=set();self.last_status=0.;self.last_error=None
        access=json.loads((self.root/'download_access.private.json').read_text(encoding='utf-8'))
        self.url=access['urls'][archive];parsed=urlsplit(self.url)
        if parsed.scheme!='https' or parsed.hostname!='fastmri-dataset.s3.amazonaws.com' or parsed.path!='/v2.0/'+archive:
            raise ValueError('access URL does not identify the authorized official object')
        self.stop_path=self.root/'downloads/STOP';self.chunk_bytes=4*2**20

    def _session(self):
        if not hasattr(self.local,'session'):
            self.local.session=self.network.session()
            with self.lock:self.sessions.append(self.local.session)
        return self.local.session

    def _stopped(self):
        if self.stop_path.exists():raise TransferStopped('fastMRI reference transfer stopped by local marker')

    def _status(self,force=False):
        with self.lock:
            now=time.perf_counter()
            if not force and now-self.last_status<5:return
            elapsed=now-self.started
            save_json(self.directory/'download_progress.json',{
                'archive':self.archive,'phase':'fetching_selected_XZ_blocks','full_archive_requested':False,
                'object_bytes':self.meta['bytes'],'blocks_accessed_this_process':len(self.used),
                'object_blocks':len(self.blocks),'network_bytes_this_process':self.received,
                'network_MiB_per_second':self.received/max(elapsed,1)/2**20,
                'seconds':elapsed,'retries':self.retries,'last_error_type':self.last_error,
                'transport':'physical Ethernet; proxy fallback disabled','source_ip':self.network.source_ip,
                'interface_index':self.network.interface_index,'training_dispatch':False})
            self.last_status=now

    def _piece(self,index,piece,start,length):
        self._stopped();folder=self.directory/'pieces'/f'{index:06d}';folder.mkdir(parents=True,exist_ok=True)
        done=folder/f'{piece:03d}.bin';part=folder/f'{piece:03d}.part';receipt=folder/f'{piece:03d}.json'
        identity={'archive':self.archive,'etag':self.meta['etag'],'offset':start,'length':length}
        record=json.loads(receipt.read_text()) if receipt.exists() else {}
        if record and any(record.get(k)!=v for k,v in identity.items()):raise SourceChanged('partial range belongs to another source object')
        if done.exists():
            data=done.read_bytes()
            if len(data)!=length or hashlib.sha256(data).hexdigest()!=record.get('sha256'):
                raise SourceChanged('cached range integrity mismatch')
            return data
        data=part.read_bytes() if part.exists() else b'';committed=int(record.get('bytes',0))
        if committed>len(data) or (committed and hashlib.sha256(data[:committed]).hexdigest()!=record.get('sha256')):
            raise SourceChanged('partial range checkpoint integrity mismatch')
        data=data[:committed]
        with part.open('wb') as stream:stream.write(data)
        digest=hashlib.sha256(data);attempt=0
        while len(data)<length:
            self._stopped();attempt+=1
            try:
                request_start=start+len(data);end=start+length-1
                headers={'Range':f'bytes={request_start}-{end}','If-Match':self.meta['etag'],'Accept-Encoding':'identity'}
                with self._session().get(self.url,headers=headers,stream=True,timeout=(20,45)) as response:
                    if response.status_code in (401,403,404,412):
                        raise SourceChanged('authorized source unavailable or changed: HTTP '+str(response.status_code))
                    if response.status_code in (408,429,500,502,503,504):
                        raise requests.ConnectionError('transient official transfer response')
                    expected=f'bytes {request_start}-{end}/{self.meta["bytes"]}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected or response.headers.get('ETag')!=self.meta['etag']:
                        raise SourceChanged('official response range or object identity mismatch')
                    with part.open('ab') as stream:
                        for chunk in response.iter_content(2**20):
                            self._stopped()
                            if not chunk:continue
                            if len(data)+len(chunk)>length:raise SourceChanged('range response exceeded requested bytes')
                            stream.write(chunk);data+=chunk;digest.update(chunk)
                            # A checkpoint always describes flushed bytes from this exact range.
                            stream.flush()
                            save_json(receipt,{**identity,'bytes':len(data),'sha256':digest.hexdigest(),'complete':False})
                            with self.lock:self.received+=len(chunk)
                            self._status()
                if len(data)<length:raise requests.ConnectionError('short authenticated source range')
            except requests.RequestException as exc:
                with self.lock:self.retries+=1;self.last_error=type(exc).__name__
                self._status(force=True)
                delay=min(30.,.5*2**min(attempt,6))
                until=time.monotonic()+delay
                while time.monotonic()<until:
                    self._stopped();time.sleep(max(0.,min(1.,until-time.monotonic())))
        part.replace(done);save_json(receipt,{**identity,'bytes':length,'sha256':digest.hexdigest(),'complete':True})
        return data

    def get(self,index):
        index=int(index)
        if not 0<=index<len(self.blocks):raise IndexError('XZ block outside authorized source')
        with self.lock:guard=self.block_locks.setdefault(index,threading.Lock())
        with guard:
            self._stopped();row=self.blocks[index]
            folder=self.directory/'blocks';folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{index:06d}.bin';receipt=folder/f'{index:06d}.json'
            if path.exists() and receipt.exists():
                info=json.loads(receipt.read_text());data=path.read_bytes()
                if (info.get('etag')!=self.meta['etag'] or len(data)!=row['padded_bytes']
                        or hashlib.sha256(data).hexdigest()!=info.get('sha256')):
                    raise SourceChanged('cached compressed block identity/integrity mismatch')
            else:
                total=int(row['padded_bytes']);start=int(row['compressed_offset'])
                jobs=[self.pool.submit(self._piece,index,j,start+offset,min(self.chunk_bytes,total-offset))
                      for j,offset in enumerate(range(0,total,self.chunk_bytes))]
                data=b''.join(job.result() for job in jobs)
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:stream.write(data)
                for attempt in range(7):
                    try:temporary.replace(path);break
                    except PermissionError:
                        if attempt==6:raise
                        time.sleep(min(.05*2**attempt,.8))
                save_json(receipt,{'block':index,'etag':self.meta['etag'],'bytes':len(data),
                    'sha256':hashlib.sha256(data).hexdigest(),'XZ_CRC_verified':False})
                # The immutable combined block plus its digest is now the resume boundary.
                pieces=self.directory/'pieces'/f'{index:06d}'
                for owned in pieces.glob('*'):
                    if re.fullmatch(r'\d{3}\.(bin|json)',owned.name):owned.unlink()
                if pieces.exists() and not any(pieces.iterdir()):pieces.rmdir()
            with self.lock:self.used.add(index)
            self._status(force=True)
            return data

    def mark_verified(self,index):
        """Called only after the consumer's liblzma CRC check succeeds."""
        path=self.directory/'blocks'/f'{int(index):06d}.json';record=json.loads(path.read_text())
        record['XZ_CRC_verified']=True;save_json(path,record)

    def close(self):
        self.pool.shutdown(wait=True)
        for session in self.sessions:session.close()
        self._status(force=True)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
