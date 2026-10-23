"""Resumable Kaggle ZIP download to F: with explicit transport selection.

By default every socket uses the physical Ethernet interface. An explicit
--proxy selects the user's local HTTP proxy for both metadata and data.
Neither mode falls back to the other or changes system-wide settings.
The ZIP is stored as verified parts, which works on exFAT without sparse files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
from urllib.parse import urlsplit
import zipfile
from collections import OrderedDict

import requests

from .common import save_json
from .direct_network import DirectNetwork

COMPETITION = "rsna-knee-abnormality-detection"


class RetryableTransferError(RuntimeError):
    """A transport response that can be retried without discarding valid bytes."""


class FatalDownloadError(RuntimeError):
    """Archive identity, range integrity, or local storage needs attention."""


class ChunkedArchive:
    """Seekable ZIP reader over completed range files; no second ZIP copy."""
    def __init__(self, directory, length, chunk_size):
        self.directory, self.length, self.chunk_size = Path(directory), length, chunk_size
        self.position = 0
        self.handles = OrderedDict()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if position < 0: raise ValueError("negative archive seek")
        self.position = position
        return position
    def read(self, size=-1):
        size = max(0,self.length-self.position) if size < 0 else min(size,max(0,self.length-self.position))
        parts=[]
        while size:
            index, offset = divmod(self.position,self.chunk_size)
            count=min(size,self.chunk_size-offset)
            if index not in self.handles:
                self.handles[index]=(self.directory/f"{index:06d}.part").open("rb")
            self.handles.move_to_end(index)
            while len(self.handles)>8:self.handles.popitem(last=False)[1].close()
            f=self.handles[index];f.seek(offset);block=f.read(count)
            if len(block)!=count:raise RuntimeError("incomplete archive range")
            parts.append(block); self.position+=count; size-=count
        return b"".join(parts)
    def close(self):
        for f in self.handles.values():f.close()
        self.handles.clear()
    def __del__(self):self.close()


class Downloader:
    def __init__(self, root, source_ip, interface, workers=6, chunk_mib=64, data_ip=None, proxy=None):
        if proxy and data_ip:raise ValueError('proxy mode cannot pin a direct data IP')
        self.root = Path(root).resolve()
        self.control = self.root / "download"
        self.control.mkdir(parents=True, exist_ok=True)
        self.archive = self.control / (COMPETITION + ".zip.part")
        self.parts = self.control / "parts"
        if proxy:
            from .proxy_network import LocalProxyNetwork
            self.net=LocalProxyNetwork(proxy)
        else:self.net=DirectNetwork(source_ip, interface)
        if data_ip:
            import ipaddress
            if not ipaddress.ip_address(data_ip).is_global:raise ValueError('data endpoint must be a public IP')
        self.data_ip=data_ip
        # The Kaggle SDK creates requests.Session instances internally. Only
        # this dedicated downloader process is changed, never system settings.
        requests.Session = self.net.session
        self.workers, self.chunk_size = workers, chunk_mib * 2**20
        self.lock, self.url_lock = threading.Lock(), threading.Lock()
        self.active_bytes, self.network_bytes = {}, 0
        self.retry_count, self.last_retry = 0, None
        self.url, self.meta, self.phase = None, {}, "initializing"
        self.started = time.monotonic()
        self.last_connection_count = -1
        self.stop = threading.Event()
        self.db = sqlite3.connect(self.control / "resume.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, bytes INTEGER, sha256 TEXT)")
        self.db.commit()

    def refresh_url(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
        with self.url_lock:
            api = KaggleApi(); api.authenticate()
            with api.build_kaggle_client() as client:
                request = ApiDownloadDataFilesRequest(); request.competition_name = COMPETITION
                redirect = client.competitions.competition_api_client.download_data_files(request)
            url = redirect.url
            host = urlsplit(url).hostname or ""
            if not (host == "storage.googleapis.com" or host.endswith(".googleapis.com")):
                raise RuntimeError("unexpected data host; refusing download")
            if self.data_ip:
                # Pin a tested service endpoint, keeping original Host/SNI and
                # TLS certificate verification. Physical-interface routing is unchanged.
                self.net.cache[host]=(time.monotonic()+86400,[self.data_ip])
            with self.net.session() as session:
                r = session.head(url, headers={"Accept-Encoding": "identity"}, timeout=(15,45), allow_redirects=True)
                r.raise_for_status()
                meta = {"competition": COMPETITION, "host": urlsplit(r.url).hostname,
                        "bytes": int(r.headers["Content-Length"]), "etag": r.headers.get("ETag"),
                        "generation": r.headers.get("x-goog-generation"),
                        "google_hash": r.headers.get("x-goog-hash"), "chunk_size": self.chunk_size}
            old_path = self.control / "archive_metadata.json"
            if old_path.exists():
                old = json.loads(old_path.read_text(encoding="utf-8"))
                if any(old.get(k) != meta[k] for k in ("bytes","etag","generation","chunk_size")):
                    raise RuntimeError("archive version or chunk size changed; preserve old download and use a new root")
            self.url, self.meta = url, meta
            save_json(old_path, meta)  # signed URL and account credentials are never persisted

    def write_status(self, **extra):
        completed, saved = self.db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM chunks").fetchone()
        with self.lock:
            inflight, transferred = sum(self.active_bytes.values()), self.network_bytes
            retry_count, last_retry = self.retry_count, self.last_retry
        elapsed = time.monotonic() - self.started
        total = self.meta.get("bytes", 0)
        value = {"phase": self.phase, "pid": os.getpid(), "competition": COMPETITION,
                 "root": str(self.root), "archive_bytes": total,
                 "verified_downloaded_bytes": saved, "inflight_bytes": inflight,
                 "downloaded_percent": 100 * (saved + inflight) / total if total else 0,
                 "completed_chunks": completed, "network_bytes_this_process": transferred,
                 "workers": self.workers, "retry_count": retry_count, "last_retry": last_retry,
                 "elapsed_seconds_this_process": elapsed,
                 "average_mib_per_second_this_process": transferred / 2**20 / max(elapsed,1),
                 "network_mode": getattr(self.net,'mode','DIRECT_ETHERNET'), "proxy_fallback": False,
                 "proxy_url": getattr(self.net,'proxy_url',None), "transport_fallback":False,
                 "source_ip": self.net.source_ip, "interface_index": self.net.interface_index,
                 "updated_unix": time.time(), **extra}
        save_json(self.control / "status.json", value)
        receipt = self.net.receipt()
        if len(receipt["connections"]) != self.last_connection_count:
            save_json(self.control / "connections.json", receipt)
            self.last_connection_count = len(receipt["connections"])
        return value

    def initialize_archive(self):
        size = self.meta["bytes"]
        self.parts.mkdir(exist_ok=True)
        downloaded = self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM chunks").fetchone()[0]
        if shutil.disk_usage(self.root).free < size - downloaded + 20 * 2**30:
            raise RuntimeError("insufficient free space for archive")
        for index,expected in self.db.execute("SELECT id,bytes FROM chunks"):
            p=self.parts/f"{index:06d}.part"
            if not p.exists() or p.stat().st_size!=expected:
                raise RuntimeError("verified archive part missing or wrong size")

    def fetch_chunk(self, index):
        start = index * self.chunk_size
        end = min(start + self.chunk_size, self.meta["bytes"]) - 1
        dest = self.parts / f"{index:06d}.part"
        temporary = self.parts / f"{index:06d}.tmp"
        if dest.exists() and dest.stat().st_size==end-start+1:
            digest=hashlib.sha256()
            with dest.open('rb') as f:
                for block in iter(lambda:f.read(4*2**20),b''):digest.update(block)
            return index,end-start+1,digest.hexdigest()
        request_bytes=8*2**20
        checkpoint=self.parts/f'{index:06d}.progress.json'
        length=end-start+1
        stored=temporary.stat().st_size if temporary.exists() else 0
        # Legacy files only committed complete 8 MiB ranges. New sidecars bind
        # arbitrary-length prefixes to both the archive identity and their SHA.
        saved=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
        if saved:
            if (saved['etag']!=self.meta['etag'] or saved['archive_bytes']!=self.meta['bytes']
                    or saved['chunk_start']!=start or not 0<=saved['bytes']<=min(stored,length)):
                raise FatalDownloadError('partial checkpoint identity or length mismatch')
            count=saved['bytes']
        else:
            count=min(stored//request_bytes*request_bytes,length//request_bytes*request_bytes)
        digest=hashlib.sha256()
        with temporary.open('r+b' if temporary.exists() else 'w+b',buffering=0) as f:
            left=count
            while left:
                block=f.read(min(left,2**20))
                if not block:raise FatalDownloadError('partial checkpoint unexpectedly truncated')
                digest.update(block);left-=len(block)
            if saved and saved['sha256']!=digest.hexdigest():
                raise FatalDownloadError('partial checkpoint SHA mismatch')
            f.truncate(count);f.seek(count)

            def preserve():
                save_json(checkpoint,{'etag':self.meta['etag'],'archive_bytes':self.meta['bytes'],
                          'chunk_start':start,'bytes':count,'sha256':digest.hexdigest()})

            preserve()
            with self.lock:self.active_bytes[index]=count
            attempt=0
            # Keep TLS/TCP connections alive between successive ranges. A slow
            # but progressing stream is valid; only idle reads time out.
            with self.net.session() as session:
                while count<length:
                    if self.stop.is_set() or (self.control/'STOP').exists():
                        preserve();raise InterruptedError('download paused')
                    segment_start=start+count;segment_end=min(segment_start+request_bytes-1,end)
                    headers={'Range':f'bytes={segment_start}-{segment_end}','Accept-Encoding':'identity'}
                    if self.meta['etag']:headers['If-Match']=self.meta['etag']
                    before=count
                    try:
                        with session.get(self.url,headers=headers,stream=True,timeout=(15,30)) as r:
                            if r.status_code==403:
                                self.refresh_url()
                                raise RetryableTransferError('signed_url_refreshed')
                            if r.status_code in (408,429,500,502,503,504):
                                raise RetryableTransferError(f'http_{r.status_code}')
                            if (r.status_code!=206 or
                                    r.headers.get('Content-Range')!=f'bytes {segment_start}-{segment_end}/{self.meta["bytes"]}'):
                                raise FatalDownloadError(f'invalid range response: HTTP {r.status_code}')
                            if r.headers.get('ETag') not in (None,self.meta['etag']):
                                raise FatalDownloadError('archive ETag changed')
                            segment_count=0;checkpoint_count=count
                            for block in r.iter_content(64*2**10):
                                if self.stop.is_set() or (self.control/'STOP').exists():
                                    preserve();raise InterruptedError('download paused')
                                if segment_count+len(block)>segment_end-segment_start+1:
                                    raise FatalDownloadError('range exceeded')
                                if f.write(block)!=len(block):raise OSError('short archive write')
                                digest.update(block);count+=len(block);segment_count+=len(block)
                                with self.lock:
                                    self.active_bytes[index]=count;self.network_bytes+=len(block)
                                if count-checkpoint_count>=2**20:
                                    preserve();checkpoint_count=count
                            if segment_count!=segment_end-segment_start+1:
                                raise RetryableTransferError('truncated_range')
                        preserve();attempt=0
                    except (requests.RequestException,RetryableTransferError) as exc:
                        preserve()
                        attempt=1 if count>before else attempt+1
                        delay=min(30,2**min(attempt-1,5))
                        # Never persist exception text from requests: it can
                        # contain the signed URL. Our own reason codes are safe.
                        reason=str(exc) if isinstance(exc,RetryableTransferError) else type(exc).__name__
                        event={'chunk':index,'consecutive_no_progress_retries':attempt,
                               'reason':reason,'retained_bytes':count,'backoff_seconds':delay}
                        with self.lock:
                            self.retry_count=getattr(self,'retry_count',0)+1
                            self.last_retry=event
                        print(json.dumps({'event':'retry','details':event}),flush=True)
                        # Transient failures keep retrying with bounded backoff;
                        # one unlucky range must not abort all other workers.
                        self.stop.wait(delay)
            os.fsync(f.fileno())
        temporary.replace(dest)
        checkpoint.unlink(missing_ok=True)
        return index,count,digest.hexdigest()

    def download(self):
        self.phase = "downloading"
        count = math.ceil(self.meta["bytes"] / self.chunk_size)
        done = {r[0] for r in self.db.execute("SELECT id FROM chunks")}
        pending = [i for i in range(count) if i not in done]
        # Prioritize the final GiB (central directory) and then the data stream.
        tail = [i for i in reversed(range(max(0,count-16),count)) if i in pending]
        pending = tail + [i for i in pending if i not in set(tail)]
        source = iter(pending)
        futures = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in range(self.workers):
                index = next(source, None)
                if index is not None:
                    futures[pool.submit(self.fetch_chunk,index)] = index
            while futures:
                complete, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in complete:
                    futures.pop(future)
                    try:
                        index, size, digest = future.result()
                    except Exception:
                        self.stop.set()
                        raise
                    self.db.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?)",(index,size,digest))
                    self.db.commit()
                    with self.lock:
                        self.active_bytes.pop(index,None)
                    following = next(source,None)
                    if following is not None:
                        futures[pool.submit(self.fetch_chunk,following)] = following
                status = self.write_status()
                if complete:
                    print(json.dumps({k:status[k] for k in ["phase","downloaded_percent","verified_downloaded_bytes","average_mib_per_second_this_process"]}),flush=True)
        if self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] != count:
            raise RuntimeError("chunk coverage incomplete")

    def extract(self):
        self.phase = "extracting"
        raw = self.root / "raw"; raw.mkdir(exist_ok=True)
        with zipfile.ZipFile(ChunkedArchive(self.parts,self.meta["bytes"],self.chunk_size)) as archive:
            entries = [x for x in archive.infolist() if not x.is_dir()]
            total = sum(x.file_size for x in entries)
            if shutil.disk_usage(self.root).free < total + 20 * 2**30:
                raise RuntimeError("insufficient disk for extraction; archive preserved")
            save_json(self.control / "zip_inventory.json", {"file_count":len(entries),"uncompressed_bytes":total,
                      "compressed_bytes":self.meta["bytes"]})
            completed, extracted = 0, 0
            def one(info):
                rel = PurePosixPath(info.filename)
                if rel.is_absolute() or ".." in rel.parts or ":" in info.filename or "\\" in info.filename:
                    raise RuntimeError("unsafe ZIP path")
                dest = raw.joinpath(*rel.parts)
                if not dest.resolve().is_relative_to(raw.resolve()):
                    raise RuntimeError("ZIP path escaped raw directory")
                if dest.exists() and dest.stat().st_size == info.file_size:
                    # Resume validation checks CRC, not just file size.
                    import zlib
                    crc = 0
                    with dest.open("rb") as f:
                        for b in iter(lambda:f.read(4*2**20),b""):
                            crc = zlib.crc32(b,crc)
                    if crc & 0xffffffff == info.CRC:
                        return info.file_size
                dest.parent.mkdir(parents=True,exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".download_tmp")
                with archive.open(info) as source, tmp.open("wb") as target:
                    shutil.copyfileobj(source,target,4*2**20)  # ZipExtFile verifies CRC at EOF
                if tmp.stat().st_size != info.file_size:
                    raise RuntimeError("uncompressed size mismatch")
                tmp.replace(dest)
                return info.file_size
            with ThreadPoolExecutor(max_workers=min(4,self.workers)) as pool:
                # executor.map eagerly queues large iterables on Python 3.12;
                # use a bounded pending set instead.
                iterator=iter(entries); fs=set()
                for _ in range(min(16,len(entries))):
                    fs.add(pool.submit(one,next(iterator)))
                last_status = 0.
                while fs:
                    ready,fs=wait(fs,timeout=5,return_when=FIRST_COMPLETED)
                    for f in ready:
                        extracted += f.result(); completed += 1
                        item=next(iterator,None)
                        if item is not None:fs.add(pool.submit(one,item))
                    if time.monotonic()-last_status>5 or not fs:
                        self.write_status(extracted_files=completed,total_files=len(entries),
                                          extracted_bytes=extracted,uncompressed_bytes=total)
                        last_status=time.monotonic()
        for name in ("train.csv","train_series.csv","test.csv","test_series.csv"):
            if not (raw/name).exists():
                raise RuntimeError(f"required competition file missing: {name}")
        self.phase="complete"
        self.write_status(raw_data=str(raw),extracted_files=completed,total_files=len(entries),
                          extracted_bytes=extracted,crc_verified=True)
        print("Download and CRC-verified extraction complete: " + str(raw),flush=True)

    def verify_and_metadata(self):
        import base64,re
        self.phase='verifying'
        digest=hashlib.md5();verified=0;last=0.
        for index in range(math.ceil(self.meta['bytes']/self.chunk_size)):
            with (self.parts/f'{index:06d}.part').open('rb') as f:
                for block in iter(lambda:f.read(8*2**20),b''):
                    digest.update(block);verified+=len(block)
                    if time.monotonic()-last>5:
                        self.write_status(verified_archive_bytes=verified);last=time.monotonic()
        expected=re.search(r'(?:^|,)\s*md5=([^,]+)',self.meta.get('google_hash') or '')
        actual=base64.b64encode(digest.digest()).decode()
        if verified!=self.meta['bytes'] or (expected and expected.group(1)!=actual):
            raise RuntimeError('archive MD5/length verification failed')
        raw=self.root/'raw';raw.mkdir(exist_ok=True)
        reader=ChunkedArchive(self.parts,self.meta['bytes'],self.chunk_size)
        with zipfile.ZipFile(reader) as archive:
            entries=archive.infolist()
            for name in ('sample_submission.csv','train.csv','train_series.csv','test.csv','test_series.csv'):
                data=archive.read(name)
                (raw/name).write_bytes(data)
            count=len(entries);raw_bytes=sum(i.file_size for i in entries)
        reader.close()
        self.phase='complete'
        self.write_status(data_layout='verified_chunked_zip',metadata_root=str(raw),
                          total_files=count,uncompressed_bytes=raw_bytes,archive_md5=actual,
                          md5_verified=bool(expected),training_cache='created directly from archive by training.local_run')
        print('Archive verified; all source MRI data is on F:; metadata extracted.',flush=True)

    def run(self, extract_raw=False):
        try:
            # Authenticated link acquisition and the archive HEAD request also
            # cross the direct network. Retry transient startup failures just
            # like range failures, without reporting existing data as 0%.
            metadata_path=self.control/'archive_metadata.json'
            if metadata_path.exists():self.meta=json.loads(metadata_path.read_text(encoding='utf-8'))
            attempt=0
            while True:
                if self.stop.is_set() or (self.control/'STOP').exists():
                    raise InterruptedError('download paused')
                try:
                    self.refresh_url()
                    break
                except requests.exceptions.SSLError:
                    raise
                except (requests.RequestException,OSError) as exc:
                    if isinstance(exc,requests.HTTPError) and exc.response is not None:
                        if exc.response.status_code not in (408,429,500,502,503,504):raise
                    attempt+=1;delay=min(30,2**min(attempt-1,5))
                    event={'stage':'archive_link','reason':type(exc).__name__,
                           'attempt':attempt,'backoff_seconds':delay}
                    with self.lock:
                        self.retry_count+=1;self.last_retry=event
                    self.phase='reconnecting';self.write_status()
                    print(json.dumps({'event':'retry','details':event}),flush=True)
                    self.stop.wait(delay)
            self.phase='initializing';self.initialize_archive();self.write_status()
            self.download()
            if extract_raw:self.extract()
            else:self.verify_and_metadata()
        except BaseException as exc:
            self.stop.set()
            self.phase="paused" if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else "failed"
            reason=str(exc) if isinstance(exc,(RuntimeError,InterruptedError)) else type(exc).__name__
            if "http" in reason:reason=type(exc).__name__
            self.write_status(error_type=type(exc).__name__,reason=reason)
            print(self.phase+": "+type(exc).__name__+"; data retained, no proxy fallback",flush=True)
            raise SystemExit(1) from None
        finally:
            self.db.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="F:/rsnaknee")
    p.add_argument("--source-ip",default="192.168.1.31")
    p.add_argument("--interface",type=int,default=11)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--chunk-mib",type=int,default=64)
    p.add_argument('--extract-raw',action='store_true',help='optional: materialize all DICOM small files; not needed by archive cache')
    p.add_argument('--data-ip',help='optional tested Google Storage endpoint; TLS hostname stays verified')
    p.add_argument('--proxy',help='explicitly use the user-authorized local HTTP proxy, e.g. http://127.0.0.1:7890')
    a=p.parse_args()
    Downloader(a.root,a.source_ip,a.interface,a.workers,a.chunk_mib,a.data_ip,a.proxy).run(a.extract_raw)
