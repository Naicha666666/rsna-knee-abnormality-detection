"""Read original DICOM bytes directly from the F: archive into Series arrays.

Avoids materializing 819,078 training DICOM small files on exFAT. The geometry,
photometric, and rendering calculations are the same as the directory reader.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import io
import json
import sqlite3
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import pydicom

from . import dicom_geometry as geom
from .common import LABELS, UID, SLOTS, file_hash, save_json
from .download_data import ChunkedArchive


def open_archive(root):
    control=Path(root)/'download'
    meta=json.loads((control/'archive_metadata.json').read_text(encoding='utf-8'))
    reader=ChunkedArchive(control/'parts',meta['bytes'],meta['chunk_size'])
    return zipfile.ZipFile(reader),reader,meta


def prepare(root='F:/rsnaknee',output=None,*,split='train',size=384,fov_mm=130.,study_ids=None):
    root=Path(root);output=Path(output or root/'cache384');output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    archive,reader,meta=open_archive(root)
    try:
        series_bytes=archive.read(f'{split}_series.csv')
        frame=pd.read_csv(io.BytesIO(series_bytes),dtype={UID:str,'SeriesInstanceUID':str})
        expected=pd.read_csv(io.BytesIO(archive.read(f'{split}.csv')),dtype={UID:str})[UID].tolist()
        if frame['SeriesInstanceUID'].duplicated().any() or len(expected)!=len(set(expected)):
            raise ValueError('duplicate source identities')
        if study_ids is not None:
            chosen=set(map(str,study_ids));expected=[u for u in expected if u in chosen]
            if chosen-set(expected):raise ValueError('pilot IDs not in source metadata')
        grouped=defaultdict(list)
        for item in archive.infolist():
            parts=item.filename.split('/')
            if len(parts)==4 and parts[0]==f'{split}_series' and parts[-1].endswith('.dcm'):
                grouped[(parts[1],parts[2])].append(item)
        with sqlite3.connect(root/'download/resume.sqlite3') as db:
            ready={r[0] for r in db.execute('SELECT id FROM chunks')}
        rows={u:x.to_dict('records') for u,x in frame.groupby(UID,sort=False)}
        for uid in expected:
            for row in rows.get(uid,[]):
                for item in grouped[(uid,row['SeriesInstanceUID'])]:
                    end=item.header_offset+len(item.filename.encode())+len(item.extra)+128+item.compress_size
                    required=set(range(item.header_offset//meta['chunk_size'],end//meta['chunk_size']+1))
                    if not required<=ready:
                        raise RuntimeError('requested MRI ranges not downloaded; no partial-study training')
        config={'size':size,'fov_mm':fov_mm,'geometry_hash':file_hash(geom.__file__),
                'builder_hash':file_hash(__file__),'source':'kaggle_archive_crc_per_entry',
                'archive_etag':meta['etag'],'archive_generation':meta['generation'],
                'series_csv_sha256':hashlib.sha256(series_bytes).hexdigest()}
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        studies=[];patients=[];start=time.perf_counter();read_bytes=0
        for number,uid in enumerate(expected):
            key=hashlib.sha256(uid.encode()).hexdigest()[:24]
            study_path=output/'studies'/f'{key}.json'
            if study_path.exists():
                saved=json.loads(study_path.read_text(encoding='utf-8'))
                if saved.get('config_hash')!=signature:
                    raise RuntimeError('cache preprocessing changed; use a new output directory')
                if all(Path(s['path']).exists() for s in saved['series']):
                    studies.append(saved);patients.append({UID:uid,'PatientID':saved['PatientID']});continue
            datasets={};orders={};errors=[];tags=set();sagittal=[];pids=set()
            t=time.perf_counter()
            for row in rows.get(uid,[]):
                sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane']
                if plane not in geom.CARDINAL_NORMAL:
                    errors.append({'series':sid,'error':'unsupported plane/localizer'});continue
                try:
                    objects={}
                    for item in grouped[(uid,sid)]:
                        raw=archive.read(item)  # CRC checked by ZipExtFile
                        read_bytes+=len(raw)
                        ds=pydicom.dcmread(io.BytesIO(raw))
                        objects[item.filename]=ds
                    records=[geom.header_from_dataset(ds,name) for name,ds in objects.items()]
                    order=geom.order_series(Path(sid),plane,provided_records=records)
                    datasets[sid]=objects;orders[sid]=order;tags.update(order.laterality_values)
                    if plane=='Sagittal' and order.ipp_x_median is not None:sagittal.append(order.ipp_x_median)
                    pids.update(str(getattr(ds,'PatientID','')).strip() for ds in objects.values())
                except Exception as exc:errors.append({'series':sid,'error':str(exc)[:300]})
            pids.discard('')
            if len(pids)>1:raise ValueError(f'conflicting PatientID within {uid}')
            try:side=geom.decide_study_side(uid,tags,sagittal).side
            except Exception as exc:
                side=None;errors.append({'series':'study','error':str(exc)[:300]})
            prepared=[]
            for row in rows.get(uid,[]):
                sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane']
                if sid not in orders:continue
                order=orders[sid];images=[];positions=[];fovs=[];names=[];bad=[]
                for name,pos in zip(order.files,order.positions_mm):
                    try:
                        payload=geom.canonicalize_payload(geom.pixels_from_dataset(datasets[sid][name],path=name),plane)
                        available=min(payload.image.shape[0]*payload.spacing[0],payload.image.shape[1]*payload.spacing[1])
                        image,receipt=geom.render_canonical_payload(payload,plane=plane,side=side,
                            crop_mm=min(fov_mm,available*.98),output_size=size)
                        if image is None:raise ValueError(receipt.reason)
                        images.append(image);positions.append(float(pos));fovs.append(min(receipt.realized_fov_x_mm,receipt.realized_fov_y_mm));names.append(Path(name).name)
                    except Exception as exc:bad.append({'file':Path(name).name,'error':str(exc)[:300]})
                if not images:
                    errors.append({'series':sid,'error':'all pixel decodes failed','details':bad[:2]});continue
                skey=hashlib.sha256((uid+'/'+sid).encode()).hexdigest()[:24]
                path=output/'series'/f'{skey}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.npy.tmp')
                with tmp.open('wb') as f:np.save(f,np.stack(images),allow_pickle=False)
                tmp.replace(path)
                fs=int(row['Fat_Suppression']) if pd.notna(row.get('Fat_Suppression')) else -1
                if fs not in (0,1):fs=-1
                prepared.append({UID:uid,'SeriesInstanceUID':sid,'path':str(path.resolve()),'slot':SLOTS[(plane,fs)],
                                 'plane':plane,'fat_suppression':fs,'fluid_sensitive':int(row['Fluid_Sensitive']) if pd.notna(row.get('Fluid_Sensitive')) else -1,
                                 'positions':positions,'fov_mm':fovs,'files':names,'errors':bad,
                                 'header_sha256':order.header_sha256,'config_hash':signature,'side':side})
            study={UID:uid,'PatientID':next(iter(pids),uid),'series':prepared,'errors':errors,'config_hash':signature}
            save_json(study_path,study);studies.append(study);patients.append({UID:uid,'PatientID':study['PatientID']})
            progress={'processed':number+1,'expected':len(expected),'seconds':time.perf_counter()-start,
                      'last_study_seconds':time.perf_counter()-t,'source_bytes_read':read_bytes,
                      'failed_studies':sum(not x['series'] for x in studies),'scope':'pilot' if study_ids is not None else 'complete_split'}
            save_json(output/'prepare_progress.json',progress)
            if number%10==0 or number+1==len(expected):print(json.dumps(progress),flush=True)
            del datasets
        result={'config':config,'config_hash':signature,'studies':studies,
                'failed_studies':[s[UID] for s in studies if not s['series']],
                'scope':'explicit_pilot_subset' if study_ids is not None else 'complete_split'}
        save_json(output/'manifest.json',result)
        pd.DataFrame(patients).to_csv(output/'patients.csv',index=False)
        return result
    finally:
        archive.close();reader.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--output')
    p.add_argument('--pilot-selection');p.add_argument('--split',choices=['train','test'],default='train')
    a=p.parse_args();ids=json.loads(Path(a.pilot_selection).read_text(encoding='utf-8'))['studies'] if a.pilot_selection else None
    prepare(a.root,a.output,split=a.split,study_ids=ids)
