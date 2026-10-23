"""Read complete proposals or a small pose-region overlay on immutable tissues."""
from pathlib import Path
import json
import numpy as np

from .common import UID, file_hash

PIXEL_KEYS={'native_tissues_u8','supervision_support_u8'}


class AnatomyReader:
    """Bind runtime proposals to the producer and the exact native source.

    A file left by an earlier run is not an accepted proposal. Source digests
    are memoized by file identity so repeated epochs do not reread MRI bytes.
    """
    def __init__(self, directory, *, weights=None):
        from .anatomy_contract import producer_signature
        self.directory=Path(directory)
        manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        declared=manifest.get('active_producer_signature') or manifest.get('contract',{}).get('producer_signature')
        if not declared:raise ValueError('anatomy producer contract missing; rebuild proposals')
        self.signature=producer_signature(weights) if weights is not None else declared
        if declared!=self.signature:raise ValueError('anatomy producer changed; rebuild proposals')
        self.rows={}
        for row in manifest['series']:
            identity=(row[UID],row['SeriesInstanceUID'])
            if identity in self.rows:raise ValueError('duplicate anatomy Series identity')
            self.rows[identity]=row
        self.digests={}

    def digest(self,path):
        path=Path(path).resolve();stat=path.stat()
        identity=(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if identity not in self.digests:self.digests[identity]=file_hash(path)
        return self.digests[identity]

    def load(self,uid,series,geometry_path,*,include_pixels=True):
        from .anatomy_contract import cache_matches
        identity=(uid,series['SeriesInstanceUID'])
        row=self.rows.get(identity)
        if row is None:raise ValueError('anatomy Series not published in the accepted manifest')
        if not cache_matches(row,self.signature):raise ValueError('stale anatomy producer; rebuild proposals')
        path=Path(row['path'])
        receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (receipt.get(UID),receipt.get('SeriesInstanceUID'))!=identity:
            raise ValueError('anatomy receipt belongs to another Series')
        if not cache_matches(receipt,self.signature):raise ValueError('stale anatomy payload receipt')
        if Path(receipt['path']).resolve()!=path.resolve():raise ValueError('anatomy payload path changed')
        for key,source in [('native_sha256',series['path']),('geometry_sha256',geometry_path)]:
            expected=row.get(key)
            if not expected or receipt.get(key)!=expected or source is None or self.digest(source)!=expected:
                raise ValueError('anatomy source changed: '+key)
        return load_anatomy(path,include_pixels)


def load_anatomy(path,include_pixels=True):
    path=Path(path).resolve()
    with np.load(path,allow_pickle=False) as archive:
        base=str(archive['_base_payload'].item()) if '_base_payload' in archive.files else None
        data={k:archive[k] for k in archive.files if k!='_base_payload' and (include_pixels or k not in PIXEL_KEYS)}
    if base:
        source=(path.parent/base).resolve()
        expected=(path.parent.parent/'anatomy_v2'/path.name).resolve()
        if source!=expected:raise ValueError('anatomy overlay source escaped its immutable source cache')
        with np.load(source,allow_pickle=False) as archive:
            inherited={k:archive[k] for k in archive.files if k not in data and (include_pixels or k not in PIXEL_KEYS)}
        inherited.update(data);data=inherited
    return data
