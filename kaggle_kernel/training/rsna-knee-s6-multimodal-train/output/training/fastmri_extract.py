"""Seekable, integrity-checked XZ blocks -> tar members -> RSS-only HDF5.

Networking belongs to fastmri_download.BlockCache. This module never reads or
records signed URLs. It does not decompress or hash an entire source HDF5.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
import time
import zlib

import h5py
import numpy as np

from .common import file_hash, save_json


def encode_vli(value):
    if not 0 <= value < 2**63: raise ValueError('invalid XZ variable integer')
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128); value >>= 7
    out.append(value)
    return bytes(out)


def decode_vli(data, offset):
    value = 0
    for i in range(9):
        if offset+i >= len(data): raise ValueError('truncated XZ variable integer')
        byte = data[offset+i]
        if i and byte == 0: raise ValueError('noncanonical XZ variable integer')
        value |= (byte & 127) << (7*i)
        if byte < 128: return value, offset+i+1
    raise ValueError('oversized XZ variable integer')


def one_block_stream(header, compressed_block, unpadded_bytes, uncompressed_bytes):
    """Keep the original block/check intact; generate only its small index/footer."""
    if len(compressed_block) != (unpadded_bytes+3)//4*4:
        raise ValueError('compressed XZ block size differs from verified index')
    index = b'\0'+encode_vli(1)+encode_vli(unpadded_bytes)+encode_vli(uncompressed_bytes)
    index += b'\0'*(-len(index) % 4)
    index += struct.pack('<I', zlib.crc32(index))
    footer_fields = struct.pack('<I', len(index)//4-1)+header[6:8]
    footer = struct.pack('<I', zlib.crc32(footer_fields))+footer_fields+b'YZ'
    return header[:12]+compressed_block+index+footer


def validate_index(metadata, prefix, index, footer):
    """Check root's parsed table against the official bytes and both CRC32s."""
    header = prefix[:12]
    if len(header) != 12 or header[:6] != b'\xfd7zXZ\0': raise ValueError('invalid XZ stream header')
    if zlib.crc32(header[6:8]) != struct.unpack('<I', header[8:12])[0]: raise ValueError('XZ header CRC mismatch')
    if len(footer) != 12 or footer[-2:] != b'YZ': raise ValueError('invalid XZ footer')
    if zlib.crc32(footer[4:10]) != struct.unpack('<I', footer[:4])[0]: raise ValueError('XZ footer CRC mismatch')
    if footer[8:10] != header[6:8]: raise ValueError('XZ header/footer flags differ')
    if (struct.unpack('<I', footer[4:8])[0]+1)*4 != len(index): raise ValueError('XZ index extent differs from footer')
    if len(index) < 8 or index[0] != 0 or zlib.crc32(index[:-4]) != struct.unpack('<I', index[-4:])[0]:
        raise ValueError('XZ index CRC mismatch')
    count, offset = decode_vli(index, 1); compressed_offset = 12; uncompressed_offset = 0; blocks = []
    for i in range(count):
        unpadded, offset = decode_vli(index, offset); size, offset = decode_vli(index, offset)
        padded = (unpadded+3)//4*4
        if not unpadded or not size: raise ValueError('empty XZ block unsupported')
        blocks.append({'block': i, 'compressed_offset': compressed_offset, 'unpadded_bytes': unpadded,
                       'padded_bytes': padded, 'uncompressed_offset': uncompressed_offset, 'uncompressed_bytes': size})
        compressed_offset += padded; uncompressed_offset += size
    if any(index[offset:-4]) or len(index)-4-offset > 3: raise ValueError('invalid XZ index padding')
    if blocks != metadata['blocks']: raise ValueError('XZ metadata table differs from verified index')
    if compressed_offset+len(index)+12 != metadata['bytes'] or uncompressed_offset != metadata['uncompressed_bytes']:
        raise ValueError('XZ extent differs from bound S3 object; concatenated streams unsupported')
    if header[6] != 0 or header[7] & 0xf0 or not lzma.is_check_supported(header[7] & 15):
        raise ValueError('unsupported XZ stream flags/check')
    if (header[7] & 15) == lzma.CHECK_NONE: raise ValueError('XZ source has no block integrity check')
    return header


class IndexedXZReader(io.RawIOBase):
    """Uncompressed seekable view; every fetched block passes liblzma's check."""
    def __init__(self, metadata, prefix, index, footer, get_block, cache_bytes=128*2**20, mark_verified=None):
        super().__init__()
        self.metadata = metadata; self.header = validate_index(metadata, prefix, index, footer)
        self.blocks = metadata['blocks']; self.offsets = [x['uncompressed_offset'] for x in self.blocks]
        self.size = metadata['uncompressed_bytes']; self.position = 0; self.get_block = get_block
        self.mark_verified = mark_verified
        self.cache_bytes = int(cache_bytes); self.cache = OrderedDict(); self.cached_bytes = 0
        self.block_receipts = {}; self.decode_seconds = 0.; self.decoded_bytes = 0
        self.binding = {'archive': metadata['name'], 'archive_bytes': metadata['bytes'], 'etag': metadata['etag'],
                        'xz_header_sha256': hashlib.sha256(self.header).hexdigest(),
                        'xz_index_sha256': hashlib.sha256(index).hexdigest()}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position+offset if whence == io.SEEK_CUR else self.size+offset if whence == io.SEEK_END else -1
        if position < 0: raise ValueError('invalid archive seek')
        self.position = int(position); return self.position

    def block_ids(self, offset, size):
        if size <= 0: return []
        if offset < 0 or offset+size > self.size: raise ValueError('read escapes uncompressed archive')
        return list(range(bisect_right(self.offsets, offset)-1, bisect_right(self.offsets, offset+size-1)))

    def _decoded_block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        row = self.blocks[i]; compressed = self.get_block(i)
        if not isinstance(compressed, bytes): compressed = bytes(compressed)
        started = time.perf_counter()
        data = lzma.decompress(one_block_stream(self.header, compressed, row['unpadded_bytes'], row['uncompressed_bytes']), format=lzma.FORMAT_XZ)
        self.decode_seconds += time.perf_counter()-started; self.decoded_bytes += len(data)
        if len(data) != row['uncompressed_bytes']: raise ValueError('XZ block decoded length differs')
        if self.mark_verified is not None: self.mark_verified(i)
        self.block_receipts[i] = {'block': i, 'compressed_sha256': hashlib.sha256(compressed).hexdigest(),
                                  'compressed_bytes': len(compressed), 'uncompressed_bytes': len(data),
                                  'xz_integrity_check_passed': True}
        while self.cache and self.cached_bytes+len(data) > self.cache_bytes:
            _, evicted = self.cache.popitem(last=False); self.cached_bytes -= len(evicted)
        if len(data) <= self.cache_bytes:
            self.cache[i] = data; self.cached_bytes += len(data)
        return data

    def read_at(self, offset, size, touched=None):
        size = min(max(0, int(size)), max(0, self.size-offset))
        if offset < 0: raise ValueError('negative archive read')
        parts = []
        for i in self.block_ids(offset, size):
            row = self.blocks[i]; data = self._decoded_block(i)
            start = max(offset-row['uncompressed_offset'], 0)
            stop = min(offset+size-row['uncompressed_offset'], len(data))
            parts.append(data[start:stop])
            if touched is not None: touched.add(i)
        return b''.join(parts)

    def read(self, size=-1):
        if size is None or size < 0: size = self.size-self.position
        data = self.read_at(self.position, size); self.position += len(data); return data

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)


class MemberView(io.RawIOBase):
    """HDF5 sees only its tar member, independently of the tar scanner cursor."""
    def __init__(self, archive, offset, size):
        super().__init__(); self.archive = archive; self.offset = offset; self.size = size
        self.position = 0; self.touched = set()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        value = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset if whence == 2 else -1
        if value < 0: raise ValueError('negative HDF5 member seek')
        self.position = int(value); return self.position

    def read(self, size=-1):
        remaining = max(0, self.size-self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        data = self.archive.read_at(self.offset+self.position, size, self.touched)
        self.position += len(data); return data

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)


def digest_value(value):
    """Value fingerprint that preserves dtype/shape and variable-length strings."""
    h = hashlib.sha256(); array = np.asarray(value)
    h.update(array.dtype.str.encode()); h.update(json.dumps(array.shape).encode())
    if array.dtype.kind == 'O':
        for item in array.flat:
            data = item if isinstance(item, bytes) else str(item).encode('utf-8')
            h.update(struct.pack('<Q', len(data))); h.update(data)
    else: h.update(array.tobytes(order='C'))
    return h.hexdigest()


def copy_attrs(source, destination):
    hashes = {}
    for name in source.attrs:
        value = source.attrs[name]
        destination.attrs.create(name, value, dtype=source.attrs.get_id(name).dtype)
        hashes[name] = digest_value(value)
    return hashes


def extract_member(archive, member, output, official_manifest_sha256):
    """Read RSS/header/attrs via HDF5 seeks; never read/hash the entire member."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    name = PurePosixPath(member['name']).name; destination = output/name; receipt_path = destination.with_suffix('.json')
    identity = {**archive.binding, 'member_name': member['name'], 'member_offset': member['offset'],
                'member_data_offset': member['offset_data'], 'member_bytes': member['size'],
                'tar_header_checksum': member['chksum'], 'tar_header_sha256': member['header_sha256'],
                'official_manifest_sha256': official_manifest_sha256}
    if destination.is_file() and receipt_path.is_file():
        old = json.loads(receipt_path.read_text(encoding='utf-8'))
        if old.get('source') != identity: raise ValueError('existing image is bound to a different source member')
        if file_hash(destination) == old.get('image_sha256'): return old
        raise ValueError('existing reduced HDF5 checksum changed')
    header_bytes = archive.read_at(member['offset'], member['offset_data']-member['offset'])
    if hashlib.sha256(header_bytes).hexdigest() != member['header_sha256']:
        raise ValueError('tar member header changed since discovery')
    with tempfile.NamedTemporaryFile(dir=output, prefix='.'+name+'.', suffix='.partial', delete=False) as handle:
        temporary = Path(handle.name)
    view = MemberView(archive, member['offset_data'], member['size']); started = time.perf_counter()
    try:
        with h5py.File(view, 'r') as source, h5py.File(temporary, 'w') as destination_file:
            if 'reconstruction_rss' not in source or 'ismrmrd_header' not in source:
                raise ValueError('official knee RSS/header fields are missing')
            rss = source['reconstruction_rss']
            if rss.ndim != 3 or tuple(rss.shape[1:]) != (320, 320) or rss.dtype.kind != 'f':
                raise ValueError('unexpected official knee RSS dimensions or dtype')
            values = rss[...]
            if not np.isfinite(values).all(): raise ValueError('nonfinite RSS source pixels')
            reduced = destination_file.create_dataset('reconstruction_rss', data=values, dtype=rss.dtype,
                compression='gzip', compression_opts=1, shuffle=True)
            rss_attrs = copy_attrs(rss, reduced); rss_hash = digest_value(values); shape = list(values.shape)
            header = source['ismrmrd_header']; header_value = header[()]
            reduced_header = destination_file.create_dataset('ismrmrd_header', data=header_value, dtype=header.dtype)
            header_attrs = copy_attrs(header, reduced_header); header_hash = digest_value(header_value)
            attrs = copy_attrs(source, destination_file)
        with h5py.File(temporary, 'r') as check:
            if digest_value(check['reconstruction_rss'][...]) != rss_hash or digest_value(check['ismrmrd_header'][()]) != header_hash:
                raise ValueError('reduced HDF5 changed source data')
            if {key: digest_value(check.attrs[key]) for key in check.attrs} != attrs:
                raise ValueError('reduced HDF5 changed source attributes')
        os.replace(temporary, destination)
        header_blocks = set(archive.block_ids(member['offset'], member['offset_data']-member['offset']))
        receipt = {'source': identity, 'retained_fields': ['reconstruction_rss', 'ismrmrd_header'],
                   'rss_shape': shape, 'rss_sha256': rss_hash, 'header_sha256': header_hash,
                   'attributes_sha256_by_name': attrs, 'rss_attributes_sha256_by_name': rss_attrs,
                   'header_attributes_sha256_by_name': header_attrs, 'image_sha256': file_hash(destination),
                   'image_bytes': destination.stat().st_size, 'source_hdf5_sha256': None,
                   'integrity_scope': 'Official member allowlist, tar header checksum, bound S3 object and verified checks on every accessed XZ block; unread kspace/HDF5 bytes are not claimed verified.',
                   'accessed_blocks': [archive.block_receipts[i] for i in sorted(view.touched | header_blocks)],
                   'seconds': time.perf_counter()-started, 'training_dispatch': False}
        save_json(receipt_path, receipt)
        return receipt
    finally:
        view.close(); temporary.unlink(missing_ok=True)


def extract_archive(reader, official_manifest, output, state_path, required_files=None, max_members=None):
    official_manifest = Path(official_manifest); names = set(official_manifest.read_text(encoding='utf-8').splitlines())
    if not names or any(not re.fullmatch(r'file\d+\.h5', name) for name in names): raise ValueError('invalid official member allowlist')
    wanted = names if required_files is None else names & set(required_files)
    source_contract = {**reader.binding, 'official_manifest_sha256': file_hash(official_manifest)}
    state_path = Path(state_path)
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'source': source_contract, 'members': {}, 'extracted': {}, 'next_tar_offset': 0, 'global_pax_headers': {}, 'scan_complete': False}
    if state['source'] != source_contract: raise ValueError('resume archive identity or official manifest changed')
    processed = 0
    def save_state():
        state.update(wanted_members=len(wanted), available_wanted=len(wanted & set(state['extracted'])),
                     distinct_verified_blocks=len(reader.block_receipts), decode_seconds=reader.decode_seconds,
                     training_dispatch=False)
        save_json(state_path, state)
    # A later request may add targets whose headers were already discovered.
    for name in sorted(wanted & set(state['members'])):
        record = extract_member(reader, state['members'][name], output, source_contract['official_manifest_sha256'])
        state['extracted'][name] = {'image_sha256': record['image_sha256'], 'image_bytes': record['image_bytes']}
    if state['scan_complete']:
        save_state(); return state
    reader.seek(state['next_tar_offset'])
    with tarfile.open(fileobj=reader, mode='r:', pax_headers=state['global_pax_headers']) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts: raise ValueError('unsafe tar member path')
            if member.isdir():
                state['next_tar_offset'] = archive.offset; state['global_pax_headers'] = archive.pax_headers; save_state(); continue
            name = path.name
            if not member.isfile() or name not in names: raise ValueError('tar file not listed in the official archive manifest')
            if name in state['members']: raise ValueError('duplicate official HDF5 member')
            record = {'name': member.name, 'offset': member.offset, 'offset_data': member.offset_data,
                      'size': member.size, 'chksum': member.chksum,
                      'header_sha256': hashlib.sha256(reader.read_at(member.offset, member.offset_data-member.offset)).hexdigest()}
            if name in wanted:
                result = extract_member(reader, record, output, source_contract['official_manifest_sha256'])
                state['extracted'][name] = {'image_sha256': result['image_sha256'], 'image_bytes': result['image_bytes']}
            state['members'][name] = record; state['next_tar_offset'] = archive.offset
            state['global_pax_headers'] = archive.pax_headers; processed += 1; save_state()
            print(json.dumps({'archive': reader.metadata['name'], 'members_scanned': len(state['members']),
                              'wanted_ready': state['available_wanted'], 'wanted_total': len(wanted)}), flush=True)
            if max_members is not None and processed >= max_members: return state
    if set(state['members']) != names: raise ValueError('tar finished without the complete official member set')
    state['scan_complete'] = True; save_state(); return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='F:/rsnaknee/reference_anatomy/fastmri_plus')
    parser.add_argument('--archive', required=True, choices=['knee_singlecoil_train.tar.xz', 'knee_singlecoil_val.tar.xz'])
    parser.add_argument('--required-files', help='optional one H5 basename per line; default retains every official member')
    parser.add_argument('--max-members', type=int)
    args = parser.parse_args(); root = Path(args.root)
    try:
        from .fastmri_download import BlockCache
        folder = root/'downloads'/args.archive
        split = 'val' if '_val.' in args.archive else 'train'
        wanted = Path(args.required_files).read_text(encoding='utf-8').splitlines() if args.required_files else None
        with BlockCache(root, args.archive) as cache:
            reader = IndexedXZReader(cache.meta, (folder/'prefix.bin').read_bytes(), (folder/'index.bin').read_bytes(),
                                     (folder/'footer.bin').read_bytes(), cache.get, mark_verified=cache.mark_verified)
            extract_archive(reader, root/'source_manifests'/f'knee_singlecoil_{split}.txt', root/'images',
                            root/'extraction'/f'{split}.json', wanted, args.max_members)
    except Exception as exc:
        # Downloader exceptions can contain request details. Never print a URL
        # or traceback from a signed request into the shared conversation/log.
        print(json.dumps({'phase': 'extraction_failed', 'error_type': type(exc).__name__,
                          'archive': args.archive, 'training_dispatch': False}), flush=True)
        return 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
