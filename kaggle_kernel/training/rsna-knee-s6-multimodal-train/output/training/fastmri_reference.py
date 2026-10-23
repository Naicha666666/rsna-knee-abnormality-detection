"""Read published fastMRI+ boxes without promoting them to disease/pixel truth.

The author's annotation viewer flips reconstruction_rss vertically. Coordinates
are continuous pixel edges on that viewer, not the untouched HDF5 array.
Unmarked slices and pixels remain unobserved. MRI access is obtained separately
through the official fastMRI data-sharing application.
"""
from pathlib import Path,PurePosixPath
import argparse
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from .common import file_hash,save_json

SOURCE='https://github.com/microsoft/fastmri-plus'
ORIENTATION_SOURCE=SOURCE+'/blob/main/ExampleScripts/example.ipynb'
SPACING_SOURCE=SOURCE+'/blob/main/ExampleScripts/fastmri_to_dicom.py'
RSS_SOURCE='https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md'


def reference_geometry(header,shape):
    """Read only geometry supported by the source header, in RSS row/col order.

    The official converter puts ISMRMRD x/y in DICOM row/column spacing order.
    The 320 RSS matrix is a centre crop, so it retains those pixel spacings.
    HFS, FrameOfReferenceUID and FOV_z are not orientation or slice positions.
    """
    shape=tuple(map(int,shape))
    if len(shape)!=3 or min(shape)<1:raise ValueError('invalid RSS volume shape')
    result={'native_shape':list(shape),'spacing_mm':None,'spacing_source':SPACING_SOURCE,
            'in_plane_fov_mm':None,'native_to_lps':None,'slice_positions_mm':None,
            'slice_spacing_mm':None,'laterality':None,'orientation_known':False,
            'rss_is_center_crop':True,'rss_source':RSS_SOURCE,'geometry_status':'header_missing'}
    if header is None:return result
    if isinstance(header,str):header=header.encode('utf-8')
    elif isinstance(header,np.ndarray):header=header.item()
    header=bytes(header)
    result['header_sha256']=hashlib.sha256(header).hexdigest()
    try:root=ET.fromstring(header)
    except ET.ParseError as e:raise ValueError('invalid ISMRMRD XML') from e
    for element in root.iter():element.tag=element.tag.split('}')[-1]
    encodings=root.findall('encoding')
    if len(encodings)!=1:
        result['geometry_status']='ambiguous_or_missing_encoding';return result
    encoding=encodings[0]
    def triple(prefix):
        values=[]
        for axis in ('x','y','z'):
            text=encoding.findtext(prefix+'/'+axis)
            if text is None:return None
            try:values.append(float(text))
            except ValueError:return None
        return values if np.isfinite(values).all() and min(values)>0 else None
    matrix=triple('reconSpace/matrixSize');fov=triple('reconSpace/fieldOfView_mm')
    result.update(recon_matrix_xyz=matrix,recon_fov_xyz_mm=fov,
                  encoded_matrix_xyz=triple('encodedSpace/matrixSize'),
                  encoded_fov_xyz_mm=triple('encodedSpace/fieldOfView_mm'),
                  patient_position=root.findtext('measurementInformation/patientPosition'),
                  protocol_name=root.findtext('measurementInformation/protocolName'),
                  sequence_type=root.findtext('sequenceParameters/sequence_type'))
    if matrix is None or fov is None or not np.equal(matrix,np.floor(matrix)).all():
        result['geometry_status']='invalid_recon_geometry';return result
    if shape[1]>matrix[0] or shape[2]>matrix[1]:
        result['geometry_status']='RSS_larger_than_recon_matrix';return result
    spacing=[fov[0]/matrix[0],fov[1]/matrix[1]]
    result.update(spacing_mm=spacing,in_plane_fov_mm=[shape[1]*spacing[0],shape[2]*spacing[1]],
                  recon_center_crop_origin_row_col=[(int(matrix[0])-shape[1])//2,(int(matrix[1])-shape[2])//2],
                  geometry_status='in_plane_spacing_known_LPS_unknown',
                  nominal_recon_partition_extent_mm=fov[2]/matrix[2])
    return result


def load_reference_volume(image_path,rows):
    """Complete RSS volume and marked source boxes; never substitutes ESC.

    Images copied out of the official archive may omit k-space, but must keep
    the original filename, RSS pixels and XML. Unmarked space stays unknown.
    """
    import h5py
    image_path=Path(image_path)
    if 'file' not in rows or rows.empty or set(rows['file'])!={image_path.stem}:
        raise ValueError('annotation and source volume identities differ')
    with h5py.File(image_path,'r') as handle:
        if 'reconstruction_rss' not in handle:raise ValueError('reference RSS image missing')
        dataset=handle['reconstruction_rss']
        if dataset.ndim!=3:raise ValueError('invalid RSS volume shape')
        volume=np.asarray(dataset[:],np.float32)
        header=handle['ismrmrd_header'][()] if 'ismrmrd_header' in handle else None
        attrs={}
        for key in ('acquisition','max','norm'):
            if key not in handle.attrs:continue
            value=handle.attrs[key]
            if isinstance(value,bytes):value=value.decode('utf-8')
            elif isinstance(value,np.generic):value=value.item()
            if isinstance(value,(str,int,float)):attrs[key]=value
    if not np.isfinite(volume).all() or volume.min()<0:raise ValueError('invalid RSS image values')
    indices=rows['slice'].to_numpy(int)
    if (indices<0).any() or (indices>=len(volume)).any():raise ValueError('reference slice outside image')
    boxes=np.stack([native_box(row,*volume.shape[1:]) for _,row in rows.iterrows()])
    image_sha256=file_hash(image_path)
    source_integrity=None;receipt_path=image_path.with_suffix('.json')
    if receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('image_sha256')!=image_sha256:
            raise ValueError('source extraction receipt does not match actual HDF5')
        source=receipt.get('source',{})
        if PurePosixPath(source.get('member_name','')).name!=image_path.name:
            raise ValueError('source extraction member identity differs')
        if receipt.get('rss_shape')!=list(volume.shape):raise ValueError('source extraction shape differs')
        blocks=receipt.get('accessed_blocks',[])
        if not blocks or not all(b.get('xz_integrity_check_passed') is True for b in blocks):
            raise ValueError('source extraction lacks verified accessed XZ blocks')
        source_integrity={'receipt_path':str(receipt_path),'receipt_sha256':file_hash(receipt_path),
                          'source':source,'verified_accessed_blocks':len(blocks),
                          'integrity_scope':receipt.get('integrity_scope'),
                          'source_hdf5_sha256':receipt.get('source_hdf5_sha256')}
    return {'native_volume':volume,'native_boxes_xyxy':boxes,'box_slice_indices':indices,
            'labels':rows.label.tolist(),'geometry':reference_geometry(header,volume.shape),
            'source_attrs':attrs,'pixel_truth':False,'outside_box_is_negative':False,
            'coordinate_source':ORIENTATION_SOURCE,'image_sha256':image_sha256,
            'source_integrity':source_integrity,
            'rss_pixels_sha256':hashlib.sha256(volume.tobytes()).hexdigest()}


def mcl_annotations(path):
    frame=pd.read_csv(path)
    frame=frame[frame.label.isin(['Ligament - MCL Low-Mod Grade Sprain','Ligament - MCL High Grade sprain'])].copy()
    if frame.empty:raise ValueError('no published MCL references')
    if not frame.study_level.eq('No').all():raise ValueError('study-level labels are not localization boxes')
    values=frame[['slice','x','y','width','height']].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values[:,3:]<=0).any():
        raise ValueError('invalid published localization box')
    if not np.equal(values[:,0],np.floor(values[:,0])).all():raise ValueError('noninteger slice')
    if not frame.file.str.fullmatch(r'file\d+').all():raise ValueError('unexpected fastMRI file identity')
    return frame


def native_box(row,height,width):
    """Half-open annotation edges -> half-open native HDF5 pixel edges."""
    x,y,w,h=map(float,[row['x'],row['y'],row['width'],row['height']])
    if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
        raise ValueError('annotation box does not fit the actual reconstruction matrix')
    return np.asarray([x,height-y-h,x+w,height-y],np.float64)


def load_reference_slice(image_path,rows,slice_index):
    import h5py
    if 'file' not in rows or set(rows['file'])!={Path(image_path).stem}:
        raise ValueError('annotation and source volume identities differ')
    selected=rows[rows['slice'].eq(int(slice_index))]
    if selected.empty:raise ValueError('unmarked slice is unknown, not a negative')
    with h5py.File(image_path,'r') as handle:
        if 'reconstruction_rss' not in handle:raise ValueError('reference RSS image missing')
        volume=handle['reconstruction_rss']
        if volume.ndim!=3 or not 0<=slice_index<len(volume):raise ValueError('reference slice outside image')
        native=np.asarray(volume[slice_index],np.float32)
    boxes=np.stack([native_box(row,*native.shape) for _,row in selected.iterrows()])
    return {'native_image':native,'native_boxes_xyxy':boxes,
            'annotation_image':native[::-1].copy(),
            'annotation_boxes_xyxy':np.asarray([[r.x,r.y,r.x+r.width,r.y+r.height] for r in selected.itertuples()],float),
            'pixel_truth':False,'outside_box_is_negative':False,
            'labels':selected.label.tolist(),'coordinate_source':ORIENTATION_SOURCE}


def prepare(annotation_csv,image_root,output):
    """Bind all source boxes to exact files; missing images remain explicit."""
    frame=mcl_annotations(annotation_csv);image_root=Path(image_root);output=Path(output)
    output.mkdir(parents=True,exist_ok=True);records=[];available=[];missing=[]
    for name,rows in frame.groupby('file',sort=True):
        image=image_root/(name+'.h5')
        record={'file':name,'path':str(image),'source_labels':sorted(rows.label.unique()),
                'slices':sorted(map(int,rows['slice'].unique())),'boxes':len(rows),
                'reference_kind':'published_radiologist_pathology_box',
                'main_competition_target':None,'acute_factor':None,'pixel_truth':False,
                'unmarked_is_negative':False,'image_available':image.is_file()}
        if image.is_file():
            for index in record['slices']:load_reference_slice(image,rows,index)
            record['image_sha256']=file_hash(image);available.append(name)
        else:missing.append(name)
        records.append(record)
    frame.to_csv(output/'mcl_boxes.csv',index=False)
    (output/'required_image_files.txt').write_text('\n'.join(x+'.h5' for x in missing)+'\n',encoding='utf-8')
    result={'source':SOURCE,'orientation_source':ORIENTATION_SOURCE,
        'annotation_sha256':file_hash(annotation_csv),'boxes':len(frame),'volumes':len(records),
        'grade_counts':frame.groupby('label').agg(boxes=('file','size'),volumes=('file','nunique')).to_dict('index'),
        'available_volumes':len(available),'missing_volumes':len(missing),'records':records,
        'phase':'references_paired' if not missing else 'awaiting_authorized_source_images',
        'scope':'single-radiologist pathology-location references, not exhaustive disease truth or fine ligament masks; no patient identities inferred',
        'access_url':'https://fastmri.med.nyu.edu/','automatic_training_labels':0}
    save_json(output/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',default='F:/rsnaknee/reference_anatomy/fastmri_plus/knee.csv')
    p.add_argument('--images',default='F:/rsnaknee/reference_anatomy/fastmri_plus/images')
    p.add_argument('--output',default='F:/rsnaknee/reference_anatomy/fastmri_plus/prepared')
    a=p.parse_args();r=prepare(a.annotations,a.images,a.output)
    print(json.dumps({k:r[k] for k in ('boxes','volumes','available_volumes','missing_volumes','phase')}))
