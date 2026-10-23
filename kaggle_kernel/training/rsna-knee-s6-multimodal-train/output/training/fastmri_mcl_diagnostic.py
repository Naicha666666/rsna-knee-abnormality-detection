"""Source-plane consumption of fastMRI+ clinical MCL pathology boxes.

The fixed whole-field / centred 130 mm / centred 100 mm comparisons exercise
the real crop renderer. No annotation chooses a crop, detection or margin.
They evaluate context retention, not MCL segmentation or disease accuracy.
The sagittal YOLO and sagittal-to-coronal atlas cannot be validated by silently
treating this standalone coronal series as their expected sagittal input.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .common import save_json,file_hash
from .fastmri_reference import mcl_annotations,load_reference_volume
from .input_provenance import view_geometry
from .native_data import render_view,covering_centers

VERSION='fastmri_mcl_source_plane_v1'
VIEWS={'whole_field':None,'center_130mm':130.,'center_100mm':100.}


def box_retention(box,crop):
    box=np.asarray(box,float);crop=np.asarray(crop,float)
    wh=np.maximum(np.minimum(box[2:],crop[2:])-np.maximum(box[:2],crop[:2]),0.)
    return float(np.prod(wh)/np.prod(box[2:]-box[:2]))


def map_box_edges(box,geometry):
    """Map continuous edges; the renderer's affine instead maps pixel centres."""
    box=np.asarray(box,float).reshape(2,2)
    x,y,_,_=geometry.crop_xyxy;px,py,_,_=geometry.valid_xyxy
    h,w=geometry.crop_shape;oh,ow=geometry.resized_shape
    return ((box-[x,y])*[ow/w,oh/h]+[px,py]).reshape(4)


def evaluate_volume(path,rows,output,*,image_size=320,figure=False):
    data=load_reference_volume(path,rows);volume=data['native_volume']
    spacing=data['geometry']['spacing_mm'];h,w=volume.shape[1:]
    boxes=data['native_boxes_xyxy'];slices=data['box_slice_indices']
    # Explicit scale conversion for the existing uint16-oriented renderer.
    # It is fixed once per volume and does not change crop coordinates.
    high=float(np.percentile(volume,99.9));scale=high if high>0 else 1.
    display=np.clip(volume/scale,0.,1.).astype(np.float32)
    centers=covering_centers(len(volume))
    triplets=np.clip(centers[:,None]+[-1,0,1],0,len(volume)-1)
    represented=set(triplets.ravel().tolist())
    result={'version':VERSION,'file':Path(path).stem,'image_sha256':data['image_sha256'],
            'rss_pixels_sha256':data['rss_pixels_sha256'],'geometry':data['geometry'],
            'source_attrs':data['source_attrs'],'source_integrity':data['source_integrity'],
            'boxes':len(rows),'marked_slices':sorted(set(slices.tolist())),
            'reference_kind':'single_radiologist_pathology_location_boxes',
            'pixel_truth':False,'outside_box_is_negative':False,'acute_factor':None,
            'native_to_lps':None,'slice_spacing_mm':None,'training_dispatch':False,
            'localizer_status':'sagittal_YOLO_not_applied_to_standalone_coronal_input',
            'atlas_status':'no_same_patient_sagittal_source_or_known_LPS_transform',
            'intensity_adapter':{'policy':'RSS divided by volume percentile 99.9 then clipped [0,1]',
                                 'denominator':scale,'render_window':[0.,1.]},
            'depth_support':{'windows':len(centers),'represented_native_slices':len(represented),
                             'all_reference_slices_present':set(slices.tolist()).issubset(represented)},
            'views':{},'references':[]}
    for i,(box,z,label) in enumerate(zip(boxes,slices,data['labels'])):
        result['references'].append({'index':i,'slice':int(z),'source_label':label,
                                     'native_xyxy':box.tolist(),'retention':{}})
    for name,crop_mm in VIEWS.items():
        if spacing is None and crop_mm is not None:
            result['views'][name]={'status':'unavailable_header_spacing_unknown'};continue
        # Full-field coverage is valid even without physical spacing. Rendering
        # in that case explicitly uses pixel aspect only, never reports mm.
        metric_spacing=spacing if spacing is not None else (1.,1.)
        geometry=view_geometry((h,w),metric_spacing,image_size,crop_mm=crop_mm)
        crop=np.asarray(geometry.crop_xyxy)
        coverage=[box_retention(b,crop) for b in boxes]
        rendered_hash=hashlib.sha256();rendered_nonzero=0
        for z in sorted(set(slices.tolist())):
            inds=np.clip([z-1,z,z+1],0,len(volume)-1)
            pixels,_,receipt=render_view(display[inds],metric_spacing,image_size,
                crop_mm=crop_mm,window=(0.,1.),return_geometry=True)
            if not np.array_equal(receipt['native_crop_xyxy'],crop):
                raise RuntimeError('reference and pixel consumers use different crops')
            rendered_hash.update(pixels.tobytes());rendered_nonzero+=int(np.count_nonzero(pixels))
        result['views'][name]={'status':'evaluated','native_crop_xyxy':crop.tolist(),
            'native_image_area_fraction':float(np.prod(crop[2:]-crop[:2])/(h*w)),
            'expert_box_area_retention_mean':float(np.mean(coverage)),
            'expert_box_area_retention_min':float(np.min(coverage)),
            'fully_retained_boxes':int(np.sum(np.asarray(coverage)>=1-1e-9)),
            'reference_boxes':len(boxes),'all_reference_boxes_fully_retained':min(coverage)>=1-1e-9,
            'source_crop_fov_mm':list(np.asarray(geometry.crop_shape)*spacing) if spacing is not None else None,
            'rendered_valid_xyxy':list(geometry.valid_xyxy),'output_to_native':geometry.receipt()['output_to_native'].tolist(),
            'rendered_marked_triplets_sha256':rendered_hash.hexdigest(),'rendered_nonzero_values':rendered_nonzero}
        for ref,box,value in zip(result['references'],boxes,coverage):
            ref['retention'][name]={'fraction':value,'output_box_edges_unclipped':map_box_edges(box,geometry).tolist()}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if figure:draw_reference(data,result,output/(Path(path).stem+'.png'))
    return result


def draw_reference(data,result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    # Fixed middle marked slice, not the easiest box or best retained slice.
    z=result['marked_slices'][len(result['marked_slices'])//2]
    image=data['native_volume'][z]
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,flip in zip(axes,(False,True)):
        ax.imshow(image[::-1] if flip else image,cmap='gray',vmin=0,vmax=np.percentile(image,99.9))
        for ref in result['references']:
            if ref['slice']!=z:continue
            x0,y0,x1,y1=ref['native_xyxy']
            if flip:y0,y1=image.shape[0]-y1,image.shape[0]-y0
            ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=1.8))
        for name,color in [('center_130mm','cyan'),('center_100mm','orange')]:
            view=result['views'][name]
            if view['status']!='evaluated':continue
            x0,y0,x1,y1=view['native_crop_xyxy']
            if flip:y0,y1=image.shape[0]-y1,image.shape[0]-y0
            ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1.3,label=name))
        ax.set_title('Author annotation display (flipud)' if flip else 'Original RSS native pixels');ax.axis('off')
    axes[0].legend(loc='lower center',fontsize=8)
    fig.suptitle(f"{result['file']} slice {z} | green: expert pathology boxes, not MCL pixel truth",fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def aggregate(records,expected):
    def stats(values):
        return {'mean':float(np.mean(values)),'median':float(np.median(values)),
                'min':float(np.min(values)),'max':float(np.max(values))} if values else None
    summary={'version':VERSION,'expected_volumes':len(expected),'expected_boxes':int(sum(expected.values())),
        'evaluated_volumes':len(records),'evaluated_boxes':sum(r['boxes'] for r in records),
        'pending_volumes':sorted(set(expected)-{r['file'] for r in records}),
        'evaluation_scope':'expert pathology-box context retention, not ligament segmentation or diagnostic accuracy',
        'training_dispatch':False,'views':{}}
    for name in VIEWS:
        valid=[r for r in records if r['views'][name]['status']=='evaluated']
        coverage=[x['retention'][name]['fraction'] for r in valid for x in r['references']]
        summary['views'][name]={'evaluated_volumes':len(valid),'evaluated_boxes':len(coverage),
            'box_area_retention':stats(coverage),
            'native_image_area_fraction':stats([r['views'][name]['native_image_area_fraction'] for r in valid]),
            'fully_retained_boxes':sum(x>=1-1e-9 for x in coverage),
            'volumes_all_boxes_fully_retained':sum(r['views'][name]['all_reference_boxes_fully_retained'] for r in valid),
            'incomplete_volume_ids':[r['file'] for r in valid if not r['views'][name]['all_reference_boxes_fully_retained']]}
        strata={}
        for record in valid:
            acquisition=record['source_attrs'].get('acquisition','unknown')
            for ref in record['references']:
                group=acquisition+' | '+ref['source_label']
                entry=strata.setdefault(group,{'volumes':set(),'retention':[]})
                entry['volumes'].add(record['file']);entry['retention'].append(ref['retention'][name]['fraction'])
        summary['views'][name]['source_acquisition_and_grade_strata']={key:{
            'volumes':len(value['volumes']),'boxes':len(value['retention']),
            'box_area_retention':stats(value['retention']),
            'fully_retained_boxes':sum(x>=1-1e-9 for x in value['retention'])}
            for key,value in strata.items()}
    return summary


def run(annotations,images,output,*,watch=False,poll_seconds=20.):
    frame=mcl_annotations(annotations);expected=frame.groupby('file').size().to_dict()
    images=Path(images);output=Path(output);(output/'volumes').mkdir(parents=True,exist_ok=True)
    # These identify the actual fixed renderer and adapter, not model weights.
    import inspect
    contract={'version':VERSION,'annotations_sha256':file_hash(annotations),
              'reader_sha256':file_hash(Path(__file__).with_name('fastmri_reference.py')),
              'diagnostic_sha256':file_hash(__file__),
              'renderer_sha256':hashlib.sha256(inspect.getsource(render_view).encode()).hexdigest(),
              'geometry_sha256':hashlib.sha256(inspect.getsource(view_geometry).encode()).hexdigest()}
    while True:
        records=[];errors={}
        for name,rows in frame.groupby('file',sort=True):
            path=images/(name+'.h5');cached=output/'volumes'/(name+'.json')
            receipt_path=path.with_suffix('.json')
            if not path.is_file() or not receipt_path.is_file():continue
            stat=path.stat();fingerprint={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                         'extraction_receipt_sha256':file_hash(receipt_path)}
            try:
                old=json.loads(cached.read_text()) if cached.is_file() else None
                if old and old.get('contract')==contract and old.get('source_stat')==fingerprint:result=old
                else:
                    result=evaluate_volume(path,rows,output/'figures',figure=True)
                    result.update(contract=contract,source_stat=fingerprint);save_json(cached,result)
                    print(json.dumps({'file':name,'boxes':len(rows),'spacing':result['geometry']['spacing_mm'],
                                      'center100_retention':result['views']['center_100mm'].get('expert_box_area_retention_min')}),flush=True)
                records.append(result)
            except (ValueError,OSError,KeyError) as error:errors[name]=str(error)
        summary=aggregate(records,expected);summary['errors']=errors;summary['contract']=contract
        summary['phase']='complete' if len(records)==len(expected) else 'awaiting_or_rejecting_source_images'
        save_json(output/'summary.json',summary)
        if not watch or summary['phase']=='complete':return summary
        time.sleep(poll_seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',default='F:/rsnaknee/reference_anatomy/fastmri_plus/knee.csv')
    p.add_argument('--images',default='F:/rsnaknee/reference_anatomy/fastmri_plus/images')
    p.add_argument('--output',default='F:/rsnaknee/analysis/fastmri_mcl_reference')
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    r=run(a.annotations,a.images,a.output,watch=a.watch)
    print(json.dumps({key:r[key] for key in ('expected_volumes','evaluated_volumes','evaluated_boxes','phase')}))
