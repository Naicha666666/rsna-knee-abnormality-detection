"""Frozen coronal bilateral-view candidate on all 125 fastMRI+ MCL references.

The two initial inspected files define development; every other source file
remains in the evaluation list, including failures. Any matching anonymous
patient key is grouped with development. No reference chooses a view centre.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .common import file_hash,save_json
from .coronal_views import coronal_bilateral_views,rectangle_union_area,box_union_retention
from .fastmri_reference import mcl_annotations,load_reference_volume
from .input_provenance import view_geometry
from .native_data import render_view

VERSION='coronal_bilateral_100mm_plus_global_v1'
DEVELOPMENT_FILES=['file1000593','file1000888']


def anonymous_patient_key(path):
    import h5py
    with h5py.File(path,'r') as handle:
        value=handle.attrs.get('patient_id')
    if value is None:return None
    if isinstance(value,bytes):value=value.decode('utf-8')
    return hashlib.sha256(('fastMRI patient_id:'+str(value)).encode('utf-8')).hexdigest()


def evaluate(path,rows,baseline,image_size=320):
    data=load_reference_volume(path,rows);volume=data['native_volume'];h,w=volume.shape[-2:]
    if data['image_sha256']!=baseline['image_sha256']:raise ValueError('baseline source image changed')
    spacing=data['geometry']['spacing_mm']
    result={'version':VERSION,'file':path.stem,'image_sha256':data['image_sha256'],
        'patient_group_sha256':anonymous_patient_key(path),'source_attrs':data['source_attrs'],
        'source_integrity':data['source_integrity'],'pixel_truth':False,'training_dispatch':False,
        'method':'whole native field plus fixed screen-left/right 100mm views; no side identity or GT centre',
        'boxes':len(rows),'views':[],'references':[]}
    if spacing is None:
        result.update(status='rejected_physical_detail_spacing_unknown',global_retention=1.);return result
    views=coronal_bilateral_views((h,w),spacing,image_size,local_mm=100.)
    scale=baseline['intensity_adapter']['denominator'];display=np.clip(volume/scale,0,1).astype(np.float32)
    rects=[]
    for view_id,crop,center in views:
        g=view_geometry((h,w),spacing,image_size,crop_mm=crop,center=center)
        rect=g.crop_xyxy
        if crop is not None:rects.append(rect)
        digest=hashlib.sha256();nonzero=0
        for z in sorted(set(data['box_slice_indices'].tolist())):
            inds=np.clip([z-1,z,z+1],0,len(volume)-1)
            rendered,_,receipt=render_view(display[inds],spacing,image_size,crop_mm=crop,center=center,
                                          window=(0,1),return_geometry=True)
            if not np.array_equal(receipt['native_crop_xyxy'],rect):raise ValueError('view pixels differ from reported geometry')
            digest.update(rendered.tobytes());nonzero+=int(np.count_nonzero(rendered))
        result['views'].append({'view_id':view_id,'crop_mm':crop,'center_row_col':list(center),
            'native_crop_xyxy':list(rect),'output_to_native':g.receipt()['output_to_native'].tolist(),
            'native_image_area_fraction':float(np.prod(g.crop_shape)/(h*w)),
            'rendered_marked_triplets_sha256':digest.hexdigest(),'rendered_nonzero_values':nonzero})
    effective=rects or [(0,0,w,h)]  # deduped detail is exactly the already retained whole field
    total_source_area=sum(v['native_image_area_fraction'] for v in result['views'])
    result.update(status='evaluated',all_views_union_area_fraction=1.,
        effective_detail_union_area_fraction=rectangle_union_area(effective)/(h*w),
        additional_detail_views=len(rects),unique_views_per_native_window=len(views),
        total_sampled_source_area_with_repeats_fraction=total_source_area,
        model_pixels_per_native_window=3*image_size*image_size*len(views),
        model_pixel_ratio_to_global_plus_one_detail=len(views)/2.,
        additional_detail_is_deduped_global=not bool(rects))
    for box,z,label,original in zip(data['native_boxes_xyxy'],data['box_slice_indices'],data['labels'],baseline['references']):
        x0,y0,x1,y1=box
        hrects=[[max(x0,r[0]),0,min(x1,r[2]),1] for r in effective]
        vrects=[[max(y0,r[1]),0,min(y1,r[3]),1] for r in effective]
        result['references'].append({'slice':int(z),'source_label':label,'native_xyxy':box.tolist(),
            'global_retention':1.,'bilateral_detail_union_retention':box_union_retention(box,effective),
            'horizontal_axis_retention':rectangle_union_area(hrects)/(x1-x0),
            'vertical_axis_retention':rectangle_union_area(vrects)/(y1-y0),
            'center_100mm_retention':original['retention']['center_100mm']['fraction'],
            'center_130mm_retention':original['retention']['center_130mm']['fraction']})
    return result


def aggregate(records,expected,development_groups):
    def stat(values):
        return {'mean':float(np.mean(values)),'median':float(np.median(values)),
                'min':float(np.min(values)),'max':float(np.max(values))} if values else None
    valid=[r for r in records if r['status']=='evaluated']
    def cohort(rows):
        refs=[x for r in rows for x in r['references']]
        pairs=[(x['center_100mm_retention'],x['bilateral_detail_union_retention']) for x in refs]
        return {'volumes':len(rows),'known_patient_groups':len({r['patient_group_sha256'] for r in rows if r['patient_group_sha256']}),
            'unknown_patient_key_volumes':sum(r['patient_group_sha256'] is None for r in rows),
            'boxes':len(refs),'box_area_retention':stat([x['bilateral_detail_union_retention'] for x in refs]),
            'fully_retained_boxes':sum(x['bilateral_detail_union_retention']>=1-1e-9 for x in refs),
            'vertical_truncation_boxes':int(sum(x['vertical_axis_retention']<1-1e-9 for x in refs)),
            'horizontal_truncation_boxes':int(sum(x['horizontal_axis_retention']<1-1e-9 for x in refs)),
            'improved_vs_center100_boxes':sum(new>old+1e-9 for old,new in pairs),
            'worsened_vs_center100_boxes':sum(new<old-1e-9 for old,new in pairs),
            'center130_box_area_retention':stat([x['center_130mm_retention'] for x in refs]),
            'detail_union_native_area_fraction':stat([r['effective_detail_union_area_fraction'] for r in rows]),
            'total_source_area_with_repeats_fraction':stat([r['total_sampled_source_area_with_repeats_fraction'] for r in rows]),
            'unique_views_per_native_window':stat([r['unique_views_per_native_window'] for r in rows]),
            'model_pixel_ratio_to_global_plus_one_detail':stat([r['model_pixel_ratio_to_global_plus_one_detail'] for r in rows]),
            'incomplete_detail_volume_ids':[r['file'] for r in rows if any(x['bilateral_detail_union_retention']<1-1e-9 for x in r['references'])]}
    development=[r for r in valid if r['file'] in DEVELOPMENT_FILES or r['patient_group_sha256'] in development_groups]
    evaluation=[r for r in valid if r not in development]
    return {'version':VERSION,'expected_volumes':len(expected),'expected_boxes':int(sum(expected.values())),
        'evaluated_volumes':len(valid),'rejected_volumes':[r['file'] for r in records if r['status']!='evaluated'],
        'pending_volumes':sorted(set(expected)-{r['file'] for r in records}),
        'development_files_frozen':DEVELOPMENT_FILES,'development_patient_groups':sorted(development_groups),
        'method_frozen_after_two_initial_images':True,'all_ready':cohort(valid),
        'development':cohort(development),'remaining_source_evaluation':cohort(evaluation),
        'training_dispatch':False,'scope':'expert pathology-box context retention, not MCL masks, patient-side inference, or disease accuracy'}


def run(root,baseline_dir,output,watch=False):
    root=Path(root);output=Path(output);baseline_dir=Path(baseline_dir)
    frame=mcl_annotations(root/'knee.csv');expected=frame.groupby('file').size().to_dict()
    (output/'volumes').mkdir(parents=True,exist_ok=True)
    contract={'method':VERSION,'annotations_sha256':file_hash(root/'knee.csv'),
              'candidate_sha256':file_hash(Path(__file__).with_name('coronal_views.py')),
              'diagnostic_sha256':file_hash(__file__)}
    while True:
        records=[];errors={};groups=set()
        for name in DEVELOPMENT_FILES:
            path=root/'images'/(name+'.h5')
            if path.is_file():
                key=anonymous_patient_key(path)
                if key:groups.add(key)
        for name,rows in frame.groupby('file',sort=True):
            path=root/'images'/(name+'.h5');baseline_path=baseline_dir/'volumes'/(name+'.json')
            if not path.is_file() or not path.with_suffix('.json').is_file() or not baseline_path.is_file():continue
            cached=output/'volumes'/(name+'.json');bound={**contract,'baseline_sha256':file_hash(baseline_path)}
            try:
                old=json.loads(cached.read_text()) if cached.is_file() else None
                if old and old.get('contract')==bound:result=old
                else:
                    result=evaluate(path,rows,json.loads(baseline_path.read_text()));result['contract']=bound
                    save_json(cached,result)
                    print(json.dumps({'file':name,'status':result['status'],
                        'detail_retention_min':min(x['bilateral_detail_union_retention'] for x in result['references']) if result['references'] else None}),flush=True)
                records.append(result)
            except (ValueError,OSError,KeyError) as error:errors[name]=str(error)
        summary=aggregate(records,expected,groups);summary['errors']=errors;summary['contract']=contract
        summary['phase']='complete' if len(records)==len(expected) else 'awaiting_or_rejecting_source_images'
        save_json(output/'summary.json',summary)
        if not watch or summary['phase']=='complete':return summary
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='F:/rsnaknee/reference_anatomy/fastmri_plus')
    p.add_argument('--baseline',default='F:/rsnaknee/analysis/fastmri_mcl_reference')
    p.add_argument('--output',default='F:/rsnaknee/analysis/fastmri_mcl_bilateral')
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    run(a.root,a.baseline,a.output,a.watch)
