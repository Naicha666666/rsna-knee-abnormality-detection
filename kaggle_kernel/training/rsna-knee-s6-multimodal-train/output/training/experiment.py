"""One research entry: encode -> train a head, or fine-tune its complete image model."""
from pathlib import Path
import argparse
import json

from .common import UID,save_json,file_hash
from .recipes import RECIPES,configuration,encoder_key,input_contract


def label_inputs(root,name,config,*,pilot=False):
    """Resolve actual training inputs and the unchanged criterion reference."""
    root=Path(root)
    label_root=Path(config.get('labels_dir',root/('labels_pilot' if pilot else 'labels')))
    if pilot and config.get('labels_dir'):label_root=label_root.with_name(label_root.name+'_pilot')
    labels=label_root/('selected_labels.csv' if pilot and not config.get('labels_dir') else 'labels.csv')
    if name=='distilled':
        reference_root=Path(configuration('criteria',root)['labels_dir'])
        if pilot:reference_root=reference_root.with_name(reference_root.name+'_pilot')
        reference=reference_root/'evaluation_reference.csv'
    else:reference=label_root/'evaluation_reference.csv'
    return labels,reference


def resolve_inputs(root,name,config,*,pilot=False,input_package=None,allow_partial=False):
    """Resolve the real frozen inputs without dispatching training."""
    root=Path(root);current=root/'input_ready/current.json'
    if input_package is None and config.get('supervision_semantics')=='competition_criterion' and name!='distilled' and current.exists():input_package=current
    if input_package:
        if name=='distilled':raise ValueError('distillation needs its separate outer-fold target lineage')
        from .prepare_input_data import resolve_package
        result=resolve_package(input_package,config,allow_partial=pilot or allow_partial)
        config.update(labels_dir=result['directory'],input_package_path=result['package_path'],
            input_package_revision=result['revision'],input_package_variant=result['variant'],
            input_manifest=result['manifest'],labels_file=result['labels'],evaluation_reference=result['evaluation_reference'])
        return result
    labels,reference=label_inputs(root,name,config,pilot=pilot);manifest=root/'cache_native/manifest.json'
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if not pilot and data['scope']!='complete_split':raise ValueError('full experiment requires complete native cache')
    return {'manifest':str(manifest),'labels':str(labels),'evaluation_reference':str(reference),
        'expert_evaluation_reference':str(labels.with_name('expert_evaluation_reference.csv')),
        'partial':data['scope']!='complete_split','revision':None}


def experiment_output_path(root,name,stage,fold,inputs,*,pilot=False,image_size=384):
    size_tag=f'_{image_size}' if image_size!=384 else ''
    tag=('_inputs_'+inputs['revision'][:12]) if inputs['revision'] else ''
    prefix='partial_' if inputs['partial'] and not pilot else 'pilot_' if pilot else ''
    return Path(root)/'runs'/f'{prefix}{name}_{stage}{size_tag}_fold{fold}{tag}'


def execute(root,name,*,stage='head',fold='0',pilot=False,epochs=None,resume=False,init_checkpoint=None,image_size=384,stop_after_epochs=None,input_package=None,allow_partial=False):
    hold=Path(__file__).with_name('TRAINING_HOLD.json')
    if hold.exists() and json.loads(hold.read_text(encoding='utf-8')).get('active'):raise RuntimeError('Training is held by user.')
    from .engine import train
    root=Path(root);config=configuration(name,root,stage=stage,epochs=epochs,fold=fold)
    config['model']['image_size']=image_size
    inputs=resolve_inputs(root,name,config,pilot=pilot,input_package=input_package,allow_partial=allow_partial)
    labels=Path(inputs['labels']);reference=Path(inputs['evaluation_reference']);manifest=Path(inputs['manifest'])
    output=experiment_output_path(root,name,stage,fold,inputs,pilot=pilot,image_size=image_size)
    save_json(output/'experiment_input_binding.json',{**inputs,'reference_sha256':file_hash(reference),
        'labels_sha256':file_hash(labels),'manifest_sha256':file_hash(manifest),'input_policy':input_contract(config),
        'intensity_augmentation_policy':config.get('intensity_augmentation_policy','legacy')})
    feature_manifest=None
    if stage=='head':
        from .feature_bank import build
        bank=root/'features'/f"{'pilot_' if pilot else ''}{encoder_key(config)}_{file_hash(manifest)[:12]}"
        # The encoder producer has no knee label exposure or fitting.
        producer=configuration('global_local',root,stage='head')
        producer.update(input_contract(config))
        for key in ('spatial_pool','backbone','input_normalization','pretrained_file'):
            if key in config['model']:producer['model'][key]=config['model'][key]
        producer['model']['image_size']=image_size
        build(manifest,producer,bank);feature_manifest=bank/'manifest.json'
    history=train(config,manifest,labels,output,fold=fold,resume=resume,pretrained=True,
        smoke=pilot or inputs['partial'],evaluation_labels_path=reference,feature_manifest=feature_manifest,init_checkpoint=init_checkpoint,
        stop_after_epochs=stop_after_epochs)
    return output,history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recipe',choices=list(RECIPES))
    p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--stage',choices=['head','finetune'],default='head')
    p.add_argument('--fold',default='0');p.add_argument('--pilot',action='store_true');p.add_argument('--epochs',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--init-checkpoint')
    p.add_argument('--size',type=int,choices=[384,512],default=384)
    p.add_argument('--input-package',help='Frozen package.json, current.json, or input_ready directory')
    p.add_argument('--allow-partial',action='store_true',help='Explicit development subset; never full-data readiness')
    a=p.parse_args();output,_=execute(a.root,a.recipe,stage=a.stage,fold=a.fold,pilot=a.pilot,
        epochs=a.epochs,resume=a.resume,init_checkpoint=a.init_checkpoint,image_size=a.size,input_package=a.input_package,allow_partial=a.allow_partial)
    print('Finished '+str(output),flush=True)


if __name__=='__main__':main()
