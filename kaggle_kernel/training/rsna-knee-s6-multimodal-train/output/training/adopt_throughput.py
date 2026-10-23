"""Adopt an accepted execution revision while preserving the fitted state."""
import argparse,json,time
from pathlib import Path
from .common import save_json,file_hash
from .orthofoundation_main import implementation_hashes


def adopt(bundle,report,candidate):
    bundle,report,candidate=map(Path,(bundle,report,candidate));proof=json.loads(report.read_text(encoding='utf-8'))
    if not proof['accepted'] or proof['speedup']<=1:raise ValueError('No validated throughput improvement to adopt')
    config=json.loads((candidate/'replay_scope.json').read_text(encoding='utf-8'));options=proof.get('options',config['execution_options'])
    plan_path=bundle/'plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'));old_hash=file_hash(plan_path)
    revision={'adopted_unix':time.time(),'previous_plan_sha256':old_hash,'previous_implementation_sha256':plan['implementation_sha256'],
        'options':options,'proof':str(report.resolve()),'proof_sha256':file_hash(report),
        'continued_optimizer_step':proof['steps'],'learning_configuration_changed':False,'data_and_order_changed':False,
        'storage_change':'New checkpoints and run artifacts on internal NVMe; original USB checkpoint is retained.'}
    save_json(bundle/'plan_before_throughput.json',plan)
    for fold in plan['folds']:
        entry_path=Path(fold['path']);entry=json.loads(entry_path.read_text(encoding='utf-8'))
        entry['previous_output_directory']=entry['output_directory']
        destination=candidate if fold['fold']==0 else candidate.parent/f"orthofoundation_anatomy_fold{fold['fold']}_prefetch2"
        entry['output_directory']=str(destination.resolve())
        save_json(entry_path,entry);fold['sha256']=file_hash(entry_path)
    plan['implementation_sha256']=implementation_hashes();plan['execution_revision']=revision
    save_json(plan_path,plan);save_json(bundle/'execution_revision_20261002.json',revision)
    save_json(bundle/'execution_options.json',{'options':options,'validation_report':str(report.resolve()),'validation_report_sha256':file_hash(report)})
    save_json(bundle.parent/'current.json',{'plan':str(plan_path),'sha256':file_hash(plan_path)})
    panel_path=bundle.parent/'status.json';panel=json.loads(panel_path.read_text(encoding='utf-8'))
    panel['state_zh']='吞吐改造已验收 · 恢复训练'
    panel['rows']=[row for row in panel['rows'] if row[0] not in ('执行方式','检查点存储')]
    panel['rows'] += [['执行方式',f"{options['prefetch_studies']} 路完整检查预取；编码块 {options['encoder_chunk']}；梯度仍累积24例"],['检查点存储','内置 NVMe；继续原有优化器、EMA和随机状态']]
    save_json(panel_path,panel);return revision


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='F:/rsnaknee/orthofoundation_final/v7_4_anatomy');p.add_argument('--report',required=True);p.add_argument('--candidate',required=True);a=p.parse_args();print(json.dumps(adopt(a.bundle,a.report,a.candidate)))
