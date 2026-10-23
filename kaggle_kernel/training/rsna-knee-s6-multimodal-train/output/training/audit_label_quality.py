"""Frozen report decisions versus image Gold, without expert-label overwrite.

Read-only with respect to source labels, development membership and training.
Unknown report decisions remain abstentions. This is development alignment,
not an independent estimate of text extraction or MRI label correctness.
"""
from collections import Counter
from datetime import datetime,timezone
from io import BytesIO
from pathlib import Path
import argparse,hashlib,html,json,math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .common import UID,LABELS,save_json,file_hash

ZH=['ACL','MCL','内侧半月板','外侧半月板','内侧OA','外侧OA','髌股OA','积液','滑膜炎','Baker囊肿','骨挫伤','骨折']


def divide(a,b):return a/b if b else None


def wilson(success,total):
    if not total:return None
    z=1.95996398454;p=success/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d;radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-radius,center+radius]


def metrics(rows):
    results=[]
    for label in LABELS:
        part=[r for r in rows if r['label']==label];c=Counter(r['outcome'] for r in part)
        positives=sum(r['official']==1 for r in part);negatives=len(part)-positives
        known=[r for r in part if r['report_criterion'] is not None]
        tp,fp,tn,fn=(c[k] for k in ('TP','FP','TN','FN'))
        auc=roc_auc_score([r['official'] for r in known],[r['report_criterion'] for r in known]) if len({r['official'] for r in known})==2 else None
        results.append({'label':label,'gold_studies':len(part),'gold_positive':positives,'gold_negative':negatives,
            'known':len(known),'unknown':c['unknown'],'coverage':divide(len(known),len(part)),
            'TP':tp,'FP':fp,'TN':tn,'FN':fn,
            'unknown_gold_positive':sum(r['official']==1 and r['report_criterion'] is None for r in part),
            'unknown_gold_negative':sum(r['official']==0 and r['report_criterion'] is None for r in part),
            'agreement_known':divide(tp+tn,len(known)),'agreement_known_wilson95':wilson(tp+tn,len(known)),
            'positive_precision_known':divide(tp,tp+fp),'positive_precision_wilson95':wilson(tp,tp+fp),
            'negative_predictive_value_known':divide(tn,tn+fn),
            'sensitivity_on_answered_gold_positives':divide(tp,tp+fn),
            'specificity_on_answered_gold_negatives':divide(tn,tn+fp),
            'correct_positive_coverage_of_all_gold_positive':divide(tp,positives),
            'hard_auc_answered_subset_only':auc})
    return results


def run(root,output,*,include_public=False):
    root=Path(root);status_path=root/'supervision_prepare_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8'));labels=Path(status['labels']);release=labels.parent
    paths=[labels,release/'contract.json',release/'evidence.jsonl',root/'raw/train.csv']
    hashes={str(p):file_hash(p) for p in paths}
    blobs={p.name:p.read_bytes() for p in paths}
    if hashlib.sha256(blobs['labels.csv']).hexdigest()!=status['labels_sha256']:raise ValueError('ready label release changed')
    frame=pd.read_csv(BytesIO(blobs['labels.csv']),dtype={UID:str,'group_id':str}).set_index(UID)
    raw=pd.read_csv(BytesIO(blobs['train.csv']),dtype={UID:str}).set_index(UID)
    contract=json.loads(blobs['contract.json']);gold=raw[raw[LABELS].notna().any(axis=1)]
    if len(gold)!=58 or gold[LABELS].notna().sum().sum()!=696:raise ValueError('unexpected released Gold denominator')
    if raw.index.duplicated().any() or frame.index.duplicated().any() or set(frame.index)!=set(raw.index):raise ValueError('invalid source identity')
    evidence=[json.loads(line) for line in blobs['evidence.jsonl'].decode('utf-8').splitlines()]
    mapped={(r[UID],r['label']):r for r in evidence}
    if len(mapped)!=len(evidence) or set(mapped)!={(u,l) for u in frame.index for l in LABELS}:raise ValueError('evidence is incomplete or duplicated')
    # Current compiler has explicit report_facts; older frozen releases preserve
    # criterion itself while compiled_target alone receives the Gold overwrite.
    rows=[]
    for uid in gold.index:
        for label in LABELS:
            record=mapped[(uid,label)];fact=record.get('report_facts',record);prediction=fact['criterion']
            if prediction is not None and prediction not in (0,1):raise ValueError('report criterion is not ternary')
            expected=int(gold.loc[uid,label]);compiled=frame.loc[uid,label]
            if compiled!=expected or frame.loc[uid,label+'__weight']!=1:raise ValueError('released Gold altered in compiled training labels')
            outcome='unknown' if prediction is None else 'TP' if prediction==expected==1 else 'TN' if prediction==expected==0 else 'FP' if prediction==1 else 'FN'
            quotes=fact.get('evidence',[]);report=str(raw.loc[uid,'Report'])
            if any(q not in report for q in quotes):raise ValueError('nonliteral evidence quote')
            rows.append({UID:uid,'label':label,'official':expected,'report_criterion':prediction,
                'compiled_target_NOT_used_for_alignment':int(compiled),'outcome':outcome,
                'reason':fact.get('reason'),'certainty':fact.get('certainty'),'quotes':quotes,
                'report_sha256':hashlib.sha256(report.encode()).hexdigest(),
                'current_fold':int(frame.loc[uid,'fold']),'development_only_flag':bool(frame.loc[uid,'development_only']),
                'alignment_scope':'previously exposed Gold development diagnostic; no new rule correction in this run'})
    per_label=metrics(rows);counts=Counter(r['outcome'] for r in rows)
    # Snapshot coverage actually usable for training, not only raw corpus totals.
    coverage=[]
    for scope,mask in [('all_non_gold',~frame.is_gold),('training_eligible_non_gold',~frame.is_gold & frame.fold.ge(0))]:
        subset=frame[mask]
        for label in LABELS:
            known=subset[label+'__weight'].gt(0)
            coverage.append({'scope':scope,'label':label,'studies':len(subset),'positive':int((known&subset[label].eq(1)).sum()),
                'negative':int((known&subset[label].eq(0)).sum()),'unknown':int((~known).sum()),
                'known_weight_sum':float(subset.loc[known,label+'__weight'].sum())})
    previous=root/'analysis/expert_report_alignment_current/comparison_before_official.jsonl'
    exposure=set()
    if previous.exists():exposure={json.loads(line)[UID] for line in previous.read_text(encoding='utf-8').splitlines()}
    clean=set(gold.index)-exposure
    # Existing dense public score tables are diagnostic comparators only. Their
    # label-generation exposure is unverified; neither is an untouched test.
    baselines=[];skipped_baselines=[]
    public_paths=[Path(__file__).parent/'data/labels/pilkwang/report_labels_v2.csv',Path(__file__).parent/'data/labels/stevenleehans/llm_labels_v4_blend.csv'] if include_public else []
    for path in public_paths:
        if not path.exists():continue
        table=pd.read_csv(path,dtype={UID:str}).set_index(UID);auc={}
        missing=set(gold.index)-set(table.index)
        if table.index.duplicated().any() or missing:
            skipped_baselines.append({'path':str(path),'sha256':file_hash(path),'missing_gold_studies':len(missing),
                'duplicate_ids':int(table.index.duplicated().sum()),'reason':'no complete unique Gold alignment; not evaluated'})
            continue
        for label in LABELS:
            values=table.loc[gold.index,label].to_numpy(float)
            if not np.isfinite(values).all():raise ValueError('public comparator score missing')
            auc[label]=float(roc_auc_score(gold[label],values))
        baselines.append({'path':str(path),'sha256':file_hash(path),'per_label_full58_auc':auc,'macro_full58_auc':float(np.mean(list(auc.values()))),
            'scope':'frozen public score columns, no target changes; label-generation Gold exposure unverified; cannot directly compare with selective report agreement'})
    known=len(rows)-counts['unknown']
    current_compiler=file_hash(Path(__file__).with_name('criteria.py'))
    after={str(p):file_hash(p) for p in paths}
    if after!=hashes:raise ValueError('source snapshot changed during audit; rerun against a stable release')
    out=Path(output)/f"{release.name}_{hashes[str(labels)][:12]}";out.mkdir(parents=True,exist_ok=True)
    for name in ('labels.csv','contract.json','evidence.jsonl'):(out/('source_'+name)).write_bytes(blobs[name])
    for name,records in [('gold_comparison.jsonl',rows),('gold_discordances.jsonl',[r for r in rows if r['outcome'] in ('FP','FN')])]:
        (out/name).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    pd.DataFrame(per_label).to_csv(out/'gold_per_label.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'training_coverage.csv',index=False)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'label_release':str(release),'contract_version':contract['version'],
        'source_sha256':hashes,'source_unchanged':True,'current_compiler_matches_frozen_release':current_compiler==contract['compiler_sha256'],
        'audit_input':'Persisted report criterion before official target overwrite; report_facts used when available. No current re-extraction mixed into frozen results.',
        'gold_studies':58,'gold_label_cells':696,'known':known,'unknown':counts['unknown'],
        'gold_full_text_semantic_review_cells':sum(mapped[(uid,label)].get('semantic_review') is not None for uid in gold.index for label in LABELS),
        'full_pipeline_comparison_limit':'Gold official targets can exclude these rows from unknown-target model review. Compare the same report-only extraction workflow before judging whole-pipeline quality; this replay evaluates persisted pre-overwrite facts only.',
        'coverage':known/696,**{k:counts[k] for k in ('TP','FP','TN','FN')},
        'agreement_known':(counts['TP']+counts['TN'])/known,
        'compiled_gold_agreement_NOT_an_extraction_metric':1.,'gold_previously_exposed':len(exposure&set(gold.index)),
        'gold_not_found_in_prior_audit':len(clean),'independent_gold_estimate_available':False,
        'current_gold_development_flag_count':int(frame.loc[gold.index,'development_only'].sum()),
        'important_exposure_limit':'A false development_only flag does not undo prior extractor/selection exposure. All 58 were compared in an earlier diagnostic.',
        'current_reserved_image_review_studies':int(frame.fold.eq(-3).sum()),
        'reserve_is_gold':False,'reserve_limit':'Reserved patients still need independent image adjudication. Do not inspect or use them to repair extraction in this audit.',
        'per_label':per_label,'training_coverage':coverage,'public_score_diagnostics':baselines,'skipped_public_tables':skipped_baselines,
        'macro_auc_for_our_abstaining_labels':None,'macro_auc_unavailable_reason':'No continuous full58 predictions for all classes. Unknowns not silently mapped to 0 or 0.5. Answered-subset hard AUC is reported separately and may be undefined.',
        'unknown_reasons':dict(Counter(r['reason'] for r in rows if r['outcome']=='unknown')),
        'discordance_is_not_proven_extraction_error':True,'rules_or_labels_changed':False,'optimizer_steps':0,
        'interval_limit':'Wilson intervals describe per-label answered-sample uncertainty only; no correction for development reuse or selection bias.'}
    save_json(out/'summary.json',summary)
    tr=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [ZH[i],f"{r['gold_positive']}/{r['gold_negative']}",f"{r['known']}/58",f"{100*r['agreement_known']:.1f}%" if r['agreement_known'] is not None else '—',r['FP'],r['FN'],r['unknown_gold_positive'],r['unknown_gold_negative']])+'</tr>' for i,r in enumerate(per_label))
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>12类标签质量对照</title><style>body{{font:16px/1.8 system-ui;background:#f5f6f8;color:#25313b}}main{{max-width:1050px;margin:28px auto;background:white;padding:30px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #dce2e8;padding:8px}}th{{background:#edf3f7}}.note{{background:#fff4de;padding:16px}}a{{color:#176b9f}}</style><main><h1>12类标签质量：{html.escape(release.name)}</h1>
    <p>58次官方专家检查，696个已知参考值。读取冻结证据中官方覆盖前的报告判断；不是拿已覆盖的训练标签验证自身。本次没有修改抽取规则、标签或训练模型。</p>
    <p class="note">这58例此前全部参与过报告对齐诊断，属于开发回放。Gold的影像参考与报告文本不是同一种信息，分歧不自动证明抽取器或专家有错。{known}/696格给出判断（{known/696:.1%}），其中{counts['TP']+counts['TN']}/{known}与Gold一致（{summary['agreement_known']:.1%}）；其余{counts['unknown']}格保持未知。不能将后一个比例称为全体标签准确率。</p>
    <table><tr><th>标签</th><th>Gold正/负</th><th>有判断</th><th>已判断一致率</th><th>FP</th><th>FN</th><th>未知中的Gold正</th><th>未知中的Gold负</th></tr>{tr}</table>
    <p>FP：报告判断阳性而Gold阴性；FN：报告判断阴性而Gold阳性。未知单列，不并入FN/TN。比如只回答阴性的类别，即使一致率100%，也不能说明其阳性提取能力可靠。</p>
    <h2>如何继续验证</h2><ol><li>文本质量：对训练开发范围做按病种、语言、正/负/未知分层的盲审，判别否定、部位、程度、急慢性和同一病变条件；保存原句和无法判断原因。用抽样权重估计总体，定向错例不能冒充随机噪声率。</li><li>影像对齐：保留当前{summary['current_reserved_image_review_studies']}个隔离检查，交由独立影像判读建立参考；目前只是预留，不是新增Gold。</li><li>训练价值：恢复训练后固定患者划分、模型、输入和预算，只替换监督，比较12类OOF AUC和错序；报告语义更对不保证测试排序提高。</li></ol>
    <p>原始表、混淆计数、选择性precision/recall、Wilson区间、有效训练正负未知覆盖、公用soft-label表的Gold AUC均保存在本目录。公用表生成时是否使用Gold未知，不能把其分数当独立泛化证据。</p>
    <p>源码当前是否与此冻结发布一致：{summary['current_compiler_matches_frozen_release']}。本次只评该发布的持久化证据，未把正在修改的规则混进结果。</p>
    <p><a href="summary.json">完整指标和来源SHA</a> · <a href="gold_per_label.csv">12类明细</a> · <a href="gold_discordances.jsonl">全部分歧及原句</a> · <a href="training_coverage.csv">训练监督覆盖</a></p></main></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')
    with (out/'report.html').open('a',encoding='utf-8') as page:
        page.write('<p style="max-width:1050px;margin:20px auto;padding:18px;background:#fff4de">验证口径：Gold的696格中，'
                   +str(summary['gold_full_text_semantic_review_cells'])+'格有后续全文语义复核记录。Gold官方值可能使它们未进入未知目标复核队列；此表只能评价保存的覆盖前报告分支，不能直接代表在其他报告上运行了更多复核步骤的完整流程。</p>')
    print(json.dumps({'output':str(out),'known':known,'unknown':counts['unknown'],'agreement_known':summary['agreement_known'],'per_label':per_label,
                      'public_macro':[{'path':r['path'],'macro':r['macro_full58_auc']} for r in baselines]},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--output',default='F:/rsnaknee/analysis/label_quality_gold_20261001')
    p.add_argument('--include-public',action='store_true',help='explicit archived comparison only; never a label input')
    a=p.parse_args();run(a.root,a.output,include_public=a.include_public)
