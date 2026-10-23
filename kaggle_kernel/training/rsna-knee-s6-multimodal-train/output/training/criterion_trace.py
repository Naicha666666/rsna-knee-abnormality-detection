"""Preserve report facts before expert-derived supervision overlays.

Statement IDs identify literal report spans, not verified image lesions.
Unknown lesion binding stays unknown; it is never reconstructed from a label.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import re
from .clinical_factors import REQUIRED

VERSION='report-image-supervision-trace-v1'


def report_snapshot(fact, report):
    saved=deepcopy(asdict(fact));units=[]
    for quote in saved['evidence']:
        positions=[{'start':m.start(),'end':m.end()} for m in re.finditer(re.escape(quote),report)]
        if not quote or not positions:raise ValueError('report evidence is not literal source text')
        units.append({'statement_id':hashlib.sha256((fact.label+'\0'+quote).encode()).hexdigest()[:20],
            'quote':quote,'source_spans':positions,'target_structure':fact.label,
            'lesion_id':None,'lesion_identity_status':'not_established_by_statement_id'})
    saved['evidence_units']=units
    saved['conditions']={key:{'target':saved['attributes'].get(key,{}).get('target'),
        'weight':saved['attributes'].get(key,{}).get('weight',0.),
        'source':saved['attributes'].get(key,{}).get('source','not_observed'),
        'scope':deepcopy(saved['attributes'].get(key,{}).get('scope')),
        'statement_ids':[u['statement_id'] for u in units if u['quote'] in saved['attributes'].get(key,{}).get('evidence',[])]}
        for key in REQUIRED[fact.label]}
    saved['missing_conditions']=[k for k,v in saved['conditions'].items() if v['target'] is None or v['weight']<=0]
    saved['is_expert_image_truth']=False
    return saved


def evidence_trace(report_fact, official, *, public_fallback=None):
    label=report_fact['label']
    return {'trace_version':VERSION,'rubric_source':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343',
        'report_facts':deepcopy(report_fact),
        'official_image_reference':None if official is None else {'value':int(official),'source':'released_expert_image_review'},
        'official_derived_conditions':({key:{'target':1,'source':'official_positive_entails_criterion','quoted_in_report':False}
            for key in REQUIRED[label]} if official==1 else {}),
        'public_presence_fallback':deepcopy(public_fallback),
        'boundary':'Report observations, teacher fallback and expert-label entailments are different evidence sources.'}
