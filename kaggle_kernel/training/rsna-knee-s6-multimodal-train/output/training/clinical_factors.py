"""Separate clinical predicates, with independent missingness and evidence.

These are criterion components, not a universal severity score. Official
positives imply required predicates; official negatives do NOT negate them all.
"""
import re
from .common import LABELS

REQUIRED={
 'ACL':['high_grade'], 'MCL':['high_grade','acute'],
 'Medial Meniscus':['surface_or_morphology'], 'Lateral Meniscus':['surface_or_morphology'],
 'Medial OA':['depth_gt_half','extent_ge_10mm'], 'Lateral OA':['depth_gt_half','extent_ge_10mm'],
 'PF OA':['depth_gt_half','extent_ge_10mm'], 'Effusion':['amount_moderate_or_large'],
 'Synovitis':['inflammation','thickening'], "Baker's":['size_moderate_or_large'],
 'Contusion':['traumatic_edema','no_fracture_line'], 'Fracture':['acute','fracture_line'],
}
FACTORS=[(c,f) for c in LABELS for f in REQUIRED[c]]
FACTOR_COLUMNS=[c+'__factor_'+f for c,f in FACTORS]
FACTOR_LABEL_INDEX=[LABELS.index(c) for c,f in FACTORS]


def lesion_extent_mm(text):
    """Bind a linear measurement to a cartilage defect, not a nearby cyst/edema."""
    values=[]
    text=re.sub(r'((?:cartilage|chondral)[^,;]{0,55}(?:loss|defect|lesion|area))\s*,\s*(?=\d)',r'\1 ',text)
    for part in re.split(r'\b(?:with|and|associated|plus)\b|[,;]',text):
        if not re.search(r'(?:cartilage|chondral).*(?:loss|defect|lesion|area)|(?:loss|defect|lesion).*cartilage',part):continue
        if re.search(r'cyst|osteophyte|marrow|bone edema|joint.space|cm\s*\^\s*2|mm\s*\^\s*2|\bdeep\b|\bdepth\b',part):continue
        if re.search(r'thickness\s*(?:of|is|measures)?\s*\d',part):continue
        for dims,unit in re.findall(r'(\d+(?:\.\d+)?(?:\s*(?:[x×*]|by)\s*\d+(?:\.\d+)?)*)\s*(cm|mm)\b',part):
            values.extend(float(x)*(10 if unit=='cm' else 1) for x in re.split(r'\s*(?:[x×*]|by)\s*',dims))
    return max(values) if values else None


def infer_scoped(label,text):
    from .criteria import norm,has,LOW,HIGH,ACUTE,CHRONIC,UNCERTAIN,NEG,TEAR,quantity_scope,quantity_levels,explicit_turkish_fluid_absence,low_ligament_grade,cartilage_evidence_scope,cartilage_lesion_scopes,mixed_cartilage_depth,whole_ligament_continuity
    t=norm(text);out={}
    if label.endswith('OA'):t=cartilage_evidence_scope(t)
    if label=='Effusion':t=quantity_scope(label,t)
    if has(UNCERTAIN,t):return out
    def put(name,value):out[name]={'target':int(value),'weight':.7,'reason':'explicit_report_predicate'}
    if label.endswith('OA') and len(cartilage_lesion_scopes(t))>1:
        local=[infer_scoped(label,p) for p in cartilage_lesion_scopes(t)]
        qualifies=any(r.get('depth_gt_half',{}).get('target')==1 and r.get('extent_ge_10mm',{}).get('target')==1 for r in local)
        if any(r.get('depth_gt_half',{}).get('target')==1 for r in local):put('depth_gt_half',1)
        if qualifies:put('extent_ge_10mm',1)
        return out
    if label in ('ACL','MCL') and whole_ligament_continuity(t):put('high_grade',0)
    if label in ('MCL','Fracture'):
        if has(r'no acute|without acute',t):put('acute',0)
        elif has(ACUTE,t) and not has(CHRONIC,t):put('acute',1)
        elif has(CHRONIC,t) and not has(ACUTE,t):put('acute',0)
    # More complex negation is left masked rather than assigning false facts.
    negative=has(NEG,t)
    if label in ('ACL','MCL') and not negative:
        if has(HIGH,t) and low_ligament_grade(t):pass  # conflicting grades of one scoped injury
        elif has(HIGH,t) and has(TEAR+'|sprain|injur',t):put('high_grade',1)
        elif low_ligament_grade(t) and has(r'sprain|tear|degener|signal|thicken|injur|esguince',t):put('high_grade',0)
        elif has(r'(?:>|over|more than|greater than)\s*50\s*%\s*(?:of (?:the )?)?fib',t):put('high_grade',1)
    if 'Meniscus' in label:
        if has(r'(?:without|not).*reach.*surfac|without.*surfac|intrasubstance.*no.*tear',t):put('surface_or_morphology',0)
        elif not negative and (has(r'truncat|displaced (?:meniscal )?fragment|bucket.handle|flipped fragment',t) or
              (has(r'(?:contact|reach|extend).*surfac',t) and has(r'(?:two|2|multiple|consecutive)\s+(?:image|slice)',t))):put('surface_or_morphology',1)
    if label.endswith('OA'):
        if has(r'(?:no|without)\s+(?:focal\s+)?high.grade\s+(?:cartilage|chondral)\s+(?:defect|loss|lesion)',t):put('depth_gt_half',0)
        if not negative:
            if has(HIGH+r'|(?:>|over|more than|greater than)\s*50\s*%',t):put('depth_gt_half',1)
            elif has(r'low.grade|(?:<|less than|under)\s*50\s*%',t):put('depth_gt_half',0)
            # Use measurements tied to the lesion/area; thickness, cyst sizes,
            # and joint-space width must not become lesion-extent supervision.
            if has(r'(?:cartilage|chondral).*(?:loss|defect|lesion|area)|(?:loss|defect|lesion).*cartilage',t) and not has(r'cyst|osteophyte|joint.space|thickness\s*(?:of|is|measures)?\s*\d',t):
                extent=lesion_extent_mm(t)
                if extent is not None and not mixed_cartilage_depth(t):put('extent_ge_10mm',extent>=10)
    if label=='Effusion' and explicit_turkish_fluid_absence(t):put('amount_moderate_or_large',0)
    elif label in ('Effusion',"Baker's") and not negative:
        key='amount_moderate_or_large' if label=='Effusion' else 'size_moderate_or_large'
        quantity=quantity_scope(label,t)
        low,high=quantity_levels(label,quantity)
        if high!=low and (high or low):put(key,high)
    if label=='Synovitis':
        for key,pattern in [('inflammation',r'synovitis|sinovitis|sinovite|inflamm|inflam|滑膜炎')]:
            if has(pattern,t) and not negative:put(key,1)
            elif has(r'(?:no|without)\s+(?:synovial\s+)?(?:'+pattern+r')',t):put(key,0)
        membrane=r'(?:synovial(?:e)?|synovium|synovia|sinovial)'
        thick=(membrane+r'(?:\s+(?:is|has|shows|demonstrates|mild|diffuse|focal|marked|moderate|severe|some)){0,3}\s+(?:thicken\w*|hypertroph\w*)|'
               r'(?:thicken\w*|hypertroph\w*|engros\w*)(?:\s+(?:of|the|de|la|del|diffuse|mild)){0,5}\s+'+membrane+'|滑膜.{0,4}增厚')
        if has(thick,t) and not negative:put('thickening',1)
        elif has(r'(?:no|without|sin|sem)\s+(?:evidence of\s+)?(?:synovial\s+)?(?:thickening|hypertrophy|engrosamiento)',t):put('thickening',0)
    if label=='Contusion':
        if not negative and has(r'bone contusion|bone bruise|contusion osea|traumatic.*(?:edema|oedema)',t):put('traumatic_edema',1)
        elif has(r'(?:no|without)\s+(?:bone )?(?:contusion|bruise)',t):put('traumatic_edema',0)
        elif has(r'degenerative.*(?:edema|oedema)|(?:edema|oedema).*degenerative',t):put('traumatic_edema',0)
        if has(r'without (?:a )?(?:discrete )?fracture line|no fracture line',t):put('no_fracture_line',1)
        elif (has(r'fracture line.{0,100}(?:surrounding|associated|adjacent|accompanying).{0,50}marrow (?:edema|oedema)',t)
              and not has(r'(?:no|without)\s+(?:a\s+)?(?:discrete\s+|definite\s+|focal\s+)?fracture line',t)
              and not has(r'\b(?:separate|elsewhere|other lesion)\b',t)):
            put('no_fracture_line',0)
    if label=='Fracture':
        if has(r'no fracture|without.*fracture line',t):put('fracture_line',0)
        elif not negative and has(r'fracture line|cortical (?:break|disruption)',t):put('fracture_line',1)
    return {k:v for k,v in out.items() if k in REQUIRED[label]}


def merge_scopes(label,scopes):
    result={}
    for scope,quote in scopes:
        for key,fact in infer_scoped(label,scope).items():
            result.setdefault(key,[]).append({**fact,'quote':quote})
    merged={}
    for key,items in result.items():
        vals={x['target'] for x in items};conflict=len(vals)>1
        merged[key]={'target':None if conflict else next(iter(vals)),'weight':0. if conflict else min(x['weight'] for x in items),
                     'source':'report','state':'conflict' if conflict else 'explicit','evidence':[x['quote'] for x in items]}
    return merged


def apply_official(label,factors,official):
    result={k:dict(v) for k,v in factors.items()}
    if official==1:
        for f in REQUIRED[label]:result[f]={'target':1,'weight':1.,'source':'official_positive_entails_criterion',
            'state':'official','evidence':['Official '+label+' positive; required predicate '+f]}
    elif official==0 and all(result.get(f,{}).get('target')==1 for f in REQUIRED[label]):
        # Expert negative contradicts the complete report-derived conjunction.
        # It does not identify WHICH predicate is false, so mask all of them.
        for f in REQUIRED[label]:result[f]={**result[f],'weight':0.,'state':'conflicts_with_official_criterion'}
    return result
