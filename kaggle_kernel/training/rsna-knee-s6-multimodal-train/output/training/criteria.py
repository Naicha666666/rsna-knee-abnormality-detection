"""Evidence -> finding presence and competition-criterion targets.

Conservative report compiler. Unsupported/unspecified facts stay unknown;
structured reviewed evidence can be supplied with literal report quotations.
This is weak supervision, not an automatic replacement for image adjudication.
"""
from dataclasses import dataclass,asdict,field
from pathlib import Path
import argparse,json,re,unicodedata
import numpy as np
import pandas as pd
from .common import LABELS,UID,save_json,file_hash

VERSION='criterion-evidence-v7.4-final-condition-review'


def labels_directory(root):
    """One active compiler version for producers, trainers and comparisons."""
    return Path(root)/('labels_clinical_'+VERSION.split('-')[2].replace('.','_'))
PATTERNS={
 'ACL':r'\bacl\b|\blca\b|anterior cruciate|cruzad[oa] anterior|croise anterieur|anterior capraz|on capraz|前交叉',
 'MCL':r'\bmcl\b|medial collateral|medial (?:and|&) (?:fibular|lateral) collateral|collateral medial|colateral (?:intern[oa]|medial)|medial kollateral|内侧副',
 'Medial Meniscus':r'medial menisc\w*|menisc\w*(?:\s+\w+){0,2}\s+(?:medial|intern[oa])|medial menisk\w*|内侧半月板',
 'Lateral Meniscus':r'lateral menisc\w*|menisc\w*(?:\s+\w+){0,2}\s+(?:lateral|extern[oa])|lateral menisk\w*|外侧半月板',
 'Medial OA':r'medial (?:compartment|femoral|tibial)|(?:compartim\w*|femorotibial|condilo femoral|platillo tibial) medial|内侧间室',
 'Lateral OA':r'lateral (?:compartment|femoral|tibial)|(?:compartim\w*|femorotibial|condilo femoral|platillo tibial) lateral|外侧间室',
 'PF OA':r'patell?ofemoral|patello.femoral|femoropatel\w*|\bpatelar\b|\bpatellar\b|trochlea\w*|troclea\w*|rotula|髌股',
 'Effusion':r'effusion|derrame|efusion|\bgelenkerguss\b|eklem[^.;\n]{0,20}sivi|关节积液',
 'Synovitis':r'synovitis|sinovitis|sinovite|synovial|sinovial|synovium|滑膜',
 "Baker's":r'baker|popliteal cyst|quistes? popliteos?|cisto popliteo|腘窝囊肿',
 'Contusion':r'bone (?:contusion|bruise)|marrow (?:edema|oedema)|contusion osea|edema oseo|edema medular|骨挫伤|骨髓水肿',
 'Fracture':r'fractur\w*|fraktur\w*|cortical (?:break|disruption)|骨折',
}
NORMAL=r'\b(?:normal|normaldir|intakt(?:es|er|e)?|intact|preserved|integro|integra|conservad[oa]s?|preservad[oa]s?|sin alteraciones)\b|完整|正常'
NEG=r'\b(?:no|without|absent|absence of|sin|sem|pas de|aucun|degil(?:dir)?)\b|未见|无'
UNCERTAIN=r'\b(?:suspect\w*|suspicious|possib\w*|equivocal|questionable|probabl\w*|cannot exclude|rule out|r/o|sospech\w*|dudoso)\b|可疑|不能排除'
LOW=r'\b(?:mild|small|trace|trivial|minimal|slight|low.grade|leve|pequen\w*|minim\w*|discret\w*|grade\s*(?:i|1)\b(?!\s*[-–]\s*(?:ii|iii|2|3)\b)|grado\s*(?:i|1)\b(?!\s*[-–]\s*(?:ii|iii|2|3)\b))\b|少量|轻度|低级别'
HIGH=r'\b(?:high.grade|complete|full.thickness|complet\w*|alto grado|total|rupture complete|grade\s*(?:iii|3)|grado\s*(?:iii|3))\b|完全|高等级|全层'
ACUTE=r'\b(?:acute|agud\w*|recent|reciente|akut)\b|急性|新鲜'
CHRONIC=r'\b(?:chronic|old|healed|remote|cron\w*|antigu\w*|consolidat\w*|sequela\w*)\b|陈旧|慢性|愈合'
TEAR=r'\b(?:tear\w*|torn|ruptur\w*|rotur\w*|desgarr\w*|disrupt\w*|yirtik)\b|撕裂|断裂'
ABNORMAL=r'\b(?:sprain|injur\w*|degener\w*|thicken\w*|edema|oedema|abnormal|esguince|lesion\w*|engros\w*)\b|损伤|增厚|退变'
CARTILAGE=r'\b(?:cartilag\w*|chondr\w*|condr\w*)|软骨'
NO_TEAR=(r'\b(?:no|without|sin|sem)\s+(?:(?:frank|definite|a)\s+|evidence of\s+|signs of\s+|'
         r'signos de\s+|signos propios de\s+|caracteres propios de\s+|criterios de (?:una )?\s*)?'
         r'(?:tear\w*|ruptur\w*|rotur\w*)|\b(?:is|are)\s+not\s+(?:torn|ruptured|disrupted)\b')


def norm(text):
    text=unicodedata.normalize('NFKD',str(text).replace('ı','i'))
    text=''.join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r'\s+',' ',re.sub(r'(?<=\d),(?=\d)','.',text.casefold())).strip()


def report_clauses(original):
    """Keep literal evidence while joining soft line wraps and preserving decimals."""
    start=0
    for match in re.finditer(r'\.(?!\d)|(?<!\d)\.|;|\r?\n+',original):
        token=match.group()
        if token=='.':
            left=original[start:match.start()].rstrip();right=original[match.end():].lstrip()
            # A typographic stop after a measurement unit can separate the
            # lesion's dimension from its immediately following location.
            if re.search(r'\d\s*(?:mm|cm)$',left,re.I) and re.match(r'(?:at|along|in|over|within)\b',right):continue
        if '\n' in token:
            left=original[start:match.start()].rstrip()
            right=original[match.end():].lstrip()
            # Lower-case continuation after a wrapped line belongs to its
            # preceding sentence; headings and bulleted findings still split.
            if right and right[0].islower() and not left.endswith(':') and '\n\n' not in token:
                continue
        clause=original[start:match.start()].strip()
        if clause and not re.fullmatch(r'\d+',clause):yield clause
        start=match.end()
    tail=original[start:].strip()
    if tail:yield tail


def shared_negative_end(text):
    """Extent of a coordinated absence statement, before any positive contrast."""
    prefix=re.split(r'\b(?:but|however|whereas|with)\b',text,maxsplit=1)[0].strip()
    item=(r'(?:fractures?(?: lines?)?|bone bruises?|bone contusions?|(?:bone )?marrow (?:edema|oedema)|'
          r'(?:knee )?(?:joint )?effusions?|synovitis|synovial thickening)')
    pattern=(r'\s*(?:[>\-*]\s*)?(?:there\s+(?:is|are)\s+)?(?:no|without|sin|sem)\s+(?:(?:acute|significant|definite)\s+)?'
             r'(?:(?:evidence|signs) of\s+)?'+item+
             r'(?:\s*(?:,|or|and|nor|&)\s*'+item+r')*'+
             r'(?:\s+(?:is|are)\s+(?:seen|noted|identified|detected|demonstrated|observed|present))?\s*')
    # An anatomical heading does not interrupt the scope of "No A or B".
    statement=prefix.rsplit(':',1)[-1].strip().rstrip(',').strip()
    spanish_item=r'(?:derrame(?: articular)?|quistes? popliteos?)'
    spanish=(r'(?:no hay|sin)\s+(?:evidentes?\s+)?'+spanish_item+
             r'(?:\s*(?:,|ni|y|o)\s*'+spanish_item+r')*')
    if re.fullmatch(spanish,statement):return len(prefix)
    return len(prefix) if re.fullmatch(pattern,statement) else 0


def has(pattern,text):return bool(re.search(pattern,text))


def quantity_scope(label,text):
    """A cyst's leakage/rupture severity is not its size."""
    if label=="Baker's":
        if re.match(r'(?:small|minimal|mild|moderate|large)\s+(?:amount of\s+)?(?:edema|oedema|fluid)\s+(?:along|around|surrounding|outside)\b',text):
            # The quantity belongs to tissue/fluid OUTSIDE the named cyst.
            # Retain cyst presence, but do not give it that quantity.
            return "popliteal cyst" if re.search(r'popliteal cyst|baker',text) else text
        return re.split(r'\b(?:with|con|showing|demonstrating)\b',text,maxsplit=1)[0]
    if label=='Effusion':
        # The degree of interval change is not the current fluid quantity.
        text=re.sub(r'\b(?:mild|minor|slight|small)\s+(?:interval\s+)?(?:reduction|decrease)\s+in\s+(?=(?:moderate|large)\b)','',text)
        # An absence of loose bodies is unrelated to the amount of joint fluid.
        text=re.sub(r',\s*(?:there (?:are|is)\s+)?no\s+intra.articular\s+(?:loose\s+)?bod(?:y|ies)(?:\s+(?:seen|identified|noted))?\s*$','',text)
        # An explicit moderate joint effusion is not negated by a smaller
        # amount described in a named subregion. The converse remains unknown.
        qualifying=re.search(r'\b(?:moderate|large)(?:\s+(?:amount|of|knee|joint|suprapatellar)){0,5}\s+effusion\b',text)
        subregion=re.search(r'\s+(?:with|and)\s+(?:minimal|small|mild)\s+(?:fluid distended\s+)?suprapatellar\s+(?:bursa|bursal fluid)\b',text)
        if qualifying and subregion and qualifying.end()<=subregion.start():text=text[:subregion.start()]
    return text


def quantity_levels(label,text):
    text=quantity_scope(label,text)
    high=has(r'\b(?:moderate(?:ly)?|large|massive(?:ly)?|abundant|moderad[oa]s?|abundante)\b|中量|大量|较大'+(r'|\bmassig(?:er|es|e|en)?\b' if label=='Effusion' else ''),text)
    low=has(LOW+(r'|\bminor\b|\bhafif\b|\bgering(?:er|es|e|en)?\b' if label=='Effusion' else ''),text)
    return low,high


def explicit_turkish_fluid_absence(text):
    """Exact report assertions, not occurrence of a Turkish negative suffix anywhere."""
    statement=text.rsplit(':',1)[-1].strip()
    statement=re.sub(r'^[>\-*]\s*','',statement).strip()
    return bool(re.fullmatch(r'(?:diz )?eklem(?:i ici|inde|de) sivi (?:miktari normal(?:dir)?|artisi saptanmamistir)',statement))


def coordinated_normal_structures(text):
    """Share a terminal normal predicate only over an explicit anatomy-only list."""
    statement=text.rsplit(':',1)[-1].strip()
    item=(r'(?:(?:on|anterior|arka) capraz bag|(?:lateral ve medial|lateral|medial) kollateral ligaman|'
          r'(?:lateral|medial) meniskus|acl|pcl|mcl|lcl)')
    if re.fullmatch(item+r'(?:\s*(?:,|ve|und|and|&)\s*'+item+r')*\s+(?:normaldir|normal|intakt)',statement):return True
    english=(r'(?:acl|pcl|mcl|lcl|(?:anterior|posterior) cruciate(?: ligaments?)?|'
             r'medial and (?:lateral|fibular) collateral ligaments?|(?:medial|lateral) collateral ligaments?|'
             r'(?:medial|lateral) meniscus)')
    return bool(re.fullmatch(r'(?:the )?'+english+r'(?:\s*(?:,\s*(?:and\s+)?|and\s+)'+english+r')*'
                            r'\s+(?:are|appear(?: to be)?|remain)\s+(?:intact|normal|preserved)',statement))


def low_ligament_grade(text):
    """Bind low grade to ligament injury, not adjacent edema or fiber gap."""
    grade=r'\bgrad[eo]\s*(?:i|1)\b(?!\s*[-–]\s*(?:ii|iii|2|3)\b)'
    injury=r'(?:tear\w*|sprain|injur\w*|esguince|rotura|ruptura)'
    adjective=r'(?:low.grade|mild|leve|slight)'
    ligament=r'(?:(?:acl|mcl|anterior cruciate(?: ligament)?|medial collateral(?: ligament)?)\s+)?'
    return has(grade,text) or has(adjective+r'\s+(?:partial\s+)?'+ligament+injury,text) or has(injury+r'\s+(?:de\s+)?'+adjective,text)


def cartilage_sites(label,text):
    if label=='PF OA':
        return {site for site,pattern in [('patella',r'\bpatell(?:ar|a)\b|\bpatelar\b|\brotula\b'),
                                          ('trochlea',r'\btrochlea\w*|\btroclea\w*')] if has(pattern,text)}
    return {site for site,pattern in [('femur',r'\bfemoral\b|\bcondyles?\b|\bcondilos?\b'),
                                     ('tibia',r'\btibial\b|\bplateaus?\b|\bplatillos?\b')] if has(pattern,text)}


def complete_cartilage_negative_coverage(label,quotes,local_context=None):
    """A heading cannot enlarge an observation of only one articular surface.

    PF requires patella and trochlea; either femorotibial compartment requires
    its femoral and tibial surfaces. A whole-compartment cartilage statement
    without a narrower named surface covers the compartment.
    """
    required={'patella','trochlea'} if label=='PF OA' else {'femur','tibia'}
    covered=set()
    for quote in quotes:
        text=norm(quote);sites=cartilage_sites(label,text)
        bounded=has(r'\bfocal\b|\bfoci\b|\bulcer\w*\b|\d\s*(?:mm|cm)\b',text)
        broad_absence=has(r'(?:no|without)\s+(?:focal\s+)?(?:high.grade\s+)?(?:chondrosis|(?:cartilage|chondral)\s+(?:defect|loss|lesion|injury))',text)
        complete_inventory=has(r'(?:otherwise|remaining|rest of|elsewhere).{0,45}(?:cartilage|normal|preserved)|'
                               r'only\s+(?:cartilage\s+)?(?:abnormality|defect|lesion)',text)
        if bounded and not broad_absence and not complete_inventory:continue
        if sites:covered.update(sites)
        elif (local_context or {}).get(quote)==label:covered.update(required)
        elif label=='PF OA' and has(r'patell?ofemoral|femoropatel|髌股',text):covered.update(required)
        elif label!='PF OA':
            side='medial' if label=='Medial OA' else 'lateral'
            if has(side+r' (?:compartment|femorotibial|tibiofemoral)|(?:compartim\w*|femorotibial|tibiofemoral) '+side,text):covered.update(required)
    return covered==required


def cartilage_evidence_scope(text):
    """Absence of secondary osseous change does not negate a cartilage lesion."""
    return re.sub(r'\b(?:sin cambios oseos(?: secundarios)?|without (?:associated |underlying |subchondral |bone |marrow )*(?:edema|oedema|cystic changes))\b','',text)


def cartilage_lesion_scopes(text):
    """Keep explicitly different lesions separate; these are text units, not ROIs."""
    pieces=re.split(r'\b(?:and|with|plus)\s+(?:a\s+)?(?:separate|another|distinct|different)\b',text)
    return pieces if len(pieces)>1 and all(has(CARTILAGE+r'|defect|lesion|fissure|loss',p) for p in pieces) else [text]


def mixed_cartilage_depth(text):
    low=has(r'low.grade|(?:less than|under|<)\s*50\s*%',text)
    high=has(r'high.grade|full.thickness|(?:more than|greater than|over|>)\s*50\s*%',text)
    return low and high


def local_intact_ligament(text):
    if has(r'\ball (?:the |its )?fib(?:er|re)s\b|\bboth bundles\b|\bentire ligament\b',text):return False
    return (has(NORMAL+r'|continuous|continuity',text) and
        has(r'\b(?:anterior|posterior|proximal|distal|remaining|some)\s+(?:(?:acl|mcl|ligament)\s+)?(?:fib(?:er|re)s|bundles?|portion)|'
            r'fib(?:er|re)s\s+(?:anteriorly|posteriorly|proximally|distally)',text))


def whole_ligament_continuity(text):
    """No disruption plus abnormal signal can be a negative tear example."""
    no_break=r'(?:no|without)\s+(?:fib(?:er|re)\s+)?(?:discontinuity|disruption)(?:\s+(?:or|and)\s+(?:tear|rupture))?'
    intact=has(r'\ball (?:the |its )?fib(?:er|re)s\b.{0,35}\b(?:intact|continuous|preserved)\b|\bboth bundles\b.{0,25}\bintact\b',text)
    cleaned=re.sub(no_break,'',text);cleaned=re.sub(NO_TEAR,'',cleaned)
    return (intact or has(no_break,text)) and not local_intact_ligament(text) and not has(TEAR+'|'+HIGH+'|'+UNCERTAIN,cleaned)

def scope_boundary(text,end,start):
    between=text[end:start]
    separators=list(re.finditer(r',|\b(?:and|but|however|with|whereas|without|no)\b',between))
    if separators:
        match=separators[-1]
        return end+(match.start() if match.group() in ('without','no') else match.end())
    # Never bisect words, which formerly truncated severity adjectives.
    spaces=list(re.finditer(r'\s+',between))
    if spaces:return end+min(spaces,key=lambda m:abs(m.start()-len(between)/2)).end()
    return start


@dataclass
class Fact:
    label:str
    presence:int|None=None
    criterion:int|None=None
    certainty:str='unknown'
    reason:str='not_stated'
    evidence:list[str]=field(default_factory=list)
    source:str='report_rule'
    attributes:dict=field(default_factory=dict)


def decision(label,text):
    """Decision within an anatomical scope, never across the whole report."""
    if label.endswith('OA'):text=cartilage_evidence_scope(text)
    if label=='Effusion':text=quantity_scope(label,text)
    if 'Meniscus' in label:
        text=re.sub(r'\bwithout\s+(?:an?\s+|associated\s+)?parameniscal\s+cysts?\b','',text)
    uncertain=has(UNCERTAIN,text)
    if label=='Effusion' and explicit_turkish_fluid_absence(text):return 0,0,'explicit_turkish_fluid_absence'
    injury=has(TEAR+'|'+ABNORMAL,text)
    if label in ('ACL','MCL') and not has(UNCERTAIN,text):
        if whole_ligament_continuity(text):
            return int(has(ABNORMAL,text)),0,'whole_ligament_continuity_without_tear'
        if local_intact_ligament(text) and not has(TEAR+'|'+ABNORMAL,text):
            return None,None,'local_intact_fibers_not_whole_ligament_negative'
        if label=='MCL' and injury and has(CHRONIC,text) and not has(ACUTE,text):return 1,0,'nonacute_MCL'
    # Negative statements about severity alone are not absence of a finding.
    not_finding=has(r'(?:\bno\b|\bwithout\b|\bsin\b|\bsem\b|未见|无)\s+(?:(?:evidence|signs) of\s+)?'+
                   r'(?:(?:knee|joint|articular|any)\s+)*(?:a\s+)?(?:discrete\s+)?(?:effusion|derrame|fractur\w*|synovitis|sinovitis|bone contusion|bone bruise|baker\w*|popliteal cyst)',text)
    # "Otherwise normal marrow" must not erase an explicit fracture, effusion,
    # or cartilage defect in the same clause.
    normal=(label in ('ACL','MCL','Medial Meniscus','Lateral Meniscus') and has(NORMAL,text) and not injury and not has(r'\bdegil(?:dir)?\b',text)
            and not has(r'(?:signal.*(?:reach|contact|extend))|displaced|truncated',text))
    if not_finding or normal:return 0,0,'explicit_negative'
    if uncertain:return None,None,'uncertain_statement'
    low=has(LOW,text);high=has(HIGH,text);acute=has(ACUTE,text);old=has(CHRONIC,text)
    no_tear=has(NO_TEAR,text)
    if label=='Fracture' and has(r'no acute fracture|without acute fracture',text):return None,0,'explicit_no_acute_fracture'
    if label.endswith('OA') and has(CARTILAGE,text):
        if has(r'(?:no|without)\s+(?:focal\s+)?high.grade\s+(?:cartilage|chondral)\s+(?:defect|loss|lesion)',text):
            return None,0,'explicit_no_high_grade_cartilage_defect'
        if (has(r'(?:no|without)\s+(?:focal\s+)?(?:cartilage|chondral)\s+(?:defect|loss|lesion|injury)|no focal chondrosis',text)
            or (has(NORMAL,text) and has(r'sin defectos condrales',text))):
            return 0,0,'explicit_no_cartilage_defect'
    # Negated severity is NOT a positive finding. A complicated scope is
    # unknown until reviewed, rather than relying on keyword voting.
    if has(NEG,text) and not no_tear and not has(r'without.*surfac|not.*reach.*surfac',text):
        return None,None,'unresolved_negative_scope'
    if label in ('ACL','MCL'):
        if no_tear and not has(r'sprain|injur|esguince|degener',text):return 0,0,'explicit_no_tear'
        if not injury:return None,None,'no_injury_evidence'
        # Mild surrounding edema/thickening can coexist with a high-grade tear.
        explicit_low_tear=low_ligament_grade(text)
        if explicit_low_tear and high:return 1,None,'conflicting_severity'
        if high:low=False
        if low_ligament_grade(text) or no_tear:return 1,0,'below_high_grade_threshold'
        if label=='MCL' and old:
            return (1,None,'mixed_MCL_temporal_scope') if acute else (1,0,'nonacute_MCL')
        greater_half=has(r'(?:>|over|more than|greater than|mas del?)\s*50\s*%',text)
        if has(TEAR,text) and (high or greater_half):
            if label=='ACL' or acute:return 1,1,'explicit_qualifying_tear'
            return 1,None,'acute_status_not_stated'
        return 1,None,'tear_grade_not_established'
    if 'Meniscus' in label:
        if has(r'degener|intramenisc|intrasubstance|intrassubst|退变',text) and (no_tear or has(r'without.*surfac|sin.*superfic|not.*reach.*surfac|未达.*表面',text)):
            return 1,0,'degeneration_without_surface_tear'
        if no_tear:
            finding=has(r'intrasubstance|intrasustancia|aumento de senal|alteracion de senal|degener|fibril|fray\w*|discoid',text)
            return int(finding),0,'meniscal_abnormality_without_tear' if finding else 'explicit_no_meniscal_tear'
        if has(TEAR,text):return 1,1,'explicit_meniscal_tear_diagnosis'
        if has(ABNORMAL+'|chondro|condro',text):return 1,None,'meniscal_abnormality_unspecified'
    if label.endswith('OA'):
        if not has(CARTILAGE,text):return None,None,'cartilage_not_described'
        lesions=cartilage_lesion_scopes(text)
        if len(lesions)>1:
            local=[decision(label,part) for part in lesions]
            if any(y==1 for _,y,_ in local):return 1,1,'one_explicit_cartilage_lesion_qualifies'
            return 1,None,'separate_cartilage_lesions_no_qualifying_conjunction'
        if has(NORMAL,text) and not has(r'thinn|loss|defect|fissur|adelgaz|perdida|eros|丢失|变薄',text):return 0,0,'cartilage_preserved'
        present=has(r'thinn|loss|defect|fissur|chondr|condr|adelgaz|perdida|eros|丢失|变薄|损伤',text)
        if not present:return None,None,'cartilage_target_unclear'
        if low and not high:return 1,0,'explicit_low_grade_cartilage_change'
        depth=high or has(r'(?:>|over|more than|greater than|mas del?)\s*50\s*%',text)
        from .clinical_factors import lesion_extent_mm
        extent=lesion_extent_mm(text);numeric_extent=extent is not None and extent>=10 and not mixed_cartilage_depth(text)
        broad=r'(?:extensive|diffuse|large area|extens\w*|difus\w*|throughout|广泛)'
        severe=r'(?:full.thickness|high.grade|espesor (?:subtotal y )?total|全层)'
        extensive=False
        for pattern in (broad+r'(.{0,45})'+severe,severe+r'(.{0,45})'+broad):
            for match in re.finditer(pattern,text):
                # "Mild diffuse thinning WITH SMALL FOCI of full-thickness
                # fissuring" does not establish a large high-grade lesion.
                if not has(r'\b(?:with|con|but|focal|foci|small|pequen\w*|areas|patchy)\b',match.group(1)):
                    extensive=True
        if mixed_cartilage_depth(text):extensive=False
        if depth and (numeric_extent or extensive):return 1,1,'high_grade_with_extent' if numeric_extent else 'high_grade_extensive_report'
        return 1,None,'depth_or_extent_missing'
    if label=='Effusion':
        low,high=quantity_levels(label,text)
        if low and high:return 1,None,'conflicting_effusion_amount'
        if low:return 1,0,'small_effusion_below_threshold'
        if high:return 1,1,'qualifying_effusion_amount'
        return 1,None,'effusion_amount_unspecified'
    if label=='Synovitis':
        if has(r'synovitis|sinovitis|sinovite|滑膜炎',text):return 1,1,'explicit_synovitis_diagnosis'
        if has(r'thicken|engros|增厚',text) and has(r'inflamm|inflam|炎',text):return 1,1,'inflammation_and_thickening'
        return 1,None,'synovial_description_incomplete'
    if label=="Baker's":
        text=quantity_scope(label,text);low=has(LOW,text)
        if low:return 1,0,'small_cyst_below_threshold'
        if has(r'moderate|large|moderad|中等|较大',text):return 1,1,'qualifying_cyst_size'
        return 1,None,'cyst_size_unspecified'
    if label=='Contusion':
        if has(r'bone (?:contusion|bruise)|contusion osea|骨挫伤',text):return 1,1,'explicit_bone_contusion_diagnosis'
        if has(r'marrow|edema|骨髓',text):return 1,None,'edema_without_traumatic_attribution'
    if label=='Fracture':
        if old and acute:return 1,None,'conflicting_fracture_age'
        if old:return 1,0,'old_or_healed_fracture'
        if acute:return 1,1,'explicit_acute_fracture'
        return 1,None,'fracture_age_unspecified'
    return None,None,'no_decisive_evidence'


def _extract_report_core(report):
    original=str(report or '');facts={c:[] for c in LABELS};region=None;mcl_scopes=[];scopes={c:[] for c in LABELS}
    cartilage_context={c:{} for c in LABELS if c.endswith('OA')}
    findings=re.search(r'\b(?:findings|hallazgos|achados)\s*:',original,re.I)
    if findings:original=original[findings.end():]  # indications/questions are not imaging findings
    carry=None;in_impression=False;impression_facts={c:[] for c in LABELS}
    for clause in report_clauses(original):
        if not clause.strip():continue
        text=norm(clause)
        if has(r'^(?:impression|impresion|imp|conclusion|interpretation|comment)\s*:',text):in_impression=True
        if has(r'^\s*(?:indication|history|clinical information|comparison|technique)\s*:',text):continue
        if len(text)<140 and ':' in text:
            if 'medial compartment' in text or 'compartimento medial' in text:region='Medial OA'
            elif 'lateral compartment' in text or 'compartimento lateral' in text:region='Lateral OA'
            elif has(PATTERNS['PF OA'],text):region='PF OA'
            elif has(r'ligament|menisc|fluid|intercondyl|osseous|extensor',text):region=None
        # Narrative compartment paragraphs often use a comma rather than a
        # colon. Keep that explicit region for the following cartilage clause.
        heading=re.match(r'^(?:in )?(?:the )?(medial|lateral) compartment\b',text)
        if heading:region=heading.group(1).title()+' OA'
        elif re.match(r'^(?:the )?patell?ofemoral (?:joint|compartment|cartilage)\b',text):region='PF OA'
        elif re.match(r'^(?:in )?(?:the )?(?:intercondylar|extensor|central supporting|others|ligaments|menisci)\b',text):region=None
        hits=[]
        for label,pattern in PATTERNS.items():
            for match in re.finditer(pattern,text):hits.append((match.start(),match.end(),label))
        if (region=='PF OA' and has(CARTILAGE,text)
                and not has(r'\b(?:tibial|tibia|plateau|femorotibial|tibiofemoral|medial compartment|lateral compartment)\b',text)):
            # Medial/lateral femoral or patellar surfaces within an explicit
            # PF cartilage paragraph do not name femorotibial compartments.
            hits=[h for h in hits if not h[2].endswith('OA')]+[(0,0,'PF OA')]
        explicit_oa_hits=any(c.endswith('OA') for _,_,c in hits)
        if hits and (text.strip().endswith(':') or text.strip() in ('acl','mcl')):
            carry=hits[0][2] if len({x[2] for x in hits})==1 else None
            continue
        if not hits and carry:
            hits=[(0,0,carry)]
        elif hits:carry=None
        if region and has(CARTILAGE,text) and not any(c.endswith('OA') for _,_,c in hits):hits.append((0,0,region))
        if region and has(CARTILAGE,text) and not explicit_oa_hits:
            # Retain the immediately active compartment heading. This is
            # explicit local paragraph context, not a search elsewhere in the
            # report. Literal quotations remain unchanged in output evidence.
            cartilage_context[region][clause.strip()]=region
        hits.sort()
        negative_end=shared_negative_end(text)
        # Repeated aliases of the same structure must not split its depth,
        # extent, or polarity into unrelated fragments.
        merged=[]
        for start,end,label in hits:
            # The same structure may be mentioned again in a new assertion.
            # Do not merge across a contrast or the end of a shared negative:
            # an earlier uncertain/absent finding cannot absorb a later one.
            separate_assertion=(bool(merged) and
                ((has(r'\b(?:but|however|whereas)\b',text[merged[-1][1]:start]) and has(UNCERTAIN,text[:start])) or
                 merged[-1][1]<=negative_end<=start))
            if merged and merged[-1][2]==label and not separate_assertion:
                merged[-1]=(merged[-1][0],max(end,merged[-1][1]),label)
            else:merged.append((start,end,label))
        hits=merged
        shared_normal=(bool(re.match(r'\s*(?:'+NORMAL+r')',text)) or coordinated_normal_structures(text)) and not has(TEAR+'|'+ABNORMAL+'|'+LOW+'|'+HIGH,text)
        negative_statement=text[:negative_end].rsplit(':',1)[-1].strip() if negative_end else text
        qualified_negative=has(r'^\s*(?:[>\-*]\s*)?(?:there\s+(?:is|are)\s+)?(?:no|without|sin|sem)\s+(?:significant|definite)\b',negative_statement)
        for j,(start,end,label) in enumerate(hits):
            # Adjacent OA/meniscus aliases can refer to the same phrase; retain
            # the full cartilage clause for its named compartment only.
            if label.endswith('OA') and not has(CARTILAGE,text):continue
            left=0 if j==0 else scope_boundary(text,hits[j-1][1],start)
            right=len(text) if j+1==len(hits) else scope_boundary(text,end,hits[j+1][0])
            scope=text if shared_normal or len(hits)==1 else text[left:right]
            if label in ('ACL','MCL'):
                # Non-target objects can also terminate a ligament assertion.
                # Do not borrow later bursal/cartilage grades in long findings.
                scope=re.split(r',\s*suprapatellar\s+(?:burs\w*|fat pad)\b',scope,maxsplit=1)[0]
            # Preserve the anatomical scope: a complete ligament tear must not
            # provide the cartilage-loss depth in another part of this clause.
            if start<negative_end:
                p,y,reason=(None,None,'qualified_negative_statement') if qualified_negative else (0,0,'shared_explicit_negative')
                if label=='Fracture' and has(r'\b(?:no|without)\s+acute fracture\b',text):p=None;y=0;reason='explicit_no_acute_fracture'
            else:p,y,reason=decision(label,scope)
            if label=='MCL':mcl_scopes.append(scope)
            # A shared negative clause must retain its polarity after anatomical
            # scoping (e.g. 'No fracture or bone bruise').
            attribute_scope=('no significant '+scope if qualified_negative and start<negative_end
                             else scope if p!=0 or (label=='Effusion' and explicit_turkish_fluid_absence(scope)) else 'no '+scope)
            scopes[label].append((attribute_scope,clause.strip()))
            if label=='Contusion' and has(r'without (?:a )?(?:discrete )?fracture line|no fracture line',text):
                scopes[label].append((text,clause.strip()))
            if label=='Contusion' and has(r'fracture line.{0,100}(?:surrounding|associated|adjacent|accompanying).{0,50}(?:marrow (?:edema|oedema))',text):
                scopes[label].append((text,clause.strip()))
            facts[label].append((p,y,reason,clause.strip()))
            if in_impression:impression_facts[label].append((p,y,reason,clause.strip()))
    def complete_negative_coverage(label,quotes):
        # A literal sentence may mention two different compartments. Coverage
        # follows the already resolved anatomical scope, while literal source
        # quotations remain attached to the returned evidence.
        chosen=[(scope,quote) for scope,quote in scopes[label] if quote in quotes]
        contextual={scope:label for scope,quote in chosen if cartilage_context[label].get(quote)==label}
        return complete_cartilage_negative_coverage(label,[scope for scope,_ in chosen],contextual)
    output={}
    for label,items in facts.items():
        ps={p for p,_,_,_ in items if p is not None};ys={y for _,y,_,_ in items if y is not None}
        conflicting=len(ys)>1 or any('conflicting' in r for _,_,r,_ in items)
        # Study labels are existential. A mild patellar lesion is compatible
        # with a qualifying trochlear lesion; they are not contradictory grades
        # of one lesion. Only resolve when locations are explicitly disjoint.
        separate_positive=False
        if label.endswith('OA') and ys=={0,1}:
            positives=[q for _,v,_,q in items if v==1];negatives=[q for _,v,_,q in items if v==0]
            psites=[cartilage_sites(label,norm(q)) for q in positives]
            nsites=[cartilage_sites(label,norm(q)) for q in negatives]
            if all(psites+nsites) and all(a.isdisjoint(b) for a in psites for b in nsites):
                conflicting=False;separate_positive=True;ys={1}
                scopes[label]=[(s,q) for s,q in scopes[label] if q in positives]
        # A positive finding with an unestablished criterion is not resolved
        # by an unrelated normal clause about the same structure.
        uncertain=any(y is None and (p==1 or 'uncertain' in r) for p,y,r,_ in items)
        # A definite final low-grade diagnosis can resolve an earlier R/O of
        # that same low-grade injury; it cannot override definite opposing facts.
        resolved_summary=(ys=={0} and any(y==0 and r=='below_high_grade_threshold' for _,y,r,_ in impression_facts[label])
                          and all(y is not None or r=='uncertain_statement' for _,y,r,_ in items))
        if resolved_summary:uncertain=False
        y=None if conflicting or (uncertain and ys=={0}) else next(iter(ys),None)
        presence=1 if 1 in ps else (0 if ps=={0} else None)
        reason='conflicting_evidence' if conflicting else (';'.join(dict.fromkeys(r for _,_,r,_ in items)) or 'not_stated')
        if separate_positive:reason+=';distinct_cartilage_sites_qualifying_lesion'
        if resolved_summary:reason+=';final_low_grade_diagnosis'
        if label=='Effusion' and ps=={0,1} and ys=={0}:
            presence=None;reason+=';presence_conflict_criterion_below_threshold'
        if label.endswith('OA') and y==0 and items:
            quotes=[q for _,v,_,q in items if v==0]
            if not complete_negative_coverage(label,quotes):
                y=None;presence=1 if 1 in ps else None;reason+=';partial_'+label.replace(' OA','').replace(' ','_')+'_cartilage_coverage'
        output[label]=Fact(label,presence,y,'conflict' if conflicting else 'explicit' if y is not None else 'unknown',
                           reason,list(dict.fromkeys(q for *_,q in items)))
        if y==1 and 'high_grade_extensive_report' in reason:
            output[label].certainty='supported_inference';output[label].source='report_extent_inference'
    # The host's MCL criteria explicitly name disrupted fibers with adjacent
    # edema. Combine separate MCL sentences, not unrelated bone/joint edema.
    mcl=output['MCL'];scoped=' '.join(mcl_scopes)
    severe=any(has(HIGH,s) and has(TEAR+'|sprain|injur',s) and not has(NEG+'|'+UNCERTAIN,s) for s in mcl_scopes)
    acute_reaction=any(has(r'edema|oedema|focal fluid',s) and not has(NEG+'|'+UNCERTAIN,s) for s in mcl_scopes)
    if (mcl.criterion is None and mcl.certainty!='conflict' and severe and acute_reaction and not has(CHRONIC,scoped)):
        mcl.criterion=1;mcl.presence=1;mcl.certainty='supported_inference'
        mcl.reason='high_grade_MCL_with_local_acute_reaction';mcl.source='report_conjunction_weak'
    from .clinical_factors import merge_scopes,REQUIRED
    for label,fact in output.items():
        fact.attributes=merge_scopes(label,scopes[label])
        if label.endswith('OA'):
            depth=fact.attributes.get('depth_gt_half',{});extent=fact.attributes.get('extent_ge_10mm',{})
            if depth.get('target')==extent.get('target')==1 and not (set(depth.get('evidence',[]))&set(extent.get('evidence',[]))):
                fact.attributes['extent_ge_10mm']={**extent,'target':None,'weight':0.,'unbound_target':1,
                    'state':'extent_not_bound_to_high_grade_region'}
                fact.reason+=';unbound_depth_and_extent'
            # These auxiliary predicates describe the whole Study target.
            # A negative observation at one surface cannot supervise absence
            # across both surfaces or re-enable the masked main target below.
            for key,value in list(fact.attributes.items()):
                if value.get('target')==0 and not complete_negative_coverage(label,value.get('evidence',[])):
                    fact.attributes[key]={**value,'local_target':0,'target':None,'weight':0.,
                                          'state':'partial_structure_negative_not_study_negative'}
        if fact.certainty!='conflict':
            predicates=[fact.attributes.get(k,{}) for k in REQUIRED[label]]
            negative=any(x.get('target')==0 and x.get('weight',0)>0 for x in predicates)
            positive=all(x.get('target')==1 and x.get('weight',0)>0 for x in predicates)
            if negative and fact.criterion==1:
                # Separate lesions can coexist. Never turn a Study containing
                # a positive finding into a hard negative due to another scope.
                fact.criterion=None;fact.certainty='conflict'
                fact.reason+=';component_conflicts_with_positive_statement'
            elif negative:
                fact.criterion=0;fact.certainty='explicit';fact.source='explicit_criterion_components'
                fact.reason+=';required_predicate_failed'
            elif positive and fact.presence==1:
                # Same-sentence support is a conservative text binding, not an
                # image lesion identity. Independent positive marginals cannot
                # create a joint lesion that no assertion described.
                joint_quotes=set.intersection(*(set(x.get('evidence',[])) for x in predicates))
                if len(predicates)>1 and not joint_quotes:
                    fact.reason+=';positive_predicates_not_bound_in_report'
                else:
                    fact.criterion=1;fact.certainty='explicit';fact.source='explicit_criterion_components'
                    fact.reason+=';all_required_predicates_explicit'
    return output


def extract(report):
    from .report_language_recovery import extract_recovered
    from .report_scope_recovery import recover_scoped_facts
    from .report_consistency import reconcile_report_facts
    return reconcile_report_facts(report,recover_scoped_facts(report,extract_recovered(report,_extract_report_core)))


def compile_labels(official_path,primary_path,partition_path,output,reviewed=None,expert_anchors=True,development_ids=None):
    official=pd.read_csv(official_path,dtype={UID:str}).set_index(UID)
    # Legacy positional slot retained for callers, but the table is never read.
    # Criterion, presence, and factor supervision must come from our own text
    # evidence or the separately identified official image reference.
    from .label_source_policy import POLICY
    anchor=pd.read_csv(partition_path,dtype={UID:str,'group_id':str})
    if development_ids is None:
        dev_path=Path(__file__).parent/'data/construction_development_ids.json'
        development_ids=json.loads(dev_path.read_text(encoding='utf-8'))['study_ids'] if dev_path.exists() else []
    from .reviewed_factors import load_reviewed_factors,apply_reviewed,DEFAULT_PATH as factor_registry
    factor_entries=load_reviewed_factors()
    factor_development=json.loads(factor_registry.read_text(encoding='utf-8'))['development_study_ids']
    from .report_review_recovery import load_review_registry,apply_review_recovery
    recovered_entries,recovered_development=load_review_registry()
    from .semantic_review_overlay import load_semantic_registry,apply_semantic_review
    semantic_entries,semantic_metadata=load_semantic_registry()
    from .semantic_supplement import load as load_supplement
    supplement_entries,supplement_metadata=load_supplement()
    semantic_manual_path=Path(__file__).parent/'data/semantic_manual_resolutions.json'
    semantic_manual=json.loads(semantic_manual_path.read_text(encoding='utf-8')) if semantic_manual_path.exists() else {}
    assertion_registry=Path(__file__).parent/'data/report_assertion_recovery_review.json'
    assertion_review=json.loads(assertion_registry.read_text(encoding='utf-8')) if assertion_registry.exists() else {}
    assertion_development=assertion_review.get('development_study_ids',assertion_review.get('reviewed_study_ids',[]))
    development_ids=sorted((set(development_ids)|set(factor_development)|set(recovered_development)|set(assertion_development)
                           |set(semantic_manual.get('development_study_ids',[]) if semantic_metadata.get('available') else []))&set(official.index))
    from .unknown_review_isolation import load_reserve,reserve_patients
    reserve_ids,reserve_sha256=load_reserve()
    known_groups=anchor.set_index(UID).group_id.to_dict()
    development_groups={known_groups.get(u,u) for u in development_ids}
    if expert_anchors:
        from .labels import group_folds
        gold=official[official[LABELS].notna().any(axis=1)].copy()
        group_map=anchor.set_index(UID).group_id.to_dict()
        gold=gold[[group_map.get(u,u) not in development_groups for u in gold.index]]
        groups=np.asarray([group_map.get(u,u) for u in gold.index],str)
        targets=gold[LABELS].to_numpy(float);observed=np.isfinite(targets).astype(float)
        splits=group_folds(np.nan_to_num(targets),observed,groups,5,42)
        for k in range(5):
            if any(np.unique(targets[(splits==k)&(observed[:,j]>0),j]).size!=2 for j in range(12)):
                raise ValueError('expert patient grouping cannot define all 12 fold AUCs; revise fold count without splitting patients')
        mapping=dict(zip(groups,splits))
        chosen=anchor.group_id.isin(mapping)
        anchor.loc[chosen,'fold']=anchor.loc[chosen,'group_id'].map(mapping)
    anchor.loc[anchor.group_id.isin(development_groups),'fold']=5
    overrides={}
    if reviewed:
        for line in Path(reviewed).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);uid=row[UID];label=row['label']
            if label not in LABELS or uid not in official.index:raise ValueError('reviewed evidence identity invalid')
            if row.get('quote','') not in str(official.loc[uid,'Report']) or not row.get('quote'):raise ValueError('reviewed quote not found verbatim')
            if row.get('criterion') not in (None,0,1) or row.get('presence') not in (None,0,1):raise ValueError('invalid reviewed facts')
            overrides[(uid,label)]=row
    rows=[];evidence=[]
    for _,partition in anchor.iterrows():
        uid=partition[UID];report=str(official.loc[uid].get('Report',''))
        facts=extract(report);row={k:partition[k] for k in (UID,'group_id','is_gold','fold')}
        row['reference_kind']='criterion_weak_reference'
        row['supervision_role']='expert_anchor' if expert_anchors and partition.is_gold else ('reserved_expert' if partition.is_gold else 'report_weak')
        row['development_only']=bool(partition.group_id in development_groups)
        for label in LABELS:
            fact=facts[label];override=overrides.get((uid,label))
            if override:fact=Fact(label,override.get('presence'),override.get('criterion'),'reviewed','reviewed_evidence',[override['quote']],override.get('reviewer','reviewed'))
            fact=apply_reviewed(fact,report,factor_entries.get((uid,label)))
            fact=apply_review_recovery(fact,report,recovered_entries.get((uid,label)))
            previous_source=fact.source;previous_certainty=fact.certainty
            fact,semantic_flags=apply_semantic_review(fact,report,semantic_entries.get((uid,label)))
            original_semantic_flags=dict(semantic_flags)
            supplement_record=supplement_entries.get((uid,label));supplement_flags=None
            if supplement_record is not None:
                fact,supplement_flags=apply_semantic_review(fact,report,supplement_record)
                for field in ('presence_weight','criterion_weight'):
                    if supplement_flags.get(field) is not None:semantic_flags[field]=supplement_flags[field]
            from .criterion_trace import report_snapshot,evidence_trace
            report_only=report_snapshot(fact,report)
            public_fallback=None
            presence=fact.presence;pw=.8 if presence is not None else 0.
            if presence is not None and previous_source=='reviewed_report_recovery':pw=.7
            if semantic_flags.get('presence_weight') is not None:pw=semantic_flags['presence_weight']
            target=fact.criterion;weight=.8 if target is not None else 0.
            if previous_certainty=='supported_inference':weight=.5
            if previous_certainty=='reviewed_report_weak' and target is not None:weight=.7
            if 'extensive_report' in fact.reason:weight=.5
            if semantic_flags.get('criterion_weight') is not None:weight=semantic_flags['criterion_weight']
            gold=official.loc[uid,label]
            from .clinical_factors import apply_official,REQUIRED
            attributes=apply_official(label,fact.attributes,int(gold) if pd.notna(gold) else None)
            for name in REQUIRED[label]:
                value=attributes.get(name,{})
                row[label+'__factor_'+name]=value.get('target',np.nan)
                row[label+'__factor_'+name+'_weight']=value.get('weight',0.)
            fact.attributes=attributes
            if pd.notna(gold):
                target=int(gold);weight=1.;fact.source='official_reference'
                if target==1:presence=1;pw=1.  # confirmed disease entails finding presence
            if partition.is_gold and pd.isna(gold):target=None;weight=0.
            row[label]=np.nan if target is None else target;row[label+'__weight']=weight
            row[label+'__presence']=np.nan if presence is None else presence;row[label+'__presence_weight']=pw
            row[label+'__criterion_state']=fact.certainty if pd.isna(gold) else 'official'
            row[label+'__semantic_review_state']=('supplemental_'+supplement_flags['semantic_status'] if supplement_flags else semantic_flags.get('semantic_status') or 'not_in_full_review_scope')
            # A hard negative is a supported finding below the criterion, not
            # membership in an incomplete list of English reason strings.
            row[label+'__hard_negative']=int(target==0 and presence==1 and weight>=.5 and pw>=.5)
            semantic_record=semantic_entries.get((uid,label))
            evidence.append({UID:uid,**asdict(fact),'compiled_target':target,'compiled_weight':weight,
                **evidence_trace(report_only,int(gold) if pd.notna(gold) else None,public_fallback=public_fallback),
                'semantic_supplement':None if supplement_record is None else {
                    'record':supplement_record,'flags':supplement_flags,'is_image_truth':False},
                'semantic_review':None if semantic_record is None else {
                    'report_id':semantic_record['report_id'],'report_sha256':semantic_record['report_sha256'],
                    'status':semantic_flags['semantic_status'],'explanation':semantic_record['explanation'],
                    'missing_criteria':sorted(set(semantic_record['missing_criteria'])|{
                        name.split(':',1)[1] for name in semantic_flags.get('scope_masked_components',[]) if name.startswith('attribute:')}),
                    'secondary_status':semantic_flags.get('secondary_status'),
                    'secondary_decision':semantic_record.get('secondary_decision'),
                    'flags':original_semantic_flags,'is_image_truth':False}})
        rows.append(row)
    result=reserve_patients(pd.DataFrame(rows),reserve_ids);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    recovery_hashes={name+'_sha256':file_hash(Path(__file__).parent/path) for name,path in {
        'verified_report_language':'verified_report_language.py',
        'assertion_recovery':'report_assertion_recovery.py',
        'assertion_recovery_review':'data/report_assertion_recovery_review.json',
        'report_review_recovery':'report_review_recovery.py',
        'report_review_registry':'data/report_review_recovery.json',
        'review_isolation':'unknown_review_isolation.py'}.items()}
    recovery_hashes['image_review_reserve_sha256']=reserve_sha256
    recovery_hashes['semantic_review_registry_sha256']=semantic_metadata.get('registry_sha256')
    recovery_hashes['semantic_review_overlay_sha256']=file_hash(Path(__file__).with_name('semantic_review_overlay.py'))
    recovery_hashes['semantic_supplement_registry_sha256']=supplement_metadata.get('registry_sha256')
    recovery_hashes['semantic_supplement_consumer_sha256']=file_hash(Path(__file__).with_name('semantic_supplement.py'))
    recovery_hashes['criterion_trace_sha256']=file_hash(Path(__file__).with_name('criterion_trace.py'))
    recovery_hashes['report_consistency_sha256']=file_hash(Path(__file__).with_name('report_consistency.py'))
    recovery_hashes['report_scope_recovery_sha256']=file_hash(Path(__file__).with_name('report_scope_recovery.py'))
    if (out/'evaluation_reference.csv').exists() and (out/'contract.json').exists():
        old=json.loads((out/'contract.json').read_text())
        signature={'compiler_sha256':file_hash(__file__),'official_sha256':file_hash(official_path),
                   'primary_sha256':None,'partition_sha256':file_hash(partition_path),'label_source_policy':POLICY,
                   'reviewed_sha256':file_hash(reviewed) if reviewed else None,'expert_anchors':expert_anchors}
        signature['factor_compiler_sha256']=file_hash(Path(__file__).with_name('clinical_factors.py'))
        signature['language_recovery_sha256']=file_hash(Path(__file__).with_name('report_language_recovery.py'))
        signature['reviewed_factor_registry_sha256']=file_hash(factor_registry)
        signature['reviewed_factor_consumer_sha256']=file_hash(Path(__file__).with_name('reviewed_factors.py'))
        signature['supervision_development_ids']=development_ids
        signature.update(recovery_hashes)
        if any(old.get(k)!=v for k,v in signature.items()):raise ValueError('fixed reference changed; use a new versioned output directory')
    result.to_csv(out/'labels.csv',index=False)
    with (out/'evidence.jsonl').open('w',encoding='utf-8') as f:
        for row in evidence:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    reference=result.copy()
    for c in LABELS:reference[c+'__weight']=(reference[c+'__weight']>0).astype(float)
    reference.to_csv(out/'evaluation_reference.csv',index=False)
    expert_reference=reference.copy()
    expert_reference.loc[~expert_reference.is_gold.astype(bool),[c+'__weight' for c in LABELS]]=0.
    expert_reference.to_csv(out/'expert_evaluation_reference.csv',index=False)
    counts=[];train=result
    for c in LABELS:
        known=train[c+'__weight']>0
        counts.append({'label':c,'positive':int((known&train[c].eq(1)).sum()),'negative':int((known&train[c].eq(0)).sum()),
                       'unknown':int((~known).sum()),'presence_observed':int(train[c+'__presence_weight'].gt(0).sum())})
    pd.DataFrame(counts).to_csv(out/'coverage.csv',index=False)
    training_counts=[];training_rows=result[result.fold>=0]
    for c in LABELS:
        known=training_rows[c+'__weight']>0
        training_counts.append({'label':c,'positive':int((known&training_rows[c].eq(1)).sum()),
            'negative':int((known&training_rows[c].eq(0)).sum()),'unknown':int((~known).sum())})
    pd.DataFrame(training_counts).to_csv(out/'training_coverage.csv',index=False)
    from .clinical_factors import FACTOR_COLUMNS
    factor_counts=[{'factor':c,'positive':int(((result[c]==1)&(result[c+'_weight']>0)).sum()),
                   'negative':int(((result[c]==0)&(result[c+'_weight']>0)).sum()),
                   'unknown':int((result[c+'_weight']==0).sum())} for c in FACTOR_COLUMNS]
    pd.DataFrame(factor_counts).to_csv(out/'factor_coverage.csv',index=False)
    save_json(out/'contract.json',{'version':VERSION,'compiler_sha256':file_hash(__file__),**recovery_hashes,
        'reviewed_factor_registry_sha256':file_hash(factor_registry),
        'reviewed_factor_consumer_sha256':file_hash(Path(__file__).with_name('reviewed_factors.py')),
        'official_sha256':file_hash(official_path),'primary_sha256':None,'partition_sha256':file_hash(partition_path),
        'label_source_policy':POLICY,'labels_sha256':file_hash(out/'labels.csv'),
        'community_supervision_cells':0,'public_fallback_enabled':False,
        'reviewed_sha256':file_hash(reviewed) if reviewed else None,'scope':'evidence-derived weak targets; unknown is not negative',
        'expert_anchors':expert_anchors,'split_rule':'expert patients balanced across five folds; training never includes its own validation patients',
        'factor_compiler_sha256':file_hash(Path(__file__).with_name('clinical_factors.py')),
        'language_recovery_sha256':file_hash(Path(__file__).with_name('report_language_recovery.py')),
        'supervision_development_ids':development_ids,'development_rule':'whole development patient group is training-only fold 5; never external-fold validation or teacher target',
        'image_review_reserved_ids':result.loc[result.review_holdout,UID].tolist(),
        'image_review_reserve_rule':'complete patient groups excluded from fitting, model selection and teacher targets; pending image adjudication, not yet reference truth',
        'full_report_semantic_review':semantic_metadata,
        'coverage_scope':'all released studies; separate training_coverage.csv excludes negative folds',
        'supervision':'independent finding-presence and competition-criterion targets','counts':counts})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--official',default='F:/rsnaknee/raw/train.csv')
    p.add_argument('--primary',default=None,help='deprecated compatibility slot; community labels are not read')
    p.add_argument('--partitions',required=True);p.add_argument('--output',default='F:/rsnaknee/labels_criterion');p.add_argument('--reviewed')
    a=p.parse_args();compile_labels(a.official,a.primary,a.partitions,a.output,a.reviewed)
