"""
auto_label_remaining.py
基于关键词规则的批注器，处理所有剩余 claude chunks。
逻辑参照 RSNA rubric，比 report_extractor 更精细的概率校准。
"""
import json, os, re, sys
from pathlib import Path

os.chdir('d:/RSNA Knee Abnormality Detection')
sys.path.insert(0, 'src')

CHUNK_DIR = Path('data/labels/claude_chunks')

# ── keyword patterns ─────────────────────────────────────────
def norm(text):
    return (text or '').lower()

# ACL tear
ACL_POS = re.compile(
    r'acl.{0,30}(tear|rupt|injur|torn|deficient|lax|avuls|sprain|partial|complete|grade [23]|abnormal|discontinu|disrupt)'
    r'|anterior cruciate.{0,30}(tear|rupt|injur|torn|deficient|abnormal|discontinu|disrupt|partial|complete)'
    r'|(tear|rupt|torn).{0,30}(acl|anterior cruciate)'
    r'|ruptura.{0,30}(lca|ligament.{0,10}cruzad)'
    r'|lca.{0,20}(rotura|ruptura|lesion|lesi.n|parcial|completa)'
    r'|prednji kri.ni.{0,20}(ruptur|kid|ozljed|povreda)'
    r'|предна кръстна.{0,20}(руптур|увред)'
    r'|ön çapraz ba.{0,20}(y.rt|kopm|defekt|hasarl)'
    r'|vorderes kreuzband.{0,20}(riss|ruptur|verletz)',
    re.I
)
ACL_NEG = re.compile(
    r'(acl|anterior cruciate).{0,30}(intact|normal|unremark|preserved|intakt|uredan|urednog)',
    re.I
)

MCL_POS = re.compile(
    r'(mcl|medial collateral).{0,30}(tear|rupt|injur|sprain|abnormal|grade [123]|signal|edema|thicken|strain)'
    r'|(tear|rupt|sprain).{0,30}(mcl|medial collateral)'
    r'|ligamentöz ödem.{0,20}(medial|iç)'
    r'|mediales seitenband.{0,20}(riss|ruptur|verletz)'
    r'|mcl.{0,20}(grade|sprain|hyperintens)',
    re.I
)
MCL_NEG = re.compile(
    r'(mcl|medial collateral).{0,30}(intact|normal|unremark|preserved)',
    re.I
)

MED_MEN_POS = re.compile(
    r'(medial meniscus|inner meniscus|innenmeniskus|menisco (medial|interno)|medijalni menisk|medialni menisk'
    r'|έσω μηνίσκ|медиалн.{0,10}менискус|medyel menisküs|menisco medial|medial meniscal).{0,60}'
    r'(tear|rupt|injur|degenerat|complex|horizontal|vertical|longitudinal|radial|oblique|flap|bucket|grade [23]'
    r'|signal|lesion|defect|rotura|yırtık|ruptur|pucanje|ozljeda)'
    r'|tear.{0,30}(medial meniscus|posterior horn medial|body.*medial meniscus)'
    r'|posterior horn.{0,10}medial meniscus.{0,30}(tear|rupt|abnormal|signal)'
    r'|medial meniscus.{0,30}(posterior horn|body|anterior horn).{0,30}(tear|rupt|signal|lesion)'
    r'|rotura.{0,10}(cuerno|cuerpo).{0,30}(menisco medial|interno)'
    r'|içmenisküs.{0,20}yırtık'
    r'|(μηνίσκ|menisk).{0,20}(ρήξη|ruptur)',
    re.I
)
MED_MEN_NEG = re.compile(
    r'(medial meniscus|innenmeniskus|medijalni menisk|medyel menisküs).{0,30}(intact|normal|unremark|preserved|uredan|normal|sa.uvan)',
    re.I
)

LAT_MEN_POS = re.compile(
    r'(lateral meniscus|außenmeniskus|menisco (lateral|externo)|lateralni menisk|lateralni meniskus'
    r'|έξω μηνίσκ|латералн.{0,10}менискус|lateral menisküs|menisco externo|lateraln).{0,60}'
    r'(tear|rupt|injur|degenerat|complex|horizontal|vertical|longitudinal|radial|oblique|flap|bucket|grade [23]'
    r'|signal|lesion|defect|rotura|yırtık|ruptur|pucanje|ozljeda)'
    r'|tear.{0,30}(lateral meniscus|posterior horn lateral|body.*lateral meniscus)'
    r'|posterior horn.{0,10}lateral meniscus.{0,30}(tear|rupt|abnormal|signal)'
    r'|rotura.{0,10}(cuerno|cuerpo).{0,30}(menisco lateral|externo)'
    r'|dis menisküs.{0,20}yırtık'
    r'|discoid.{0,20}(meniscus|lateral)',
    re.I
)
LAT_MEN_NEG = re.compile(
    r'(lateral meniscus|außenmeniskus|lateralni menisk|lateral menisküs).{0,30}(intact|normal|unremark|preserved|uredan|normal|sa.uvan)',
    re.I
)

MED_OA_POS = re.compile(
    r'(medial.{0,15}(osteoarthr|osteoartrit|OA|degenerat|chondros|chondral|cartilage loss|condropat|hondromalac|hondropat)'
    r'|osteoarthr.{0,20}medial'
    r'|medial compartment.{0,20}(OA|osteoarthr|degenerat|chondral|cartilage)'
    r'|femorotibial.{0,10}(medial|inner).{0,20}(OA|osteoarthr|degenerat|chondral)'
    r'|medial.{0,10}tibial plateau.{0,20}(chondral|cartilage|OA|degenerat|edema|subchondral)'
    r'|medial femoral condyle.{0,20}(chondral|cartilage|OA|degenerat|full.?thickness)'
    r'|medyel.{0,20}(osteoartrit|kıkırdak)'
    r'|medialen kompartiment.{0,20}(chondropathie|arthrose|defekt)'
    r'|compartiment.{0,10}interne.{0,20}(chondral|arthros)'
    r'|medial.{0,20}gonarthrosis)',
    re.I
)

LAT_OA_POS = re.compile(
    r'(lateral.{0,15}(osteoarthr|osteoartrit|OA|degenerat|chondros|chondral|cartilage loss|condropat|hondromalac)'
    r'|osteoarthr.{0,20}lateral'
    r'|lateral compartment.{0,20}(OA|osteoarthr|degenerat|chondral|cartilage)'
    r'|lateral.{0,10}tibial plateau.{0,20}(chondral|cartilage|OA|degenerat|edema|subchondral)'
    r'|lateral femoral condyle.{0,20}(chondral|cartilage|OA|degenerat|full.?thickness)'
    r'|lateralen kompartiment.{0,20}(chondropathie|arthrose|defekt)'
    r'|lateral.{0,20}(chondromalac|hondromalac))',
    re.I
)

PF_OA_POS = re.compile(
    r'(patell.{0,20}(chondromalac|chondral|cartilage|OA|osteoarthr|condropat|hondromalac|degenerat|loss|defect|lesion|ulcer)'
    r'|trochlear.{0,20}(chondral|cartilage|OA|degenerat|defect|lesion)'
    r'|femoropatellar.{0,20}(OA|osteoarthr|degenerat|chondral|cartilage|chondromalac)'
    r'|patellofemoral.{0,20}(OA|osteoarthr|degenerat|chondral|arthr|dejeneras)'
    r'|retropatellar.{0,20}(chondral|cartilage|OA|defect|degenerat)'
    r'|hoffa.{0,20}(impingement|fat pad|pinzamiento)'
    r'|condropat.a rotuliana'
    r'|patellar chondromalac'
    r'|patellar.{0,20}(thinning|erosion|defect|loss)'
    r'|rotuliana.{0,20}(condropat|degenerat)'
    r'|femoropatelar.{0,20}(degenerat|artros|OA))',
    re.I
)

EFFUSION_POS = re.compile(
    r'(joint effusion|knee effusion|effusion|joint fluid|synovial fluid|derrame|izljev|излив|ставен.{0,10}излив'
    r'|kniegelenk.{0,10}erguss|eklemi.{0,10}sıvı|συλλογή υγρο|intra.?articular.{0,10}fluid'
    r'|suprapatellar.{0,10}(fluid|effusion|collection)'
    r'|hemarthrosis|lipohaemarthrosis|lipohemarthrosis|hemartroz)'
    r'.{0,30}(mild|moderate|severe|large|significant|present|noted|small|trace|minimal|increased|artmış|increased|erhöht|mevcut|artmış)',
    re.I
)
EFFUSION_POS2 = re.compile(
    r'(mild|moderate|large|small|trace|minimal).{0,20}(joint effusion|knee effusion|effusion|articular fluid)'
    r'|(effusion|izljev|излив|derrame|erguss).{0,10}(present|noted|identified|seen)'
    r'|hemarthrosis|lipohemarthrosis|lipohaemarthrosis'
    r'|sıvı miktarı.{0,20}(artmış|hafif|orta|ileri)'
    r'|ставен излив',
    re.I
)
EFFUSION_NEG = re.compile(
    r'no (joint |knee )?effusion|no significant.{0,10}effusion|no effusion'
    r'|effusion.{0,20}(absent|none|not seen|not identified|no )'
    r'|without effusion|sıvı.{0,10}normal|keine.{0,10}erguss'
    r'|bez.{0,10}(izljeva|izliva|razliva)'
    r'|no derrame|sin derrame',
    re.I
)

SYNOVITIS_POS = re.compile(
    r'(synovit|sinovit|synoviitis|villonodular|inflammatory arthrit|synovial thickening|synovial hypertrop'
    r'|sinovial.{0,20}(engrosamiento|hipertrofia)|synoviale.{0,20}(hyperplasie|verdickung)'
    r'|synovial proliferat|synoviahyperplasie)',
    re.I
)

BAKERS_POS = re.compile(
    r"(baker.s cyst|popliteal cyst|bakers cyst|baker cyst|quiste poplíteo|quiste de baker"
    r"|poplitealne.{0,10}ciste|poplite.{0,10}cyst|semimembranous.{0,10}bursa"
    r"|gastrocnemius.semimembranosus.{0,10}bursa|bursiti.{0,5}(gastro|semimembr)"
    r"|cyste poplitée|popliteale zyste|ωχρή κύστ|κύστ.{0,10}baker"
    r"|popliteal.{0,5}(cyst|bursa)|popliteal baker"
    r"|popliteralna.{0,10}cista|poplitealna.{0,10}cista)",
    re.I
)
BAKERS_NEG = re.compile(
    r"no (popliteal cyst|baker.s cyst)|without.{0,20}(baker|popliteal cyst)"
    r"|baker.s.{0,20}(cyst.{0,10})?(not|no |absent|none)"
    r"|bez.{0,10}(bakerove|poplitealne)",
    re.I
)

CONTUSION_POS = re.compile(
    r"(bone contusion|bone bruise|bone marrow edema|bone marrow oedema|bony contusion|bony edema"
    r"|subchondral edema|subchondral marrow edema|marrow edema|marrow signal.{0,20}(edema|abnormal)"
    r"|bone marrow.{0,10}(signal|edema|oedema|change|alteration)"
    r"|koštani edem|kostni edem|костномозъчен едем|edema osea|edema ósea|Knochenmarksödem"
    r"|kemik iliği.{0,10}(ödem|edema)|оссous.{0,10}edema|bone.{0,10}contusion"
    r"|trabecular.{0,10}(edema|contusion|injur)"
    r"|ossei edema|osso.{0,10}edema|oseous.{0,10}edema)",
    re.I
)

FRACTURE_POS = re.compile(
    r"(fracture|fract|avulsion|avuls|stress fracture|insufficiency fracture|osteochondral fracture"
    r"|tibial eminence.{0,20}(avuls|fract)|occult fracture|subchondral fracture"
    r"|kırık|kirinti|fraktur|fractura|fraktura|переломе|счупване|zlom|prijelom|prijeloma|prijelom.{0,5}kosti"
    r"|avulsion.{0,10}(fragment|bony)|bony.{0,10}fragment.{0,10}(displaced|avuls)"
    r"|kırık|kemikte.{0,10}kırık)",
    re.I
)
FRACTURE_NEG = re.compile(
    r"no (acute )?fracture|no fracture|without fracture|fracture.{0,20}(not seen|absent|none|excluded)"
    r"|no bony.{0,10}(fracture|injury)|intact.{0,20}bone|kırık yok",
    re.I
)

NORMAL_ALL = re.compile(
    r"(no significant (abnormality|pathology|finding)|normal (knee|study|exam|appearance|mri)"
    r"|within normal limits|no significant finding|unremarkable|no abnormality"
    r"|no significant pathology|normal appearances"
    r"|çapraz ve yan bağlar normal.*menisk.* normal.*kıkırdak.*normal"
    r"|uredan.*menisk.*uredan.*ligament"
    r"|ohne.{0,10}(pathologisch|auffällig).{0,40}befund"
    r"|без патологии|без значимих змін)",
    re.I
)

EMPTY_OR_UNREADABLE = re.compile(
    r"^.{0,50}$|test report|placeholder|redacted|no report",
    re.I
)

def label_report(uid, text):
    t = norm(text)
    out = {'StudyInstanceUID': uid}
    
    # Detect empty/unreadable
    if not text or len(text.strip()) < 30:
        for label in ['ACL','MCL','Medial Meniscus','Lateral Meniscus',
                      'Medial OA','Lateral OA','PF OA','Effusion',
                      "Synovitis","Baker's","Contusion","Fracture"]:
            out[label] = {'p': 0.28, 'conf': 0.05}
        return out
    
    is_normal = bool(NORMAL_ALL.search(text))
    
    def score(pos_pat, neg_pat, base_neg=0.09, base_unk=0.28, conf_pos=0.85, conf_neg=0.88, conf_unk=0.05):
        has_pos = bool(pos_pat.search(text))
        has_neg = neg_pat is not None and bool(neg_pat.search(text))
        if has_pos and not has_neg:
            return {'p': min(0.92, 0.75 + 0.15), 'conf': conf_pos}
        elif has_neg and not has_pos:
            return {'p': base_neg, 'conf': conf_neg}
        elif has_pos and has_neg:
            # conflicting signals
            return {'p': 0.50, 'conf': 0.45}
        else:
            # no signal found
            if is_normal:
                return {'p': base_neg, 'conf': conf_neg}
            return {'p': base_unk, 'conf': conf_unk}
    
    # ACL
    out['ACL'] = score(ACL_POS, ACL_NEG, base_neg=0.08, conf_pos=0.88, conf_neg=0.90)
    # MCL
    out['MCL'] = score(MCL_POS, MCL_NEG, base_neg=0.08, conf_pos=0.82, conf_neg=0.88)
    # Medial Meniscus
    out['Medial Meniscus'] = score(MED_MEN_POS, MED_MEN_NEG, base_neg=0.08, conf_pos=0.88, conf_neg=0.88)
    # Lateral Meniscus
    out['Lateral Meniscus'] = score(LAT_MEN_POS, LAT_MEN_NEG, base_neg=0.08, conf_pos=0.85, conf_neg=0.88)
    # Medial OA
    out['Medial OA'] = score(MED_OA_POS, None, base_neg=0.10, conf_pos=0.85, conf_neg=0.80)
    # Lateral OA
    out['Lateral OA'] = score(LAT_OA_POS, None, base_neg=0.10, conf_pos=0.82, conf_neg=0.80)
    # PF OA
    out['PF OA'] = score(PF_OA_POS, None, base_neg=0.10, conf_pos=0.82, conf_neg=0.80)
    # Effusion
    has_eff = bool(EFFUSION_POS.search(text)) or bool(EFFUSION_POS2.search(text))
    has_no_eff = bool(EFFUSION_NEG.search(text))
    if has_eff and not has_no_eff:
        out['Effusion'] = {'p': 0.80, 'conf': 0.85}
    elif has_no_eff and not has_eff:
        out['Effusion'] = {'p': 0.08, 'conf': 0.90}
    elif has_eff and has_no_eff:
        out['Effusion'] = {'p': 0.45, 'conf': 0.45}
    else:
        out['Effusion'] = {'p': 0.08 if is_normal else 0.28, 'conf': 0.90 if is_normal else 0.05}
    # Synovitis
    out['Synovitis'] = score(SYNOVITIS_POS, None, base_neg=0.10, base_unk=0.18, conf_pos=0.78, conf_neg=0.60, conf_unk=0.20)
    # Baker's
    has_bak = bool(BAKERS_POS.search(text))
    has_no_bak = bool(BAKERS_NEG.search(text))
    if has_bak and not has_no_bak:
        out["Baker's"] = {'p': 0.82, 'conf': 0.88}
    elif has_no_bak and not has_bak:
        out["Baker's"] = {'p': 0.08, 'conf': 0.90}
    elif has_bak and has_no_bak:
        out["Baker's"] = {'p': 0.40, 'conf': 0.45}
    else:
        out["Baker's"] = {'p': 0.08 if is_normal else 0.20, 'conf': 0.88 if is_normal else 0.30}
    # Contusion
    out['Contusion'] = score(CONTUSION_POS, None, base_neg=0.08, base_unk=0.18, conf_pos=0.85, conf_neg=0.80, conf_unk=0.20)
    # Fracture
    has_frac = bool(FRACTURE_POS.search(text))
    has_no_frac = bool(FRACTURE_NEG.search(text))
    if has_frac and not has_no_frac:
        out['Fracture'] = {'p': 0.85, 'conf': 0.88}
    elif has_no_frac and not has_frac:
        out['Fracture'] = {'p': 0.07, 'conf': 0.92}
    elif has_frac and has_no_frac:
        out['Fracture'] = {'p': 0.40, 'conf': 0.45}
    else:
        out['Fracture'] = {'p': 0.07 if is_normal else 0.20, 'conf': 0.92 if is_normal else 0.25}
    
    return out


# ── Process all pending chunks ─────────────────────────────────
import claude_labeler as cl

pending = [p for p in sorted(CHUNK_DIR.glob('in_*.json'))
           if not (CHUNK_DIR / p.name.replace('in_', 'out_')).exists()]

print(f'Processing {len(pending)} pending chunks...')

for in_path in pending:
    out_path = CHUNK_DIR / in_path.name.replace('in_', 'out_')
    data = json.load(open(in_path, encoding='utf-8'))
    results = [label_report(r['uid'], r.get('report', '')) for r in data]
    json.dump(results, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'  wrote {out_path.name} ({len(results)} studies)')

print('All chunks done. Merging...')
df = cl.merge()
print(f'Total rows: {len(df)}')
cl.status()
