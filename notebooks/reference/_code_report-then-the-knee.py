"""Report -> twelve graded targets, in nine languages.

v2. Differences from the public lexicon are all coverage: the compartment scoping for
osteoarthritis, the pathology vocabulary for cartilage, an asserted-negative path for
findings a report explicitly clears, and a backoff for synovitis, which most reports
never name.
"""
from __future__ import annotations

import re
import unicodedata

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

_PRE = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
})


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def unwrap(text: str) -> str:
    """Rejoin lines that a fixed-width layout broke mid-sentence.

    A large share of this corpus arrives hard-wrapped at some column, so a sentence is
    split across two lines with no punctuation at the break. Splitting on newlines then
    severs the finding from its anatomy - `... y parte de la raiz` / `meniscal posterior
    del menisco lateral con extrusion asociada` puts the tear in one clause and the
    meniscus in the next, and neither clause says anything on its own. A line that does
    not end in sentence punctuation is a continuation, not a statement.
    """
    if not isinstance(text, str):
        return ""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if out and out[-1] and not re.search(r"[.;:!?>*•]$", out[-1]) \
                and len(out[-1].split()) >= 4 and s and not s[:1].isupper():
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return "\n".join(out)


def clauses(text: str):
    """Split into clauses; attach `header:` lines to the value beneath them."""
    norm = normalize(unwrap(text) if FEATURES["unwrap"] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


# Each rule below that is not obviously right is behind a flag, so that the notebook can
# turn it off and re-measure rather than assert that it helps.
FEATURES = {
    "unwrap": True,               # rejoin hard-wrapped lines before splitting
    "directional_negation": True,  # scope negation by direction instead of by clause
    "oa_inherit": True,            # unlocalised cartilage statements reach all three
    "graded_pathology": True,      # read numeric grades on a per-structure scale
    "synovitis_backoff": True,     # order the silent majority by inflammatory context
}


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
# Negation is scoped by direction, not by clause. `Subchondral insufficiency fracture at
# the medial tibial plateau without articular surface collapse` contains a negator and
# asserts a fracture: `without` governs what follows it, and the fracture precedes it.
# Reading negation at clause scope turns that sentence into a denial - and it is the
# house style of one of the larger reporting sites here, so the error is systematic
# rather than occasional.
PRE_NEG = _rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b",
    r"\bno evidence\b", r"\bfree of\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se\b",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    r"\bkeine?[nmrs]?\b", r"\bohne\b", r"\bnicht\b", r"\bkein\b",
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν", r"\bουτε\b",
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

# Turkish and a few Croatian forms put the negator at the end of the sentence, so these
# govern what precedes them instead.
POST_NEG = _rx(
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b",
    r"gozlenmemekte", r"mevcut degil", r"eslik etmiyor", r"\bizlenmedi\b",
    r"izlenmemistir", r"saptanmamistir", r"gorulmemistir", r"\bnema znakova\b",
    r"bez znakova",
)

NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, r"\bunremarkable\b")

NEG_WINDOW = 90


def _negated(clause: str, start: int, end: int) -> bool:
    """True when a negation trigger governs the span [start, end) of this clause."""
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            # `... intact but with a tear` - a contrast conjunction closes the scope.
            if not re.search(r"\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|"
                             r"ωστοσο|αλλα|но)\b", clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    return False

NORMALITY = _rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt", r"\bprimjeren",
    r"\bodrzanog kontinuiteta", r"\bodržan",
    r"φυσιολογικ", r"ακεραι", r"δεν παρατηρουνται", r"δεν σημειωνονται",
    r"unauffallig", r"regelrecht", r"\bo\.?b\.?\b",
    r"нормал", r"запазен", r"съхранен", r"\bбез особености\b", r"интактн",
    r"\bgaaf\b", r"\bnormaal\b",
)

# "Negator + abnormality-noun" is how most of these languages assert normality:
# `sin alteraciones`, `ohne Auffalligkeiten`, `geen afwijkingen`, `bez osobitosti`.
# Read literally each one is a negation, so the old guard (`normality AND NOT negation`)
# threw every one of them away and left the structure looking unmentioned instead of
# explicitly clear. That matters to a rank metric: a ligament a radiologist looked at and
# called intact must rank below one the report never mentions, not equal to it.
NORMAL_PHRASE = _rx(
    r"\bsin alteracion", r"\bsin cambios\b", r"\bsin particularidad",
    r"\bsin hallazgos\b", r"\bsin lesion", r"\bsin signos de (rotura|lesion)",
    r"\bcontinu[oa]s?\b", r"\bcontinuidad conservada\b",
    r"\bno abnormalit", r"\bno significant abnormalit", r"\bunremarkable\b",
    r"\bno evidence of (tear|injury|abnormalit)",
    r"\bohne auffalligkeit", r"\bkein nachweis\b", r"\bohne befund\b",
    r"\bgeen afwijking", r"\bzonder afwijking",
    r"\bsans anomalie", r"\bpas d[e']anomalie",
    r"\bbez osobitosti\b", r"\bbez znakova (rupture|lezije)\b",
    r"\bbez patoloskih\b",
    r"χωρις αλλοιωσ", r"χωρις παθολογ", r"δεν παρατηρουνται (αξιολογα|παθολογ)",
    r"\bбез особености\b", r"\bбез патологич", r"\bбез данни за\b",
    r"\bozel bir ozellik yok", r"\bpatolojik bulgu (yok|izlenmemis)",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected?\b",
    r"cannot (be )?exclude", r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b",
    r"\br/o\b", r"\bdd\b", r"\blikely\b", r"\bsuggest", r"\bcompatible with\b",
    r"\bposible\b", r"sin criterios categoricos", r"\bdudos", r"\bsugier",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim", r"\bdusundur",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b", r"\bmoze odgovarati\b",
    r"πιθαν", r"υποπτ",
    r"\bmoglich", r"\bverdachtig", r"\bfraglich", r"\bv\.?a\.?\b", r"\bwohl\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект",
    r"\bmogelijk\b", r"\bverdacht\b",
)

# ==== CELL ====

# --------------------------------------------------------------------------- #
# pathology vocabulary
# --------------------------------------------------------------------------- #
TEAR = _rx(
    r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
    r"\bavuls", r"\bmacerat", r"\bbuckethandle\b", r"bucket handle",
    r"\brotura\b", r"\broturas\b", r"\bruptura", r"\bdesgarro", r"\broto\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"\briss\b", r"einriss", r"\bruptur", r"zerreiss", r"\blasion", r"\bausriss",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"\brupturu\b",
    r"devamsizlik", r"\brupture\b", r"\bdevamliligi secilememis",
    r"\bpuknuce", r"\bprekid\b", r"\bpukotin", r"\bruptur",
    r"ρηξη", r"ρηξις", r"ρηγμα", r"ασυνεχεια",
    r"руптура", r"разкъсв", r"разрив", r"скъсв", r"\bлезия\b",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bmukoid\b", r"degenerativn", r"εκφυλ", r"дегенерат",
    r"\bμυξοειδ", r"\bμυξωδ", r"\bmeniskopat", r"\bmeniscopath",
    r"\bmuco ?ide\b", r"aufgefasert", r"\bdejenerasyon\b",
)

INJURY = _rx(
    r"\binjur", r"\bsprain", r"\blesion", r"\blasion", r"\bedema\b", r"\boedema\b",
    r"\bodem\b", r"\bedem\b", r"\bοιδημα", r"\bодем", r"\bедем", r"\bstrain\b",
    r"\bhigh signal\b", r"\bsignal alteration\b", r"\bhiperintens", r"\bhyperintens",
    r"aumento de senal", r"alteracion de senal", r"cambio de senal",
    r"\bsignalanhebung", r"\bsignalalteration", r"verhoogd signaal", r"sinyal artis",
    r"αυξημενο σημα", r"повишен сигнал", r"\besguince\b",
    r"\bthicken", r"\bzadebljanje\b", r"\bverdikking\b", r"\bdistenzij",
    r"\blaksite\b", r"\blaxity\b", r"\bpartial\b", r"\bparcijaln", r"\bparcial",
    r"\bpartiel", r"\bpartiell",
)

# A numeric grade is the most precise thing a knee report says, and it means different
# things in different places: grade 3 of a meniscus is a tear by definition, grade 1 or 2
# is intrasubstance signal that never reaches the surface and is not one. A ligament runs
# the other way round - grade 1 is a stretch, grade 2 a partial tear. So the grade is
# read as a number and interpreted per structure rather than folded into one pathology
# vocabulary.
_GRADE_RX = re.compile(
    r"(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)"
    r"[\s:]*(?:grade\s*)?([1-4]|iv|iii|ii|i)\b"
)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def _grade_of(clause: str):
    """Highest numeric grade stated in a clause, or None."""
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best

# --------------------------------------------------------------------------- #
# anatomy
# --------------------------------------------------------------------------- #
ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b",
        r"cruzado anterior", r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband", r"\bvkb\b",
        r"vorderes kreuzband", r"vorderen kreuzband", r"vordere kreuzband",
        r"on capraz", r"\bocb\b", r"anterior capraz",
        r"prednji krizni", r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ", r"προσθιου χιαστου", r"χιαστο[^ ]* συνδεσμ",
        r"\bχιαστ\w*",
        r"предна кръстна", r"предната кръстна", r"предна кръста",
        r"cruciate ligaments", r"ligamentos cruzados", r"ligaments croises",
        r"kruisbanden", r"kreuzbander", r"capraz baglar", r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ", r"χιαστων συνδεσμ", r"кръстните връзки", r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral medial", r"colateral interno", r"\blcm\b",
        r"collateral medial", r"collateral interne",
        r"mediale collaterale", r"binnenband", r"\b(mediale|laterale) banden\b",
        r"\bcollaterale banden\b",
        r"innenband", r"mediales? kollateral",
        r"\bic yan bag", r"medial kollateral", r"\biyb\b", r"medyal kollateral",
        r"medijalni kolateraln", r"medijalnog kolateraln",
        r"εσω πλαγι", r"εσωτερικο πλαγι", r"\bπλαγι\w* συνδεσμ", r"\bπλαγιοι\b",
        r"медиален колатерал", r"вътрешна странична", r"\bколатерал\w*",
        r"\bcolaterales\b", r"\bcollateraux\b", r"\bcollateralen\b", r"\bkolateralni\b",
        r"collateral ligaments", r"ligamentos colaterales", r"ligaments collateraux",
        r"collaterale banden", r"kollateralbander", r"seitenbander", r"yan baglar",
        r"kolateraln[a-z]* ligament[a-z]*", r"πλαγιοι συνδεσμ", r"πλαγιων συνδεσμ",
        r"колатерални връзки", r"страничните връзки",
    ),
    "Medial Meniscus": _rx(
        r"medial meniscus", r"\bmm\b(?= tear)", r"medial menisc",
        r"menisco medial", r"menisco interno",
        r"menisque medial", r"menisque interne",
        r"mediale meniscus", r"binnenmeniscus",
        r"innenmeniskus", r"medialen? meniskus", r"innenmeniskushinterhorn",
        r"medyal menisk", r"\bic menisk",
        r"medijalni meniskus", r"medijalnog meniskusa", r"medijalnom meniskusu",
        r"medijaln\w* menisk\w*", r"\bmedijalnog meniska\b", r"medijalni menisk",
        r"εσω μηνισκ", r"μηνισκ[^ ]* του εσω", r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус", r"медиален менискус", r"вътрешния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"amfoteroi\w* mhnisk", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus", r"lateral menisc",
        r"menisco lateral", r"menisco externo",
        r"menisque lateral", r"menisque externe",
        r"laterale meniscus", r"buitenmeniscus",
        r"aussenmeniskus", r"lateralen? meniskus", r"aussenmeniskushinterhorn",
        r"lateral menisk", r"\bdis menisk",
        r"lateralni meniskus", r"lateralnog meniskusa", r"lateralnom meniskusu",
        r"lateraln\w* menisk\w*", r"\blateralnog meniska\b",
        r"εξω μηνισκ", r"μηνισκ[^ ]* του εξω", r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус", r"латерален менискус", r"външния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
}

# Osteoarthritis is written as cartilage damage far more often than as a diagnosis.
OA_EVIDENCE = _rx(
    r"osteoarthrit", r"\barthros", r"\bgonarthros", r"\bosteoarthros",
    r"chondropath", r"chondromalac", r"condropat", r"condromalac", r"\bchondros",
    r"\bchondrosis\b", r"chondral (loss|defect|ulcer|thinning|injury|fissur|wear)",
    r"cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)",
    r"(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage",
    r"articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)",
    r"osteophyt", r"osteofit", r"osteofyt", r"osteofito", r"osteophyten", r"spurring",
    r"joint space narrowing", r"pinzamiento articular", r"reduced joint space",
    r"kikirdak kayb", r"kikirdak incelme", r"kondropati", r"kondral", r"kikirdak dejener",
    r"eklem aralig\w* daral", r"eklem mesafesi daral", r"kikirdak kalinlig\w* azal",
    r"kraakbeen", r"gonartrose", r"artrose", r"\bknorpel", r"arthrose", r"gonarthrose",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit", r"artrotsk", r"artrotick",
    r"\boa promjen", r"\boa\b", r"degenerativne promjene hrskav",
    r"χονδρ[^ ]*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ", r"χονδρομαλακ",
    r"αρθρικου χονδρου", r"εξαλειψη του αρθρικου χονδρου", r"διαβρωση του αρθρικου χονδρ",
    r"λεπτυνση[^.]{0,30}χονδρ", r"φθορα[^.]{0,20}χονδρ",
    r"артроз", r"хондропат", r"остеофит", r"хрущял[^.]{0,40}(изтън|увред|дефект|липс)",
    r"изтъняване[^.]{0,30}хрущял", r"хондромалац",
    r"ulcera[s]? condral", r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade", r"icrs\b", r"outerbridge", r"\bdenudation\b", r"denudacij",
    r"erozivne promjene", r"\berosion of[^.]{0,20}cartilage",
    r"kraakbeenlijden", r"kraakbeenverlies",
)

# --------------------------------------------------------------------------- #
# where in the joint a cartilage statement sits
# --------------------------------------------------------------------------- #
# Tibiofemoral structures. Used only inside a clause that already carries OA evidence,
# so bare "condyle" is safe here and would not be elsewhere.
TF_SITE = _rx(
    r"compartment", r"compartimento", r"compartiment", r"kompartman", r"kompartiment",
    r"kompartment", r"odjelj", r"διαμερισμα", r"компартм", r"\bотдел",
    r"femorotibial", r"tibiofemoral", r"femoro tibial", r"femorotibiaal",
    r"femorotibijaln", r"феморотибиал", r"\bft zglob", r"tibiofemoraln",
    r"condyle", r"condilo", r"kondyl", r"kondil", r"condyl", r"κονδυλ",
    r"кондил", r"\bplateau", r"\bplato\b", r"platillo", r"meseta", r"плато",
    r"tibiaplateau", r"tibijaln\w* plato", r"tibyal plato", r"tibia plato",
    r"κνημιαι", r"μηριαι", r"weightbearing", r"weightbaring", r"zona de carga",
    r"dragende deel", r"agirlik tasiyan", r"\bfemur\b", r"\btibia\b", r"\bfemoral\b",
    r"\btibial\b", r"\bfemura\b", r"\btibije\b", r"\bmesarthrio\b", r"μεσαρθριο",
)

# Patellofemoral structures.
PF_SITE = _rx(
    r"patellofemoral", r"femoropatellar", r"femoropatelar", r"patelofemoral",
    r"retropatellar", r"retrorotulian", r"trochlea", r"troclea", r"troklea",
    r"trochlear", r"trohlej", r"τροχιλ", r"\bpatella", r"\bpatellar", r"rotulian",
    r"\brotula\b", r"\bpatele\b", r"patellofemoraal", r"femoropatellair",
    r"επιγονατιδ", r"μηροεπιγονατιδ", r"пател", r"феморопател",
    r"anterior compartment", r"compartimento anterior", r"prednj\w* odjeljk",
    r"\bfp zglob", r"\bpf zglob", r"\bfaset", r"\bfacet", r"patellofemoraln",
)

SIDE_MEDIAL = _rx(
    r"\bmedial\w*", r"\bmedyal\w*", r"\bmedijaln\w*", r"\bmediaal\w*",
    r"\bmediale\w*", r"\binterno\b", r"\binterna\b", r"\binternos\b", r"\binterne\b",
    r"\binnen\w*", r"\bic\b", r"\bunutarnj\w*", r"\bεσω\w*", r"\bεσωτερικ\w*",
    r"\bмедиал\w*", r"\bвътреш\w*", r"\bbinnen\w*", r"\bmediaal\b", r"\bmediales?\b",
)
SIDE_LATERAL = _rx(
    r"\blateral\w*", r"\bexterno\b", r"\bexterna\b", r"\bexternos\b", r"\bexterne\b",
    r"\bdis\b", r"\blateraln\w*", r"\baussen\w*", r"\bbuiten\w*", r"\bεξω\w*",
    r"\bεξωτερικ\w*", r"\bлатерал\w*", r"\bвъншн\w*", r"\bvanjsk\w*",
)
SIDE_ANTERIOR = _rx(
    r"\banterior\w*", r"\bant\b", r"\bon\b", r"\bprednj\w*", r"\bvorder\w*",
    r"\bvoorste\b", r"\bπροσθι\w*", r"\bпредн\w*", r"\banteriyor\w*", r"\bavant\b",
    r"\banterieur\w*",
)

GLOBAL_OA = _rx(
    r"tri ?compartment", r"all three compartment", r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros", r"\bgonartros", r"\bgonarthrose", r"\bgonartrose", r"gonartro",
    r"goanrtrot", r"gonartrot",
    r"osteoarthritis of the knee", r"artrosis (de |)(la )?rodilla", r"knee osteoarthrit",
    r"\bdiz osteoartrit", r"\bgonartroz", r"artroza koljena",
    r"οστεοαρθριτιδα", r"αρθριτιδα του γονατος", r"εκφυλιστικη οστεοαρθριτ",
    r"артроза на колянната", r"гонартроз",
    r"degenerative joint disease", r"\bdjd\b", r"three compartments",
    r"compartmens", r"compartments",
)

# --------------------------------------------------------------------------- #
# self-declaring findings
# --------------------------------------------------------------------------- #
DIRECT = {
    "Effusion": _rx(
        r"\beffusion", r"joint fluid", r"intra ?articular fluid", r"\bhydrops\b",
        r"\bhemarthros", r"\bhaemarthros",
        r"derrame articular", r"\bderrame\b", r"liquido articular", r"hemartrosis",
        r"epanchement",
        r"gewrichtsvocht", r"\bvocht\b", r"gewrichtseffusie", r"opzetting van suprapatell",
        r"gelenkerguss", r"\berguss\b", r"gelenksergu", r"gelenksflussigkeit",
        r"eklem\w* ic\w* sivi", r"efuzyon", r"eklem sivisi", r"eklem mesafesinde sivi",
        r"sivi (miktari|artisi|birikimi)", r"sivi artis", r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev", r"\bizliv", r"zglobn[^ ]* tekucin", r"\bhidrops\b",
        r"αρθρικ[^ ]* υγρ", r"υγρου ενδαρθρικα", r"ενδαρθρικ[^ ]* υγρ", r"ποσοτητα υγρου",
        r"ενδαρθρικ", r"αρθρικη συλλογη", r"υγρο στην αρθρωση", r"υγρου στην αρθρωση",
        r"συλλογη υγρου", r"ενθαρθρικ",
        r"ставен излив", r"излив", r"ставна течност", r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
        r"thicken\w* synovial", r"hypertroph\w* of the synovium",
        r"synoviale? (verdikking|proliferatie)", r"verdikkingen van (het )?synovium",
        r"synovialitis", r"synovialis(verdickung|proliferation)", r"reizsynovial",
        r"sinovijalitis", r"sinovitis", r"zadebljanje sinovij", r"proliferacij\w* sinovij",
        r"sinovijaln\w* proliferacij",
        r"υμενιτιδα", r"συνοβιτιδα", r"υμενικ[^ ]* υπερτροφ", r"αρθρικου υμεν",
        r"παχυνση[^.]{0,20}υμεν", r"υμενα",
        r"синовит", r"синовиал[^ ]* (задебел|пролифер)",
        r"\bpannus\b", r"\bhoffit", r"sinovyal\w* (kalinlas|proliferas)",
        r"sinovyal hipertrof", r"\bartrit\b", r"\barthritis\b",
    ),
    "Baker's": _rx(
        r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
        r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
        r"popliteal kist", r"\bbakerova\b", r"poplitealn[^ ]* cist", r"popliteal\w* cist",
        r"κυστη baker", r"πολυχωρη συνοβιακη κυστη", r"κυστη του baker",
        r"συνοβιακη κυστη", r"κυστη τυπου baker",
        r"киста на бейкър", r"бейкърова киста", r"поплитеална киста", r"бекеров",
        r"gastrocnemio ?semimembranos", r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion", r"bone bruise", r"bone marrow (o?edema|contusion)",
        r"marrow o?edema", r"\bkontuz", r"medular bone o?edema", r"osseous contusion",
        r"contusion osea", r"edema oseo", r"edema de medula osea", r"contusiones oseas",
        r"oedeme osseux", r"contusion osseuse",
        r"botcontusie", r"botoedeem", r"beenmergoedeem", r"botmergoedeem",
        r"knochenmarkodem", r"knochenodem", r"knochenmarksodem", r"kontusion",
        r"kemik kontuzyonu", r"kemik iligi odemi", r"kemik odemi", r"kemik iliginde odem",
        r"kontuzyonel kemik", r"kemik iligi odemleri",
        r"kostani edem", r"edem kosti", r"kontuzij", r"kostane srzi[^.]{0,20}edem",
        r"οστεομυελικ[^ ]* οιδημα", r"οστικο οιδημα", r"μυελικο οιδημα", r"οστικο μωλωπ",
        r"костномозъчен едем", r"костен едем", r"контузионен", r"костно мозъчен едем",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfract\b",
        r"\bfractura", r"\bfracturas\b",
        r"\bfractuur", r"\bbreuk\b",
        r"\bfraktur", r"\bbruch\b",
        r"\bkirik\b", r"\bkirigi\b", r"\bkiri[kg]\w*",
        r"\bprijelom", r"impresijsk[^ ]* fraktur", r"impaktcij",
        r"καταγμα", r"καταγματ",
        r"фрактур", r"счупван", r"фисур",
        r"insufficiency fracture", r"stress fracture", r"avulsion fracture",
        r"subchondral fracture", r"subkondral kiri", r"impaction (fracture|injury)",
        r"osteochondral (fracture|impaction)", r"\bsegond\b", r"impactiefractuur",
        r"subchondrale impression", r"subchondraler? impress",
    ),
}

DECOY = {
    "Fracture": _rx(r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"parameniscal"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = ["Medial OA", "Lateral OA", "PF OA"]

# --------------------------------------------------------------------------- #
# stems, for morphology the phrase lexicons cannot reach
# --------------------------------------------------------------------------- #
# A report that clears or tears both menisci in one breath - "normaal voorkomen
# menisci", "Normal medial and lateral menisci" - names neither side, so a
# side-qualified lexicon leaves both targets silent. The cruciates and collaterals
# already had their plural forms; the menisci were simply missed.
#
# It is only consulted when the clause names no side at all. Otherwise "tear of the
# medial meniscus, menisci otherwise intact" would fire the plural for the lateral side
# off a clause that is about the medial one.
PLURAL_MENISCI = _rx(
    r"\bmenisci\b", r"\bmeniscos\b", r"\bmenisques\b", r"\bmenisken\b",
    r"\bmeniskusi\b", r"\bmenisk\w*ler\b", r"\bμηνισκοι\b", r"\bμηνισκων\b",
    r"\bменискуси\b", r"\bменискусите\b", r"\bmenisci\w*\b",
)
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
                    r"capraz bag\w*", r"krizn\w*", r"χιαστ\w*", r"кръстн\w*",
                    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\bocb\b", r"\bacb\b")
STEM_COLLATERAL = _rx(r"collateral\w*", r"colateral\w*", r"kollateral\w*",
                      r"collaterale\w*", r"kolateraln\w*", r"yan bag\w*",
                      r"πλαγι\w*", r"колатерал\w*", r"странич\w*",
                      r"innenband\w*", r"binnenband\w*", r"\bmcl\b", r"\blcm\b",
                      r"\biyb\b")

STEM_FRACTURE = _rx(r"fractur\w*", r"fraktur\w*", r"fractuur\w*", r"\bfract\b",
                    r"kiri[kgğ]\w*", r"prijelom\w*", r"lom kosti", r"\bbreuk\w*",
                    r"\bbruch\w*", r"καταγμα\w*", r"καταγματ\w*", r"фрактур\w*",
                    r"счупван\w*", r"fisur\w* (osea|oseas|kost)", r"fissur\w* kost")

# Decoys that steal a cruciate / collateral stem match for the wrong ligament.
POSTERIOR_ONLY = _rx(r"\bpcl\b", r"\blcp\b", r"\bhkb\b", r"\bacb\b",
                     r"posterior cruciate", r"cruzado posterior", r"croise posterieur",
                     r"achterste kruisband", r"hinteres kreuzband", r"arka capraz",
                     r"straznji krizn", r"οπισθι[οα]\w* χιαστ", r"задна кръстн",
                     r"задната кръстн")
LATERAL_COLL_ONLY = _rx(r"\blcl\b", r"\bfcl\b", r"lateral collateral",
                        r"fibular collateral", r"colateral lateral", r"colateral externo",
                        r"buitenband", r"aussenband", r"dis yan bag",
                        r"lateralni kolateraln", r"εξω πλαγι", r"латерален колатерал")

# ==== CELL ====

def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False


STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
}


class _Matcher:
    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None


ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {
    t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == "Fracture" else rx)
    for t, rx in DIRECT.items()
}

# --------------------------------------------------------------------------- #
# severity
# --------------------------------------------------------------------------- #
SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bdiscrete\b", r"\blow ?grade\b", r"\bincipient\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bfina\b", r"\bfino\b", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\baz miktarda\b", r"\bsilik\b",
    r"\bmanj\w*", r"\bblago\b", r"\bdiskretn", r"\bmalo\b", r"\bpocetn",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b", r"\blicht\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ", r"\bαρχομεν",
    r"\bминимал", r"\bлек", r"\bмалк", r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bmoderate\b", r"\bgross\b", r"\bsignificant\b", r"\babundant\b", r"\btense\b",
    r"\bcomplete\b", r"\bfull ?thickness\b", r"\bhigh ?grade\b", r"\badvanced\b",
    r"\bmoderad", r"\bimportante\b", r"\bsevera?\b", r"\bmarcad", r"\bcuantios",
    r"\bespesor total\b", r"\bcompleta?\b",
    r"\bbelirgin\b", r"\byaygin\b", r"\bileri\b", r"\bciddi\b", r"\bbol\b", r"\bkomplet",
    r"\bopsezan\b", r"\bveliki\b", r"\bizrazit", r"\bznacajn", r"\bumjeren",
    r"\buznapredoval", r"\bpotpun", r"\bkompleksn",
    r"\bausgepragt", r"\bdeutlich", r"\bmassiv", r"\bmassig", r"\bgross",
    r"\buitgebreid", r"\bgevorderd", r"\bveel\b", r"\bmatige?\b", r"\bvolledig",
    r"\bμετρι", r"\bμεγαλ", r"\bεκτεταμεν", r"\bευμεγεθ", r"\bσοβαρ", r"\bπληρη",
    r"\bголям", r"\bизразен", r"\bзначим", r"\bумерен", r"\bобилен", r"\bпълн",
)

GRADE_HIGH = re.compile(r"grade?[ao]?\s*(3|4|iii|iv)\b|icrs grade (iii|iv|3|4)|"
                        r"stupnja iv|stupnja iii|\bgrado (3|4)\b|\bgrad (3|4)\b|"
                        r"\bgrade (3|4)\b")

DEGENERATIVE_MARROW = _rx(
    r"subchondral", r"subcondral", r"subkondral", r"supkondraln", r"subchondraln",
    r"υποχονδρι", r"υπαρθρικ", r"субхондрал", r"subchondrale?", r"subartikuler",
    r"\bcyst", r"\bquist", r"\bzyste\b", r"\bcistic", r"reactive", r"reactivo",
    r"degenerative", r"degenerativ", r"reaktiv", r"\bcisti\b",
)

TRAUMA = _rx(
    r"\bbruise\b", r"\bcontusion", r"\bkontuz", r"\btrauma", r"\bimpaction\b",
    r"\bpivot shift\b", r"\bkissing\b", r"\bacute\b", r"\bagudo\b", r"\bakut",
    r"\bpivot kaymasi\b", r"\bcontusion osseuse\b", r"\bbone bruise\b",
    r"\bbotcontusie\b", r"\bконтузион", r"\bμωλωπ", r"\bkontuzij", r"\bimpaktcij",
    r"\bimpakcij", r"\bfall\b", r"\binjury\b", r"\bimpression\b",
)

# Effusion-adjacent inflammatory signs, used only when a report never names synovitis.
SYNOVIAL_PROXY = _rx(
    r"bursit", r"burzit", r"\bbursa\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)",
    r"suprapatellar (bursitis|effusion|recess)", r"suprapatellar bursa",
    r"suprapatellar bursada", r"suprapatelarno", r"suprapatellaire recessus",
    r"hoffa", r"hoffit", r"plica", r"plika", r"πλικα", r"fat pad[^.]{0,20}(edema|oedema)",
    r"kapsul", r"capsul", r"καψ", r"капсул", r"\bpannus\b", r"\bsinov", r"\bsynov",
)


def _polarity(clause: str, span=None) -> str:
    """positive / negative / uncertain for a term matched at `span` in this clause."""
    if UNCERTAIN.search(clause):
        return "uncertain"
    if span is None or not FEATURES["directional_negation"]:
        if NEGATION.search(clause):
            return "negative"
    elif _negated(clause, span[0], span[1]):
        return "negative"
    if NORMALITY.search(clause):
        # `meniscus normal` denies; `normal alignment ... full thickness tear` does not.
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return "positive"
        return "negative"
    return "positive"


def _severity(clause: str) -> float:
    """How emphatic a clause is about the finding it asserts.

    A numeric grade is deliberately not read here. It belongs to whichever structure the
    grade was written for, and a clause that grades the cartilage while mentioning the
    effusion in passing - `PF arthrotic change with reduced joint space, a smaller
    effusion and grade IV chondromalacia` - would otherwise report a severe effusion.
    """
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    if high and low:
        return 0.8
    return 0.75


def _grade(n_pos, n_neg, n_unc, best):
    """Map counted evidence onto a score in (0, 1) and a confidence."""
    if n_pos or n_unc:
        score = min(0.97, 0.50 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05
    return score, conf


def _paired_weight(clause: str, meniscus: bool) -> float:
    """How strongly one clause asserts damage to a meniscus or a cruciate/collateral.

    Ordered, not calibrated. What has to hold is that a tear outranks a graded lesion,
    that the grade is read on the right scale for the structure, and that intrasubstance
    degeneration lands below both - the annotator marks a torn meniscus and leaves a
    degenerate one, and a lexicon that scores the two alike throws that ordering away.
    """
    g = _grade_of(clause) if FEATURES["graded_pathology"] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.30
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    else:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.85 if g >= 2 else 0.30
        elif DEGEN.search(clause):
            base = 0.40
        else:
            base = 0.55
    if SEV_HIGH.search(clause) and not SEV_LOW.search(clause):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and not SEV_HIGH.search(clause):
        base *= 0.7
    return base


def _score_paired(cls, tgt):
    """Evidence for one of the four side-specific ligament / meniscus targets."""
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = "Meniscus" in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) \
                and not ANY_SIDE.search(c):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        # Anchor the negation test on the pathology word, not on the anatomy word.
        # `Medial meniscus is not torn` negates the tear, and the negator stands after
        # the noun: scoping from the noun would read the sentence as an assertion.
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == "positive":
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return s, cf, n_pos, n_neg


def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None,
                   context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == "positive":
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.30)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return s, c, n_pos, n_neg


def _score_oa(cls):
    """Osteoarthritis, scoped to the three compartments.

    A cartilage statement is attributed by what else the clause names rather than by a
    compartment phrase: `medial` next to a tibiofemoral structure sends it to the medial
    compartment, `patella` or `trochlea` to the patellofemoral one, and a statement that
    names neither is a whole-joint assertion that counts for all three at a discount.
    That last case is what the public lexicon leaves silent, and it is the majority of
    the corpus - reports say `tricompartmental chondrosis`, not `chondrosis of the
    medial femorotibial compartment`.
    """
    acc = {t: {"pos": 0, "neg": 0, "unc": 0, "best": 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = 0, 0, 0.0

    for c in cls:
        m = OA_EVIDENCE.search(c)
        if not m:
            continue
        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None
        hits = []
        if tf_med:
            hits.append("Medial OA")
        if tf_lat:
            hits.append("Lateral OA")
        if pf:
            hits.append("PF OA")

        if not hits:
            # No compartment named. Whole-joint statements ("tricompartmental
            # chondrosis", "gonarthrose") speak for all three; an unlocalised cartilage
            # remark is weaker evidence but still evidence, so it is carried at a
            # discount rather than dropped.
            if pol == "positive":
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == "negative":
                g_neg += 1
            continue

        for t in hits:
            if pol == "positive":
                acc[t]["pos"] += 1
                acc[t]["best"] = max(acc[t]["best"], sev)
            elif pol == "negative":
                acc[t]["neg"] += 1
            else:
                acc[t]["unc"] += 1
                acc[t]["best"] = max(acc[t]["best"], 0.30)

    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = a["pos"], a["neg"], a["unc"], a["best"]
        if not (pos or unc) and g_pos and FEATURES["oa_inherit"]:
            # Inherit the whole-joint statement, unless this compartment was separately
            # and explicitly cleared.
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out


def extract(report: str) -> dict:
    """Twelve (score, confidence) pairs, plus the counts the coverage gauge reads."""
    cls = clauses(report)
    out = {}

    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt in ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"):
        if tgt == "Contusion":
            s, c, npos, nneg = _score_clauses(
                cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt),
                context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None,
                                              DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    # --- synovitis backoff --------------------------------------------------- #
    # Eighty-eight per cent of reports never write the word, and the annotator marks it
    # on nearly half the studies: the label cannot be read off the term alone. What the
    # report does say is whether the joint is wet and irritated - an effusion, a
    # distended bursa, an inflamed fat pad - and that ordering is the only signal
    # available on the silent majority. It enters at low confidence, so it shapes the
    # ranking without asserting a finding.
    if (FEATURES["synovitis_backoff"] and out["Synovitis__npos"] == 0
            and out["Synovitis__nneg"] == 0):
        proxy = sum(1 for c in cls if SYNOVIAL_PROXY.search(c)
                    and _polarity(c) == "positive")
        eff = out["Effusion"]
        prior = 0.30 + 0.30 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out["Synovitis"] = min(0.72, prior)
        out["Synovitis__conf"] = 0.18

    return out

# ==== CELL ====

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    """Locate the competition mount, wherever it was attached."""
    for c in [Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for d1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [d1] + sorted(p for p in d1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError("competition mount not found")


ROOT = find_root()
log(f"input root: {ROOT}")

# A submission that never writes scores nothing at all, which is strictly worse than
# scoring badly. Every stage below is wrapped so that it cannot raise, and a valid file
# exists from the first second - overwritten only once real predictions are ready.
_test_df = pd.read_csv(ROOT / "test.csv")
_bench = _test_df[["StudyInstanceUID"]].copy()
for _c in TARGETS:
    _bench[_c] = 0.5
_bench.to_csv("submission.csv", index=False)
log(f"benchmark submission.csv written ({len(_bench)} rows)")

STAGE_OK = {}


def stage(name):
    """Run a stage; record failure instead of raising it.

    A rerun that dies halfway has still paid for everything it did before the failure,
    and the benchmark file above is worth more than a traceback. So each stage records
    whether it succeeded and the next one checks before proceeding.
    """
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                out = fn(*a, **k)
                STAGE_OK[name] = True
                log(f"stage '{name}' ok in {time.time() - t:.1f}s")
                return out
            except Exception:
                import traceback
                traceback.print_exc()
                STAGE_OK[name] = False
                log(f"stage '{name}' FAILED after {time.time() - t:.1f}s")
                return None
        return run
    return deco


train_df = pd.read_csv(ROOT / "train.csv")
log(f"train {train_df.shape}  test {_test_df.shape}")

t = time.time()
LAB = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
LAB["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
LAB = LAB.set_index("StudyInstanceUID")
log(f"read {len(LAB)} reports in {time.time() - t:.1f}s")

GOLD = train_df.dropna(subset=TARGETS).set_index("StudyInstanceUID")[TARGETS]
log(f"{len(GOLD)} studies carry the twelve annotations")

pos = (LAB[TARGETS] > 0.5).mean()
sil = pd.Series({t_: float(((LAB[t_ + "__npos"] == 0) & (LAB[t_ + "__nneg"] == 0)).mean())
                 for t_ in TARGETS})
print(pd.DataFrame({"derived positive rate": pos.round(3),
                    "silence rate": sil.round(3),
                    "annotated positive rate": GOLD.mean().round(3)}).to_string())

# ==== CELL ====

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams.update({"figure.dpi": 120, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, ACC, WARN = "#22303f", "#2b7a9b", "#c25a3d"


def agreement(lab, n_boot=2000, seed=0):
    """Per-target AUC against the annotated studies, with a bootstrap interval."""
    rng = np.random.default_rng(seed)
    g = lab.loc[GOLD.index]
    rows = []
    for t_ in TARGETS:
        y = GOLD[t_].values.astype(int)
        p = g[t_].values
        if len(set(y)) < 2:
            rows.append((t_, np.nan, np.nan, np.nan, int(y.sum()), int((1 - y).sum())))
            continue
        a = roc_auc_score(y, p)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) > 1:
                bs.append(roc_auc_score(y[i], p[i]))
        rows.append((t_, a, np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                     int(y.sum()), int((1 - y).sum())))
    return pd.DataFrame(rows, columns=["target", "auc", "lo", "hi", "npos", "nneg"])


AGREE = agreement(LAB).dropna(subset=["auc"])
print(AGREE.round(3).to_string(index=False))
print(f"\nmacro agreement AUC: {AGREE.auc.mean():.4f}   "
      f"mean silence rate: {sil.mean() * 100:.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6),
                       gridspec_kw={"width_ratios": [1.25, 1]})
o = AGREE.sort_values("auc")
y = np.arange(len(o))
ax[0].hlines(y, o.lo, o.hi, color=ACC, lw=3, alpha=.35)
ax[0].plot(o.auc, y, "o", color=ACC, ms=5)
ax[0].axvline(0.5, color=INK, lw=.8, ls=":")
ax[0].axvline(o.auc.mean(), color=WARN, lw=1, ls="--")
ax[0].text(o.auc.mean(), -0.9, f" macro {o.auc.mean():.3f}", color=WARN, fontsize=7)
ax[0].set_ylim(-1.4, len(o) - 0.4)
ax[0].set_yticks(y)
ax[0].set_yticklabels([f"{t_}  ({p}+/{n}-)" for t_, p, n in zip(o.target, o.npos, o.nneg)])
ax[0].set_xlim(0.35, 1.02)
ax[0].set_xlabel("AUC of the derived score against the annotation")
ax[0].set_title("gauge one: agreement, n = 58\n"
                "bars are 95% bootstrap intervals — they are this wide on purpose",
                loc="left", fontsize=8)

o2 = sil.sort_values()
ax[1].barh(np.arange(len(o2)), o2.values * 100, color=INK, alpha=.8, height=.65)
ax[1].set_yticks(np.arange(len(o2)))
ax[1].set_yticklabels(o2.index)
ax[1].set_xlabel("% of studies where no rule fired at all")
ax[1].set_title("gauge two: coverage, n = 4 407\n"
                "no labels needed, so it runs on the whole corpus",
                loc="left", fontsize=8)
for i, v in enumerate(o2.values * 100):
    ax[1].text(v + 1, i, f"{v:.0f}", va="center", fontsize=6.5, color=INK)
fig.tight_layout()
plt.show()

# ==== CELL ====

def macro_and_silence():
    lab = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    lab["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    lab = lab.set_index("StudyInstanceUID")
    g = lab.loc[GOLD.index]
    a = float(np.nanmean([
        roc_auc_score(GOLD[t_].values.astype(int), g[t_].values)
        if GOLD[t_].nunique() > 1 else np.nan for t_ in TARGETS]))
    s = float(np.mean([((lab[t_ + "__npos"] == 0) & (lab[t_ + "__nneg"] == 0)).mean()
                       for t_ in TARGETS]))
    return a, s


base_a, base_s = macro_and_silence()
rows = [("all rules on", base_a, base_s, 0.0, 0.0)]
for k in list(FEATURES):
    FEATURES[k] = False
    a, s = macro_and_silence()
    rows.append((f"without {k}", a, s, a - base_a, s - base_s))
    FEATURES[k] = True

ABL = pd.DataFrame(rows, columns=["configuration", "macro AUC", "mean silence",
                                  "d(AUC)", "d(silence)"])
print(ABL.round(4).to_string(index=False))

# ==== CELL ====

_SCRIPT = {"el": re.compile(r"[Ͱ-Ͽ]"), "bg/ru": re.compile(r"[Ѐ-ӿ]")}
_STOP = {
    "en": r"\b(the|and|is|with|there is|normal)\b",
    "es": (r"\b(del|los|las|con|sin|senal|rodilla|hallazgos|tecnica|resultados"
           r"|impresion|menisco|rotura)\b"),
    "fr": r"\b(des|les|avec|sans|genou|aucune)\b",
    "nl": r"\b(van|het|een|geen|met|voorste|knie)\b",
    "de": r"\b(der|die|und|mit|ohne|kein|keine|nachweis)\b",
    "tr": r"\b(ve|ile|izlenmistir|mevcut|normaldir|diz|bulgular)\b",
    "hr": r"\b(se|te|uz|bez|prikaz|uredan|koljena|meniska)\b",
}
_STOP = {k: re.compile(v) for k, v in _STOP.items()}


def guess_language(report):
    """A crude language tag, used only to *audit* the reading - never to do it.

    §2 argued against routing a report to a per-language rule set, because that commits
    to a guess before any evidence is read. None of that applies here: this classifier
    never touches extraction. It exists so the silence rate can be broken down, and a
    tag that is wrong now and then blurs the breakdown rather than corrupting a label.
    Script settles Greek and Cyrillic outright; the Latin-script languages are separated
    by counting function words, which is ugly and sufficient for a histogram.
    """
    n = normalize(report)
    for tag, rx in _SCRIPT.items():
        if rx.search(n):
            return tag
    score = {k: len(rx.findall(n)) for k, rx in _STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 2 else "?"


LANG = pd.Series([guess_language(r) for r in train_df["Report"].fillna("")],
                 index=train_df["StudyInstanceUID"])
print(LANG.value_counts().to_string())

SIL = pd.DataFrame({t: ((LAB[t + "__npos"] == 0) & (LAB[t + "__nneg"] == 0)).values
                    for t in TARGETS}, index=LAB.index)
by_lang = SIL.groupby(LANG.reindex(SIL.index).values).mean() * 100
by_lang = by_lang.loc[LANG.value_counts().index.intersection(by_lang.index)]

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6),
                       gridspec_kw={"width_ratios": [1.7, 1]})
im = ax[0].imshow(by_lang.values, cmap="RdYlBu_r", vmin=0, vmax=100, aspect="auto")
ax[0].set_xticks(range(len(TARGETS)))
ax[0].set_xticklabels(TARGETS, rotation=55, ha="right", fontsize=6.5)
ax[0].set_yticks(range(len(by_lang)))
ax[0].set_yticklabels([f"{l}  (n={int((LANG == l).sum())})" for l in by_lang.index],
                      fontsize=7)
ax[0].grid(False)
for i in range(by_lang.shape[0]):
    for j in range(by_lang.shape[1]):
        v = by_lang.values[i, j]
        ax[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                   color="white" if v > 62 or v < 12 else INK)
ax[0].set_title("gauge two, broken down: % of studies where no rule fired\n"
                "a common finding silent in one language and not another is a "
                "lexicon gap — or a reporting style", loc="left", fontsize=8)
fig.colorbar(im, ax=ax[0], fraction=0.02, pad=0.01)

# Silence has two causes needing different work, so split them: of the studies where
# nothing fired, in how many does the report name *this* structure anyway?
#
# The test has to be the target's own anatomy matcher, not a bare stem for the organ. A
# report discussing only the medial meniscus contains the word "meniscus", and counting
# that as a missed *lateral* meniscus would score correct silence as a bug - the wrong
# direction entirely for a gauge whose job is to find real gaps.
CLAUSES = {u: clauses(r) for u, r in
           zip(train_df["StudyInstanceUID"], train_df["Report"].fillna(""))}


def names_it(uid, target):
    cs = CLAUSES[uid]
    if any(ANAT_MATCH[target].search(c) for c in cs):
        return True
    if "Meniscus" in target:
        return any(PLURAL_MENISCI.search(c) and not ANY_SIDE.search(c) for c in cs)
    return False


rows = []
for t_ in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"]:
    sel = SIL[t_].values
    if not sel.sum():
        continue
    named = np.array([names_it(u, t_) for u in SIL.index[sel]])
    rows.append((t_, int(sel.sum()), 100 * named.mean()))
D = pd.DataFrame(rows, columns=["target", "silent", "names it anyway"])
y = np.arange(len(D))
ax[1].barh(y, 100 - D["names it anyway"], color="#8a97a3", label="never mentioned")
ax[1].barh(y, D["names it anyway"], left=100 - D["names it anyway"], color=WARN,
           label="mentioned, missed")
ax[1].set_yticks(y)
ax[1].set_yticklabels([f"{t_}\n({n} silent)" for t_, n in zip(D.target, D.silent)],
                      fontsize=6.5)
ax[1].set_xlabel("% of the silent studies")
ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
ax[1].set_title("why it was silent\nonly the orange half is a lexicon gap",
                loc="left", fontsize=8)
fig.tight_layout()
plt.show()
print(D.round(1).to_string(index=False))

# ==== CELL ====

import gc
import re as _re
from concurrent.futures import ThreadPoolExecutor

import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- geometry -------------------------------------------------------------- #
# The centre crop has to be smaller than the field of view of nearly every series, or it
# silently does nothing: where the requested crop is larger than the image, the crop is
# skipped and that series keeps its physical scale unnormalised - with no error and no
# log line. The figure two cells below plots the acquired field of view
# (Rows x PixelSpacing) over the whole training corpus and marks this value against it,
# together with the exact percentage of series it would fail to crop, so the choice can
# be checked rather than taken on trust. 130 mm still contains the joint.
CROP_MM = 130.0

# --- cache ----------------------------------------------------------------- #
# bytes = n_study x n_slot x slices x P^2. The exponent on P is what makes this a real
# constraint: the cache grows with the SQUARE of resolution and only linearly with
# slices, so coverage is the cheap axis and resolution the expensive one.
#
# Measured, not assumed. Resolution was tested twice at fixed coverage - 224 px against
# 266 px - and moved the holdout by a thousandth of macro AUC both times, inside the
# noise. Coverage was then tested at fixed resolution and moved it by 0.033, and the
# curve was still climbing at the last point measured: 3 -> 6 slices gained 0.019, and
# 6 -> 12 gained a further 0.014.
#
# So the trade is pushed one step further in the direction the evidence points: a coarser
# grid buying twice the slices again. That is an extrapolation, and it has a way of being
# wrong - resolution not mattering between 224 and 266 does not prove it stops mattering
# below 224, and somewhere there is a floor. So the arms below include a 12-slice arm at
# this grid, directly comparable with the 12-slice arm of the previous version at 224 px.
# If that comparison drops, the floor has been found and the notebook says so.
# 168 = 12 x 14 keeps the DINOv2 patch grid exact.
CACHE_IMG = 168
GROUP = 3                  # slices per encoder input, stacked as the three channels
N_GROUP_MAX = 8            # -> 24 slices per slot
CACHE_BUDGET_GB = float(os.environ.get("CACHE_BUDGET_GB", 17.0))
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32         # slice ordering is latency-bound on the mount, not CPU-bound
ORDER_BUDGET_S = 3000     # the real corpus used 2318s of a 2400s cap: too close

# --- training -------------------------------------------------------------- #
EPOCHS = int(os.environ.get("EPOCHS", 16))
BATCH_STUDIES = 8
LR_HEAD = 1e-3
LR_BACKBONE = 8e-6         # the encoder is adapted, not retrained
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = float(os.environ.get("TIME_BUDGET_H", 7.6)) * 3600
SMOKE = int(os.environ.get("SMOKE", 0))

# Arms, cheapest first. Each one checks the remaining budget before it starts and the
# ensemble is rewritten after every arm, so a run that is cut short still leaves the best
# submission it had time to earn rather than nothing.
#
# The first three arms are a controlled coverage experiment: identical encoder, identical
# grid, identical cache, identical cost per epoch - they differ only in how many of the
# cached slice groups each may draw from and average over. Three points rather than two,
# because two points cannot tell "still climbing" from "saturated", and that distinction
# is the whole guidance for whoever runs this next.
#
# `large` was tried and is not here. It scored below `base` on the holdout while costing
# two and a half hours, so capacity is not monotonic on this corpus and the budget is
# better spent elsewhere. The second `base` arm differs only by seed, which buys ensemble
# diversity at a known cost rather than a speculative one.
ARMS = [
    # the resolution safety check: same 12 slices as the previous version, coarser grid
    {"name": "s168g4", "variant": "small", "img": 168, "lr_bb": 8e-6, "groups": 4},
    # the coverage step
    {"name": "s168g8", "variant": "small", "img": 168, "lr_bb": 8e-6, "groups": 8},
    {"name": "b168g8", "variant": "base", "img": 168, "lr_bb": 5e-6, "groups": 8},
    {"name": "b168g8b", "variant": "base", "img": 168, "lr_bb": 5e-6, "groups": 8,
     "seed": 7},
    {"name": "b168g8c", "variant": "base", "img": 168, "lr_bb": 5e-6, "groups": 8,
     "seed": 13},
]
if SMOKE:
    # A smoke run keeps the whole control path and shrinks only the work: one epoch and
    # one arm by default, but both overridable, so the arm bookkeeping - deduplication,
    # selection, per-arm groups - can be exercised without a GPU.
    EPOCHS = int(os.environ.get("EPOCHS", 1))
    ARMS = ARMS[:int(os.environ.get("SMOKE_ARMS", 1))]

SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
# The alternative: plane x the single delivered flag, ignoring the recovered weighting.
# Under this scheme one slot mixes T1 with non-fat-suppressed PD/T2.
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]
SLOTS = SLOTS_PUBLIC if os.environ.get("SLOT_SCHEME") == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = _re.compile(r"[_\-.]")
_FATSAT_RX = _re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                         r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = _re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = _re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = _re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

def gpu_probe():
    """Check that the GPU can run a kernel - before an hour of I/O is spent assuming it.

    This is not paranoia, it is a bug this notebook hit. Kaggle's generic `GPU`
    accelerator may allocate a P100, and current PyTorch builds no longer ship kernels
    for Pascal (sm_60). Every symptom you would test for looks healthy:
    `cuda.is_available()` is True, the device name and capability query fine, tensors
    allocate, and the model moves onto the device without complaint. The *first kernel
    launch* then fails with `no kernel image is available for execution on the device`.

    Discovering that after the header pass and the decode costs 56 minutes and the whole
    run. Discovering it with a 64x64 matmul costs a millisecond, so it is done here,
    before anything expensive, and it launches a real kernel rather than asking a
    property.
    """
    if not torch.cuda.is_available():
        return False, "no CUDA device visible"
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as exc:
        return False, f"device query failed: {exc}"
    try:
        a = torch.randn(64, 64, device="cuda")
        float((a @ a).sum().item())
        float(torch.rand(1, device="cuda").item())
        return True, f"{name} (sm_{major}{minor})"
    except Exception as exc:
        return False, (f"{name} (sm_{major}{minor}) allocates but cannot launch a "
                       f"kernel: {type(exc).__name__}. This torch build has no code "
                       f"for sm_{major}{minor} - select a T4 accelerator.")


GPU_OK, GPU_MSG = gpu_probe()
DEV = torch.device("cuda" if GPU_OK else "cpu")
log(f"torch {torch.__version__}  |  gpu: {GPU_MSG}")
log(f"device {DEV}  |  arms {[a['name'] for a in ARMS]}  |  epochs {EPOCHS}"
    f"  |  budget {TIME_BUDGET / 3600:.1f} h")

# Fine-tuning three vision transformers on CPU is not a slower plan, it is not a plan.
# So if the accelerator is unusable the notebook says so at the top and stops before the
# expensive stages, leaving the benchmark submission and a legible reason, rather than
# spending an hour of I/O to arrive at the same file.
RUN_HEAVY = bool(GPU_OK or SMOKE)
if not RUN_HEAVY:
    log("=" * 72)
    log("NO USABLE GPU - skipping the imaging pipeline.")
    log(f"reason: {GPU_MSG}")
    log("submission.csv keeps the 0.5 benchmark. Sections 1-2 above still ran:")
    log("the report reader needs no accelerator and its measurements stand.")
    log("=" * 72)

# ==== CELL ====

HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns"]


def probe(item):
    """One header read per series - the middle slice - for protocol and geometry."""
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t_ in HDR_TAGS:
            v = getattr(ds, t_, None)
            if v is None:
                row[t_] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t_] = "|".join(str(x) for x in v)
            else:
                row[t_] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame()
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        return pd.DataFrame(list(pool.map(probe, items)))


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    if not len(df):
        return df
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions is matched as exact
    # tokens; a substring test on "SAT" fires on series that are not fat-suppressed.
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(x.strip() in FATSAT_OPTS for x in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr_ = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te_ = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1 = desc.str.contains(_T1_RX)
    t2 = desc.str.contains(_T2_RX)
    pdw = desc.str.contains(_PD_RX)
    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr_ < 800, "T1",
                             np.where(te_ > 60, "T2",
                               np.where(tr_ >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    df["fov_mm"] = pd.to_numeric(df["Rows"], errors="coerce") * df["px"]
    return df


def pick_slots(series_df, plane_map):
    """One series per slot per study.

    Ties break toward the stack with the most slices: a denser stack samples the joint
    better, and the slice sampler below has more to spread over.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if len(cand) == 0 and fluid is False:
                # T1 slots are the scarcest; fall back to any non-fat-sat series in the
                # plane before giving up on the slot entirely.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


def lat_of(h):
    """Study -> 'L' / 'R' / None.

    The tag is present on some studies and absent on others, and is sometimes an empty
    string rather than missing, which is not the same as NaN.
    """
    d = {}
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        d[st] = v[0] if v else None
    return d


@stage("headers")
def _headers():
    global HTR, HTE, SLOTS_TR, SLOTS_TE, LAT_TR, LAT_TE, TEST_DF, TRAIN_SERIES
    TEST_DF = pd.read_csv(ROOT / "test.csv")
    TRAIN_SERIES = pd.read_csv(ROOT / "train_series.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    both = pd.concat([TRAIN_SERIES, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    log("header pass: test")
    HTE = annotate(walk("test_series"))
    log(f"  {len(HTE)} test series")
    log("header pass: train")
    HTR = annotate(walk("train_series"))
    log(f"  {len(HTR)} train series")

    SLOTS_TE = pick_slots(HTE, plane_map)
    SLOTS_TR = pick_slots(HTR, plane_map)
    LAT_TE, LAT_TR = lat_of(HTE), lat_of(HTR)
    cov = pd.Series([len(v) for v in SLOTS_TR.values()])
    log(f"train slots per study: mean {cov.mean():.2f}  min {cov.min()}  max {cov.max()}")
    lat_known = np.mean([v is not None for v in LAT_TR.values()])
    log(f"laterality known for {lat_known * 100:.1f}% of training studies")


if RUN_HEAVY:
    _headers()

# ==== CELL ====

if STAGE_OK.get("headers"):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2),
                           gridspec_kw={"width_ratios": [1.1, 1, 1]})

    # 1. how often each slot exists at all
    names = [s[0] for s in SLOTS]
    have = np.array([[1.0 if n in SLOTS_TR[st] else 0.0 for n in names]
                     for st in SLOTS_TR])
    frac = have.mean(0) * 100
    ax[0].barh(np.arange(len(names)), frac, color=INK, alpha=.85, height=.65)
    ax[0].set_yticks(np.arange(len(names)))
    ax[0].set_yticklabels(names)
    ax[0].set_xlabel("% of training studies that have this slot")
    ax[0].set_xlim(0, 105)
    for i, v in enumerate(frac):
        ax[0].text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=6.5, color=INK)
    ax[0].set_title("the presence mask is not a formality", loc="left", fontsize=8)

    # 2. how many slots a study has
    cnt = have.sum(1)
    ax[1].hist(cnt, bins=np.arange(-.5, N_SLOT + 1.5), color=ACC, alpha=.85, rwidth=.85)
    ax[1].set_xlabel("slots present per study")
    ax[1].set_ylabel("studies")
    ax[1].set_title(f"median {np.median(cnt):.0f} of {N_SLOT}", loc="left", fontsize=8)

    # 3. the field of view, which is what fixes the crop
    fov = HTR["fov_mm"].dropna()
    fov = fov[(fov > 40) & (fov < 500)]
    ax[2].hist(fov, bins=60, color=INK, alpha=.75)
    ax[2].axvline(CROP_MM, color=WARN, lw=1.6)
    below = float((fov < CROP_MM).mean() * 100)
    ax[2].text(CROP_MM + 4, ax[2].get_ylim()[1] * .88,
               f"crop {CROP_MM:.0f} mm\nlarger than the image\nfor {below:.1f}% of series",
               color=WARN, fontsize=6.8)
    ax[2].set_xlabel("acquired field of view, Rows x PixelSpacing (mm)")
    ax[2].set_ylabel("series")
    ax[2].set_title("physical scale varies; the crop must fit under it",
                    loc="left", fontsize=8)
    fig.tight_layout()
    plt.show()

    print(f"field of view: median {fov.median():.0f} mm, "
          f"1st-99th percentile {fov.quantile(.01):.0f}-{fov.quantile(.99):.0f} mm; "
          f"a {CROP_MM:.0f} mm crop is skipped for {below:.1f}% of series")
    print(f"pixel spacing spans {HTR.px.quantile(.01):.2f} - "
          f"{HTR.px.quantile(.99):.2f} mm  "
          f"({HTR.px.quantile(.99) / max(HTR.px.quantile(.01), 1e-9):.1f}x)")
    print(f"a {CROP_MM:.0f} mm crop at {CACHE_IMG} px is "
          f"{CROP_MM / CACHE_IMG:.3f} mm/pixel; at 224 px it is "
          f"{CROP_MM / 224:.3f} mm/pixel")
    print(HTR.groupby("weight").size().sort_values(ascending=False).to_string())

# ==== CELL ====

ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]


def order_slices(rec):
    """Return the series' files sorted along the through-plane axis."""
    files, d = rec["files"], rec["dir"]
    keyed, ds = [], None
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        # A series with no usable geometry keeps its arbitrary order. That is worse than
        # sorting and better than dropping the series; the count is logged.
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t_: t_[0])], True


def read_slot(rec, n_slice, out_size):
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Returns uint8 [n_slice, out, out], normalised per-series to its 1st-99th percentile.
    Percentiles rather than min/max because MR intensity has no absolute scale and one
    bright vessel would otherwise compress the whole dynamic range.
    """
    files = rec.get("ordered") or rec["files"]
    d, px = rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    # Spread the samples over the central 60% of the stack: the outer slices of a knee
    # series are mostly soft tissue outside the joint.
    lo, hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo \
        else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            a = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
                + float(getattr(ds, "RescaleIntercept", 0) or 0)
        except Exception:
            a = None
        planes.append(a)
    shp = next((p.shape for p in planes if p is not None), None)
    if shp is None:
        return None
    planes = [p if (p is not None and p.shape == shp) else np.zeros(shp, np.float32)
              for p in planes]
    vol = np.stack(planes)

    # constant physical extent, then resize
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx, half = h // 2, w // 2, want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    t_ = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t_ = F.interpolate(t_, size=(out_size, out_size), mode="bilinear",
                       align_corners=False)
    # uint8, not float32: intensity is already normalised into [0, 1], so eight bits cost
    # nothing that the bilinear resize has not already cost, and the cache is a quarter
    # the size - which is the whole reason two groups of slices fit at all.
    return (t_.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def normalise_laterality(img, plane, lat):
    """Map every knee onto a left-knee convention."""
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])

# ==== CELL ====

RESERVE_GB = 7.0    # torch, the CUDA context, the frames, and the per-batch host copies


def available_gb():
    """Free memory as the kernel reports it, not as the documentation promises."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return CACHE_BUDGET_GB + RESERVE_GB


def plan_cache(n_study):
    """Choose how many slices per slot the memory budget allows.

    `n_study` is the train and test corpora together, because both caches are held at
    once - the test predictions are written as each arm finishes, so that a run cut
    short still leaves a submission, and that costs holding the test cache throughout.

    The number that matters is not the one in the config: it is what the machine
    actually has free. Submitting a code competition re-runs this notebook against a
    hidden test set of unknown size, and a cache sized for the three-study placeholder
    in `test.csv` would be sized wrong there by a factor of several hundred. Getting
    this wrong is not a degraded score - it is a SIGKILL, which no try/except in this
    notebook can catch, and a run that scores nothing at all.

    Resolution is held fixed and coverage gives way, in that order and deliberately.
    The pixel grid has a hard threshold under it - §5's Nyquist argument - while
    dropping from two groups of slices to one degrades the sample of the joint
    gracefully.
    """
    budget = min(CACHE_BUDGET_GB, max(3.0, available_gb() - RESERVE_GB))
    log(f"memory: {available_gb():.1f} GB available, reserving {RESERVE_GB:.0f} GB "
        f"-> cache budget {budget:.1f} GB for {n_study} studies")

    # Coverage gives way first, resolution second, and only if a single group of three
    # slices still does not fit. A patch grid must stay a multiple of 14 to keep the
    # DINOv2 tokenisation exact, so the search steps in 14s.
    img = CACHE_IMG
    while img > 154 and n_study * N_SLOT * img * img * GROUP > budget * 1024 ** 3:
        img -= 14
    per_slice = n_study * N_SLOT * img * img
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    if groups < N_GROUP_MAX or img != CACHE_IMG:
        log(f"  layout reduced: {groups} group(s) of {GROUP} at {img} px "
            f"({CROP_MM / img:.3f} mm/pixel)")
    return groups, img


N_STUDY_TOTAL = (len(SLOTS_TR) + len(SLOTS_TE)) if STAGE_OK.get("headers") else 1
if STAGE_OK.get("headers"):
    N_GROUP, CACHE_IMG = plan_cache(N_STUDY_TOTAL)
else:
    N_GROUP = 1
CACHE_SLICES = GROUP * N_GROUP
log(f"cache layout: {N_GROUP} group(s) x {GROUP} slices = {CACHE_SLICES} per slot "
    f"at {CACHE_IMG} px  ->  "
    f"{N_STUDY_TOTAL * N_SLOT * CACHE_SLICES * CACHE_IMG ** 2 / 1024 ** 3:.1f} GB")


def build_cache(slot_map, lat_map, tag, limit=None):
    """Decode every (study, slot) once into one uint8 array."""
    studies = sorted(slot_map)
    if limit:
        studies = studies[:limit]
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, CACHE_IMG, CACHE_IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.2f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]

    # Ordering first, and as its own pass. It reads one header per slice of every chosen
    # series - far more file opens than the decode that follows - and on a network mount
    # that is latency rather than work, so it gets its own wider pool and its own budget.
    t_ord = time.time()
    log(f"{tag}: ordering {len(jobs)} slot-series "
        f"({sum(len(j[3]['files']) for j in jobs)} slice headers)")
    ok = done = 0
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), 1024):
            block = jobs[c0:c0 + 1024]
            for (_, _, _, rec), (files, good) in zip(
                    block, pool.map(lambda j: order_slices(j[3]), block)):
                rec["ordered"] = files
                ok += int(good)
                done += 1
            if time.time() - t_ord > ORDER_BUDGET_S:
                log(f"{tag}: ordering budget spent at {done}/{len(jobs)}; "
                    f"the rest keep file order")
                break
    log(f"{tag}: ordered {ok}/{len(jobs)} by geometry in {time.time() - t_ord:.0f}s")

    log(f"{tag}: decoding {len(jobs)} slot-series")
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), 512):
            block = jobs[c0:c0 + 512]
            for (st, k, plane, _), img in zip(
                    block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, CACHE_IMG),
                                    block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(
                    img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < 512:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    gc.collect()
    return studies, cache, mask


@stage("cache")
def _cache():
    global ST_TE, CTE, MTE, ST_TR, CTR, MTR
    lim = 240 if SMOKE else None
    ST_TE, CTE, MTE = build_cache(SLOTS_TE, LAT_TE, "test")
    ST_TR, CTR, MTR = build_cache(SLOTS_TR, LAT_TR, "train", limit=lim)
    log(f"cached {len(ST_TR)} train studies, {len(ST_TE)} test studies; "
        f"slot presence {MTR.mean() * 100:.1f}%")
    log(f"cache resident: {(CTR.nbytes + CTE.nbytes) / 1024 ** 3:.2f} GB; "
        f"{available_gb():.1f} GB still free")


if STAGE_OK.get("headers"):
    _cache()

# ==== CELL ====

if STAGE_OK.get("cache") and len(ST_TR):
    # A study with as many slots as possible, so the figure shows what the head is given.
    j = int(np.argmax(MTR.sum(1)))
    names = [s[0] for s in SLOTS]
    fig, axes = plt.subplots(N_GROUP, N_SLOT, figsize=(1.55 * N_SLOT, 1.7 * N_GROUP),
                             squeeze=False)
    for g in range(N_GROUP):
        for k in range(N_SLOT):
            ax_ = axes[g][k]
            ax_.set_xticks([])
            ax_.set_yticks([])
            ax_.grid(False)
            if MTR[j, k] < .5:
                ax_.set_facecolor("#eceff2")
                ax_.text(.5, .5, "absent", ha="center", va="center", fontsize=6.5,
                         color="#8a97a3", transform=ax_.transAxes)
            else:
                # the three cached slices of this group, shown as an RGB composite -
                # which is exactly what the encoder receives
                rgb = CTR[j, k, g * GROUP:(g + 1) * GROUP].transpose(1, 2, 0)
                ax_.imshow(rgb)
            if g == 0:
                ax_.set_title(names[k], fontsize=6.5, pad=3)
            if k == 0:
                ax_.set_ylabel(f"group {g}", fontsize=6.5)
    fig.suptitle("one study as the model sees it: six protocol slots x "
                 f"{N_GROUP} groups of {GROUP} neighbouring slices, "
                 "normalised to a left knee and a 130 mm field",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    plt.show()
    print(f"cache {CTR.nbytes / 1024 ** 3:.2f} GB train + "
          f"{CTE.nbytes / 1024 ** 3:.2f} GB test")

# ==== CELL ====

class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study."""

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask, return_att=False):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        z = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        return (z, att) if return_att else z


class Model(nn.Module):
    """Encoder plus head, trained end to end.

    The bag is flattened for the encoder and folded back before the head, so the encoder
    never sees the study structure and the head never sees pixels.
    """

    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim, N_SLOT, len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None, return_att=False):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any arm needs; the rest
            # downsample from it, so every arm sees the same pixels through a different
            # sampling grid rather than a different crop.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1).reshape(B, S, -1)
        return self.head(feat, mask, return_att)


def find_dinov2(variant):
    """Locate a mounted DINOv2 checkpoint directory by variant name."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            hits.append(Path(root))
    exact = [h for h in hits if f"/{variant}/" in str(h).lower()
             or str(h).lower().rstrip("/").endswith(variant)]
    if exact:
        return exact[0]
    # Deliberately no fallback to "some other DINOv2". A loose match here would load
    # whichever checkpoint happened to be mounted, so an arm asking for `large` would
    # quietly train a second copy of `base` - and then be rank-averaged with the first,
    # doubling one model's vote under a name that says otherwise. An arm that cannot
    # find its own encoder should fail and be skipped, which the caller handles.
    return None


def build_model(variant, unfreeze_last=UNFREEZE_LAST):
    from transformers import AutoModel
    p = find_dinov2(variant)
    if p is None:
        raise FileNotFoundError(f"DINOv2 '{variant}' weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    trainable = sum(q.numel() for q in bb.parameters() if q.requires_grad)
    log(f"backbone {variant} from {p.name}: {n_layer} blocks, last {unfreeze_last} "
        f"trainable ({trainable / 1e6:.1f}M params), hidden {bb.config.hidden_size}")
    return Model(bb, bb.config.hidden_size * 2)


def take_group(rows, g):
    return rows[:, :, g * GROUP:(g + 1) * GROUP]


def augment(imgs):
    """Vertical flip and a small intensity scale, applied to a whole bag at once.

    No horizontal flip: laterality was normalised onto a single convention upstream, and
    flipping horizontally would reintroduce exactly the nuisance axis that removed.
    """
    if torch.rand(1).item() < 0.5:
        imgs = torch.flip(imgs, dims=[-2])
    scale = 1.0 + (torch.rand(1, device=imgs.device) - 0.5) * 0.2
    return (imgs.float() * scale).clamp(0, 255).to(imgs.dtype)


@torch.no_grad()
def predict(model, cache, mask, idx, img_size, n_group=None, both=False):
    """Average the logits over the slice groups this arm is allowed to see.

    Training draws one group per step, which is augmentation along the stack; inference
    averages over all of an arm's groups, so a prediction does not depend on which three
    slices a single draw happened to pick. `n_group` is per-arm so that two arms reading
    the same cache can be given different amounts of the joint.

    With `both`, the same forward passes are also aggregated by max, at no extra cost.
    Mean and max are not interchangeable here and neither is right for all twelve
    targets: a meniscal tear or a fracture is focal, visible on the groups that cross it
    and absent from the rest, so averaging four groups dilutes the one that saw it by
    four; an effusion or tricompartmental cartilage loss is diffuse and every group sees
    it, where the mean is the better estimator and the max is just the noisiest group.
    So the choice is made per target, on the holdout, in §8.
    """
    ng = N_GROUP if n_group is None else min(n_group, N_GROUP)
    model.eval()
    mean_out, max_out = [], []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        rows = torch.from_numpy(cache[sel]).to(DEV)
        m = torch.from_numpy(mask[sel]).to(DEV)
        zs = []
        for g in range(ng):
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                zs.append(model(take_group(rows, g), m, img_size).float())
        z = torch.stack(zs)                      # [groups, batch, targets]
        mean_out.append(torch.sigmoid(z.mean(0)).cpu().numpy())
        max_out.append(torch.sigmoid(z.max(0).values).cpu().numpy())
    if not mean_out:
        z0 = np.zeros((0, len(TARGETS)), np.float32)
        return z0 if not both else (z0, z0)
    mean_p = np.concatenate(mean_out)
    if not both:
        return mean_p
    return mean_p, np.concatenate(max_out)


def macro_auc(y, p):
    v = [roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
         for j in range(y.shape[1])]
    return float(np.nanmean(v)), np.array(v, dtype=float)

# ==== CELL ====

import hashlib


@stage("targets")
def _targets():
    global Y, W, TR_IDX, VA_IDX, YV, GI, GOLD_Y
    Y = np.zeros((len(ST_TR), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    conf_cols = [t_ + "__conf" for t_ in TARGETS]
    for i, st in enumerate(ST_TR):
        if st in GOLD.index:
            Y[i] = GOLD.loc[st].values
            W[i] = 3.0
        elif st in LAB.index:
            r = LAB.loc[st]
            Y[i] = r[TARGETS].values
            W[i] = 0.25 + 0.75 * r[conf_cols].values
    keep = np.where(W.sum(1) > 0)[0]

    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5
                    for s in ST_TR])
    VA_IDX = np.array([i for i in keep if grp[i] == 0])
    TR_IDX = np.array([i for i in keep if grp[i] != 0])
    if len(VA_IDX) == 0 or len(TR_IDX) < BATCH_STUDIES:
        cut = max(1, len(keep) // 5)
        VA_IDX, TR_IDX = keep[:cut], keep[cut:]
    YV = (Y[VA_IDX] > 0.5).astype(int)

    pos = {s: i for i, s in enumerate(ST_TR)}
    va_set = set(VA_IDX.tolist())
    GI = np.array([pos[s] for s in GOLD.index if s in pos and pos[s] in va_set],
                  dtype=int)
    GOLD_Y = (GOLD.loc[[ST_TR[i] for i in GI]].values.astype(int)
              if len(GI) else None)

    log(f"supervised {len(keep)} of {len(ST_TR)} studies "
        f"(annotated {sum(s in GOLD.index for s in ST_TR)})")
    log(f"train {len(TR_IDX)} / holdout {len(VA_IDX)} studies, grouped on report text")
    log(f"annotation check: {len(GI)} annotated studies fell in the holdout")
    log(f"holdout positive rate per target: "
        f"{np.round(YV.mean(0), 2).tolist()}")


if STAGE_OK.get("cache"):
    _targets()

# ==== CELL ====

def write_submission(pred, studies, path):
    """One submission file from one prediction matrix, as per-column ranks."""
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    sub = TEST_DF[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub


def rank_mean(preds):
    return np.mean([pd.DataFrame(p).rank(pct=True).values for p in preds], axis=0)


def train_arm(cfg):
    """Fine-tune one configuration; return its holdout history and test predictions."""
    # An arm can ask for no more resolution than the cache was actually built at; if
    # §5's planner had to shrink the grid, the arms shrink with it rather than
    # upsampling cached pixels back to a size that no longer carries the detail.
    cfg = dict(cfg, img=min(cfg["img"], CACHE_IMG))
    batch = int(cfg.get("batch", BATCH_STUDIES))
    ng = min(int(cfg.get("groups", N_GROUP)), N_GROUP)
    log(f"=== {cfg['name']}: {cfg['variant']}, batch {batch}, {cfg['img']} px, "
        f"{CROP_MM / cfg['img']:.3f} mm/pixel, "
        f"{CROP_MM / cfg['img'] * 14:.2f} mm per patch token, "
        f"{ng * GROUP} slices/slot ===")
    torch.manual_seed(int(cfg.get("seed", SEED)))
    np.random.seed(int(cfg.get("seed", SEED)))
    model = build_model(cfg["variant"]).to(DEV)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": cfg["lr_bb"]},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    steps = max(EPOCHS * (len(TR_IDX) // batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg["lr_bb"], LR_HEAD], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

    hist, best, best_state, best_per, best_val = [], -1.0, None, None, None
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(TR_IDX)
        tot = nstep = 0
        for b in range(0, len(perm) - batch + 1, batch):
            sel = perm[b:b + batch]
            rows = torch.from_numpy(CTR[sel]).to(DEV)
            g = int(torch.randint(ng, (1,)).item())
            imgs = augment(take_group(rows, g))
            m = torch.from_numpy(MTR[sel]).to(DEV)
            y = torch.from_numpy(Y[sel]).to(DEV)
            w = torch.from_numpy(W[sel]).to(DEV)
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                loss = (F.binary_cross_entropy_with_logits(
                    model(imgs, m, cfg["img"]), y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.item())
            nstep += 1
            if time.time() - T0 > TIME_BUDGET:
                break

        pv = predict(model, CTR, MTR, VA_IDX, cfg["img"], ng)
        d, per = macro_auc(YV, pv)
        g_auc = float("nan")
        if GOLD_Y is not None and len(GI) > 3:
            g_auc = macro_auc(GOLD_Y,
                              predict(model, CTR, MTR, GI, cfg["img"], ng))[0]
        hist.append({"epoch": ep + 1, "loss": tot / max(nstep, 1), "holdout": d,
                     "annot": g_auc})
        log(f"  epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}  "
            f"holdout {d:.4f}  annot(n={len(GI)}) {g_auc:.4f}  "
            f"[{(time.time() - T0) / 3600:.2f} h]")

        # Selection reads the holdout alone; see §7 for why the annotation check cannot
        # arbitrate between epochs.
        if d > best:
            best, best_per, best_val = d, per, pv
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if time.time() - T0 > TIME_BUDGET:
            log("  time budget reached")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Choose mean- or max-over-groups per target, on the holdout. Free: both come out of
    # the same forward passes. With one group the two are identical and the choice is a
    # no-op, which is what makes `s224g1` a clean control.
    vm, vx = predict(model, CTR, MTR, VA_IDX, cfg["img"], ng, both=True)
    _, per_mean = macro_auc(YV, vm)
    _, per_max = macro_auc(YV, vx)
    use_max = np.nan_to_num(per_max) > np.nan_to_num(per_mean)
    best_val = np.where(use_max, vx, vm)
    gain = float(np.nanmean(np.where(use_max, per_max, per_mean)) -
                 np.nanmean(per_mean))
    log(f"  group aggregation: max chosen for "
        f"{[t for t, u in zip(TARGETS, use_max) if u] or 'no target'}"
        f"  (+{gain:.4f} holdout)")

    tm, tx = predict(model, CTE, MTE, np.arange(len(ST_TE)), cfg["img"], ng, both=True)
    tp = np.where(use_max, tx, tm)
    best_per = np.where(use_max, per_max, per_mean)
    best = float(np.nanmean(best_per))
    log(f"  {cfg['name']}: holdout {best:.4f}")
    del model, opt, sched, scaler, best_state
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"best": best, "per_target": best_per, "hist": hist, "test": tp,
            "val": best_val, "slices": ng * GROUP,
            "use_max": [t for t, u in zip(TARGETS, use_max) if u]}


RESULTS, TEST_PREDS = {}, {}


@stage("train")
def _train():
    seen = set()
    for cfg in ARMS:
        left = TIME_BUDGET - (time.time() - T0)
        if left < 900:
            log(f"skipping {cfg['name']}: {left / 60:.0f} min left")
            continue
        # If the planner shrank the grid or the slice budget, two arms can collapse onto
        # the same configuration. Running the second one would spend an hour reproducing
        # the first one's predictions and then average them with itself.
        #
        # The key has to name every axis an arm can vary, not just the grid. An earlier
        # version keyed on (encoder, resolution) alone, which silently swallowed the
        # entire coverage experiment - `s224g1` and `s224g4` differ in neither - and the
        # second seed with it. A deduplication rule that is coarser than the experiment
        # deletes the experiment.
        key = (cfg["variant"], min(cfg["img"], CACHE_IMG),
               min(int(cfg.get("groups", N_GROUP)), N_GROUP),
               int(cfg.get("seed", SEED)), int(cfg.get("batch", BATCH_STUDIES)))
        if key in seen:
            log(f"skipping {cfg['name']}: identical to an arm already run {key}")
            continue
        seen.add(key)
        try:
            r = train_arm(cfg)
        except Exception:
            import traceback
            traceback.print_exc()
            log(f"arm {cfg['name']} failed; continuing with the others")
            gc.collect()
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
            continue
        RESULTS[cfg["name"]] = r
        TEST_PREDS[cfg["name"]] = r["test"]
        write_submission(r["test"], ST_TE, f"submission_{cfg['name']}.csv")
        # Rewrite the ensemble after every arm, so a run cut short by the wall clock
        # still leaves the best file it had time to earn.
        ens = rank_mean(list(TEST_PREDS.values()))
        write_submission(ens, ST_TE, "submission.csv")
        log(f"  submission.csv now the rank mean of {len(TEST_PREDS)} arm(s)")
    if not TEST_PREDS:
        raise RuntimeError("no arm completed")


SELECTED = []


@stage("ensemble")
def _ensemble():
    """Choose which arms to average, on the holdout, instead of averaging all of them.

    An equal-weight mean over every arm assumes every arm deserves an equal vote. That
    assumption was tested and failed: a DINOv2-large arm scored well below base here, and
    averaging it in at equal weight drags the mean toward a model already known to be
    worse. Greedy forward selection with replacement (Caruana et al.) fixes both halves
    of the problem at once - a bad arm is simply never picked, and a good arm picked
    twice is a good arm weighted twice.

    Selection reads the same report-derived holdout that chose the epochs, so it can
    overfit it; with four candidates and ~900 studies that risk is small, but it is real,
    which is why a pick has to improve the holdout by more than EPS to be accepted, and
    why the unselected equal-weight mean is written alongside for comparison.
    """
    global SELECTED
    names = [n for n in TEST_PREDS if RESULTS[n].get("val") is not None]
    if not names:
        return
    EPS = 1e-4
    chosen, best = [], -1.0
    for _ in range(2 * len(names)):
        pick, pick_s = None, best + EPS
        for n in names:
            s, _ = macro_auc(YV, rank_mean([RESULTS[k]["val"] for k in chosen + [n]]))
            if s > pick_s:
                pick, pick_s = n, s
        if pick is None:
            break
        chosen.append(pick)
        best = pick_s
    if not chosen:
        chosen = [max(names, key=lambda n: RESULTS[n]["best"])]
        best, _ = macro_auc(YV, RESULTS[chosen[0]]["val"])
    SELECTED = chosen

    flat, _ = macro_auc(YV, rank_mean([RESULTS[k]["val"] for k in names]))
    from collections import Counter
    log(f"ensemble: {dict(Counter(chosen))}")
    log(f"  holdout macro AUC  selected {best:.4f}   equal-weight-all {flat:.4f}   "
        f"best single {max(RESULTS[n]['best'] for n in names):.4f}")
    write_submission(rank_mean([RESULTS[k]["test"] for k in names]), ST_TE,
                     "submission_equalweight.csv")
    sub = write_submission(rank_mean([RESULTS[k]["test"] for k in chosen]), ST_TE,
                           "submission.csv")
    log(f"  submission.csv = rank mean of {len(chosen)} picks over "
        f"{len(set(chosen))} distinct arm(s); {sub.shape}")


if STAGE_OK.get("targets"):
    _train()

if STAGE_OK.get("train"):
    _ensemble()

# ==== CELL ====

if RESULTS:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4),
                           gridspec_kw={"width_ratios": [1, 1.35, .9]})
    cols = {"s168g4": "#8a97a3", "s168g8": ACC, "b168g8": WARN,
            "b168g8b": "#6b8f3a", "b168g8c": "#9a6fb0"}

    for name, r in RESULTS.items():
        h = pd.DataFrame(r["hist"])
        c = cols.get(name, INK)
        ax[0].plot(h.epoch, h.holdout, "-o", ms=3, color=c, label=f"{name} holdout")
        ax[0].plot(h.epoch, h.annot, ":", lw=1, color=c, alpha=.6)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("macro AUC")
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("solid: report-derived holdout (selects)\n"
                    "dotted: annotation check (reports only)", loc="left", fontsize=8)

    w = 0.8 / max(len(RESULTS), 1)
    xs = np.arange(len(TARGETS))
    for i, (name, r) in enumerate(RESULTS.items()):
        ax[1].bar(xs + i * w, r["per_target"], width=w, color=cols.get(name, INK),
                  alpha=.85, label=name)
    ax[1].axhline(0.5, color=INK, lw=.8, ls=":")
    ax[1].set_xticks(xs + w * (len(RESULTS) - 1) / 2)
    ax[1].set_xticklabels(TARGETS, rotation=55, ha="right", fontsize=6.5)
    ax[1].set_ylabel("holdout AUC")
    ax[1].set_ylim(0.4, 1.0)
    ax[1].legend(fontsize=6.5, frameon=False)
    ax[1].set_title("per target — §1 says the weakest one is the expensive one",
                    loc="left", fontsize=8)

    names = list(TEST_PREDS)
    if len(names) > 1:
        R = np.array([pd.DataFrame(TEST_PREDS[n]).rank(pct=True).values.ravel()
                      for n in names])
        C = np.corrcoef(R)
        im = ax[2].imshow(C, cmap="Blues", vmin=max(0, C.min() - .02), vmax=1)
        ax[2].set_xticks(range(len(names)))
        ax[2].set_xticklabels(names, fontsize=6.5)
        ax[2].set_yticks(range(len(names)))
        ax[2].set_yticklabels(names, fontsize=6.5)
        ax[2].grid(False)
        for a in range(len(names)):
            for b in range(len(names)):
                ax[2].text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if C[a, b] > .8 else INK)
        ax[2].set_title("rank correlation between arms\nlower is why the mean helps",
                        loc="left", fontsize=8)
    else:
        ax[2].axis("off")
    fig.tight_layout()
    plt.show()

    summary = pd.DataFrame({n: {"best holdout macro AUC": r["best"],
                                "slices/slot": r.get("slices"),
                                "epochs run": len(r["hist"]),
                                "picked": SELECTED.count(n)}
                            for n, r in RESULTS.items()}).T
    print(summary.round(4).to_string())
    per = pd.DataFrame({n: r["per_target"] for n, r in RESULTS.items()}, index=TARGETS)
    print("\nper-target holdout AUC")
    print(per.round(3).to_string())

    # The controlled coverage comparison, stated as a number rather than a claim.
    curve = [(RESULTS[n]["slices"], RESULTS[n]["best"])
             for n in ("s168g4", "s168g8") if n in RESULTS]
    if len(curve) >= 2:
        curve.sort()
        print("\ncoverage curve, everything else held fixed "
              "(same encoder, grid, cache and cost per epoch):")
        for i, (sl, sc) in enumerate(curve):
            d = f"   {sc - curve[i - 1][1]:+.4f} vs previous" if i else ""
            print(f"   {sl:2d} slices/slot   holdout {sc:.4f}{d}")
        a, b = RESULTS["s168g4"], RESULTS["s168g8"]
        pt = pd.Series(b["per_target"] - a["per_target"], index=TARGETS).sort_values()
        print("  per target, most helped by more slices:")
        print("   ", ", ".join(f"{k} {v:+.3f}" for k, v in pt.tail(4)[::-1].items()))
        print("  least:")
        print("   ", ", ".join(f"{k} {v:+.3f}" for k, v in pt.head(3).items()))

# ==== CELL ====

sub = pd.read_csv("submission.csv")
assert len(sub) == len(_bench), (len(sub), len(_bench))
assert list(sub.columns) == ["StudyInstanceUID"] + TARGETS, list(sub.columns)
assert sub[TARGETS].isna().sum().sum() == 0
assert np.isfinite(sub[TARGETS].values).all()
print(f"submission.csv  {sub.shape}  "
      f"nulls {int(sub[TARGETS].isna().sum().sum())}  "
      f"range [{sub[TARGETS].values.min():.3f}, {sub[TARGETS].values.max():.3f}]")
print(sub.head().to_string())
log(f"done in {(time.time() - T0) / 3600:.2f} h; stages {STAGE_OK}")