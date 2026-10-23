from __future__ import annotations

import re
import unicodedata

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# Turkish dotted/dotless i must be folded before casefolding, otherwise "İZLENMEZ"
# and "izlenmez" diverge. ß and the Croatian/Serbian d-with-stroke likewise.
_PRE = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
})


def normalize(text: str) -> str:
    """Fold case, diacritics and separators; keep Greek and Cyrillic letters.

    NFKD decomposition strips Latin accents and Greek tonos alike (ά -> α), which is what
    we want: reports are inconsistent about accents. It also maps the MICRO SIGN U+00B5
    to a real mu, which matters because most Greek reports here use the wrong codepoint.
    """
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")                    # soft hyphen
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def clauses(text: str):
    """Split into clauses, then attach `header:` lines to the value that follows.

    A report line reading `Fractures :` followed by `Aucune.` is one statement. Splitting
    on punctuation alone separates the anatomy from its negation and flips the label.
    """
    norm = normalize(text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]

    merged = []
    for i, c in enumerate(raw):
        # A fragment ending in a colon is a heading for the next fragment. Structured
        # English reports write long ones - "lateral compartment (meniscus, collateral
        # ligament complex, cartilage):" is eight words - so the cap is generous, and
        # the heading is kept as its own clause too in case it carries the finding.
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
        merged.append(c)
    # Comma-separated enumerations inside a long clause hide separate assertions.
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# ==== CELL ====

NEGATION = _rx(
    # en
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b",
    r"\bno evidence\b", r"\bunremarkable\b", r"\bfree of\b",
    # es
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b",
    # fr
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b", r"\babsence\b",
    # nl
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    # de
    r"\bkeine?\b", r"\bohne\b", r"\bnicht\b",
    # tr
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b",
    r"gozlenmemekte", r"mevcut degil", r"eslik etmiyor", r"\bizlenmedi\b",
    # hr / sr / bs
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    # el (accents already stripped)
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν",
    # bg / ru
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

NORMALITY = _rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt",
    r"φυσιολογικ", r"ακεραι",
    r"unauffallig", r"regelrecht", r"\bintakt\b",
    r"нормал", r"запазен", r"съхранен", r"\bбез особености\b",
    r"\bgaaf\b", r"\bnormaal\b",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected\b",
    r"cannot (be )?exclude", r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b",
    r"\bposible\b", r"sin criterios categoricos", r"\bdudos",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b",
    r"πιθαν", r"υποπτ",
    r"\bmoglich", r"\bverdachtig", r"\bfraglich", r"\bV\.a\.\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект",
    r"\bmogelijk\b", r"\bverdacht\b",
)

# Pathology vocabulary shared by the paired rules.
TEAR = _rx(
    r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
    r"\bavuls",
    r"\brotura\b", r"\broturas\b", r"\bruptura", r"\bdesgarro", r"\broto\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"\briss\b", r"einriss", r"\bruptur", r"zerreiss", r"\blasion",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"\brupturu\b",
    r"\bpuknuce", r"\bruptur", r"\bprekid\b", r"\bpukotin",
    r"ρηξη", r"ρηξις", r"ρηγμα",
    r"руптура", r"разкъсв", r"разрив", r"скъсв",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bmukoid\b", r"degenerativn", r"εκφυλιστ", r"дегенерат",
    r"\bmuco ?ide\b", r"aufgefasert",
)

INJURY = _rx(
    r"\binjur", r"\bsprain", r"\blesion", r"\blasion", r"\bedema\b", r"\boedema\b",
    r"\bodem\b", r"\bedem\b", r"\bοιδημα", r"\bодем", r"\bедем", r"\bstrain\b",
    r"\bhigh signal\b", r"\bsignal alteration\b", r"\bhiperintens", r"\bhyperintens",
    r"\bthicken", r"\bzadebljanje\b", r"\bverdikking\b", r"\bdistenzij",
    r"\blaksite\b", r"\blaxity\b", r"\bpartial\b", r"\bparcijaln", r"\bparcial",
    r"\bpartiel", r"\bpartiell",
)


# ==== CELL ====

ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b",
        r"cruzado anterior", r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband", r"\bvkb\b",
        r"vorderes kreuzband", r"vorderen kreuzband", r"vordere kreuzband",
        r"on capraz", r"\bocb\b",
        r"prednji krizni", r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ", r"προσθιου χιαστου", r"χιαστο[^ ]* συνδεσμ",
        r"предна кръстна", r"предната кръстна",
        # Plural, unqualified: reports routinely clear both cruciates in one clause
        # ("Ligamentos cruzados y colaterales dentro de limites normales"). Without
        # this, Spanish ACL was silent on 88% of its reports and Dutch on 70%.
        r"cruciate ligaments", r"ligamentos cruzados", r"ligaments croises",
        r"kruisbanden", r"kreuzbander", r"capraz baglar", r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ", r"χιαστων συνδεσμ", r"кръстните връзки", r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral medial", r"colateral interno", r"\blcm\b",
        r"collateral medial", r"collateral interne",
        r"mediale collaterale", r"binnenband",
        r"innenband", r"mediales? kollateral",
        r"\bic yan bag", r"medial kollateral", r"\biyb\b",
        r"medijalni kolateraln", r"medijalnog kolateraln",
        r"εσω πλαγι", r"εσωτερικο πλαγι",
        r"медиален колатерал", r"вътрешна странична",
        # Same plural pattern as the cruciates.
        # "Ligamentos cruzados y colaterales" separates the noun from its adjective, so
        # the adjective has to stand alone as a cue.
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
        r"εσω μηνισκ", r"μηνισκ[^ ]* του εσω", r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус", r"медиален менискус", r"вътрешния менискус",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus", r"lateral menisc",
        r"menisco lateral", r"menisco externo",
        r"menisque lateral", r"menisque externe",
        r"laterale meniscus", r"buitenmeniscus",
        r"aussenmeniskus", r"lateralen? meniskus",
        r"lateral menisk", r"\bdis menisk",
        r"lateralni meniskus", r"lateralnog meniskusa", r"lateralnom meniskusu",
        r"εξω μηνισκ", r"μηνισκ[^ ]* του εξω", r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус", r"латерален менискус", r"външния менискус",
    ),
}

# Osteoarthritis is rarely written as "osteoarthritis". It is written as cartilage loss,
# chondropathy grade, joint space narrowing, or osteophytes - scoped to a compartment.
OA_EVIDENCE = _rx(
    r"osteoarthrit", r"\barthros", r"\bgonarthros", r"\bosteoarthros",
    r"chondropath", r"chondromalac", r"condropat", r"condromalac",
    r"cartilage loss", r"cartilage thinning", r"chondral (loss|defect|ulcer|thinning)",
    r"osteophyt", r"osteofit", r"osteofyt", r"osteofito", r"osteophyten",
    r"joint space narrowing", r"pinzamiento articular",
    r"kikirdak kayb", r"kikirdak incelme", r"kondropati", r"kondral",
    r"kraakbeen(lijden|verlies)", r"gonartrose", r"artrose",
    r"knorpel(verlust|schaden|defekt)", r"arthrose", r"gonarthrose",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit",
    r"χονδρ[^ ]*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ",
    r"αρθρικου χονδρου", r"εξαλειψη του αρθρικου χονδρου",
    r"артроз", r"хондропат", r"остеофит", r"хрущял[^.]{0,30}(изтън|увред|дефект)",
    r"ulcera[s]? condral", r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade", r"outerbridge",
)

COMPARTMENT = {
    "Medial OA": _rx(
        r"medial (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial medial", r"femorotibial interno",
        r"mediaal femorotibiaal", r"mediale femorotibial",
        r"medial femorotibial", r"medialen kompartiment", r"innere[sn]? kompartiment",
        r"medyal femorotibial", r"ic kompartman", r"medyal kompartman",
        r"medijaln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εσω διαμερισμα", r"εσω κνημιαι", r"εσω μηριαι",
        r"медиалн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"medial (femoral|tibial) (condyle|plateau)", r"condilo femoral medial",
        r"medialen? (femurkondyl|tibiaplateau)", r"mediale femorale condyl",
    ),
    "Lateral OA": _rx(
        r"lateral (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial lateral", r"femorotibial externo",
        r"lateraal femorotibiaal", r"laterale femorotibial",
        r"lateral femorotibial", r"lateralen kompartiment", r"aussere[sn]? kompartiment",
        r"lateral femorotibial", r"dis kompartman", r"lateral kompartman",
        r"lateraln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εξω διαμερισμα", r"εξω κνημιαι", r"εξω μηριαι",
        r"латералн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"lateral (femoral|tibial) (condyle|plateau)", r"condilo femoral lateral",
        r"lateralen? (femurkondyl|tibiaplateau)", r"laterale femorale condyl",
    ),
    "PF OA": _rx(
        r"patellofemoral", r"femoropatellar", r"femoropatelar", r"patelofemoral",
        r"retropatellar", r"retrorotulian", r"\btrochlea", r"\btroclea", r"\btroklea",
        r"\bpatella\b", r"\bpatellar\b", r"\brotulian", r"\brotula\b", r"\bpatele\b",
        r"\bpatellae?\b", r"patellofemoraal", r"femoropatellair",
        r"επιγονατιδ", r"μηροεπιγονατιδ", r"τροχιλ",
        r"пател", r"феморопател", r"тролх",
        r"anterior compartment", r"compartimento anterior", r"prednj[^ ]* odjeljk",
    ),
}

# Self-declaring findings: the term itself is the finding.
DIRECT = {
    "Effusion": _rx(
        r"\beffusion", r"joint fluid", r"intra ?articular fluid", r"\bhydrops\b",
        r"derrame articular", r"\bderrame\b", r"liquido articular",
        r"epanchement",
        r"gewrichtsvocht", r"\bvocht\b", r"\bhydrops\b", r"gewrichtseffusie",
        r"gelenkerguss", r"\berguss\b", r"gelenksergu",
        # "diz eklemi ici sivi miktari ... artmis" and "eklem icerisinde yaygin sivi
        # artisi" both occur; the noun takes a possessive suffix, so `eklem ` alone
        # misses. Match the stem plus any suffix.
        r"eklem\w* ic\w* sivi", r"efuzyon", r"eklem sivisi",
        r"sivi (miktari|artisi|birikimi)", r"sivi artis", r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev", r"\bizliv", r"zglobn[^ ]* tekucin", r"\bhidrops\b",
        r"αρθρικ[^ ]* υγρ", r"υγρου ενδαρθρικα", r"ενδαρθρικ[^ ]* υγρ", r"ποσοτητα υγρου",
        r"ενδαρθρικ", r"αρθρικη συλλογη", r"υγρο στην αρθρωση", r"υγρου στην αρθρωση",
        r"ставен излив", r"излив", r"ставна течност", r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
        r"synovitis", r"synoviale? (verdikking|proliferatie)",
        r"synovialitis", r"synovialis(verdickung|proliferation)",
        r"sinovijalitis", r"sinovitis", r"zadebljanje sinovij",
        r"υμενιτιδα", r"συνοβιτιδα", r"υμενικ[^ ]* υπερτροφ", r"αρθρικου υμεν",
        r"синовит", r"синовиал[^ ]* (задебел|пролифер)",
        r"verdikkingen van (het )?synovium", r"pannus",
    ),
    "Baker's": _rx(
        r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
        r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
        r"popliteal kist", r"\bbakerova\b", r"poplitealn[^ ]* cist",
        r"κυστη baker", r"πολυχωρη συνοβιακη κυστη", r"κυστη του baker",
        r"киста на бейкър", r"бейкърова киста", r"поплитеална киста",
        r"gastrocnemio ?semimembranos", r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion", r"bone bruise", r"bone marrow (o?edema|contusion)",
        r"\bkontuz", r"medular bone o?edema", r"marrow o?edema",
        r"contusion osea", r"edema oseo", r"edema de medula osea",
        r"oedeme osseux", r"contusion osseuse",
        r"botcontusie", r"botoedeem", r"beenmergoedeem", r"botmergoedeem",
        r"knochenmarkodem", r"knochenodem", r"kontusion", r"bone bruise",
        r"kemik kontuzyonu", r"kemik iligi odemi", r"kemik odemi",
        r"kostani edem", r"edem kosti", r"kontuzij",
        r"οστεομυελικ[^ ]* οιδημα", r"οστικο οιδημα", r"μυελικο οιδημα",
        r"костномозъчен едем", r"костен едем", r"контузионен",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfract\b",
        r"\bfractura", r"\bfracturas\b",
        r"\bfractuur", r"\bbreuk\b",
        r"\bfraktur", r"\bbruch\b",
        r"\bkirik\b", r"\bkirigi\b", r"\bkirik\b",
        r"\bfraktur", r"\bprijelom", r"impresijsk[^ ]* fraktur",
        r"καταγμα", r"καταγματ",
        r"фрактур", r"счупван", r"фисур",
        r"insufficiency fracture", r"stress fracture", r"avulsion fracture",
        r"subchondral fracture", r"subkondral kiri",
    ),
}

# Terms that look like a finding but are not the finding being scored.
DECOY = {
    "Fracture": _rx(r"no fracture", r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"ganglion"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = {"Medial OA", "Lateral OA", "PF OA"}


# ==== CELL ====

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
                    r"capraz bag\w*", r"krizn\w*", r"χιαστ\w*", r"кръстн\w*",
                    r"\bacl\b", r"\bpcl\b", r"\blca\b", r"\blcp\b", r"\bvkb\b",
                    r"\bhkb\b", r"\bocb\b", r"\bacb\b")
STEM_COLLATERAL = _rx(r"collateral\w*", r"colateral\w*", r"kollateral\w*",
                      r"collaterale\w*", r"kolateraln\w*", r"yan bag\w*",
                      r"πλαγι\w*", r"колатерал\w*", r"странич\w*",
                      r"innenband\w*", r"aussenband\w*", r"binnenband\w*",
                      r"\bmcl\b", r"\blcl\b", r"\blcm\b", r"\biyb\b")

SIDE_MEDIAL = _rx(r"\bmedial\w*", r"\bmedyal\w*", r"\bmedijaln\w*", r"\bmediaal\w*",
                  r"\bmediale\w*", r"\binterno\w*", r"\binterne\w*", r"\binnen\w*",
                  r"\bic\b", r"\bunutarnj\w*", r"\bεσω\w*", r"\bεσωτερικ\w*",
                  r"\bмедиал\w*", r"\bвътреш\w*", r"\btibial collateral\b",
                  r"\bbinnen\w*", r"\bmediaal\b")
SIDE_LATERAL = _rx(r"\blateral\w*", r"\bexterno\w*", r"\bexterne\w*", r"\bdis\b",
                   r"\blateraln\w*", r"\baussen\w*", r"\bbuiten\w*", r"\bεξω\w*",
                   r"\bεξωτερικ\w*", r"\bлатерал\w*", r"\bвъншн\w*",
                   r"\bfibular collateral\b", r"\bvanjsk\w*")
SIDE_ANTERIOR = _rx(r"\banterior\w*", r"\bant\b", r"\bon\b", r"\bprednj\w*",
                    r"\bvorder\w*", r"\bvoorste\b", r"\bπροσθι\w*", r"\bпредн\w*",
                    r"\banteriyor\w*", r"\bavant\b", r"\bant[eé]rieur\w*")

# Fracture is the target whose stem varies most across the corpus.
STEM_FRACTURE = _rx(r"fractur\w*", r"fraktur\w*", r"fractuur\w*", r"\bfract\b",
                    r"kiri[kgğ]\w*", r"prijelom\w*", r"lom kosti", r"\bbreuk\w*",
                    r"\bbruch\w*", r"καταγμα\w*", r"καταγματ\w*", r"фрактур\w*",
                    # NOT a bare `fissur\w*`: "fisuras condrales" and "full thickness
                    # fissures in the articular cartilage" describe cartilage, not bone.
                    # The stem has to be anchored to a bone word to mean a fracture.
                    r"счупван\w*", r"fisur\w* (osea|oseas|kost)", r"fissur\w* kost")

STEM_OA_COMPARTMENT = _rx(r"compartment\w*", r"compartimento\w*", r"compartiment\w*",
                          r"kompartman\w*", r"kompartiment\w*", r"odjelj\w*",
                          r"διαμερισμα\w*", r"компартм\w*", r"\bотдел\w*",
                          r"femorotibial\w*", r"femorotibiaal\w*", r"tibiofemoral\w*",
                          r"femoro tibial\w*", r"κνημιαι\w*", r"μηριαι\w*",
                          r"femoral condyl\w*", r"tibial plateau\w*",
                          r"condilo femoral", r"platillo tibial", r"tibiaplateau\w*",
                          r"femurkondyl\w*", r"femoralne? kondil\w*",
                          r"tibijaln\w* plato", r"femoral kondil\w*",
                          r"tibia plato", r"tibyal plato")


def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    """True if a stem match has a qualifier within `window` characters either side.

    Character windows rather than token windows, because word order differs: English
    puts the side before the noun, Greek and Bulgarian often after, and Turkish
    attaches it as a separate preceding adjective.
    """
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False


# concept -> (stem, side) pairs used in addition to the phrase lexicons above
STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
    "Medial OA": (STEM_OA_COMPARTMENT, SIDE_MEDIAL),
    "Lateral OA": (STEM_OA_COMPARTMENT, SIDE_LATERAL),
}


# ==== CELL ====

SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bmimimal\b", r"\bdiscrete\b", r"\bfocal\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\bminimal\b", r"\baz miktarda\b", r"\bsilik\b",
    r"\bmanja\b", r"\bmanji\b", r"\bblago\b", r"\bdiskretn", r"\bmalo\b",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ",
    r"\bминимал", r"\bлек", r"\bмалк", r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bmoderate\b", r"\bgross\b", r"\bsignificant\b", r"\babundant\b", r"\btense\b",
    r"\bmoderad", r"\bimportante\b", r"\bsevera?\b", r"\bmarcad", r"\bcuantios",
    r"\bbelirgin\b", r"\byaygin\b", r"\bileri\b", r"\bciddi\b", r"\bbol\b",
    r"\bopsezan\b", r"\bveliki\b", r"\bizrazit", r"\bznacajn", r"\bumjeren",
    r"\bausgepragt", r"\bdeutlich", r"\bmassiv", r"\bmassig", r"\bgross",
    r"\buitgebreid", r"\bgevorderd", r"\bveel\b", r"\bmatige?\b",
    r"\bμετρι", r"\bμεγαλ", r"\bεκτεταμεν", r"\bευμεγεθ", r"\bσοβαρ",
    r"\bголям", r"\bизразен", r"\bзначим", r"\bумерен", r"\bобилен",
)

# OA is often asserted for the whole joint rather than per compartment
# ("tricompartmental osteoarthritis", "gonarthrose", "incipient OA of all three
# compartments"). Those statements are evidence for all three OA targets.
GLOBAL_OA = _rx(
    r"tri ?compartment", r"all three compartment", r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros", r"\bgonartros", r"\bgonarthrose", r"\bgonartrose",
    r"osteoarthritis of the knee", r"artrosis (de |)(la )?rodilla", r"knee osteoarthrit",
    r"\bdiz osteoartrit", r"\bgonartroz", r"artroza koljena",
    r"οστεοαρθριτιδα", r"αρθριτιδα του γονατος",
    r"артроза на колянната", r"гонартроз",
    r"degenerative joint disease", r"\bdjd\b",
)

# A bare "bone marrow oedema" is not a contusion when it sits under a cartilage
# defect: subchondral oedema beneath a worn compartment is reactive degenerative signal,
# and reading it as a bruise turns every osteoarthritic knee into a trauma case.
DEGENERATIVE_MARROW = _rx(
    r"subchondral", r"subcondral", r"subkondral", r"supkondraln", r"subchondraln",
    r"υποχονδρι", r"субхондрал", r"subchondrale?",
    r"\bcyst", r"\bquist", r"\bzyste\b", r"\bcistic", r"reactive", r"reactivo",
)

TRAUMA = _rx(
    r"\bbruise\b", r"\bcontusion", r"\bkontuz", r"\bcontusion osea\b",
    r"\btrauma", r"\bimpaction\b", r"\bpivot shift\b", r"\bkissing\b",
    r"\bacute\b", r"\bagudo\b", r"\bakut", r"\bpivot kaymasi\b",
    r"\bcontusion osseuse\b", r"\bbone bruise\b", r"\bbotcontusie\b",
    r"\bконтузион", r"\bμωλωπ", r"\bkontuzij",
)


# ==== CELL ====

def _polarity(clause: str, anchor_end: int) -> str:
    """Classify one clause as positive, negative or uncertain for a matched term.

    Scope is the whole clause. Clause segmentation already keeps statements short, and
    a window in characters mis-scopes badly across languages with different word orders -
    Turkish puts its negator at the end of the sentence, English at the front.
    """
    if UNCERTAIN.search(clause):
        return "uncertain"
    if NEGATION.search(clause):
        return "negative"
    if NORMALITY.search(clause):
        # "meniscus normal" negates; "normal ... but tear" does not.
        if TEAR.search(clause) or re.search(r"\bgrade [34]\b", clause):
            return "positive"
        return "negative"
    return "positive"


class _Matcher:
    """Phrase lexicon first, stem+side proximity as the fallback.

    Exposes `.search` so it drops into the same slot as a compiled pattern.
    """

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


ANAT_MATCH = {
    tgt: _Matcher(ANAT[tgt], *STEM_RULES[tgt]) for tgt in PAIRED
}
COMPARTMENT_MATCH = {
    "Medial OA": _Matcher(COMPARTMENT["Medial OA"], *STEM_RULES["Medial OA"]),
    "Lateral OA": _Matcher(COMPARTMENT["Lateral OA"], *STEM_RULES["Lateral OA"]),
    "PF OA": _Matcher(COMPARTMENT["PF OA"]),
}
DIRECT_MATCH = {
    tgt: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if tgt == "Fracture" else rx)
    for tgt, rx in DIRECT.items()
}


def _severity(clause: str) -> float:
    """Weight one positive mention by how emphatic the sentence is.

    Ordered, not calibrated. A "moderate effusion" must outrank a "trace effusion" and
    both must outrank silence; the absolute numbers do not matter to AUC.
    """
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    return 0.75                       # unqualified mention


def _is_structure_heading(clause: str) -> bool:
    """A short line ending in a colon names a section; it asserts nothing by itself.

    clauses() emits both the heading joined to the value beneath it and the heading on its
    own. For the five targets whose anatomy word *is* the finding - effusion, synovitis,
    Baker's, contusion, fracture - the bare heading matches the term with no negation in
    scope, so a structured report reading `Fractures :` / `Aucune.` scored a confident
    positive off the heading while the joined clause correctly read the negation. Short
    headings are therefore skipped as assertions; the joined clause carries the meaning.
    """
    return clause.endswith(":") and len(clause.split()) <= 5


def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None,
                   context_bonus=None):
    """Accumulate graded evidence over clauses for one target.

    Returns (score, confidence, n_pos, n_neg). Positives are graded by severity and by
    optional context regexes; negatives only matter when nothing positive was found,
    because reports assert normality for every structure they check.
    """
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMALITY.search(c) and not NEGATION.search(c):
                n_neg += 1
            continue
        pol = _polarity(c, m.end())
        if pol == "positive" and _is_structure_heading(c):
            continue
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

    if n_pos or n_unc:
        # 0.52 .. 0.95, ordered by the strongest single mention, nudged by repetition.
        score = min(0.95, 0.50 + 0.42 * best + 0.03 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05          # silence sits above asserted-negative
    return score, conf, n_pos, n_neg


def extract(report: str) -> dict:
    """Extract twelve (score, confidence) pairs from one report."""
    cls = clauses(report)
    out = {}
    path_paired = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)

    for tgt in TARGETS:
        if tgt in PAIRED:
            s, c, npos, nneg = _score_clauses(cls, ANAT_MATCH[tgt], path_paired)
        elif tgt in OA_TARGETS:
            s, c, npos, nneg = _score_clauses(cls, COMPARTMENT_MATCH[tgt], OA_EVIDENCE)
        elif tgt == "Contusion":
            # Reactive subchondral oedema under a cartilage defect is osteoarthritis,
            # not a bruise. Explicit trauma wording pushes the other way.
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt),
                                              context_penalty=DEGENERATIVE_MARROW,
                                              context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    # --- cross-target corrections ------------------------------------------ #
    # A whole-joint osteoarthritis statement is evidence for every compartment that was
    # not separately assessed. Without this, "incipient OA of all three compartments"
    # scores zero on all three OA targets.
    g_hits = [c for c in cls if GLOBAL_OA.search(c) and _polarity(c, 0) == "positive"]
    if g_hits:
        gscore = 0.50 + 0.42 * max(_severity(c) for c in g_hits)
        for tgt in OA_TARGETS:
            if out[tgt + "__npos"] == 0 and out[tgt + "__nneg"] == 0:
                out[tgt] = max(out[tgt], gscore * 0.92)
                out[tgt + "__conf"] = max(out[tgt + "__conf"], 0.4)

    # Synovitis is frequently visible on the images and absent from the text, so silence
    # is weak evidence of absence here in a way it is not for other findings. Effusion is
    # its most reliable textual proxy - the two share a mechanism - so a silent synovitis
    # inherits a fraction of the effusion evidence instead of falling to the floor.
    if out["Synovitis__npos"] == 0 and out["Synovitis__nneg"] == 0:
        out["Synovitis"] = max(out["Synovitis"], 0.28 + 0.45 * (out["Effusion"] - 0.28))

    return out


# ==== CELL ====

# Additions are unioned onto the compiled patterns rather than edited into them, so the
# diff against the original lexicon stays readable and every entry can be reverted alone.

NEGATION = _rx(
    NEGATION.pattern,
    r"\bnone\b", r"\bnil\b",                       # "Effusion: none."
    r"not (identified|seen|visuali[sz]ed|demonstrated|detected|present|appreciated)",
    r"\bnegative\b",
    r"\bningun[ao]?\b", r"no se (identifica|aprecia|visualiza|reconoce)",
    r"\bnenhum\w*",
    r"niet zichtbaar", r"\bafwezigheid\b",
    r"izlenmemistir", r"gorulmemistir",
    # Rejected: r"\bnon\b". It negates "fracture non deplacee", which is a
    # non-displaced fracture - a positive finding with a qualifier, not an absence.
)

NORMALITY = _rx(
    NORMALITY.pattern,
    r"\bwnl\b", r"sans particularite", r"sin particularidades",
)

# A decoy skips the clause; a negation scores it. "no fracture" belongs to the second.
DECOY["Fracture"] = _rx(r"microfractur", r"\bfracture (risk|prophyla)")

_checks = {
    ("Fracture", "Fractures :\nAucune."): "lt",
    ("Fracture", "No fracture is identified."): "lt",
    ("Fracture", "Fractures :\nFracture non deplacee du plateau tibial."): "gt",
    ("Fracture", "Microfracture of the trochlea was performed."): "silent",
    ("Effusion", "Effusion:\nNone."): "lt",
    ("Effusion", "Joint effusion: nil."): "lt",
    ("Effusion", "Moderate joint effusion."): "gt",
    ("Fracture", "Fracturas :\nNinguna."): "lt",
    ("Baker's", "Quiste de Baker :\nPresente."): "gt",
}
for (tgt, text), want in _checks.items():
    v = extract(text)[tgt]
    ok = (v < 0.3) if want == "lt" else (v > 0.5) if want == "gt" else (0.25 < v < 0.32)
    assert ok, f"{want}: {tgt} scored {v:.2f} on {text!r}"
print("lexicon repairs verified on 9 regression cases")

# ==== CELL ====

# --- extractor self-test -------------------------------------------------------------
# Every claim the section above makes about the extractor is checked here on a synthetic
# report in the relevant language. Rerun after any lexicon edit: a rule that stops firing
# does not raise, it silently emits a negative.

def _s(text):
    return extract(text)

_cases = {
    "en_tear":     "Complete tear of the anterior cruciate ligament.",
    "en_intact":   "The anterior cruciate ligament is intact.",
    "es_group":    "Ligamentos cruzados y colaterales dentro de limites normales.",
    "fr_heading":  "Fractures :\nAucune.",
    "tr_effusion": "Eklem icerisinde yaygin sivi artisi mevcuttur.",
    "el_meniscus": "Ρηξη του εσω μηνισκου.",
    "eff_large":   "Large joint effusion.",
    "eff_trace":   "Trace joint effusion.",
    "eff_silent":  "The menisci are intact.",
    "global_oa":   "Tricompartmental osteoarthritis.",
    "degen_marrow": "Subchondral bone marrow oedema beneath a full thickness cartilage defect.",
    "true_bruise": "Bone bruise of the lateral femoral condyle after acute trauma.",
}
R = {k: _s(v) for k, v in _cases.items()}

assert R["en_tear"]["ACL"] > 0.5, R["en_tear"]["ACL"]
assert R["en_intact"]["ACL"] < 0.3, R["en_intact"]["ACL"]
# plural group statement clears both cruciates and both collaterals at once
assert R["es_group"]["ACL"] < 0.3 and R["es_group"]["MCL"] < 0.3, (
    R["es_group"]["ACL"], R["es_group"]["MCL"])
# heading on one line, value on the next: the merge in clauses() is what keeps this negative
assert R["fr_heading"]["Fracture"] < 0.3, R["fr_heading"]["Fracture"]
assert R["tr_effusion"]["Effusion"] > 0.5, R["tr_effusion"]["Effusion"]
assert R["el_meniscus"]["Medial Meniscus"] > 0.5, R["el_meniscus"]["Medial Meniscus"]
# severity is ordered, which is all AUC reads
assert R["eff_large"]["Effusion"] > R["eff_trace"]["Effusion"] > R["eff_silent"]["Effusion"], (
    R["eff_large"]["Effusion"], R["eff_trace"]["Effusion"], R["eff_silent"]["Effusion"])
# a whole-joint statement reaches every compartment
assert min(R["global_oa"][t] for t in ("Medial OA", "Lateral OA", "PF OA")) > 0.5
# reactive subchondral oedema must not read as a bruise as strongly as explicit trauma
assert R["degen_marrow"]["Contusion"] < R["true_bruise"]["Contusion"], (
    R["degen_marrow"]["Contusion"], R["true_bruise"]["Contusion"])

print("extractor self-test passed: negation, group statements, heading merge, Turkish and")
print("Greek morphology, severity ordering, global OA propagation, contusion decoy")

# ==== CELL ====

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

# The label extractor is defined in the cells above when this runs as a notebook. As a
# plain script it is imported from the package source, so the two paths share one
# definition rather than keeping a copy each.
if "extract" not in dir():
    import sys
    for _p in ("src", "../src", "../../src"):
        if (Path(_p) / "report_labeler.py").is_file():
            sys.path.insert(0, _p)
            break
    from report_labeler import extract  # noqa: F401

T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


IMG = 224                  # encoder input; a multiple of the patch size 14
CROP_MM = 160.0            # physical extent of the centre crop; a knee FOV is 140-180
GROUP = 3                  # slices per encoder input, stacked as the three channels
N_GROUP_MAX = 3            # groups per slot, before the memory budget is applied
CACHE_BUDGET_GB = 12.0     # ceiling for the training cache
HDR_THREADS = 16
PIX_THREADS = 12

EPOCHS = 12
BATCH_STUDIES = 8          # a study is a bag of up to N_SLOT slot images
LR_BACKBONE = 8e-6         # the encoder is being adapted, not retrained
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.02
UNFREEZE_LAST = 6          # trainable transformer blocks, counted from the output end
EVAL_BATCH = 12
TIME_BUDGET = 8.0 * 3600

# Six slots: three planes crossed with the acquisition axes. The fat-suppressed
# fluid-sensitive series exist for nearly every study; the T1 and the non-suppressed
# fluid-sensitive series are scarcer, which is what the presence mask is for.
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# The alternative: plane x the provided flag, ignoring the recovered weighting. Kept as
# a switch so the choice of slot definition can be varied while everything else is held
# fixed. Under this scheme a `Struct` slot mixes T1 series with non-fat-suppressed PD/T2
# series, which carry very different tissue contrast.
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


# ==== CELL ====

# --- v4 configuration ----------------------------------------------------------------
# Only the knobs this version adds or changes. Everything above keeps its original value,
# so setting all of these to their "off" state reproduces the previous pipeline exactly.

import math
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

N_FOLDS       = 4          # report-hash groups; folds are trained until the budget runs out
MAX_FOLDS     = "auto"     # "auto" fits as many as the time budget allows, or an int
FOLD_TIME_PAD = 1.15       # safety factor on the measured per-fold time

USE_VFLIP     = False      # was True; a knee is not vertically symmetric
AUG_AFFINE    = True
AUG_ROT_DEG   = 8.0
AUG_SCALE     = 0.08
AUG_SHIFT     = 0.05
AUG_INTENSITY = 0.10

EMA_DECAY     = 0.997      # 0 disables the moving average
RANK_LOSS_W   = 0.05       # 0 disables the pairwise term
RANK_POS, RANK_NEG = 0.60, 0.40      # graded-target cutoffs that define a usable pair

LAT_FALLBACK        = "auto"   # "auto" | "on" | "off": infer side from x when the tag is absent
LAT_MIN_AGREEMENT   = 0.85     # required agreement with the tag before "auto" enables it
LAT_MIN_OFFSET_MM   = 5.0      # |x| below this is not a usable side cue

SELECTION     = "worse_of_two"   # "worse_of_two" | "derived"
GOLD_WEIGHT   = 3.0              # weight of an annotated study relative to a derived one

# ------------------------------------------------------------------ icefire theme
ICEFIRE  = sns.color_palette("icefire", 12)
ICE      = ICEFIRE[2]
FIRE     = ICEFIRE[9]
MIDLINE  = "#39414d"
CMAP     = "icefire"
CMAP_SEQ = sns.blend_palette(["#eef1f6", ICE, FIRE], as_cmap=True)

sns.set_theme(style="white", palette=ICEFIRE, font_scale=0.95)
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c8cdd4", "axes.labelcolor": MIDLINE,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 10,
    "text.color": MIDLINE, "xtick.color": MIDLINE, "ytick.color": MIDLINE,
    "grid.color": "#e8eaee", "axes.grid": True, "axes.grid.axis": "y",
    "legend.frameon": False, "figure.facecolor": "white",
})


def grad(n, reverse=False):
    """n colours along the icefire ramp, skipping its near-black centre."""
    cm = plt.get_cmap(CMAP)
    cold = np.linspace(0.02, 0.28, n - n // 2)
    warm = np.linspace(0.78, 0.99, n // 2)
    cols = [cm(p) for p in np.concatenate([cold, warm])][:n]
    return cols[::-1] if reverse else cols


# `log` is defined in the next cell, so this one reports with print().
print(f"v4: folds={N_FOLDS} vflip={USE_VFLIP} ema={EMA_DECAY} rank_w={RANK_LOSS_W} "
      f"lat_fallback={LAT_FALLBACK}")

# ==== CELL ====

def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    # last resort: two-level scan, because the mount is nested one deeper than usual
    for depth1 in sorted(p for p in Path("/kaggle/input").iterdir() if p.is_dir()):
        for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
            if (cand / "test.csv").is_file():
                return cand
    raise FileNotFoundError("competition mount not found")


def find_dinov2(variant="small"):
    """Locate a mounted DINOv2 checkpoint directory by variant name."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None


ROOT = find_root()
log(f"input root: {ROOT}")


def plan_cache(n_study):
    """Choose how many slices per slot the memory budget allows.

    The cache is n_study x n_slot x slices x IMG^2 bytes. Coverage is the cheap axis -
    linear - and resolution the expensive one, so when the budget binds it is the slice
    count that gives way rather than the pixel grid. Deciding once, from the training
    corpus size, keeps train and test caches on the same group layout.
    """
    per_slice = n_study * N_SLOT * IMG * IMG
    afford = int(CACHE_BUDGET_GB * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    if groups < N_GROUP_MAX:
        log(f"cache budget {CACHE_BUDGET_GB:.0f} GB allows {groups} group(s) of {GROUP}, "
            f"not {N_GROUP_MAX}")
    return groups


N_GROUP = plan_cache(len(pd.read_csv(ROOT / "train.csv")))
CACHE_SLICES = GROUP * N_GROUP
log(f"cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot")


# ==== CELL ====

train_df_diag = pd.read_csv(ROOT / "train.csv")
_t = time.time()
lab_diag = pd.DataFrame([extract(r) for r in train_df_diag["Report"].fillna("")])
log(f"extracted {len(lab_diag)} reports in {time.time() - _t:.1f}s")

silence = pd.Series(
    {t: float(((lab_diag[t + "__npos"] == 0) & (lab_diag[t + "__nneg"] == 0)).mean())
     for t in TARGETS})

gold_diag = train_df_diag.set_index("StudyInstanceUID")[TARGETS]
gold_diag = gold_diag[gold_diag.notna().all(axis=1)]


def hanley_mcneil_se(auc, n_pos, n_neg):
    """Standard error of an AUC estimate; the reason the annotated subset cannot tune."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


from sklearn.metrics import roc_auc_score

lab_diag.index = train_df_diag["StudyInstanceUID"].values
rows = []
for t in TARGETS:
    y = gold_diag[t].astype(int).values
    p = lab_diag.loc[gold_diag.index, t].values
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
    se = hanley_mcneil_se(auc, int(y.sum()), int((1 - y).sum()))
    rows.append({"target": t, "agreement_auc": auc, "se": se,
                 "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                 "silence_rate": silence[t]})
diag = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
order = diag.sort_values("agreement_auc")
ax = axes[0]
ax.barh(order.target, order.agreement_auc, color=grad(12), edgecolor="white", height=.7)
ax.errorbar(order.agreement_auc, np.arange(12), xerr=1.96 * order.se, fmt="none",
            ecolor=MIDLINE, elinewidth=1.2, capsize=3)
ax.axvline(.5, color=MIDLINE, ls=":", lw=1)
ax.set_xlim(0, 1.05); ax.set_xlabel("AUC of the extracted score against the annotation")
ax.set_title(f"Agreement, with 95% Hanley-McNeil intervals (n={len(gold_diag)})")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)

ax = axes[1]
order2 = diag.sort_values("silence_rate", ascending=False)
ax.barh(order2.target, order2.silence_rate * 100, color=grad(12, reverse=True),
        edgecolor="white", height=.7)
ax.set_xlabel("% of reports where no rule fired at all")
ax.set_title("Silence rate — the gauge with enough sample to steer by")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()

display(diag.sort_values("silence_rate", ascending=False).round(3))
print(f"mean agreement AUC {diag.agreement_auc.mean():.3f}, "
      f"mean 95% half-width +-{1.96 * diag.se.mean():.3f}")
print("A target whose interval is wider than the gap you are chasing cannot arbitrate a "
      "lexicon change. Use the silence column for that.")

del train_df_diag, lab_diag, gold_diag
gc.collect()

# ==== CELL ====

HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "ImageLaterality",
            "ImagePositionPatient", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept"]


def probe(item):
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
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
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
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions must be matched as
    # exact tokens; a substring test on "SAT" fires on non-fat-sat series.
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df


# ==== CELL ====

# --- laterality resolution -----------------------------------------------------------
# The tag is authoritative where it exists. Where it does not, the patient x-coordinate
# can stand in, but only if it agrees with the tag on the studies that have both: a wrong
# flip is worse than no flip, and knee coils often place the joint near isocentre, where
# the sign carries no information at all.

def _tag_side(g):
    v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
    if "ImageLaterality" in g.columns:
        v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
    v = [x[0] for x in v if x and x[0] in ("L", "R")]
    return v[0] if v else None


def _position_side(g):
    xs = []
    for s in g.get("ImagePositionPatient", pd.Series(dtype=object)).dropna():
        try:
            xs.append(float(str(s).split("|")[0]))
        except Exception:
            pass
    if not xs:
        return None
    x = float(np.median(xs))
    if abs(x) < LAT_MIN_OFFSET_MM:
        return None                      # centred in the coil: the sign means nothing
    return "R" if x < 0 else "L"         # LPS: the right knee sits at negative x


def laterality_maps(h):
    """Return (side_by_study, diagnostics). Uses the fallback only if it earns it."""
    tag, pos = {}, {}
    for st, g in h.groupby("StudyInstanceUID"):
        tag[st] = _tag_side(g)
        pos[st] = _position_side(g)

    both = [st for st in tag if tag[st] and pos[st]]
    agree = float(np.mean([tag[st] == pos[st] for st in both])) if both else np.nan
    have_tag = float(np.mean([v is not None for v in tag.values()]))

    if LAT_FALLBACK == "on":
        use = True
    elif LAT_FALLBACK == "off":
        use = False
    else:
        use = bool(both) and np.isfinite(agree) and agree >= LAT_MIN_AGREEMENT

    side = {st: (tag[st] or (pos[st] if use else None)) for st in tag}
    covered = float(np.mean([v is not None for v in side.values()]))
    info = {"tag_coverage": have_tag, "agreement": agree, "n_compared": len(both),
            "fallback_used": use, "final_coverage": covered}
    log(f"laterality: tag on {have_tag:.1%} of studies, x-sign agrees with it on "
        f"{agree:.1%} of {len(both)} comparable studies, fallback "
        f"{'enabled' if use else 'disabled'} -> {covered:.1%} normalised")
    return side, info

# ==== CELL ====

def pick_slots(series_df, plane_map):
    """One series per slot per study.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, and the three-slice sampler below benefits from the margin.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None means "do not condition on weighting" - the public scheme,
            # where the single provided flag stands in for both axes at once.
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if len(cand) == 0 and fluid is False:
                # T1 slots are the scarcest; fall back to any non-fat-sat, non-fluid
                # series in the plane before giving up on the slot entirely.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


# ==== CELL ====

def read_slot(rec, n_slice=None, out_size=None):
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Returns float32 [n_slice, out, out] normalised per-series to its 1st-99th
    percentile. Percentiles rather than min/max because MR intensity has no absolute
    scale and a single bright vessel would otherwise compress the whole dynamic range.

    Reading is the expensive half of this pipeline, so the caller reads once at the
    largest configuration it needs and derives the smaller ones from the returned buffer
    rather than re-reading.
    """
    # was `N_SLICE`, which is never defined: a latent NameError on the default path.
    n_slice = CACHE_SLICES if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    # Spread the samples over the central 60% of the stack: the outer slices of a knee
    # series are mostly soft tissue outside the joint.
    lo, hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = np.zeros((out_size, out_size), dtype=np.float32)
        planes.append(a)

    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    # constant physical extent, then resize: PixelSpacing varies 3.4x across the corpus
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    # uint8, not float32. These buffers queue up between the reader threads and the
    # encoder, and at this size a float32 slot-series is several megabytes. Intensity is
    # already normalised into [0, 1] here, so eight bits cost nothing that a bilinear
    # resize has not already cost, and the queue is a quarter the size.
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


# ==== CELL ====

def normalise_laterality(img, plane, lat):
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks are not
    mirror images of each other - the slice order runs medial-to-lateral in opposite
    directions - so the channel order is reversed instead.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


# ==== CELL ====

def build_cache(slot_map, plane_map, lat_map, tag):
    """Decode every (study, slot) once into an in-memory uint8 array.

    Fine-tuning revisits the same pixels every epoch. Reading them from the mount each
    time would make the epoch count a function of I/O rather than of learning, so they
    are decoded once and held as bytes: intensity has already been normalised into
    [0, 1], and eight bits cost nothing a bilinear resize has not already cost.

    CACHE_SLICES positions are kept per slot, which the training loop reads as N_GROUP
    groups of GROUP consecutive channels.
    """
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    log(f"{tag}: decoding {len(jobs)} slot-series")

    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(
                    block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane,
                                                          lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    gc.collect()
    return studies, cache, mask


# ==== CELL ====

class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences - cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially - so pooling
    the slots identically would dilute the one that carries the evidence with the rest.

    The aggregation is deliberately this simple. Richer alternatives were measured
    against it and lost: with a study-level label there is no signal telling the model
    which part of a study matters, so extra attention parameters have nothing to learn
    from and spend their capacity fitting noise.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


# ==== CELL ====

class Model(nn.Module):
    """Encoder plus head, trained end to end.

    A study arrives as a bag of slot images. The bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.
    """

    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim, N_SLOT, len(TARGETS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1).reshape(B, S, -1)
        return self.head(feat, mask)


# ==== CELL ====

def build_model():
    """Load the encoder and open the last few blocks for training.

    Only the last UNFREEZE_LAST blocks and the final norm are trainable. The early
    blocks of a self-supervised transformer are generic edge and texture filters; there
    is not enough supervision here to improve them and quite enough to damage them.
    """
    from transformers import AutoModel
    p = find_dinov2("small")
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - UNFREEZE_LAST):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size * 2
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {UNFREEZE_LAST} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim}")
    return Model(bb, dim)


# ==== CELL ====

def take_group(cache_rows, g):
    """Slice GROUP consecutive channels out of the cached slices."""
    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]


# `augment` is defined further down, after the reasoning for dropping the flip.


@torch.no_grad()
def predict(model, cache, mask, idx, dev):
    """Average the logits over the groups of each slot.

    Training sees one group at a time, which acts as augmentation along the stack;
    inference averages over all of them, which is the same aggregation the frozen
    pipeline used and the one that measured best.
    """
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        rows = torch.from_numpy(cache[sel]).to(dev)
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(take_group(rows, g), m).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / N_GROUP).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.nanmean([roc_auc_score(y[:, j], p[:, j])
                             if len(set(y[:, j])) > 1 else np.nan
                             for j in range(y.shape[1])]))


# ==== CELL ====

def augment(imgs):
    """Small affine and intensity jitter over a whole bag. No flips of either kind.

    Horizontal flip would undo the laterality normalisation of section 5. Vertical flip
    would exchange the femoral and tibial surfaces, which is the axis the compartmental
    osteoarthritis targets are defined on.
    """
    B, S, C, H, W = imgs.shape
    x = imgs.float()

    if AUG_AFFINE:
        ang = math.radians((torch.rand(1).item() - 0.5) * 2 * AUG_ROT_DEG)
        sc = 1.0 + (torch.rand(1).item() - 0.5) * 2 * AUG_SCALE
        tx = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
        ty = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
        cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]],
                             dtype=torch.float32, device=imgs.device)
        theta = theta.unsqueeze(0).repeat(B * S, 1, 1)
        flat = x.reshape(B * S, C, H, W)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False).reshape(B, S, C, H, W)

    scale = 1.0 + (torch.rand(1, device=imgs.device) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)

    if USE_VFLIP and torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    return x.round().to(imgs.dtype)


class Ema:
    """Exponential moving average of the trainable weights.

    Buffers are copied rather than averaged: the normalisation constants are fixed, and
    averaging them would only accumulate floating point error.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        # Warm-up: without it the average stays anchored to the random initialisation for
        # the first few hundred steps, which is most of a short fold.
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
            else:
                v.copy_(s)

    def target(self, model):
        """The weights to evaluate and to keep: the average, or the model if disabled."""
        return self.model if self.decay > 0 else model


def rank_loss(logits, y, w):
    """Pairwise ranking term on confidently graded pairs inside the batch.

    AUC counts ordered positive-negative pairs, so this optimises the metric directly.
    A target with no usable pair in the batch contributes nothing rather than a constant.
    """
    parts = []
    usable = w > 0
    for j in range(logits.shape[1]):
        pos = logits[(y[:, j] > RANK_POS) & usable[:, j], j]
        neg = logits[(y[:, j] < RANK_NEG) & usable[:, j], j]
        if len(pos) and len(neg):
            parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
    return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)


# --- self-tests ----------------------------------------------------------------------
_t = torch.randint(0, 255, (2, 3, GROUP, 32, 32), dtype=torch.uint8)
_a = augment(_t)
assert _a.shape == _t.shape and _a.dtype == _t.dtype
# an asymmetric marker must not end up mirrored in either axis
_m = torch.zeros(1, 1, GROUP, 32, 32, dtype=torch.uint8)
_m[..., :6, :] = 255                                  # bright band along the top
_out = augment(_m).float()
assert _out[..., :16, :].mean() > _out[..., 16:, :].mean(), "vertical orientation was lost"
_m2 = torch.zeros(1, 1, GROUP, 32, 32, dtype=torch.uint8)
_m2[..., :6] = 255                                    # bright band on the left
_out2 = augment(_m2).float()
assert _out2[..., :16].mean() > _out2[..., 16:].mean(), "horizontal orientation was lost"

_lg = torch.tensor([[2.0], [-2.0]])
_y = torch.tensor([[1.0], [0.0]])
_w = torch.ones(2, 1)
assert rank_loss(_lg, _y, _w) < rank_loss(-_lg, _y, _w), "ranking loss has the wrong sign"
assert float(rank_loss(_lg, torch.tensor([[1.0], [1.0]]), _w)) == 0.0, "no pair, no loss"
print("augmentation preserves both anatomical axes; ranking loss is oriented correctly")

# ==== CELL ====

def write_benchmark_submission():
    """Write the 0.5 benchmark file immediately.

    A submission that never writes scores nothing at all. The try/except around main()
    covers exceptions, but a kill for memory is a SIGKILL and never reaches it, so a valid
    file exists from the first second and is overwritten only once predictions are ready.
    """
    t = pd.read_csv(ROOT / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)


def write_submission(rank_sum, n_models, st_te, test_df):
    """Rank-mean of the finished folds, aligned onto the test index."""
    P = rank_sum / max(n_models, 1)
    sub = pd.DataFrame(P, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", st_te)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv("submission.csv", index=False)
    return sub


def train_one_fold(fold, Ctr, Mtr, Y, W, tr, va, gi_va, gold_y_va, dev):
    """Train one fold and return (ema_state, history, val_predictions)."""
    model = build_model().to(dev)
    ema = Ema(model, EMA_DECAY)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    steps = max(EPOCHS * (len(tr) // BATCH_STUDIES), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    yv = (Y[va] > 0.5).astype(int)
    best, best_state, history = -1.0, None, []

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tot, nstep = 0.0, 0
        for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
            sel = perm[b:b + BATCH_STUDIES]
            rows = torch.from_numpy(Ctr[sel]).to(dev)
            g = int(torch.randint(N_GROUP, (1,)).item())
            imgs = augment(take_group(rows, g))
            m = torch.from_numpy(Mtr[sel]).to(dev)
            y = torch.from_numpy(Y[sel]).to(dev)
            w = torch.from_numpy(W[sel]).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(imgs, m)
                loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).mean()
                if RANK_LOSS_W > 0:
                    loss = loss + RANK_LOSS_W * rank_loss(z.float(), y, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item()
            nstep += 1

        eval_model = ema.target(model)
        pv = predict(eval_model, Ctr, Mtr, va, dev)
        d = macro_auc(yv, pv)

        # The annotated reference for this fold is only the annotated studies held out of
        # it. Scoring the ones it trained on is what made the previous rule optimistic.
        g_auc = float("nan")
        if gold_y_va is not None and len(gi_va) >= 6:
            g_auc = macro_auc(gold_y_va, predict(eval_model, Ctr, Mtr, gi_va, dev))

        history.append({"fold": fold, "epoch": ep + 1, "loss": tot / max(nstep, 1),
                        "derived": d, "annot": g_auc})
        log(f"fold {fold} epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}"
            f"  derived {d:.4f}  annot(held out) {g_auc:.4f}")

        if SELECTION == "derived" or not np.isfinite(g_auc):
            score = d
        else:
            score = min(d, g_auc)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            log(f"  best so far ({SELECTION} {score:.4f})")
        if time.time() - T0 > TIME_BUDGET:
            log("time budget reached inside fold")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in ema.target(model).state_dict().items()}
    del model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_state, history, best


def main():
    write_benchmark_submission()
    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    train_df = pd.read_csv(ROOT / "train.csv")
    train_series = pd.read_csv(ROOT / "train_series.csv")
    log(f"train {train_df.shape} test {test_df.shape}")

    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    log("header pass: test")
    hte = annotate(walk("test_series"))
    log(f"  {len(hte)} test series")
    log("header pass: train")
    htr = annotate(walk("train_series"))
    log(f"  {len(htr)} train series")

    lat_tr, lat_info_tr = laterality_maps(htr)
    lat_te, _ = laterality_maps(hte)
    globals()["LAT_INFO"] = lat_info_tr

    slots_te, slots_tr = pick_slots(hte, plane_map), pick_slots(htr, plane_map)
    cov = pd.Series([len(v) for v in slots_tr.values()]).describe()
    log(f"train slots per study: mean {cov['mean']:.2f} min {cov['min']:.0f} "
        f"max {cov['max']:.0f}")
    globals()["SLOT_COVERAGE"] = pd.Series(
        {name: float(np.mean([name in v for v in slots_tr.values()]))
         for name, *_ in SLOTS})
    globals()["WEIGHT_TABLE"] = htr.groupby(["weight", "fatsat"]).size()

    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_tr, "train")
    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_te, "test")

    # ---- targets ---------------------------------------------------------- #
    t_lab = time.time()
    lab = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    lab["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    lab = lab.set_index("StudyInstanceUID")
    log(f"derived labels for {len(lab)} studies in {time.time() - t_lab:.1f}s")

    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]

    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = gold.loc[st].values, GOLD_WEIGHT
        elif st in lab.index:
            r = lab.loc[st]
            Y[i] = r[TARGETS].values
            W[i] = 0.25 + 0.75 * r[[t + "__conf" for t in TARGETS]].values
    keep = np.where(W.sum(1) > 0)[0]
    log(f"supervised {len(keep)} of {len(st_tr)} studies (annotated {len(gold)})")

    # Folds are grouped on report text: some reports are byte-identical across studies and
    # yield one target vector for all of them, so splitting such a group would score a
    # model on a target whose source it had already trained on.
    import hashlib
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
                    for s in st_tr])
    gpos = {s: i for i, s in enumerate(st_tr)}
    gi_all = np.array([gpos[s] for s in gold.index if s in gpos])
    gold_y_all = (gold.loc[[st_tr[i] for i in gi_all]].values.astype(int)
                  if len(gi_all) else None)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank_sum = np.zeros((len(st_te), len(TARGETS)), np.float64)
    n_models, sub = 0, None
    oof_gold = np.full((len(st_tr), len(TARGETS)), np.nan, np.float32)
    histories, fold_scores = [], []

    max_folds = N_FOLDS if MAX_FOLDS == "auto" else int(MAX_FOLDS)
    for fold in range(min(N_FOLDS, max_folds)):
        va = np.array([i for i in keep if grp[i] == fold])
        tr = np.array([i for i in keep if grp[i] != fold])
        if len(va) == 0 or len(tr) < BATCH_STUDIES:
            log(f"fold {fold}: not enough studies, skipped")
            continue
        gi_va = np.array([i for i in gi_all if grp[i] == fold])
        gold_y_va = (gold.loc[[st_tr[i] for i in gi_va]].values.astype(int)
                     if len(gi_va) else None)
        log(f"=== fold {fold}: train {len(tr)} / holdout {len(va)} "
            f"(annotated held out: {len(gi_va)}) ===")

        t_fold = time.time()
        state, history, best = train_one_fold(fold, Ctr, Mtr, Y, W, tr, va,
                                              gi_va, gold_y_va, dev)
        fold_time = time.time() - t_fold
        histories.extend(history)
        fold_scores.append(best)

        model = build_model().to(dev)
        model.load_state_dict(state)
        if len(gi_va):
            oof_gold[gi_va] = predict(model, Ctr, Mtr, gi_va, dev)
        P = predict(model, Cte, Mte, np.arange(len(st_te)), dev)
        rank_sum += pd.DataFrame(P).rank(pct=True).values
        n_models += 1
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sub = write_submission(rank_sum, n_models, st_te, test_df)
        log(f"fold {fold} done in {fold_time / 60:.1f} min; submission has {n_models} model(s)")

        elapsed = time.time() - T0
        if elapsed + FOLD_TIME_PAD * fold_time > TIME_BUDGET:
            log(f"stopping after {n_models} fold(s): another would need "
                f"{FOLD_TIME_PAD * fold_time / 60:.0f} min and "
                f"{(TIME_BUDGET - elapsed) / 60:.0f} min remain")
            break

    globals()["HISTORY"] = pd.DataFrame(histories)
    if gold_y_all is not None and len(gi_all):
        seen = np.isfinite(oof_gold[gi_all]).all(axis=1)
        if seen.sum() >= 8:
            oof_auc = macro_auc(gold_y_all[seen], oof_gold[gi_all][seen])
            log(f"out-of-fold macro AUC on {int(seen.sum())} annotated studies: {oof_auc:.4f}")
            globals()["OOF_GOLD"] = (gold_y_all[seen], oof_gold[gi_all][seen])
    log(f"fold selection scores: {[round(s, 4) for s in fold_scores]}")
    if sub is None:
        # No fold finished. The benchmark file written at the top is still on disk, so the
        # run leaves a scoreable submission rather than nothing.
        log("no fold completed; the 0.5 benchmark submission stands")
        return
    log(f"submission.csv {sub.shape}; models {n_models}")
    print(sub.head().to_string())

# ==== CELL ====

try:
    main()
except Exception:
    traceback.print_exc()
    # A submission that fails to write scores nothing at all, so fall back to the
    # benchmark file rather than dying.
    t = pd.read_csv(find_root() / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)
    print("wrote fallback submission.csv")
log("done")


# ==== CELL ====

# --- what the run actually did -------------------------------------------------------
# Reads the globals main() left behind. Safe to run after a partial run; anything the run
# did not reach is simply skipped.

if "HISTORY" in globals() and len(HISTORY):
    h = HISTORY
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for k, f in enumerate(sorted(h.fold.unique())):
        sub_h = h[h.fold == f]
        axes[0].plot(sub_h.epoch, sub_h.derived, "o-", color=grad(max(len(h.fold.unique()), 3))[k],
                     label=f"fold {f}")
        axes[1].plot(sub_h.epoch, sub_h.annot, "o-", color=grad(max(len(h.fold.unique()), 3))[k],
                     label=f"fold {f}")
    axes[0].set_title("Derived-target holdout AUC"); axes[0].set_xlabel("epoch")
    axes[1].set_title("Annotated holdout AUC (held out of this fold)"); axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    print("The right panel is the honest one and it is noisy by construction: roughly a")
    print("dozen annotated studies per fold. It guards the selection; it does not drive it.")

if "OOF_GOLD" in globals():
    y, p = OOF_GOLD
    rows = []
    for j, t in enumerate(TARGETS):
        if len(set(y[:, j])) > 1:
            auc = roc_auc_score(y[:, j], p[:, j])
            se = hanley_mcneil_se(auc, int(y[:, j].sum()), int((1 - y[:, j]).sum()))
        else:
            auc, se = np.nan, np.nan
        rows.append({"target": t, "oof_auc": auc, "se": se, "n_pos": int(y[:, j].sum())})
    oof_table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    o = oof_table.sort_values("oof_auc")
    ax.barh(o.target, o.oof_auc, color=grad(12), edgecolor="white", height=.7)
    ax.errorbar(o.oof_auc, np.arange(len(o)), xerr=1.96 * o.se, fmt="none",
                ecolor=MIDLINE, elinewidth=1.2, capsize=3)
    ax.axvline(.5, color=MIDLINE, ls=":", lw=1)
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Out-of-fold AUC against the annotations, macro {oof_table.oof_auc.mean():.4f}")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    plt.tight_layout(); plt.show()
    display(oof_table.round(3))

if "SLOT_COVERAGE" in globals():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sc = SLOT_COVERAGE.sort_values(ascending=False)
    ax.bar(sc.index, sc.values * 100, color=grad(len(sc)), edgecolor="white", width=.6)
    ax.set_ylim(0, 112); ax.set_ylabel("% of studies"); ax.set_title("Slot availability")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(sc.values * 100):
        ax.text(i, v + 2.5, f"{v:.0f}", ha="center", fontsize=9, color=MIDLINE)
    plt.tight_layout(); plt.show()

if "WEIGHT_TABLE" in globals():
    print("series by recovered weighting and fat suppression:")
    display(WEIGHT_TABLE.unstack(fill_value=0))

if "LAT_INFO" in globals():
    print("laterality:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in LAT_INFO.items()})