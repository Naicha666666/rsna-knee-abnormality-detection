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
