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