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