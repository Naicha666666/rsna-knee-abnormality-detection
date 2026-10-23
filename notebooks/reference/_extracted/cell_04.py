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
