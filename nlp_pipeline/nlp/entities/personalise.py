"""ner perso regex"""  # stdlib re

from __future__ import annotations

import re
from collections import Counter

from nlp_pipeline.nlp.entities import common as c
from nlp_pipeline.errors import DependencyError

TASK_NAME = "entities-personalise"

# lieu apres in at to from etc (ex in Paris)
_LOC = re.compile(
    r"\b(?:in|at|to|from|into|through|near|across|around|down|over|past)\s+"
    r"(?:the\s+)?([A-Z][A-Za-z'\-]+(?:\s+(?:of|the|and|-)\s*[A-Z]?[A-Za-z'\-]+){0,4})"
)

# perso apres said asked cried etc (ex said Alice)
_PERS = re.compile(
    r"\b(?:said|cried|asked|replied|whispered|muttered|shouted|thought|continued|remarked)\s+"
    r"(?:the\s+)?([A-Z][A-Za-z'\-]+(?:\s+(?:of|the|and)\s+[A-Z][A-Za-z'\-]+|\s+[A-Z][A-Za-z'\-]+){0,4})"
)

# perso via possessif (ex Alice's associe a Alice)
_POSS = re.compile(r"\b([A-Z][a-z]+)(?:'s|\u2019s)\b")

# gere les nom composé utilise les freq > 2 (ex: "White Rabbit")
_CAP = re.compile(
    r"\b([A-Z][a-z]+(?:'[a-z]+)?(?:-[A-Z][a-z]+)*"
    r"(?:\s+(?:[A-Z][a-z]+|of|the|and|-)){0,3}"
    r"[A-Z][a-z]+)\b"
)


def scan(texte: str, scores: c.Score) -> None:
    try:
        for m in _LOC.finditer(texte):
            c.vote(scores, m.group(1), "lieu")

        for m in _PERS.finditer(texte):
            c.vote(scores, m.group(1), "pers")

        for m in _POSS.finditer(texte):
            # correction importante: on enlève le 's
            c.vote(scores, m.group(1), "pers")

        freq: Counter[str] = Counter()

        for m in _CAP.finditer(texte):
            nom = c.clean(m.group(1))
            if nom and len(nom) > 2:
                freq[nom] += 1

        for nom, nb in freq.items():
            if nb >= 2:
                c.vote(scores, nom, "pers")  # vote si repete plus de 2 fois

    except Exception as exc:
        raise DependencyError("regex NER failed") from exc


def compute(book_id: int, *, use_cache: bool = True) -> dict[str, list[str]]:
    return c.run(
        TASK_NAME,
        book_id,
        scan,
        use_cache=use_cache,
        min_mentions=2,
        use_pos=False,
    )
