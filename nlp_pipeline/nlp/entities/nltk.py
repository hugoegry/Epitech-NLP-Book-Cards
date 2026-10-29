"""ner nltk ne_chunk"""

from __future__ import annotations

from functools import lru_cache

from nlp_pipeline import preprocessing
from nlp_pipeline.nlp.entities import common as c
from nlp_pipeline.errors import DependencyError

TASK_NAME = "entities-nltk"

_PERS, _LIEU = {"PERSON"}, {"GPE", "GSP", "LOCATION", "FACILITY"}


_POS_REJET = {
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "RB",
    "WRB",
    "WP",
    "WDT",
    "UH",
    "MD",
    "CC",
    "IN",
} # pos a ignorer

_BATCH = 80


@lru_cache(maxsize=1)
def _tools():
    try:
        from nltk import pos_tag, word_tokenize
        from nltk.chunk import ne_chunk

        return ne_chunk, pos_tag, word_tokenize

    except Exception as exc:
        raise DependencyError("nltk not available") from exc


def pos_ok(mot: str) -> bool:
    _, pos_tag, word_tokenize = _tools()

    try:
        tag = pos_tag(word_tokenize(mot.lower()))[0][1]
    except Exception:
        return False

    if tag in _POS_REJET:
        return False

    return not (tag.startswith("JJ") and mot and mot[0].isupper())


def scan(texte: str, scores: c.Score) -> None:
    chunker, pos_tag, word_tokenize = _tools()

    phrases = preprocessing.sentences(texte)

    buf = []

    for phrase in phrases:
        tokens = word_tokenize(phrase)

        if len(tokens) < 2:
            continue

        buf.append(tokens)

        if len(buf) >= _BATCH:
            _lot(chunker, pos_tag, buf, scores)
            buf = []

    if buf:
        _lot(chunker, pos_tag, buf, scores)


def _lot(chunker, pos_tag, buf, scores) -> None:
    from nltk.tag import pos_tag_sents

    try:
        tagged = pos_tag_sents(buf)
        trees = chunker(tagged)

    except Exception:
        return

    for arbre in trees:
        for node in arbre:
            if not hasattr(node, "label"):
                continue

            label = node.label()
            nom = " ".join(w for w, _ in node.leaves())

            if label in _PERS:
                c.vote(scores, nom, "pers")

            elif label in _LIEU:
                c.vote(scores, nom, "lieu")


def compute(book_id: int, *, use_cache: bool = True) -> dict[str, list[str]]:
    return c.run(
        TASK_NAME,
        book_id,
        scan,
        use_cache=use_cache,
    )
