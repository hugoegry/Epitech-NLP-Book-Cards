"""
ner spacy sm
"""

from __future__ import annotations

from functools import lru_cache

from nlp_pipeline import config
from nlp_pipeline.errors import DependencyError, ModelLoadingError
from nlp_pipeline.nlp.entities import common as c

TASK_NAME = "entities-spacy"


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
    except ImportError as exc:
        raise DependencyError("spacy not installed") from exc

    try:
        return spacy.load(
            config.SPACY_MODEL,
            disable=[
                "parser",
                "tagger",
                "lemmatizer",
                "attribute_ruler",
            ],
        )

    except OSError as exc:
        raise ModelLoadingError("spaCy model not found") from exc

    except Exception as exc:
        raise ModelLoadingError("spaCy initialization failed") from exc


def scan(texte: str, scores: c.Score) -> None:
    try:
        nlp = _nlp()
        c.scan_spacy(nlp, texte, scores)

    except Exception as exc:
        raise DependencyError("spaCy NER failed") from exc


def compute(book_id: int, *, use_cache: bool = True) -> dict[str, list[str]]:
    try:
        return c.run(
            TASK_NAME,
            book_id,
            scan,
            use_cache=use_cache,
        )
    except Exception as exc:
        raise DependencyError("spaCy entity extraction pipeline failed") from exc
