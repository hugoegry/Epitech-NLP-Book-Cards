"""
ner spacy md
"""

from __future__ import annotations

from functools import lru_cache

from nlp_pipeline import config
from nlp_pipeline.errors import DependencyError, ModelLoadingError
from nlp_pipeline.nlp.entities import common as c

TASK_NAME = "entities-spacy-md"


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
    except ImportError as exc:
        raise DependencyError("spacy not installed") from exc

    try:
        return spacy.load(
            config.SPACY_MODEL_MD,
            disable=[
                "parser",
                "tagger",
                "lemmatizer",
                "attribute_ruler",
            ],
        )

    except OSError as exc:
        raise ModelLoadingError("spaCy model MD not found") from exc

    except Exception as exc:
        raise ModelLoadingError("spaCy MD initialization failed") from exc


def scan(texte: str, scores: c.Score) -> None:
    try:
        nlp = _nlp()
        c.scan_spacy(nlp, texte, scores)

    except Exception as exc:
        raise DependencyError("spaCy MD NER failed") from exc


def compute(
    book_id: int,
    *,
    use_cache: bool = True,
) -> dict[str, list[str]]:

    return c.run(
        TASK_NAME,
        book_id,
        scan,
        use_cache=use_cache,
        min_mentions=config.ENTITY_MIN_MENTIONS_MD,
    )
