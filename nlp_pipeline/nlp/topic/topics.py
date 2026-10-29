"""
topics tfidf bow
"""

from __future__ import annotations

from nlp_pipeline.nlp.topic import topic_bow, topic_tfidf

from nlp_pipeline.errors import (
    InvalidMethodError,
    TopicComputationError,
)

TASK_NAME = "topics"

_METHODES = {
    "tfidf": topic_tfidf.compute,
    "bow": topic_bow.compute,
}


def compute(
    book_id: int,
    *,
    use_cache: bool = True,
    methode: str = "tfidf",
) -> dict[int, list[str]]:
    """
    Génère les topics d’un livre avec TF-IDF ou Bag of Words.
    """

    if methode not in _METHODES:
        raise InvalidMethodError(methode)

    fn = _METHODES[methode]

    try:
        result = fn(book_id, use_cache=use_cache)
    except Exception as e:
        raise TopicComputationError() from e

    return result
