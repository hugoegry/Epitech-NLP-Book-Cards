"""le but est de rassembler toutes les informations extraites précédemment dans un format structuré et synthétique."""

from __future__ import annotations

from typing import Any

from nlp_pipeline import books
from nlp_pipeline.retrieval import get_metadata
from nlp_pipeline.storage import cache
from nlp_pipeline.nlp import entities, summarize, lexdiv, similarity, topic
from nlp_pipeline.errors import (
    BookNotFoundError,
    PipelineError,
)

TASK_NAME = "card"


def compute(book_id: int, *, use_cache: bool = True) -> dict[str, Any]:
    """Construit la carte NLP complète d'un livre"""

    try:
        info = books.select(book_id)
        summary_text = summarize.compute(book_id, use_cache=use_cache)
        entities_info = entities.compute(book_id, use_cache=use_cache)
        lexdiv_info = lexdiv.compute(book_id, use_cache=use_cache)
        similarity_info = similarity.compute(book_id, use_cache=use_cache)
        topics_info = topic.compute(book_id, use_cache=use_cache)

    except Exception as e:
        raise PipelineError() from e

    if not info:
        raise BookNotFoundError()

    return {
        "info": info,
        "lexical_diversity": lexdiv_info,
        "topics": topics_info,
        "entities": entities_info,
        "summary": summary_text,
        "similarity": similarity_info,
    }
