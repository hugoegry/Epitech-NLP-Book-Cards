from __future__ import annotations

from typing import Any

from nlp_pipeline import config
from nlp_pipeline.errors import InvalidBookIdError
from nlp_pipeline.retrieval import get_clean_text, get_metadata


def select(book_id: int, *, use_cache: bool = True) -> dict[str, Any]:
    if book_id <= 0:
        raise InvalidBookIdError(f"identifiant invalide: {book_id}")

    config.ensure_directories()
    meta = get_metadata(book_id, use_cache=use_cache)
    get_clean_text(book_id, use_cache=use_cache)
    return meta