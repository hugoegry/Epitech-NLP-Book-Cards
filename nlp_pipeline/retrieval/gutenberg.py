"""download texte gutenberg via url gutendex

utf8 puis us-ascii si le premier lien marche pas
"""

from __future__ import annotations

from nlp_pipeline import config
from nlp_pipeline.errors import BookNotFoundError
from nlp_pipeline.storage import cache
from nlp_pipeline.retrieval import gutendex, http_client
from nlp_pipeline.preprocessing import strip_gutenberg_boilerplate


def _download(urls: list[str]) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            status, text = http_client.get_text(url, headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
            if status == 200 and text:
                return text
        except http_client.HttpError as exc:
            last_error = exc

    raise BookNotFoundError("impossible de telecharger le texte du livre depuis gutenberg") from last_error


def download_text(urls: list[str]) -> str:
    """expose le dl dans gutenberg"""
    return _download(urls)

def get_clean_text(book_id: int, *, use_cache: bool = True) -> str:
    """texte clean"""
    cached = cache.load_text(book_id, use_cache=use_cache)
    if cached is not None:
        return cached

    index = gutendex.fetch(book_id, use_cache=use_cache)
    urls = gutendex.text_urls(index)
    if not urls:
        raise BookNotFoundError(f"aucun lien texte pour le livre #{book_id} dans l index gutendex")

    raw = _download(urls)

    clean = strip_gutenberg_boilerplate(raw)
    cache.save_text(book_id, clean)
    return clean