"""acce api gutendex avec fb
principe cache - api - fb api local
"""

from __future__ import annotations

from nlp_pipeline import config
from nlp_pipeline.errors import BookNotFoundError, GutendexApiError
from nlp_pipeline.storage import cache
from nlp_pipeline.retrieval import http_client

INDEX_ACTION = "index"

class _UpstreamFailed(Exception):
    """switche si url ne rep pas on passe a la suivante"""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

def _book_url(template: str, book_id: int) -> str:
    return template.format(id=book_id)

def _fetch_remote(book_id: int, url_template: str) -> dict:
    """fetch gutendex recupere le json du livre"""
    url = _book_url(url_template, book_id)
    try:
        status, data = http_client.get_json(url, headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
    except http_client.HttpError as exc:
        raise _UpstreamFailed("impossible d atteindre l api") from exc

    if status == 404:
        raise BookNotFoundError(f"livre #{book_id} introuvable sur gutendex")
    if status != 200:
        raise _UpstreamFailed(f"reponse gutendex inattendue (statut {status})", status=status)
    return data


def _gutendex_sources() -> list[tuple[str, str]]:
    """liste url"""
    labels = ("public", "fallback")
    return [
        (url, labels[i] if i < len(labels) else f"alt{i}")
        for i, url in enumerate(config.GUTENDEX_URLS)
    ]


def fetch(book_id: int, *, use_cache: bool = True) -> dict:
    """recup les info du livre (cache puis api avec fb)"""
    cached = cache.load(INDEX_ACTION, book_id, use_cache=use_cache)
    if cached is not None:
        return cached

    errors: list[str] = []
    for url_template, label in _gutendex_sources():
        try:
            data = _fetch_remote(book_id, url_template)
            cache.save(INDEX_ACTION, book_id, data)
            return data
        except BookNotFoundError:
            raise
        except _UpstreamFailed as exc:
            errors.append(f"{label}: {exc}")
            continue

    detail = " ; ".join(errors) if errors else "aucune source configuree"
    raise GutendexApiError(f"impossible d acceder a gutendex {detail}")


def text_urls(data: dict) -> list[str]:
    """extract les liens depuis la rep gutendex"""
    formats = data.get("formats", {})
    urls: list[str] = []
    for key in (config.TEXT_FORMAT_UTF8, config.TEXT_FORMAT_ASCII):
        url = formats.get(key)
        if url:
            urls.append(url)
    return urls