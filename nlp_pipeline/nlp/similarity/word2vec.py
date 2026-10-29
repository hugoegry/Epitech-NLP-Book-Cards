"""
similarite word2vec
"""

from __future__ import annotations

import math
from functools import lru_cache

from nlp_pipeline import config, preprocessing
from nlp_pipeline.storage import cache
from nlp_pipeline.retrieval import get_metadata
from gensim.models import Word2Vec

from nlp_pipeline.errors import (
    DependencyError,
    InvalidBookIdError,
    EmptyVectorError,
)

TASK_NAME = "similarity-word2vec"
EMBED_TASK = "embed-word2vec"

Vecteur = list[float]
Vecteurs = dict[int, Vecteur]


def _ids_corpus() -> list[int]:
    return sorted(config.BOOK_COLLECTION.keys())


def _titre(book_id: int) -> str:
    if book_id in config.BOOK_COLLECTION:
        return config.BOOK_COLLECTION[book_id]["title"]

    from nlp_pipeline.retrieval import get_metadata

    meta = get_metadata(book_id, use_cache=True)
    return str(meta.get("title", f"book-{book_id}"))


def _verif_id(book_id: int) -> None:
    if book_id not in config.BOOK_COLLECTION:
        raise InvalidBookIdError(book_id)


def _tokens_livre(book_id: int, *, use_cache: bool) -> list[str]:
    texte = preprocessing.get_clean_text(book_id, use_cache=use_cache)
    return preprocessing.content_words(preprocessing.word_tokens(texte))


def _cosine(v1: Vecteur, v2: Vecteur) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))

    if n1 == 0.0 or n2 == 0.0:
        return 0.0

    return dot / (n1 * n2)


def _classer(book_id: int, vecteurs: Vecteurs) -> list[str]:
    if book_id not in vecteurs:
        raise InvalidBookIdError(book_id)

    ref = vecteurs[book_id]
    scores: list[tuple[int, float]] = []

    for bid, vec in vecteurs.items():
        if bid == book_id:
            continue
        scores.append((bid, _cosine(ref, vec)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [_titre(bid) for bid, _ in scores[: config.SIMILAR_TOP_K]]


@lru_cache(maxsize=1)
def _modele():
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise DependencyError("gensim is not installed") from exc

    try:
        phrases = [_tokens_livre(bid, use_cache=True) for bid in _ids_corpus()]

        return Word2Vec(
            sentences=phrases,
            vector_size=config.WORD2VEC_VECTOR_SIZE,
            window=config.WORD2VEC_WINDOW,
            min_count=config.WORD2VEC_MIN_COUNT,
            workers=1,
            epochs=config.WORD2VEC_EPOCHS,
            seed=42,
        )
    except Exception as exc:
        raise DependencyError("Word2Vec model training failed") from exc


def _vecteur_livre(book_id: int, *, use_cache: bool) -> Vecteur:
    w2v = _modele()

    mots = _tokens_livre(book_id, use_cache=use_cache)

    vecs = [w2v.wv[m] for m in mots if m in w2v.wv]

    if not vecs:
        raise EmptyVectorError()

    dim = len(vecs[0])
    moy = [0.0] * dim

    for vec in vecs:
        for i, val in enumerate(vec):
            moy[i] += float(val)

    n = len(vecs)

    return [v / n for v in moy]


def _build_vecteurs(ids: list[int], *, use_cache: bool) -> Vecteurs:
    out: Vecteurs = {}

    for bid in ids:
        try:
            cached = cache.load(EMBED_TASK, bid, use_cache=use_cache)

            if cached is not None:
                out[bid] = list(cached)
                continue

            vec = _vecteur_livre(bid, use_cache=use_cache)
            cache.save(EMBED_TASK, bid, vec)
            out[bid] = vec
        except Exception as exc:
            raise EmptyVectorError(f"Failed to build vector for book {bid}") from exc

    return out


def compute(book_id: int, *, use_cache: bool = True) -> list[str]:

    _verif_id(book_id)

    try:
        cached = cache.load(TASK_NAME, book_id, use_cache=use_cache)
        if cached:
            return list(cached)

        vecteurs = _build_vecteurs(
            _ids_corpus(),
            use_cache=use_cache,
        )

        res = _classer(book_id, vecteurs)

        cache.save(TASK_NAME, book_id, res)

        return res
    except (InvalidBookIdError, EmptyVectorError):
        raise
    except Exception as exc:
        raise EmptyVectorError("Word2Vec similarity computation failed") from exc
