"""similarite tf-idf"""

from __future__ import annotations

from functools import lru_cache

from nlp_pipeline import config, preprocessing
from nlp_pipeline.nlp.tfidf import tfidf
from nlp_pipeline.nlp.cosine_sim import cosine_similarity
from nlp_pipeline.errors import (
    InvalidBookIdError,
    SimilarityMatrixError,
    SimilarityComputationError,
)
from nlp_pipeline.storage import cache
from nlp_pipeline.nlp.topic.topic_tfidf import _params_vec
from nlp_pipeline.retrieval import get_metadata

TASK_NAME = "similarity-tfidf"


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
        raise InvalidBookIdError(f"id {book_id} pas dans la liste")


def _doc_livre(book_id: int, *, use_cache: bool) -> str:
    """meme prep que topic_tfidf mots nettoyes en string"""
    texte = preprocessing.get_clean_text(book_id, use_cache=use_cache)
    words = preprocessing.content_words(preprocessing.word_tokens(texte))
    return " ".join(words)


@lru_cache(maxsize=1)
def _matrice(tfidf_methode: str):
    try:
        ids = _ids_corpus()

        docs = [_doc_livre(bid, use_cache=True) for bid in ids]

        min_df, max_df = _params_vec(len(docs))

        _, mat = tfidf(
            docs,
            methode=tfidf_methode,
            max_features=5000,
            min_df=min_df,
            max_df=max_df,
        )

        return ids, mat

    except Exception as e:
        raise SimilarityMatrixError() from e


def compute(
    book_id: int,
    *,
    use_cache: bool = True,
    cosine_methode=config.COSINE_METHODE,
    tfidf_methode=config.TFIDF_METHODE,
) -> list[str]:

    _verif_id(book_id)

    if use_cache and (cached := cache.load(TASK_NAME, book_id, use_cache=True)):
        return list(cached)

    try:
        ids, mat = _matrice(tfidf_methode)

        idx = ids.index(book_id)

        sims = cosine_similarity(
            mat[idx],
            mat,
            methode=cosine_methode,
        ).flatten()

    except Exception as e:
        raise SimilarityComputationError() from e

    paires = [(bid, sims[i]) for i, bid in enumerate(ids) if bid != book_id]

    paires.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    res = [_titre(bid) for bid, _ in paires[: config.SIMILAR_TOP_K]]

    cache.save(TASK_NAME, book_id, res)

    return res
