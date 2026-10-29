"""similarite entre livres - word2vec, sbert ou tfidf"""

from __future__ import annotations

from nlp_pipeline import config
from nlp_pipeline.nlp.similarity import eval, sbert, tfidf, word2vec
from nlp_pipeline.errors import InvalidMethodError

TASK_NAME = sbert.TASK_NAME

_METHODES = {
    "word2vec": word2vec.compute,
    "sbert": sbert.compute,
    "tfidf": tfidf.compute,
}


def compute(book_id: int, *, use_cache: bool = True, methode: str = "word2vec", cosine_methode=config.COSINE_METHODE, tfidf_methode=config.TFIDF_METHODE) -> list[str]:

    if methode == "tfidf":
        return tfidf.compute(book_id, use_cache=use_cache, cosine_methode=cosine_methode, tfidf_methode=tfidf_methode)
    elif methode == "sbert":
        return sbert.compute(book_id, use_cache=use_cache)
    elif methode == "word2vec":
        return word2vec.compute(book_id, use_cache=use_cache)
    else:
        from nlp_pipeline.errors import InvalidMethodError
        raise InvalidMethodError(f"Unknown similarity method: {methode}")


__all__ = ["compute", "TASK_NAME", "word2vec", "sbert", "tfidf", "eval"]
