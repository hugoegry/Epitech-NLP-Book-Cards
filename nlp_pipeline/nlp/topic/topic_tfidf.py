"""
tf-idf par chapitre (sklearn + perso)
"""

from __future__ import annotations

from nlp_pipeline import config, preprocessing
from nlp_pipeline.nlp.tfidf import decoupe_chapitres, tfidf, top_mots_par_doc

from nlp_pipeline.errors import (
    EmptyDocumentError,
    VectorizerError,
    EmptyVocabularyError,
)

TOP_MOTS = config.TOPIC_TOP_WORDS


def _top_depuis_scores(scores, vocabulaire: list[str], n: int) -> list[str]:
    idx = scores.argsort()[::-1]
    mots = []

    for i in idx:
        if scores[i] <= 0:
            break
        mots.append(vocabulaire[i])
        if len(mots) >= n:
            break

    return mots


def _params_vec(n: int) -> tuple[int, float]:
    if n <= 1:
        return 1, 1.0

    min_df = 2 if n >= 6 else 1
    max_df = 1.0

    if n > min_df:
        max_df = min(0.95, max(0.85, (min_df + 0.5) / n))

    return min_df, max_df


def extraire(
    docs: list[str],
    *,
    n_mots: int = TOP_MOTS,
    tfidf_methode: str = config.TFIDF_METHODE,
) -> dict[int, list[str]]:

    if not docs:
        raise EmptyDocumentError()

    if tfidf_methode == "perso":
        try:
            return top_mots_par_doc(
                docs,
                stop_words="english",
                n_mots=n_mots,
            )
        except Exception as e:
            raise VectorizerError() from e

    try:
        n = len(docs)
        min_df, max_df = _params_vec(n)

        voc, mat = tfidf(
            docs,
            methode="sklearn",
            max_features=5000,
            min_df=min_df,
            max_df=max_df,
        )

    except ValueError as e:
        raise EmptyVocabularyError() from e

    except Exception as e:
        raise VectorizerError() from e

    res = {}

    for i in range(mat.shape[0]):
        row = mat[i]

        try:
            sc = row.toarray().flatten() if hasattr(row, "toarray") else row.flatten()
        except Exception as e:
            raise VectorizerError() from e

        res[i + 1] = _top_depuis_scores(sc, voc, n_mots)

    return res


def compute(book_id: int, *, use_cache: bool = True) -> dict[int, list[str]]:

    texte = preprocessing.get_clean_text(book_id, use_cache=use_cache)

    tfidf_methode = config.TFIDF_METHODE

    if tfidf_methode == "perso":
        try:
            chapitres = decoupe_chapitres(texte)

            if not chapitres:
                raise EmptyDocumentError()

            return top_mots_par_doc(
                chapitres,
                stop_words="english",
                n_mots=TOP_MOTS,
            )

        except Exception as e:
            raise VectorizerError() from e

    chapitres = preprocessing.chapters(texte)

    if not chapitres:
        raise EmptyDocumentError()

    docs = []

    for chapitre in chapitres:
        words = preprocessing.content_words(preprocessing.word_tokens(chapitre))
        docs.append(" ".join(words))

    return extraire(
        docs,
        tfidf_methode="sklearn",
    )
