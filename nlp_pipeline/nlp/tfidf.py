"""tf-idf"""

from __future__ import annotations

import math
from collections import Counter

import numpy

from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pipeline import config, preprocessing
from nlp_pipeline.errors import EmptyVocabularyError, VectorizerError


def _stop_set(stop_words):
    """recup stop mots depuis prep ou liste perso"""
    if stop_words is None:
        return None
    if stop_words == "english":
        return preprocessing._stopwords()
    return frozenset(stop_words)


def _doc_tokens(doc: str, stop_words) -> list[str]:
    """tokenize un doc avec word_tokens + filtre stop"""
    tokens = preprocessing.word_tokens(doc)
    if stop_words is None:
        return tokens
    if stop_words == "english":
        return preprocessing.content_words(tokens)
    stop = _stop_set(stop_words)
    return [t for t in tokens if t not in stop]


def _tfidf(docs, max_features, min_df, max_df, stop_words):
    """tfidf perso"""
    if not docs:
        raise VectorizerError("Empty document list")

    tokens_docs = [_doc_tokens(d, stop_words) for d in docs]
    n_docs = len(tokens_docs)

    df = Counter(word for tokens in tokens_docs for word in set(tokens))

    if not df:
        raise EmptyVocabularyError("No vocabulary extracted")

    vocab = sorted(df)
    idx = {word: i for i, word in enumerate(vocab)}

    mat = numpy.zeros((n_docs, len(vocab)))

    for i, tokens in enumerate(tokens_docs):
        if not tokens:
            continue

        total = len(tokens)
        tf = Counter(tokens)

        for word, count in tf.items():
            if word not in idx:
                continue
            # smoothing IDF (important)
            idf = math.log((n_docs + 1) / (df[word] + 1)) + 1
            mat[i, idx[word]] = (count / total) * idf

    return vocab, mat


def decoupe_chapitres(texte: str) -> list[str]:
    return preprocessing.chapters(texte)


def top_mots_par_doc(docs, stop_words="english", n_mots=10):
    vocab, mat = _tfidf(docs, None, None, None, stop_words)
    res = {}
    for i in range(len(docs)):
        row = mat[i]
        mots = [vocab[j] for j in row.argsort()[::-1] if row[j] > 0][:n_mots]
        if mots:
            res[i + 1] = mots
    return res


def _tfidf_sklearn(docs, max_features, min_df, max_df, stop_words):
    """tfidf sklearn"""
    try:
        vec = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df, stop_words=stop_words
        )
        mat = vec.fit_transform(docs)
        vocab = vec.get_feature_names_out().tolist()
        return vocab, mat
    except ValueError as e:
        raise EmptyVocabularyError("sklearn vectorizer failed") from e
    except Exception as e:
        raise VectorizerError("sklearn transformation failed") from e


def tfidf(
    docs,
    *,
    methode=config.TFIDF_METHODE,
    max_features=5000,
    min_df=1,
    max_df=1.0,
    stop_words=None,
):
    """eguilage sklearn ou notr metode, retourne (vocab, mat)"""
    try:
        if methode == "perso":
            return _tfidf(docs, max_features, min_df, max_df, stop_words)
        elif methode == "sklearn":
            return _tfidf_sklearn(docs, max_features, min_df, max_df, stop_words)
        else:
            raise ValueError(f"Unknown TF-IDF method: {methode}")
    except (EmptyVocabularyError, VectorizerError):
        raise
    except ValueError as e:
        raise EmptyVocabularyError(f"TF-IDF failed: {str(e)}") from e
    except Exception as e:
        raise VectorizerError("TF-IDF computation failed") from e
