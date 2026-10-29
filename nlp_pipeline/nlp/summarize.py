"""Tâche ``--summarize`` : résumé automatique d'un livre."""

from nlp_pipeline import config, preprocessing

import networkx as nx

from nlp_pipeline.nlp.cosine_sim import cosine_similarity
from nlp_pipeline.nlp.tfidf import tfidf
from nlp_pipeline.errors import (
    EmptyVocabularyError,
    InvalidCosineMethodError,
    PageRankError,
    BookNotFoundError,
)

SUMMARY_SENTENCES = config.SUMMARY_SENTENCES
CHAPTER_SENTENCES = 1
TASK_NAME = "summarize"


def sentences_clear(text: str) -> list[str]:
    """Nettoie le texte et retourne la liste des phrases non vides."""
    return [
        sentence.strip()
        for sentence in preprocessing.sentences(text)
        if sentence.strip()
    ]


def _top_sentence_indexes(
    sentences: list[str],
    n_sentences: int,
    *,
    cosine_methode=config.COSINE_METHODE,
    tfidf_methode=config.TFIDF_METHODE
) -> list[int]:

    if not sentences:
        return []

    sentences = [s for s in sentences if len(s) > 20]

    if len(sentences) <= n_sentences:
        return list(range(len(sentences)))

    try:
        voc, mat = tfidf(sentences, methode=tfidf_methode, stop_words="english")

    except ValueError as e:
        raise EmptyVocabularyError() from e

    try:
        similarity = cosine_similarity(mat, methode=cosine_methode)

    except ValueError as e:
        raise InvalidCosineMethodError() from e

    similarity[similarity < 0.1] = 0

    try:
        graph = nx.from_numpy_array(similarity)
        pagerank_scores = nx.pagerank(graph)

    except Exception as e:
        raise PageRankError() from e

    scores = {}

    for i in range(len(sentences)):
        position_score = 1 - (i / len(sentences))
        scores[i] = pagerank_scores[i] * 0.7 + position_score * 0.3

    best_indexes = sorted(scores, key=scores.get, reverse=True)[:n_sentences]
    best_indexes.sort()

    return best_indexes


def summarize_chapter(
    text: str,
    n_sentences: int,
    *,
    cosine_methode=config.COSINE_METHODE,
    tfidf_methode=config.TFIDF_METHODE
) -> list[str]:

    sentences = sentences_clear(text)

    indexes = _top_sentence_indexes(
        sentences,
        n_sentences,
        cosine_methode=cosine_methode,
        tfidf_methode=tfidf_methode,
    )

    return [sentences[i] for i in indexes]


def summarize(
    text: str,
    *,
    n_sentences: int = SUMMARY_SENTENCES,
    cosine_methode=config.COSINE_METHODE,
    tfidf_methode=config.TFIDF_METHODE
) -> str:

    chapters_list = preprocessing.chapters(text)

    if not chapters_list:
        return ""

    chapter_key_sentences = []

    for chapter in chapters_list:
        chapter = chapter.strip()
        if not chapter:
            continue

        chapter_key_sentences.extend(
            summarize_chapter(
                chapter,
                n_sentences=CHAPTER_SENTENCES,
                cosine_methode=cosine_methode,
                tfidf_methode=tfidf_methode,
            )
        )

    if not chapter_key_sentences:
        return ""

    if len(chapter_key_sentences) <= n_sentences:
        return " ".join(chapter_key_sentences)

    indexes = _top_sentence_indexes(
        chapter_key_sentences,
        n_sentences,
        cosine_methode=cosine_methode,
        tfidf_methode=tfidf_methode,
    )

    return " ".join(chapter_key_sentences[i] for i in indexes)


def compute(
    book_id: int,
    *,
    use_cache: bool = True,
    cosine_methode=config.COSINE_METHODE,
    tfidf_methode=config.TFIDF_METHODE
) -> str:

    try:
        text = preprocessing.get_clean_text(book_id, use_cache=use_cache)

    except FileNotFoundError as e:
        raise BookNotFoundError() from e

    return summarize(
        text,
        cosine_methode=cosine_methode,
        tfidf_methode=tfidf_methode,
    )
