"""
bow simple top freq par chapitre
"""

from __future__ import annotations
from collections import Counter
from nlp_pipeline import config, preprocessing

from nlp_pipeline.errors import (
    EmptyDocumentError,
    NoTokensError,
    BowComputationError,
)

TOP_MOTS = config.TOPIC_TOP_WORDS


def extraire(
    tokens_ch: list[list[str]], *, n_mots: int = TOP_MOTS
) -> dict[int, list[str]]:
    res = {}

    try:
        for i, toks in enumerate(tokens_ch):

            if not toks:
                raise NoTokensError()

            cnt = Counter(toks)

            res[i + 1] = [mot for mot, _ in cnt.most_common(n_mots)]

    except Exception as e:
        raise BowComputationError() from e

    return res


def compute(book_id: int, *, use_cache: bool = True) -> dict[int, list[str]]:
    try:
        texte = preprocessing.get_clean_text(book_id, use_cache=use_cache)

        chapitres = preprocessing.chapters(texte)

        if not chapitres:
            raise EmptyDocumentError()

        tokens_par_chapitre = []

        for chapitre in chapitres:
            words = preprocessing.content_words(preprocessing.word_tokens(chapitre))

            if not words:
                raise NoTokensError()

            tokens_par_chapitre.append(words)

        return extraire(tokens_par_chapitre)
    except (EmptyDocumentError, NoTokensError, BowComputationError):
        raise
    except Exception as e:
        raise BowComputationError() from e
