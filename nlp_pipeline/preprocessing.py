"""Prétraitement du texte : nettoyage, tokenisation, phrases.

Ce module fournit les briques de base partagées par TOUTES les tâches :
* enlever l'en-tête/pied de page Gutenberg
* découper le texte en mots
* découper le texte en phrases
* enlever les mots vides qui n'apportent pas de sens.

"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import nltk

from nlp_pipeline import config
from nlp_pipeline.errors import (
    EmptyTextError,
    InvalidChapterError,
    TokenizationError,
    NLTKResourceError,
    StopwordsError,
    BookNotFoundError,
)

# Téléchargement paresseux des données NLTK
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}


def _ensure_nltk() -> None:
    """S'assure que les données NLTK nécessaires sont téléchargées"""
    try:
        for package, path in _NLTK_RESOURCES.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)
    except Exception as e:
        raise NLTKResourceError() from e


# Nettoyage des boilerplate
_START_MARKER = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_END_MARKER = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Retire l'en-tête et le pied de page de Project Gutenberg."""
    start = _START_MARKER.search(text)
    end = _END_MARKER.search(text)
    begin = start.end() if start else 0
    finish = end.start() if end else len(text)
    if begin < finish:
        return text[begin:finish].strip()
    return text.strip()


# Tokenisation en mots
_WORD_PATTERN = re.compile(r"\b[a-z]{3,}\b")


def word_tokens(text: str) -> list[str]:
    """Tokenisation mots"""

    if not text or not text.strip():
        raise EmptyTextError()

    try:
        return _WORD_PATTERN.findall(text.lower())
    except Exception as e:
        raise TokenizationError() from e


def sentences(text: str) -> list[str]:
    """Découpe en phrases"""

    if not text or not text.strip():
        raise EmptyTextError()

    _ensure_nltk()

    try:
        flat = re.sub(r"\s+", " ", text).strip()
        return nltk.sent_tokenize(flat, language=config.LANGUAGE)
    except Exception as e:
        raise TokenizationError() from e


#
_MIN_SECTION_LENGTH = 400
_FRONT_MATTER_MARKERS = ("contents", "table of contents", "contenu")

_CHAPTER_SPLITTERS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:chapter|ch\.?|chapitre)\s+(?:[ivxlcdm]+|\d+)\.?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:chapter|ch\.?|chapitre)\s+(?:[ivxlcdm]+|\d+)\.?\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*part\s+"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|[ivxlcdm]+)"
        r"(?:\s*--|\s*$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*[IVXLCDM]+\.\s+[A-Z][A-Z \-\u2019'&,\\.]+$", re.MULTILINE),
    re.compile(r"^ [IVXLCDM]+\.\s*$", re.MULTILINE),
)


def _split_sections(text: str, pattern: re.Pattern[str]) -> list[str]:
    parts = [part.strip() for part in pattern.split(text) if part.strip()]
    return [part for part in parts if len(part) >= _MIN_SECTION_LENGTH]


def _drop_front_matter(sections: list[str]) -> list[str]:
    if not sections:
        return sections
    first = sections[0].lower()
    if len(sections[0]) < 1500 and any(
        marker in first for marker in _FRONT_MATTER_MARKERS
    ):
        return sections[1:]
    if len(sections[0]) < _MIN_SECTION_LENGTH and len(sections) > 1:
        return sections[1:]
    return sections


def _fallback_sections(text: str, target: int = 12) -> list[str]:
    blocks = sentences(text)
    if not blocks:
        stripped = text.strip()
        return [stripped] if stripped else []
    chunk_size = max(1, len(blocks) // target)
    grouped: list[str] = []
    for index in range(0, len(blocks), chunk_size):
        block = " ".join(blocks[index : index + chunk_size]).strip()
        if len(block) >= _MIN_SECTION_LENGTH:
            grouped.append(block)
    return grouped if grouped else [text.strip()]


def chapters(text: str) -> list[str]:
    """Découpe en chapitres"""

    if not text or not text.strip():
        raise EmptyTextError()

    try:
        for pattern in _CHAPTER_SPLITTERS:
            sections = _drop_front_matter(_split_sections(text, pattern))
            if len(sections) >= 2:
                return sections

        result = _fallback_sections(text)

        if not result:
            raise InvalidChapterError()

        return result

    except Exception as e:
        raise InvalidChapterError() from e


#


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """Stopwords anglais"""

    try:
        _ensure_nltk()
        from nltk.corpus import stopwords

        return frozenset(stopwords.words(config.LANGUAGE))
    except Exception as e:
        raise StopwordsError() from e


def content_words(tokens: list[str]) -> list[str]:
    """Garde uniquement les mots qui ont du sens (filtre stopwords)"""
    stop = _stopwords()
    return [t for t in tokens if t not in stop]


def get_clean_text(book_id: int, *, use_cache: bool = True) -> str:
    """Télécharge un livre nettoyé"""
    try:
        from nlp_pipeline.retrieval import get_clean_text as _get_clean_text

        return _get_clean_text(book_id, use_cache=use_cache)

    except Exception as e:
        raise BookNotFoundError() from e


# e
def fusionner_compteur(compteur: Counter[str]) -> Counter[str]:
    """fusionne variantes de nom ex Alice alice"""
    out: Counter[str] = Counter()
    ref: dict[str, str] = {}
    for nom, nb in compteur.items():
        cle = nom.strip().lower()
        if not cle:
            continue
        if cle not in ref:
            ref[cle] = nom.strip()
        out[ref[cle]] += nb
    return out


def trier_par_freq(compteur: Counter[str]) -> list[str]:
    """trie par freq decroissante puis alphabetic"""

    def r_tri(mot):
        return (-compteur[mot], mot.lower())

    return sorted(compteur.keys(), key=r_tri)


def enlever_sous_noms(noms: list[str]) -> list[str]:
    """vire rabbit si white rabbit existe"""
    out: list[str] = []
    for n in noms:
        nl = n.lower()
        if any(
            n != m and (m.lower().startswith(nl + " ") or m.lower().endswith(" " + nl))
            for m in noms
        ):
            continue
        out.append(n)
    return out
