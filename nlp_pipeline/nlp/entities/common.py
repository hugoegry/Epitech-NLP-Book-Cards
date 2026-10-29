"""util entities"""

from __future__ import annotations

import re
from collections import Counter

from nlp_pipeline import config, preprocessing
from nlp_pipeline.storage import cache
from nlp_pipeline.nlp.entities.nltk import pos_ok #hugo

# erreurs cohérentes avec ton projet
from nlp_pipeline.errors import (
    PreprocessingError,
    EmptyPredictionError,
    PipelineError,
)

Scores = dict[str, dict[str, int | str]]

LABELS_PERS = {"PERSON"}
LABELS_LIEU = {"GPE", "LOC"}


def clean(nom: str) -> str:
    try:
        nom = " ".join(nom.split())

        if nom.lower().startswith("the "):
            nom = nom[4:]

        for fin in ("'s", "\u2019s"):
            if nom.endswith(fin):
                nom = nom[: -len(fin)]

        nom = nom.strip(" \t\"'.,;:!?")

        if not nom or nom.lower() in ("n't", "nt") or ("'" in nom and len(nom) < 4):
            return ""

        return nom

    except Exception as e:
        raise PreprocessingError("Entity cleaning failed") from e


def _nom_valid(nom: str, texte: str, *, use_pos: bool) -> bool:
    try:
        if len(nom) < 2 or not nom[0].isupper() or nom.isupper():
            return False

        if use_pos and " " not in nom:
            from nlp_pipeline.nlp.entities.nltk import pos_ok

            if not pos_ok(nom):
                return False

        trouves = re.findall(re.escape(nom), texte, re.I)
        if not trouves:
            return False

        nb_maj = sum(1 for t in trouves if t[0].isupper())
        return nb_maj / len(trouves) >= config.ENTITY_MIN_CAP_RATIO

    except Exception as e:
        raise PreprocessingError("Entity validation failed") from e


def vote(scores: Scores, nom: str, categorie: str) -> None:
    try:
        nom = clean(nom)
        if not nom:
            return

        cle = nom.lower()

        if cle not in scores:
            scores[cle] = {"nom": nom, "pers": 0, "lieu": 0}
        elif len(nom) > len(str(scores[cle]["nom"])):
            scores[cle]["nom"] = nom

        scores[cle][categorie] += 1

    except Exception as e:
        raise PipelineError("Voting failed") from e


def scan_spacy(modele, texte: str, scores: Scores) -> None:
    try:
        taille = config.ENTITY_CHUNK_SIZE

        for debut in range(0, len(texte), taille):
            morceau = texte[debut : debut + taille]

            for doc in modele.pipe([morceau]):
                for ent in doc.ents:
                    if ent.label_ in LABELS_PERS:
                        vote(scores, ent.text, "pers")
                    elif ent.label_ in LABELS_LIEU:
                        vote(scores, ent.text, "lieu")

    except Exception as e:
        raise PipelineError("Spacy scan failed") from e


def classer(
    scores: Scores, texte: str, *, use_pos: bool = True
) -> tuple[Counter[str], Counter[str]]:
    try:
        ratio = config.ENTITY_VOTE_RATIO
        min_votes = config.ENTITY_MIN_VOTES

        personnages, lieux = Counter(), Counter()

        for entree in scores.values():
            nom = str(entree["nom"])

            if not _nom_valid(nom, texte, use_pos=use_pos):
                continue

            votes_pers, votes_lieu = int(entree["pers"]), int(entree["lieu"])

            if votes_pers + votes_lieu < min_votes:
                continue

            if votes_pers >= votes_lieu * ratio:
                personnages[nom] += votes_pers
            elif votes_lieu >= votes_pers * ratio:
                lieux[nom] += votes_lieu

        return personnages, lieux

    except Exception as e:
        raise PipelineError("Classification failed") from e


def rank(compteur: Counter[str], *, min_mentions: int | None = None) -> list[str]:
    try:
        seuil = config.ENTITY_MIN_MENTIONS if min_mentions is None else min_mentions

        freq, libelle = {}, {}

        for nom, nb in compteur.items():
            cle = nom.lower()
            freq[cle] = freq.get(cle, 0) + nb

            if cle not in libelle or nb >= compteur[libelle[cle]]:
                libelle[cle] = nom

        cles = sorted(
            (k for k, n in freq.items() if n >= seuil),
            key=lambda k: freq[k],
            reverse=True,
        )

        return [libelle[k] for k in cles[: config.ENTITY_MAX]]

    except Exception as e:
        raise PipelineError("Ranking failed") from e


def resultat(
    personnages: Counter[str], lieux: Counter[str], *, min_mentions: int | None = None
) -> dict[str, list[str]]:
    try:
        pers = preprocessing.enlever_sous_noms(
            rank(personnages, min_mentions=min_mentions)
        )

        pers_bas = {p.lower() for p in pers}

        loc = preprocessing.enlever_sous_noms(rank(lieux, min_mentions=min_mentions))

        loc = [
            l
            for l in loc
            if l.lower() not in pers_bas
            and not any(
                p.startswith(l.lower() + " ") or p.endswith(" " + l.lower())
                for p in pers_bas
            )
        ]

        return {"characters": pers, "locations": loc}

    except Exception as e:
        raise PipelineError("Result building failed") from e


def run(
    task: str, book_id: int, scan, *, use_cache=True, min_mentions=None, use_pos=True
) -> dict[str, list[str]]:
    try:
        if use_cache:
            cached = cache.load(task, book_id, use_cache=True)
            if cached:
                return cached

        texte = preprocessing.get_clean_text(book_id, use_cache=use_cache)

        if not texte:
            raise EmptyPredictionError()

        scores: Scores = {}

        scan(texte, scores)

        personnages, lieux = classer(scores, texte, use_pos=use_pos)

        res = resultat(personnages, lieux, min_mentions=min_mentions)

        cache.save(task, book_id, res)

        return res

    except Exception as e:
        raise PipelineError("Entity pipeline failed") from e


Score = Scores
