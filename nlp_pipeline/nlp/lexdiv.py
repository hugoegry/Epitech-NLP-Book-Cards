"""
Tâche ``--lexdiv`` : mesures de diversité lexicale.

But :
Mesurer si un auteur utilise un vocabulaire **riche et varié** ou au contraire
**répétitif**. On compte les mots et on en tire quelques indicateurs

Vocabulaire
* **token** : nombre total de mots
* **type**  : un mot distinct

Les six métriques demandées sont :
* ``tok`` : nombre total de tokens
* ``typ`` : nombre de types (taille du vocabulaire)
* ``hap`` : nombre de *hapax* = mots apparaissant **une seule fois**
* ``ttr`` : *Type-Token Ratio* = ``typ / tok``. Proche de 1 = très varié
* ``mwl`` : *Mean Word Length* = nombre moyen de caractères par mot
* ``mwf`` : *Mean Word Frequency* = ``tok / typ`` = combien de fois, en moyenne, chaque mot distinct est utilisé
"""

from wsgiref import types

from nlp_pipeline import preprocessing

from nlp_pipeline.errors import EmptyTextError, NoTokensError, DivisionByZeroError


def calcMetrique(text: str) -> dict[str, float]:
    """
    Calcule les métriques lexicales :
    tok, typ, hap, ttr, mwl, mwf
    """

    if not text or not text.strip():
        raise EmptyTextError()

    tokens = preprocessing.word_tokens(text)

    if not tokens:
        raise NoTokensError()

    types_set = set(tokens)

    # hapax
    hapax = []
    for t in types_set:
        if tokens.count(t) == 1:
            hapax.append(t)

    tok = len(tokens)
    typ = len(types_set)
    hap = len(hapax)
    if tok == 0 or typ == 0:
        raise DivisionByZeroError()
    ttr = typ / tok
    mwl = sum(len(t) for t in tokens) / tok
    mwf = tok / typ

    return {
        "tok": tok,
        "typ": typ,
        "hap": hap,
        "ttr": ttr,
        "mwl": mwl,
        "mwf": mwf,
    }


def compute(id: int, *, use_cache: bool = True) -> dict[str, float]:
    """
    Charge un texte et calcule la diversité lexicale.
    """

    text = preprocessing.get_clean_text(id, use_cache=use_cache)

    return calcMetrique(text)
