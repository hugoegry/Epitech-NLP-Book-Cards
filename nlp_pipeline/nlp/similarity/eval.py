"""
eval similarite alice
"""

from __future__ import annotations

from nlp_pipeline.errors import EmptyPredictionError

ALICE_REF = [
    "The Secret Garden",
    "Peter Pan",
    "Poirot Investigates",
    "The murder of Roger Ackroyd",
    "Through the Looking-Glass",
    "The Jungle Book",
]


def compare(
    got: list[str],
    ref: list[str] | None = None,
) -> dict[str, float]:

    if not got:
        raise EmptyPredictionError()

    exp = {x.strip().lower() for x in (ref or ALICE_REF)}

    pred = {x.strip().lower() for x in got}

    tp = pred & exp

    prec = len(tp) / len(pred) if pred else 0.0
    rec = len(tp) / len(exp) if exp else 0.0

    if prec + rec:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    return {
        "prec": round(prec, 3),
        "rec": round(rec, 3),
        "score_equilibre": round(f1, 3),
    }
