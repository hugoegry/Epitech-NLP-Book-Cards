"""evaluation entities alice"""

from nlp_pipeline.errors import EmptyPredictionError  # déjà dans ton système

ALICE_REF: dict[str, list[str]] = {
    "characters": [
        "Alice",
        "White Rabbit",
        "Mouse",
        "Dodo",
        "Duck",
        "Lory",
        "Eaglet",
        "Bill",
        "Pat",
        "Puppy",
        "Caterpillar",
        "Duchess",
        "Cook",
        "Cheshire Cat",
        "Hatter",
        "March Hare",
        "Dormouse",
        "Queen of Hearts",
        "King of Hearts",
        "Knave of Hearts",
        "Frog-Footman",
        "Fish-Footman",
        "Alice's sister",
        "Dinah",
        "Mabel",
        "William the Conqueror",
        "Edwin",
        "Morcar",
    ],
    "locations": [
        "Rabbit-Hole",
        "Hall of Doors",
        "Pool of Tears",
        "White Rabbit's House",
        "Caterpillar's Mushroom",
        "Duchess's House",
        "Mad Tea-Party",
        "Queen's Croquet-Ground",
        "Court",
        "River-bank",
        "London",
        "Paris",
        "Rome",
        "Sea",
        "Wonderland",
    ],
}


def _norm(items: list[str]) -> set[str]:
    try:
        if not items:
            return set()

        return {x.strip().lower() for x in items if isinstance(x, str) and x.strip()}

    except Exception as e:
        raise ValueError("Normalization failed in entity evaluation") from e


def _metrics(got: set[str], exp: set[str]) -> dict:
    try:
        if exp is None:
            raise EmptyPredictionError()

        tp = got & exp

        prec = len(tp) / len(got) if got else 0.0
        rec = len(tp) / len(exp) if exp else 0.0

        score_equilibre = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "prec": round(prec, 3),
            "rec": round(rec, 3),
            "score_equilibre": round(score_equilibre, 3),
        }

    except Exception as e:
        raise ValueError("Metric computation failed") from e


def compare(res: dict[str, list[str]], ref: dict[str, list[str]] | None = None) -> dict:
    try:
        if not res:
            raise EmptyPredictionError()

        ref = ref or ALICE_REF

        out = {}

        for cat in ("characters", "locations"):
            got = _norm(res.get(cat, []))
            exp = _norm(ref.get(cat, []))

            out[cat] = _metrics(got, exp)

        return out

    except EmptyPredictionError:
        # erreur propre métier
        raise

    except Exception as e:
        raise RuntimeError("Entity evaluation pipeline failed") from e
