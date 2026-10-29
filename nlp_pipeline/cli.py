"""CLI executer les etapes de la pipeline"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from nlp_pipeline import books, config
from nlp_pipeline.storage import cache
from nlp_pipeline.errors import BookwormError
from nlp_pipeline.nlp import entities, lexdiv, topic, summarize, similarity

_HELPCMD = ("-h", "--help", "--h", "help")

def parse_args(argv: list[str]) -> dict[str, Any]:
    args = {
        "command": None,
        "value": None,
        "no_cache": False,
    }

    if not argv:
        return args

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in _HELPCMD:
            show_help()
            sys.exit(0)

        if arg == "--no-cache":
            args["no_cache"] = True
            i += 1

        elif arg.startswith("--"):
            cmd = arg[2:]  # enlever "--"
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                value = argv[i + 1]
                try:
                    value = int(value)
                except ValueError:
                    pass
                args["command"] = cmd
                args["value"] = value
                i += 2
            else:
                print(f"Erreur : {arg} requiert une valeur")
                sys.exit(1)
        else:
            print(f"Erreur : argument inconnu '{arg}'")
            sys.exit(1)

    return args


def _emit(result: Any) -> None:
    """Affiche un résultat JSON ou texte."""
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _clear_cache(scope: str) -> None:
    if scope not in ("all", "data", "output"):
        print(f"Erreur : scope doit être 'all', 'data' ou 'output' (reçu: {scope})")
        sys.exit(1)

    counts = cache.clear(scope)
    labels = {
        "all": "tout le cache",
        "data": "les livres et index gutendex",
        "output": "les fichiers de sortie",
    }
    print(f"cache vidé ({labels[scope]}): {counts}")


def show_help() -> None:
    """affiche l aide cli"""
    print(
"""Nlp pipeline

Usage:
  python bookworm.py --<commande> <valeur> [--no-cache]

Commandes:
  --select <id>        recup metadata et telecharge le livre
  --lexdiv <id>        diversite lexicale
  --topic <id>         mots cle par chapitre
  --entities <id>      perso et lieux
  --summarize <id>     resumé du livre
  --card <id>          fiche du livre
  --similar <id>       livre similaires
  --clear-cache <type>  vide le cache (all data output)

Options:
  --no-cache           ignore le cache
  -h, --h, --help      affiche l aide
""")

def main(argv: list[str] | None = None) -> int:
    config.ensure_directories()

    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in _HELPCMD:
        show_help()
        return 0

    args = parse_args(argv)
    cmd = args["command"]
    val = args["value"]
    use_cache = not args["no_cache"]

    try:
        if cmd == "select":
            if not isinstance(val, int):
                print("Erreur : --select requiert un ID entier")
                return 1
            _emit(books.select(val, use_cache=use_cache))

        elif cmd == "lexdiv":
            if not isinstance(val, int):
                print("Erreur : --lexdiv requiert un ID entier")
                return 1
            _emit(lexdiv.compute(val, use_cache=use_cache))

        elif cmd == "topic":
            if not isinstance(val, int):
                print("Erreur : --topic requiert un ID entier")
                return 1
            _emit(topic.compute(val, use_cache=use_cache))

        elif cmd == "entities":
            if not isinstance(val, int):
                print("Erreur : --entities requiert un ID entier")
                return 1
            _emit(entities.compute(val, use_cache=use_cache))

        elif cmd == "summarize":
            if not isinstance(val, int):
                print("Erreur : --summarize requiert un ID entier")
                return 1
            _emit(summarize.compute(val, use_cache=use_cache))
        elif cmd == "card":
            if not isinstance(val, int):
                print("Erreur : --card requiert un ID entier")
                return 1
            from nlp_pipeline.nlp import card

            _emit(card.compute(val, use_cache=use_cache))


        elif cmd == "similar":
            if not isinstance(val, int):
                print("Erreur : --similar requiert un ID entier")
                return 1
            _emit(similarity.compute(val, use_cache=use_cache))

        elif cmd == "clear-cache":
            if not isinstance(val, str):
                print("Erreur : --clear-cache requiert 'all', 'data' ou 'output'")
                return 1
            _clear_cache(val)

        else:
            print(f"Erreur : commande inconnue '--{cmd}'")
            show_help()
            return 1

        return 0

    except BookwormError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        if config.DEBUG_MODE:
            traceback.print_exc(file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Erreur inattendue : {exc}", file=sys.stderr)
        if config.DEBUG_MODE:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())