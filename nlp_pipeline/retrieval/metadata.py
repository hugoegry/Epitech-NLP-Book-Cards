"""metadata gutendex"""

from __future__ import annotations
import json
from typing import Any

from nlp_pipeline.retrieval import gutendex

AUTHORS_KEYS = ("authors", "editors", "translators")

def get_metadata(book_id: int, *, use_cache: bool = True, all_copy: bool = False) -> dict[str, Any]:
    data = gutendex.fetch(book_id, use_cache=use_cache)
    if all_copy:
        return json.loads(json.dumps(data)) # serialise str / deserialise (cp auto)
    else:
        return dict(data)