"""cache gutenber gutendex"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from nlp_pipeline import config

ClearScope = Literal["all", "data", "output"]

def _result_path(action: str, book_id: int) -> Path:
    return config.CACHE_DIR / f"{action}-{book_id}.json"

def _book_path(book_id: int) -> Path:
    return config.BOOKS_DIR / f"{book_id}.txt"

def _is_expired(path: Path, ttl_seconds: int) -> bool:
    return (time.time() - path.stat().st_mtime) > ttl_seconds

def load(action: str, book_id: int, *, use_cache: bool = True, ttl_seconds: int | None = config.CACHE_TTL_SECONDS) -> Any | None:
    """charge le json en cache (ex rep gutendex index-{id})."""
    if not use_cache:
        return None

    path = _result_path(action, book_id)
    if not path.exists():
        return None
    if ttl_seconds is not None and _is_expired(path, ttl_seconds):
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def save(action: str, book_id: int, data: Any) -> None:
    config.ensure_directories()
    path = _result_path(action, book_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_text(book_id: int, *, use_cache: bool = True, ttl_seconds: int | None = config.CACHE_TTL_SECONDS) -> str | None:
    """load text deja telecharge"""
    if not use_cache:
        return None

    path = _book_path(book_id)
    if not path.exists():
        return None
    if ttl_seconds is not None and _is_expired(path, ttl_seconds):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None

def save_text(book_id: int, text: str) -> None:
    config.ensure_directories()
    _book_path(book_id).write_text(text, encoding="utf-8")

def _delete_files(directory: Path, *, pattern: str | None = None) -> int:
    if not directory.exists():
        return 0

    deleted = 0
    for path in directory.iterdir():
        if path.name == ".gitkeep":
            continue
        if pattern is not None and not path.match(pattern):
            continue
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted

def clear(scope: ClearScope) -> dict[str, int]:
    """vide le cache (all=(tout), data(books index), output=(sortie))"""
    counts = {"books": 0, "cache": 0, "output": 0}

    if scope in ("all", "data"):
        counts["books"] = _delete_files(config.BOOKS_DIR)

    if scope == "all":
        counts["cache"] = _delete_files(config.CACHE_DIR)
    elif scope == "data":
        counts["cache"] = _delete_files(config.CACHE_DIR, pattern="index-*.json")

    if scope in ("all", "output"):
        counts["output"] = _delete_files(config.OUTPUT_DIR)

    return counts