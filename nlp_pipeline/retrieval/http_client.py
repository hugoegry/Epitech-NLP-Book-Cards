"""cl http (requests)"""

from __future__ import annotations

from typing import Any

import requests


class HttpError(Exception):
    """prb reseau ou corps de rep invalide"""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def get(url: str, *, headers: dict[str, str], timeout: int) -> tuple[int, bytes]:
    """get renvoie code http et contenu brut"""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise HttpError(f"la requete a echoue ({exc})") from exc
    return response.status_code, response.content


def get_json(url: str, *, headers: dict[str, str], timeout: int) -> tuple[int, dict[str, Any]]:
    """get json (gutendex)"""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise HttpError(f"la requete a echoue ({exc})") from exc

    try:
        parsed = response.json()
    except ValueError as exc:
        raise HttpError("reponse json illisible", status=response.status_code) from exc
    if not isinstance(parsed, dict):
        raise HttpError("json inattendu (pas un objet)", status=response.status_code)
    return response.status_code, parsed


def get_text(url: str, *, headers: dict[str, str], timeout: int) -> tuple[int, str]:
    """get texte (livre gutenberg)"""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise HttpError(f"la requete a echoue ({exc})") from exc

    if response.status_code != 200 or not response.text:
        return response.status_code, ""
    response.encoding = response.encoding or "utf-8"
    return response.status_code, response.text