"""Récupération distante : API Gutendex, texte Gutenberg, métadonnées.

Dépendance HTTP : requests (via http_client). Cache = storage/.
Le NLP vit dans preprocessing/ et nlp/.
"""

from nlp_pipeline.retrieval.gutendex import fetch as fetch_gutendex_index
from nlp_pipeline.retrieval.gutendex import text_urls
from nlp_pipeline.retrieval.gutenberg import download_text, get_clean_text
from nlp_pipeline.retrieval.metadata import get_metadata

__all__ = ["fetch_gutendex_index", "text_urls", "download_text", "get_clean_text", "get_metadata"]