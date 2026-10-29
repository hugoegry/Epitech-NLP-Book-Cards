from __future__ import annotations

from pathlib import Path

DEBUG_MODE: bool = False 

COSINE_METHODE: str = "perso"  # sklearn / perso
TFIDF_METHODE: str = "perso"
RAISE_ON_ZERO_VECTOR: bool = False
# topics
DEFAULT_N_TOPICS: int = 5
TOPIC_TOP_WORDS: int = 10

# summarize
SUMMARY_SENTENCES: int = 5

# similarite
SIMILAR_TOP_K: int = 5
SBERT_MODEL: str = "all-MiniLM-L6-v2"
SBERT_MAX_SENTENCES: int = 200
WORD2VEC_VECTOR_SIZE: int = 100
WORD2VEC_WINDOW: int = 5
WORD2VEC_MIN_COUNT: int = 5
WORD2VEC_EPOCHS: int = 10

BOOK_COLLECTION: dict[int, dict[str, str]] = {
    11: {
        "title": "Alice's Adventures in Wonderland",
        "category": "Children / Young Adult",
    },
    12: {"title": "Through the Looking-Glass", "category": "Children / Young Adult"},
    16: {"title": "Peter Pan", "category": "Children / Young Adult"},
    55: {"title": "The Wonderful Wizard of Oz", "category": "Children / Young Adult"},
    113: {"title": "The Secret Garden", "category": "Children / Young Adult"},
    120: {"title": "Treasure Island", "category": "Children / Young Adult"},
    236: {"title": "The Jungle Book", "category": "Children / Young Adult"},
    108: {
        "title": "The Return of Sherlock Holmes",
        "category": "Crime, Mystery & Thriller",
    },
    834: {
        "title": "The Memoirs of Sherlock Holmes",
        "category": "Crime, Mystery & Thriller",
    },
    863: {
        "title": "The Mysterious Affair at Styles",
        "category": "Crime, Mystery & Thriller",
    },
    1661: {
        "title": "The Adventures of Sherlock Holmes",
        "category": "Crime, Mystery & Thriller",
    },
    61262: {"title": "Poirot Investigates", "category": "Crime, Mystery & Thriller"},
    69087: {
        "title": "The murder of Roger Ackroyd",
        "category": "Crime, Mystery & Thriller",
    },
    70114: {"title": "The Big Four", "category": "Crime, Mystery & Thriller"},
    35: {"title": "The Time Machine", "category": "Science-Fiction & Fantasy"},
    36: {"title": "The War of the Worlds", "category": "Science-Fiction & Fantasy"},
    84: {
        "title": "Frankenstein; Or, The Modern Prometheus",
        "category": "Science-Fiction & Fantasy",
    },
    159: {
        "title": "The island of Doctor Moreau",
        "category": "Science-Fiction & Fantasy",
    },
    164: {
        "title": "Twenty Thousand Leagues under the Sea",
        "category": "Science-Fiction & Fantasy",
    },
    345: {"title": "Dracula", "category": "Science-Fiction & Fantasy"},
    68283: {"title": "The call of Cthulhu", "category": "Science-Fiction & Fantasy"},
}

# entities
SPACY_MODEL: str = "en_core_web_sm"
SPACY_MODEL_MD: str = "en_core_web_md"
ENTITY_MIN_MENTIONS: int = 3
ENTITY_MIN_MENTIONS_MD: int = 2
ENTITY_MIN_VOTES: int = 2
ENTITY_VOTE_RATIO: float = 1.5
ENTITY_MIN_CAP_RATIO: float = 0.6
ENTITY_MAX: int = 25
ENTITY_CHUNK_SIZE: int = 200_000


ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
BOOKS_DIR: Path = DATA_DIR / "books"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUT_DIR: Path = DATA_DIR / "output"

GUTENDEX_URLS = (
    "https://gutendex.com/books/{id}",
    "http://92.88.3.200:8002/books/{id}",
)

TEXT_FORMAT_UTF8 = "text/plain; charset=utf-8"
TEXT_FORMAT_ASCII = "text/plain; charset=us-ascii"

CACHE_TTL_SECONDS: int = 24 * 3600
LANGUAGE: str = "english"

HTTP_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; BookwormBot/1.0; +educational)"
}
HTTP_TIMEOUT: int = 30


def ensure_directories() -> None:
    for directory in (BOOKS_DIR, CACHE_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
