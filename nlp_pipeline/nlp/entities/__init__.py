"""entities"""

from nlp_pipeline.nlp.entities.spacy import TASK_NAME, compute
from nlp_pipeline.nlp.entities import eval, nltk, personalise, spacy, spacy_md

__all__ = ["compute", "TASK_NAME", "spacy", "spacy_md", "nltk", "personalise", "eval"]