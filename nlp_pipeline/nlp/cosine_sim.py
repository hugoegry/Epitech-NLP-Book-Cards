"""cosinus"""

import math
import numpy as np

from nlp_pipeline import config
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from nlp_pipeline.errors import SimilarityComputationError, SimilarityMatrixError


def _matrix(mat):
    if hasattr(mat, "toarray"):
        return mat.toarray()
    arr = np.asarray(mat)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _cos(a, b):
    """cos entre 2 vecteur"""
    prodScalaire = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):  # zip pour parcourir les deux vecteurs en parallele
        prodScalaire += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        if config.RAISE_ON_ZERO_VECTOR:
            raise SimilarityComputationError("Zero vector detected")
        return 0.0
    return prodScalaire / (math.sqrt(na) * math.sqrt(nb))


def _cosinus(x, y=None):
    if x is None:
        raise SimilarityMatrixError("Input vector x is None")

    x = _matrix(x)
    if y is None:
        y = x
    else:
        y = _matrix(y)
    return np.array([[_cos(lx, ly) for ly in y] for lx in x])


def cosine_similarity(x, y=None, *, methode=config.COSINE_METHODE):
    """eguilage"""
    if methode == "perso":
        return _cosinus(x, y)
    elif methode == "sklearn":
        return sklearn_cosine_similarity(x, y)
    else:
        raise ValueError(f"Unknown cosine method: {methode}")
