from dataclasses import dataclass

import numpy as np

from ezmeans.config import RECOVERY_COSINE
from ezmeans.dataset import Matrix


@dataclass(frozen=True)
class Score:
    mean_cosine: float
    recovered: int


def score(points: Matrix, means: Matrix, centroids: Matrix) -> Score:
    codes = centroids.astype(np.float64)
    nearest = (points.astype(np.float64) @ codes.T).max(axis=1)
    found = (means.astype(np.float64) @ codes.T).max(axis=1)
    return Score(float(nearest.mean()), int((found >= RECOVERY_COSINE).sum()))
