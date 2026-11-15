import math
from dataclasses import dataclass

from faiss.contrib.e_means.base import _resolve_recipe

BATCH_LAW = "throughput"
REVIVAL_THRESHOLD = 0.1
INIT_USAGE = 2.0**-16
CHECK_EVERY = 5
ALPHA_MIN_FACTOR = 0.1
SPLIT_EPSILON = 1e-4
GROUP_ITERATIONS = 10


@dataclass(frozen=True)
class Recipe:
    epochs: float
    batch: int
    alpha0: float
    tau: float
    seed: int

    def __post_init__(self) -> None:
        if not self.epochs > 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not 0 < self.alpha0 <= 1:
            raise ValueError(f"alpha0 must be in (0, 1], got {self.alpha0}")


def group_count(k: int) -> int:
    return max(1, round(math.sqrt(k)))


def epoch_rows(n: int, epochs: float) -> list[int]:
    whole = math.floor(epochs)
    remainder = round((epochs - whole) * n)
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def law_recipe(n: int, k: int, epochs: float, seed: int) -> Recipe:
    batch, alpha0 = _resolve_recipe(n, k, epochs, BATCH_LAW, None, None)
    return Recipe(epochs, int(batch), float(alpha0), REVIVAL_THRESHOLD, seed)
