import argparse
import time
from dataclasses import dataclass

import faiss
import numpy as np

from ezmeans.config import DINO_BASE, SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_bvecs, read_fvecs
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

EPOCHS = 2
PROBES = 4
SEED = 0
EPSILONS: tuple[float | None, ...] = (None, 2.1, 1.0)


@dataclass(frozen=True)
class Case:
    train: tuple[int, int]
    test: tuple[int, int]
    clusters: tuple[int, ...]


CASES = {
    "sift": Case((0, 1_000_000), (0, 100_000), (4096,)),
    "dino": Case((0, 300_000), (900_000, 100_000), (1024, 4096)),
}


def load(name: str, start: int, rows: int, test: bool) -> Matrix:
    if name == "dino":
        return read_bvecs(DINO_BASE, start, rows)
    return read_fvecs(SIFT_LEARN if test else SIFT_BASE, rows)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=sorted(CASES), required=True)
    name = parser.parse_args().data
    case = CASES[name]
    train, test = load(name, *case.train, test=False), load(name, *case.test, test=True)
    print(f"{name}: train {train.shape}, test {test.shape}, T={EPOCHS}, p{PROBES}\n")
    print("| k | pruning | seconds | test MSE / exact | distance work / exact | seconds / exact |")
    print("|---|---|---|---|---|---|")
    for k in case.clusters:
        base_error, base_work, base_time = 0.0, 0.0, 0.0
        for epsilon in EPSILONS:
            start = time.perf_counter()
            centroids, work = ez_means(train, k, law_recipe(len(train), k, EPOCHS, SEED), PROBES, epsilon)
            seconds = time.perf_counter() - start
            error = mse(test, centroids)
            if epsilon is None:
                base_error, base_work, base_time = error, work, seconds
            label = "exact" if epsilon is None else f"ADSampling eps {epsilon}"
            print(
                f"| {k} | {label} | {seconds:.2f} | {error / base_error:.4f} | {work / base_work:.4f} | "
                f"{seconds / base_time:.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
