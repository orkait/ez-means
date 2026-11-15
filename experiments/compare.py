import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import faiss
import numpy as np

from ezmeans.config import CLUSTERS, SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, Spec, class_means, read_fvecs, sample
from ezmeans.metrics import score
from ezmeans.reference import Emeans, Metric, Revival, law_recipe

LLOYD_ITERATIONS = 25
VERIFY_RATIO = 1.0
KEPLER = Spec(2176, CLUSTERS, 0.025, 0.0)
KEPLER_SEEDS = range(10)
SIFT_ROWS = 100_000
SIFT_CLUSTERS = 1024
SIFT_SEEDS = range(3)


class Init(Enum):
    RANDOM = "random"
    PLUS_PLUS = "k-means++"


@dataclass(frozen=True)
class Method:
    name: str
    fit: Callable[[Matrix, int, int, Metric], Matrix]


def plus_plus(x: Matrix, k: int, seed: int) -> Matrix:
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(x)))]
    nearest = ((x - x[chosen[0]]) ** 2).sum(axis=1, dtype=np.float64)
    for _ in range(1, k):
        pick = int(rng.choice(len(x), p=nearest / nearest.sum()))
        chosen.append(pick)
        nearest = np.minimum(nearest, ((x - x[pick]) ** 2).sum(axis=1, dtype=np.float64))
    return np.ascontiguousarray(x[chosen])


def lloyd(init: Init) -> Callable[[Matrix, int, int, Metric], Matrix]:
    def fit(x: Matrix, k: int, seed: int, metric: Metric) -> Matrix:
        kmeans = faiss.Kmeans(x.shape[1], k, niter=LLOYD_ITERATIONS, seed=seed, spherical=metric is Metric.SPHERICAL)
        match init:
            case Init.RANDOM:
                kmeans.train(x)
            case Init.PLUS_PLUS:
                kmeans.train(x, init_centroids=plus_plus(x, k, seed))
        return np.asarray(kmeans.centroids, dtype=np.float32)

    return fit


def emeans(epochs: int, revival: Revival) -> Callable[[Matrix, int, int, Metric], Matrix]:
    def fit(x: Matrix, k: int, seed: int, metric: Metric) -> Matrix:
        return Emeans(k, law_recipe(len(x), k, epochs, seed), revival, VERIFY_RATIO, metric).train(x)

    return fit


METHODS = (
    Method(f"k-means (Lloyd, random init, {LLOYD_ITERATIONS} it)", lloyd(Init.RANDOM)),
    Method(f"k-means++ (Lloyd, k-means++ init, {LLOYD_ITERATIONS} it)", lloyd(Init.PLUS_PLUS)),
    Method("e-means T=2", emeans(2, Revival.MASS)),
    Method("e-means T=5", emeans(5, Revival.MASS)),
    Method("e-means T=25", emeans(25, Revival.MASS)),
    Method("e-means + verify swap T=2", emeans(2, Revival.VERIFY)),
    Method("e-means + verify swap T=5", emeans(5, Revival.VERIFY)),
)


def timed(method: Method, x: Matrix, k: int, seed: int, metric: Metric) -> tuple[Matrix, float]:
    start = time.perf_counter()
    centroids = method.fit(x, k, seed, metric)
    return centroids, time.perf_counter() - start


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def kepler() -> None:
    print(f"Kepler-like synthetic: n {KEPLER.points}, d 32, k {CLUSTERS}, spherical, {len(KEPLER_SEEDS)} seeds\n")
    print("| method | test cosine, median | recovered min / median of 40 | ms, median |")
    print("|---|---|---|---|")
    for method in METHODS:
        tests, recovered, times = [], [], []
        for seed in KEPLER_SEEDS:
            train, test = sample(KEPLER, seed)
            means = class_means(train)
            centroids, seconds = timed(method, train.points, CLUSTERS, seed, Metric.SPHERICAL)
            tests.append(score(test.points, means, centroids).mean_cosine)
            recovered.append(score(train.points, means, centroids).recovered)
            times.append(seconds * 1000.0)
        print(
            f"| {method.name} | {statistics.median(tests):.5f} | {min(recovered)} / {statistics.median(recovered):g} | "
            f"{statistics.median(times):.1f} |"
        )


def sift() -> None:
    train = read_fvecs(SIFT_LEARN, SIFT_ROWS)
    test = read_fvecs(SIFT_BASE, SIFT_ROWS)
    print(f"\nSIFT: train {train.shape}, test {test.shape}, k {SIFT_CLUSTERS}, euclidean, {len(SIFT_SEEDS)} seeds\n")
    print("| method | test MSE, median | test / k-means | s, median |")
    print("|---|---|---|---|")
    baseline = 0.0
    for method in METHODS:
        tests, times = [], []
        for seed in SIFT_SEEDS:
            centroids, seconds = timed(method, train, SIFT_CLUSTERS, seed, Metric.EUCLIDEAN)
            tests.append(mse(test, centroids))
            times.append(seconds)
        median = statistics.median(tests)
        baseline = baseline or median
        print(f"| {method.name} | {median:.1f} | {median / baseline:.4f} | {statistics.median(times):.2f} |")


if __name__ == "__main__":
    kepler()
    sift()
