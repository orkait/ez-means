import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from faiss.extra_wrappers import SuperKmeans
from sklearn.cluster import KMeans, MiniBatchKMeans

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe
from ezmeans.trainer import ez_means

ALL_POINTS = 10**9
MINI_BATCH = 4096
GROUPED_PROBES = 4
TOLERANCE = 1.001
REFERENCE = ("faiss Lloyd", "25 it")

Fit = Callable[[Matrix, int, int], Matrix]


@dataclass(frozen=True)
class Entry:
    method: str
    budget: str
    fit: Fit


@dataclass(frozen=True)
class Round:
    train: tuple[Path, int]
    test: tuple[Path, int]
    clusters: int
    seeds: tuple[int, ...]
    entries: tuple[Entry, ...]


def lloyd(iterations: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        model = faiss.Kmeans(x.shape[1], k, niter=iterations, seed=seed, max_points_per_centroid=ALL_POINTS)
        model.train(x)
        return np.asarray(model.centroids, dtype=np.float32)

    return fit


def super_kmeans(iterations: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        model = SuperKmeans(x.shape[1], k, niter=iterations, seed=seed, max_points_per_centroid=ALL_POINTS)
        model.train(x)
        return np.asarray(model.centroids, dtype=np.float32)

    return fit


def emeans(epochs: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        centroids, _ = NumpyEmeans(x.shape[1], k, nepochs=epochs, seed=seed).train(x)
        return np.asarray(centroids, dtype=np.float32)

    return fit


def sklearn_lloyd(iterations: int, init: str) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        model = KMeans(k, init=init, n_init=1, max_iter=iterations, tol=0.0, random_state=seed)
        return np.asarray(model.fit(x).cluster_centers_, dtype=np.float32)

    return fit


def sklearn_minibatch(epochs: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        model = MiniBatchKMeans(
            k,
            init="random",
            n_init=1,
            batch_size=MINI_BATCH,
            max_iter=epochs,
            max_no_improvement=None,
            tol=0.0,
            random_state=seed,
        )
        return np.asarray(model.fit(x).cluster_centers_, dtype=np.float32)

    return fit


def grouped(epochs: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        model = Emeans(
            k,
            law_recipe(len(x), k, epochs, seed),
            Revival.MASS,
            metric=Metric.EUCLIDEAN,
            assignment=Assignment.GROUPED,
            probes=GROUPED_PROBES,
        )
        return model.train(x)

    return fit


def ez(epochs: int) -> Fit:
    def fit(x: Matrix, k: int, seed: int) -> Matrix:
        centroids, _ = ez_means(x, k, law_recipe(len(x), k, epochs, seed), GROUPED_PROBES)
        return centroids

    return fit


def common(lloyd_budgets: tuple[int, ...]) -> tuple[Entry, ...]:
    return (
        *(Entry("faiss Lloyd", f"{i} it", lloyd(i)) for i in lloyd_budgets),
        *(Entry("faiss e-means (paper)", f"T={t}", emeans(t)) for t in (1, 2, 3)),
        *(Entry("sklearn KMeans, random init", f"{i} it", sklearn_lloyd(i, "random")) for i in (10, 25)),
        *(Entry("sklearn MiniBatchKMeans", f"{t} epochs", sklearn_minibatch(t)) for t in (1, 3, 10)),
        *(Entry("ez-means prototype, Python loop", f"T={t}", grouped(t)) for t in (1, 2, 3)),
        *(Entry("ez-means", f"T={t}", ez(t)) for t in (1, 2, 3, 4)),
    )


ROUNDS = {
    "small": Round(
        (SIFT_LEARN, 100_000),
        (SIFT_BASE, 100_000),
        1024,
        (0,),
        (*common((5, 10, 25)), Entry("sklearn KMeans, k-means++ init", "25 it", sklearn_lloyd(25, "k-means++"))),
    ),
    "large": Round((SIFT_BASE, 1_000_000), (SIFT_LEARN, 100_000), 4096, (0, 1), common((10, 25))),
    "remote": Round(
        (SIFT_BASE, 1_000_000),
        (SIFT_LEARN, 100_000),
        4096,
        (0, 1),
        (
            Entry("faiss Lloyd", "25 it", lloyd(25)),
            Entry("faiss SuperKMeans", "25 it", super_kmeans(25)),
            Entry("sklearn KMeans, random init", "25 it", sklearn_lloyd(25, "random")),
            *(Entry("faiss e-means (paper)", f"T={t}", emeans(t)) for t in (2, 3)),
            *(Entry("ez-means", f"T={t}", ez(t)) for t in (2, 3)),
        ),
    ),
    "super": Round(
        (SIFT_BASE, 1_000_000),
        (SIFT_LEARN, 100_000),
        4096,
        (0, 1),
        (
            Entry("faiss Lloyd", "25 it", lloyd(25)),
            *(Entry("faiss SuperKMeans", f"{i} it", super_kmeans(i)) for i in (10, 15, 25)),
            *(Entry("ez-means", f"T={t}", ez(t)) for t in (2, 3)),
        ),
    ),
}


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=sorted(ROUNDS), required=True)
    plan = ROUNDS[parser.parse_args().round]
    train, test = read_fvecs(*plan.train), read_fvecs(*plan.test)
    ratios: dict[Entry, list[float]] = {entry: [] for entry in plan.entries}
    times: dict[Entry, list[float]] = {entry: [] for entry in plan.entries}
    for seed in plan.seeds:
        errors = {}
        for entry in plan.entries:
            start = time.perf_counter()
            centroids = entry.fit(train, plan.clusters, seed)
            times[entry].append(time.perf_counter() - start)
            errors[entry] = mse(test, centroids)
        reference = next(error for entry, error in errors.items() if (entry.method, entry.budget) == REFERENCE)
        for entry, error in errors.items():
            ratios[entry].append(error / reference)
    print(f"SIFT train {train.shape}, test {test.shape}, k {plan.clusters}, seeds {plan.seeds}, 4 cores\n")
    print("| method | budget | test MSE / Lloyd-25, median | seconds, median |")
    print("|---|---|---|---|")
    for entry in plan.entries:
        print(
            f"| {entry.method} | {entry.budget} | {statistics.median(ratios[entry]):.4f} | "
            f"{statistics.median(times[entry]):.2f} |"
        )
    print(f"\nFastest run per method within {TOLERANCE - 1:.1%} of Lloyd-25 (medians):\n")
    print("| method | budget | seconds |")
    print("|---|---|---|")
    for method in dict.fromkeys(entry.method for entry in plan.entries):
        good = [
            (statistics.median(times[e]), e)
            for e in plan.entries
            if e.method == method and statistics.median(ratios[e]) <= TOLERANCE
        ]
        if good:
            seconds, entry = min(good, key=lambda pair: pair[0])
            print(f"| {method} | {entry.budget} | {seconds:.2f} |")
        else:
            print(f"| {method} | none reached | - |")


if __name__ == "__main__":
    main()
