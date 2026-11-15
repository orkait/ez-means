import statistics
import time

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Emeans, Metric, Revival, law_recipe

POOL_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTERS = 1024
EPOCHS = 3
PER_CLUSTER = (16, 32, 64, 128, 256)
SEEDS = range(3)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def fit(pool: Matrix, rows: int, seed: int) -> tuple[Matrix, float]:
    subset = pool[np.random.default_rng(seed).choice(len(pool), rows, replace=False)] if rows < len(pool) else pool
    start = time.perf_counter()
    centroids = Emeans(CLUSTERS, law_recipe(rows, CLUSTERS, EPOCHS, seed), Revival.MASS, metric=Metric.EUCLIDEAN).train(
        subset
    )
    return centroids, time.perf_counter() - start


def main() -> None:
    pool = read_fvecs(SIFT_BASE, POOL_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    full = [fit(pool, len(pool), seed) for seed in SEEDS]
    reference = statistics.median(mse(test, centroids) for centroids, _ in full)
    print(f"SIFT pool {pool.shape}, test {test.shape}, k {CLUSTERS}, e-means T={EPOCHS}, {len(SEEDS)} seeds\n")
    print("| points per cluster | training rows | test MSE / full data | training s, median |")
    print("|---|---|---|---|")
    for per in PER_CLUSTER:
        runs = [fit(pool, per * CLUSTERS, seed) for seed in SEEDS]
        ratio = statistics.median(mse(test, centroids) for centroids, _ in runs) / reference
        seconds = statistics.median(elapsed for _, elapsed in runs)
        print(f"| {per} | {per * CLUSTERS} | {ratio:.4f} | {seconds:.2f} |")
    seconds = statistics.median(elapsed for _, elapsed in full)
    print(f"| all | {len(pool)} | 1.0000 | {seconds:.2f} |")


if __name__ == "__main__":
    main()
