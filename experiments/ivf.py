import statistics
import time

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe

POOL_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTERS = 1024
EPOCHS = (1, 2)
PROBES = (1, 2, 4, 8, 16)
SEEDS = range(2)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def run(pool: Matrix, test: Matrix, epochs: int, assignment: Assignment, probes: int) -> tuple[float, float, float]:
    losses, counts, times = [], [], []
    for seed in SEEDS:
        recipe = law_recipe(len(pool), CLUSTERS, epochs, seed)
        model = Emeans(CLUSTERS, recipe, Revival.MASS, metric=Metric.EUCLIDEAN, assignment=assignment, probes=probes)
        start = time.perf_counter()
        centroids = model.train(pool)
        times.append(time.perf_counter() - start)
        losses.append(mse(test, centroids))
        counts.append(float(model.distances))
    return statistics.median(losses), statistics.median(counts), statistics.median(times)


def main() -> None:
    pool = read_fvecs(SIFT_BASE, POOL_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    print(f"SIFT pool {pool.shape}, test {test.shape}, k {CLUSTERS}, {len(SEEDS)} seeds\n")
    print("| epochs | assignment | probes | test MSE / exact | distances / exact | s, median | s / exact |")
    print("|---|---|---|---|---|---|---|")
    for epochs in EPOCHS:
        loss, count, seconds = run(pool, test, epochs, Assignment.EXACT, 1)
        print(f"| {epochs} | exact | - | 1.0000 | 1.0000 | {seconds:.2f} | 1.00 |")
        for probes in PROBES:
            other, others, elapsed = run(pool, test, epochs, Assignment.INDEXED, probes)
            print(
                f"| {epochs} | indexed | {probes} | {other / loss:.4f} | {others / count:.4f} | "
                f"{elapsed:.2f} | {elapsed / seconds:.2f} |"
            )


if __name__ == "__main__":
    main()
