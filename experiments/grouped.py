import statistics
import time

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe

POOL_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTER_COUNTS = (1024, 4096)
EPOCHS = 2
ARMS = ((Assignment.EXACT, 1), (Assignment.INDEXED, 4), (Assignment.GROUPED, 2), (Assignment.GROUPED, 4))
SEEDS = range(2)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def run(pool: Matrix, test: Matrix, k: int, assignment: Assignment, probes: int) -> tuple[float, float, float]:
    losses, counts, times = [], [], []
    for seed in SEEDS:
        recipe = law_recipe(len(pool), k, EPOCHS, seed)
        model = Emeans(k, recipe, Revival.MASS, metric=Metric.EUCLIDEAN, assignment=assignment, probes=probes)
        start = time.perf_counter()
        centroids = model.train(pool)
        times.append(time.perf_counter() - start)
        losses.append(mse(test, centroids))
        counts.append(float(model.distances))
    return statistics.median(losses), statistics.median(counts), statistics.median(times)


def main() -> None:
    pool = read_fvecs(SIFT_BASE, POOL_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    print(f"SIFT pool {pool.shape}, test {test.shape}, e-means T={EPOCHS}, {len(SEEDS)} seeds\n")
    print("| k | assignment | probes | test MSE / exact | distances / exact | s, median | s / exact |")
    print("|---|---|---|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        base_loss, base_count, base_time = 0.0, 0.0, 0.0
        for assignment, probes in ARMS:
            loss, count, seconds = run(pool, test, k, assignment, probes)
            if assignment is Assignment.EXACT:
                base_loss, base_count, base_time = loss, count, seconds
            print(
                f"| {k} | {assignment.value} | {probes} | {loss / base_loss:.4f} | {count / base_count:.4f} | "
                f"{seconds:.2f} | {seconds / base_time:.2f} |"
            )


if __name__ == "__main__":
    main()
