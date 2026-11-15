import time
from collections.abc import Callable

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe
from ezmeans.trainer import ez_means

TRAIN_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTERS = 4096
EPOCHS = 2
PROBES = 4
SEED = 0

Arm = Callable[[Matrix], tuple[Matrix, float]]


def python_arm(assignment: Assignment) -> Arm:
    def run(train: Matrix) -> tuple[Matrix, float]:
        recipe = law_recipe(len(train), CLUSTERS, EPOCHS, SEED)
        model = Emeans(CLUSTERS, recipe, Revival.MASS, metric=Metric.EUCLIDEAN, assignment=assignment, probes=PROBES)
        return model.train(train), model.distances

    return run


def ez_means_arm(train: Matrix) -> tuple[Matrix, float]:
    return ez_means(train, CLUSTERS, law_recipe(len(train), CLUSTERS, EPOCHS, SEED), PROBES)


ARMS: dict[str, Arm] = {
    "exact": python_arm(Assignment.EXACT),
    "grouped": python_arm(Assignment.GROUPED),
    "fused": python_arm(Assignment.FUSED),
    "ez-means": ez_means_arm,
}


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def main() -> None:
    train = read_fvecs(SIFT_BASE, TRAIN_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    print(f"SIFT train {train.shape}, test {test.shape}, k {CLUSTERS}, T={EPOCHS}, seed {SEED}\n")
    print("| arm | test MSE / exact | distance work / exact | seconds | seconds / exact |")
    print("|---|---|---|---|---|")
    base_error, base_work, base_time = 0.0, 0.0, 0.0
    for name, arm in ARMS.items():
        start = time.perf_counter()
        centroids, work = arm(train)
        seconds = time.perf_counter() - start
        error = mse(test, centroids)
        if not base_time:
            base_error, base_work, base_time = error, work, seconds
        print(
            f"| {name} | {error / base_error:.4f} | {work / base_work:.4f} | {seconds:.2f} | {seconds / base_time:.2f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
