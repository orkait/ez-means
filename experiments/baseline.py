import statistics
import time
from enum import Enum

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from ezmeans.config import (
    CLUSTERS,
    DIM,
    EPOCH_BUDGETS,
    RECOVERY_PASS,
    RUN_SEEDS,
    THREAD_COUNTS,
    TIMING_REPEATS,
    TIMING_SEED,
)
from ezmeans.dataset import Matrix, class_means, generate
from ezmeans.metrics import score

MILLISECONDS = 1000.0


class Method(Enum):
    EMEANS = "e-means"
    LLOYD = "lloyd"


def train(method: Method, points: Matrix, epochs: int, seed: int) -> Matrix:
    match method:
        case Method.EMEANS:
            centroids, _ = NumpyEmeans(DIM, CLUSTERS, nepochs=epochs, spherical=True, seed=seed).train(points)
            return np.asarray(centroids, dtype=np.float32)
        case Method.LLOYD:
            kmeans = faiss.Kmeans(DIM, CLUSTERS, niter=epochs, spherical=True, seed=seed)
            kmeans.train(points)
            return np.asarray(kmeans.centroids, dtype=np.float32)


def best_milliseconds(method: Method, points: Matrix, epochs: int) -> float:
    elapsed = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        train(method, points, epochs, TIMING_SEED)
        elapsed.append(time.perf_counter() - start)
    return min(elapsed) * MILLISECONDS


def main() -> None:
    data = generate()
    means = class_means(data)
    sizes = np.bincount(data.labels, minlength=CLUSTERS)
    print(
        f"data: {len(data.points)} points, class sizes {sizes.min()}-{sizes.max()}, "
        f"oracle mean cosine {score(data.points, means, means).mean_cosine:.5f}\n"
    )
    print(
        f"| method | epochs | median mean cosine | recovered min / median | seeds >= {RECOVERY_PASS} | "
        + " | ".join(f"ms at {threads} threads" for threads in THREAD_COUNTS)
        + " |"
    )
    print("|---|---|---|---|---|" + "---|" * len(THREAD_COUNTS))
    for method in Method:
        for epochs in EPOCH_BUDGETS:
            faiss.omp_set_num_threads(max(THREAD_COUNTS))
            scores = [score(data.points, means, train(method, data.points, epochs, seed)) for seed in RUN_SEEDS]
            recovered = [s.recovered for s in scores]
            timings = []
            for threads in THREAD_COUNTS:
                faiss.omp_set_num_threads(threads)
                timings.append(f"{best_milliseconds(method, data.points, epochs):.1f}")
            print(
                f"| {method.value} | {epochs} | {statistics.median(s.mean_cosine for s in scores):.5f} | "
                f"{min(recovered)} / {statistics.median(recovered):g} | "
                f"{sum(r >= RECOVERY_PASS for r in recovered)} of {len(scores)} | " + " | ".join(timings) + " |"
            )


if __name__ == "__main__":
    main()
