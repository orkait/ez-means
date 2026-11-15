import math
import statistics
import time
from dataclasses import dataclass

import faiss
import numpy as np
import numpy.typing as npt

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe

POOL_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTERS = 1024
FLAT_EPOCHS = 2
REFINE_PROBES = 4
SEEDS = range(2)


@dataclass(frozen=True)
class Plan:
    name: str
    fine_epochs: int
    refine_epochs: int


PLANS = (
    Plan("coarse 1 + fine 1", 1, 0),
    Plan("coarse 1 + fine 2", 2, 0),
    Plan("coarse 1 + fine 1 + indexed refine 1", 1, 1),
)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def emeans(k: int, rows: int, epochs: int, seed: int, assignment: Assignment = Assignment.EXACT) -> Emeans:
    return Emeans(
        k,
        law_recipe(rows, k, epochs, seed),
        Revival.MASS,
        metric=Metric.EUCLIDEAN,
        assignment=assignment,
        probes=REFINE_PROBES,
    )


def shares(sizes: npt.NDArray[np.int64], k: int) -> npt.NDArray[np.int64]:
    raw = sizes / sizes.sum() * k
    counts = np.maximum(1, np.floor(raw)).astype(np.int64)
    for cell in np.argsort(-(raw - np.floor(raw)))[: max(0, k - int(counts.sum()))]:
        counts[cell] += 1
    return np.asarray(np.minimum(counts, sizes), dtype=np.int64)


def coarse_to_fine(pool: Matrix, plan: Plan, seed: int) -> tuple[Matrix, float]:
    cells = max(1, round(math.sqrt(CLUSTERS)))
    coarse = emeans(cells, len(pool), 1, seed)
    coarse_centroids = coarse.train(pool)
    _, owner = faiss.knn(pool, coarse_centroids, 1)
    owner = owner[:, 0]
    distances = coarse.distances + len(pool) * cells
    sizes = np.bincount(owner, minlength=cells)
    parts = []
    for cell, fine_k in enumerate(shares(sizes, CLUSTERS)):
        members = np.ascontiguousarray(pool[owner == cell])
        if fine_k == 0:
            continue
        fine = emeans(int(fine_k), len(members), plan.fine_epochs, seed)
        parts.append(fine.train(members))
        distances += fine.distances
    centroids = np.ascontiguousarray(np.concatenate(parts).astype(np.float32))
    if plan.refine_epochs:
        refine = emeans(len(centroids), len(pool), plan.refine_epochs, seed, Assignment.INDEXED)
        centroids = refine.train(pool, centroids)
        distances += refine.distances
    return centroids, distances


def main() -> None:
    pool = read_fvecs(SIFT_BASE, POOL_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    flat_losses, flat_counts, flat_times = [], [], []
    for seed in SEEDS:
        model = emeans(CLUSTERS, len(pool), FLAT_EPOCHS, seed)
        start = time.perf_counter()
        centroids = model.train(pool)
        flat_times.append(time.perf_counter() - start)
        flat_losses.append(mse(test, centroids))
        flat_counts.append(model.distances)
    loss, count, seconds = (statistics.median(v) for v in (flat_losses, flat_counts, flat_times))
    print(f"SIFT pool {pool.shape}, test {test.shape}, k {CLUSTERS}, {len(SEEDS)} seeds\n")
    print("| plan | test MSE / flat | distances / flat | s | s / flat |")
    print("|---|---|---|---|---|")
    print(f"| flat e-means T={FLAT_EPOCHS} | 1.0000 | 1.0000 | {seconds:.2f} | 1.00 |")
    for plan in PLANS:
        losses, counts, times = [], [], []
        for seed in SEEDS:
            start = time.perf_counter()
            centroids, distances = coarse_to_fine(pool, plan, seed)
            times.append(time.perf_counter() - start)
            losses.append(mse(test, centroids))
            counts.append(distances)
        elapsed = statistics.median(times)
        print(
            f"| {plan.name} | {statistics.median(losses) / loss:.4f} | {statistics.median(counts) / count:.4f} | "
            f"{elapsed:.2f} | {elapsed / seconds:.2f} |"
        )


if __name__ == "__main__":
    main()
