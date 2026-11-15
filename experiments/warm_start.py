import statistics
import time

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

TRAIN_ROWS = 1_000_000
TEST_ROWS = 100_000
CLUSTERS = 4096
PROBES = 4
SEEDS = (0, 1)
REFERENCE_EPOCHS = 3.0
PLAIN_EPOCHS = (1.0, 1.4, 2.0, 3.0)
SUBSAMPLE_SHARES = (0.1, 0.25)
SUBSAMPLE_EPOCHS = 2.0
FULL_EPOCHS = (1.0, 1.4, 2.0)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def plain(x: Matrix, epochs: float, seed: int) -> tuple[Matrix, float]:
    start = time.perf_counter()
    centroids, _ = ez_means(x, CLUSTERS, law_recipe(len(x), CLUSTERS, epochs, seed), PROBES)
    return centroids, time.perf_counter() - start


def warmed(x: Matrix, share: float, epochs: float, seed: int, trusted: bool) -> tuple[Matrix, float]:
    start = time.perf_counter()
    rows = np.random.default_rng(seed).choice(len(x), round(share * len(x)), replace=False)
    small = np.ascontiguousarray(x[np.sort(rows)])
    init, _ = ez_means(small, CLUSTERS, law_recipe(len(small), CLUSTERS, SUBSAMPLE_EPOCHS, seed), PROBES)
    recipe = law_recipe(len(x), CLUSTERS, epochs, seed)
    if trusted:
        centroids, _ = ez_means(x, CLUSTERS, recipe, PROBES, init=init, init_usage=len(small) / CLUSTERS)
    else:
        centroids, _ = ez_means(x, CLUSTERS, recipe, PROBES, init=init)
    return centroids, time.perf_counter() - start


def main() -> None:
    x = read_fvecs(SIFT_BASE, TRAIN_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    reference = {seed: mse(test, plain(x, REFERENCE_EPOCHS, seed)[0]) for seed in SEEDS}
    print(f"SIFT {x.shape}, k {CLUSTERS}, p{PROBES}, seeds {SEEDS}; quality relative to plain T={REFERENCE_EPOCHS:g}\n")
    print("| run | data passes | seconds, median | test MSE / plain T=3, median |")
    print("|---|---|---|---|")
    arms: list[tuple[str, float, float | None, float, bool]] = [
        (f"plain T={t:g}", t, None, t, False) for t in PLAIN_EPOCHS
    ]
    arms += [
        (
            f"init from {s:.0%} subsample (T={SUBSAMPLE_EPOCHS:g}){', trusted' if trusted else ''}, then T={t:g}",
            s * SUBSAMPLE_EPOCHS + t,
            s,
            t,
            trusted,
        )
        for trusted in (False, True)
        for s in SUBSAMPLE_SHARES
        for t in FULL_EPOCHS
    ]
    for name, passes, share, epochs, trusted in arms:
        outcomes = [
            plain(x, epochs, seed) if share is None else warmed(x, share, epochs, seed, trusted) for seed in SEEDS
        ]
        ratios = [mse(test, centroids) / reference[seed] for seed, (centroids, _) in zip(SEEDS, outcomes, strict=True)]
        seconds = statistics.median(elapsed for _, elapsed in outcomes)
        print(f"| {name} | {passes:.2f} | {seconds:.2f} | {statistics.median(ratios):.4f} |", flush=True)


if __name__ == "__main__":
    main()
