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
CLUSTERS = (4096, 16384)
EPOCHS = (2, 3)
PROBES = 4
SEEDS = (0, 1)
SCREENS: tuple[float | None, ...] = (None, 3.0, 1.9)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def main() -> None:
    x = read_fvecs(SIFT_BASE, TRAIN_ROWS)
    test = read_fvecs(SIFT_LEARN, TEST_ROWS)
    print(f"SIFT {x.shape}, p{PROBES}, seeds {SEEDS}, medians\n")
    print("| k | T | screen | seconds | speedup | test MSE / exact | float work / exact |")
    print("|---|---|---|---|---|---|---|")
    for k in CLUSTERS:
        for epochs in EPOCHS:
            base: dict[int, tuple[float, float, float]] = {}
            for screen in SCREENS:
                rows = []
                for seed in SEEDS:
                    start = time.perf_counter()
                    centroids, work = ez_means(x, k, law_recipe(len(x), k, epochs, seed), PROBES, screen=screen)
                    seconds = time.perf_counter() - start
                    error = mse(test, centroids)
                    if screen is None:
                        base[seed] = (seconds, error, work)
                    rows.append((seconds, error / base[seed][1], work / base[seed][2], base[seed][0] / seconds))
                label = "exact" if screen is None else f"RaBitQ eps {screen}"
                seconds = statistics.median(r[0] for r in rows)
                print(
                    f"| {k} | {epochs} | {label} | {seconds:.2f} | {statistics.median(r[3] for r in rows):.2f}x | "
                    f"{statistics.median(r[1] for r in rows):.4f} | {statistics.median(r[2] for r in rows):.3f} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
