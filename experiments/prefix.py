import numpy as np
import numpy.typing as npt

from ezmeans.config import SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import Emeans, Metric, Revival, law_recipe

POOL_ROWS = 100_000
PROBE_ROWS = 5_000
CLUSTER_COUNTS = (256, 1024)
EPOCHS = 2
PREFIXES = (8, 16, 32, 48, 64)
SEED = 0
TOLERANCE = 1e-9

Wide = npt.NDArray[np.float64]


def rotation(x: Matrix) -> Matrix:
    centered = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return np.ascontiguousarray(vectors[:, ::-1], dtype=np.float32)


def squared(rows: Matrix, centroids: Matrix) -> Wide:
    wide_rows, wide_centroids = rows.astype(np.float64), centroids.astype(np.float64)
    return np.asarray(
        (wide_rows**2).sum(axis=1)[:, None]
        + (wide_centroids**2).sum(axis=1)[None, :]
        - 2.0 * (wide_rows @ wide_centroids.T),
        dtype=np.float64,
    )


def survivors(rows: Matrix, centroids: Matrix, prefix: int) -> tuple[float, float]:
    full = squared(rows, centroids)
    nearest, best = full.argmin(axis=1), full.min(axis=1)
    partial = squared(rows[:, :prefix], centroids[:, :prefix])
    alive = partial <= best[:, None] * (1.0 + TOLERANCE)
    kept = alive[np.arange(len(rows)), nearest]
    return float(alive.mean()), float(kept.mean())


def main() -> None:
    pool = read_fvecs(SIFT_LEARN, POOL_ROWS)
    turn = rotation(pool)
    probe = pool[np.random.default_rng(SEED).choice(len(pool), PROBE_ROWS, replace=False)] @ turn
    print(f"SIFT {pool.shape}, probe {len(probe)} rows: centroids that survive the exact prefix bound\n")
    print("| k | prefix dims | survivors | true nearest kept | FLOPs vs full |")
    print("|---|---|---|---|---|")
    d = pool.shape[1]
    for k in CLUSTER_COUNTS:
        recipe = law_recipe(len(pool), k, EPOCHS, SEED)
        centroids = Emeans(k, recipe, Revival.MASS, metric=Metric.EUCLIDEAN).train(pool) @ turn
        for prefix in PREFIXES:
            alive, kept = survivors(probe, centroids, prefix)
            cost = prefix / d + alive * (d - prefix) / d
            print(f"| {k} | {prefix} | {alive:.4f} | {kept:.4f} | {cost:.3f} |", flush=True)


if __name__ == "__main__":
    main()
