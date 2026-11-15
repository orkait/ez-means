import math

import faiss
import numpy as np
from prefix import rotation, squared

from ezmeans.config import SIFT_LEARN
from ezmeans.dataset import read_fvecs
from ezmeans.reference import Emeans, Metric, Revival, law_recipe

POOL_ROWS = 100_000
PROBE_ROWS = 5_000
CLUSTER_COUNTS = (1024, 4096)
EPOCHS = 2
PROBES = 4
PREFIX = 16
SEED = 0
TOLERANCE = 1e-9


def main() -> None:
    pool = read_fvecs(SIFT_LEARN, POOL_ROWS)
    turn = rotation(pool)
    probe = np.ascontiguousarray(pool[np.random.default_rng(SEED).choice(len(pool), PROBE_ROWS, replace=False)] @ turn)
    d = pool.shape[1]
    print(f"SIFT {pool.shape}, probe {len(probe)}: distance work per point as a share of brute force k*d\n")
    print("| k | method | work / brute force | same nearest as brute force |")
    print("|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        recipe = law_recipe(len(pool), k, EPOCHS, SEED)
        centroids = np.ascontiguousarray(Emeans(k, recipe, Revival.MASS, metric=Metric.EUCLIDEAN).train(pool) @ turn)
        full = squared(probe, centroids)
        truth = full.argmin(axis=1)
        lists = round(math.sqrt(k))
        coarse = faiss.Kmeans(d, lists, niter=10, seed=SEED, min_points_per_centroid=1)
        coarse.train(centroids)
        groups = np.asarray(coarse.centroids, dtype=np.float32)
        _, homes = faiss.knn(centroids, groups, 1)
        _, chosen = faiss.knn(probe, groups, PROBES)
        member = (homes[:, 0][None, :] == chosen[:, :, None]).any(axis=1)
        grouped_best = np.where(member, full, np.inf).argmin(axis=1)
        grouped_work = (lists * d + member.sum(axis=1) * d) / (k * d)
        best = full[np.arange(len(probe)), grouped_best]
        partial = squared(probe[:, :PREFIX], centroids[:, :PREFIX])
        alive = member & (partial <= best[:, None] * (1.0 + TOLERANCE))
        stacked_work = (lists * d + member.sum(axis=1) * PREFIX + alive.sum(axis=1) * (d - PREFIX)) / (k * d)
        prefix_alive = partial <= full.min(axis=1)[:, None] * (1.0 + TOLERANCE)
        prefix_work = (k * PREFIX + prefix_alive.sum(axis=1) * (d - PREFIX)) / (k * d)
        rows = (
            ("prefix only (exact)", prefix_work.mean(), 1.0),
            (f"grouped p{PROBES}", grouped_work.mean(), (grouped_best == truth).mean()),
            (f"grouped p{PROBES} + prefix {PREFIX}", stacked_work.mean(), (grouped_best == truth).mean()),
        )
        for name, work, same in rows:
            print(f"| {k} | {name} | {work:.4f} | {same:.4f} |", flush=True)


if __name__ == "__main__":
    main()
