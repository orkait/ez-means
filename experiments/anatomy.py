from collections import Counter

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from ezmeans.config import CLUSTERS, DIM, KEPLER_EPOCHS, RECOVERY_COSINE, RUN_SEEDS
from ezmeans.dataset import Matrix, class_means, generate

STRADDLE_COSINE = 0.95
MAX_SHOWN = 3


def coverage(means: Matrix, centroids: Matrix) -> tuple[Counter[int], int]:
    near = (means.astype(np.float64) @ centroids.astype(np.float64).T) >= RECOVERY_COSINE
    per_class = Counter(int(row.sum()) for row in near)
    best = (centroids.astype(np.float64) @ means.astype(np.float64).T).max(axis=1)
    straddling = int(((best < RECOVERY_COSINE) & (best >= STRADDLE_COSINE)).sum())
    return per_class, straddling


def main() -> None:
    data = generate()
    means = class_means(data)
    print("| seed | classes with 0 / 1 / 2+ centroids | centroids between classes | centroids far from any class |")
    print("|---|---|---|---|")
    for seed in RUN_SEEDS:
        centroids, _ = NumpyEmeans(DIM, CLUSTERS, nepochs=KEPLER_EPOCHS, spherical=True, seed=seed).train(data.points)
        per_class, straddling = coverage(means, np.asarray(centroids, dtype=np.float32))
        many = sum(count for covered, count in per_class.items() if covered >= 2)
        best = (np.asarray(centroids, dtype=np.float64) @ means.astype(np.float64).T).max(axis=1)
        far = int((best < STRADDLE_COSINE).sum())
        print(f"| {seed} | {per_class[0]} / {per_class[1]} / {many} | {straddling} | {far} |")


if __name__ == "__main__":
    main()
