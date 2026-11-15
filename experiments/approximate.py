import statistics

import faiss
import numpy as np

from ezmeans import kernel
from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Labels, Matrix, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe

ROWS = 100_000
CLUSTERS = 1024
PROBES = 4
LLOYD_ITERATIONS = 25
EPOCHS = 2
SEEDS = range(5)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def helper(x: Matrix, seed: int) -> Emeans:
    recipe = law_recipe(len(x), CLUSTERS, 1, seed)
    return Emeans(CLUSTERS, recipe, Revival.MASS, metric=Metric.EUCLIDEAN, assignment=Assignment.FUSED, probes=PROBES)


def grouped_labels(x: Matrix, centroids: Matrix, groups: Emeans) -> Labels:
    groups.regroup(centroids, faiss.METRIC_L2)
    offsets = np.cumsum([0, *map(len, groups.blocks)])
    _, labels, _ = kernel.assign(x, groups.groups, centroids, np.concatenate(groups.blocks), offsets, PROBES)
    return labels


def exact_labels(x: Matrix, centroids: Matrix) -> Labels:
    _, nearest = faiss.knn(x, centroids, 1)
    return np.asarray(nearest[:, 0], dtype=np.int64)


def lloyd(x: Matrix, seed: int, grouped: bool) -> Matrix:
    rng = np.random.default_rng(seed)
    centroids = np.ascontiguousarray(x[rng.choice(len(x), CLUSTERS, replace=False)])
    groups = helper(x, seed)
    for _ in range(LLOYD_ITERATIONS):
        labels = grouped_labels(x, centroids, groups) if grouped else exact_labels(x, centroids)
        sums, counts = kernel.accumulate(x, labels, CLUSTERS)
        means = sums / np.maximum(counts, 1)[:, None]
        centroids = np.ascontiguousarray(np.where(counts[:, None] > 0, means, centroids), dtype=np.float32)
    return centroids


def emeans(x: Matrix, seed: int, grouped: bool) -> Matrix:
    assignment = Assignment.GROUPED if grouped else Assignment.EXACT
    recipe = law_recipe(len(x), CLUSTERS, EPOCHS, seed)
    model = Emeans(CLUSTERS, recipe, Revival.MASS, metric=Metric.EUCLIDEAN, assignment=assignment, probes=PROBES)
    return model.train(x)


def main() -> None:
    train, test = read_fvecs(SIFT_LEARN, ROWS), read_fvecs(SIFT_BASE, ROWS)
    print(f"SIFT train {train.shape}, test {test.shape}, k {CLUSTERS}, grouped p{PROBES}, seeds {list(SEEDS)}\n")
    static = []
    for seed in SEEDS:
        centroids = emeans(train, seed, grouped=False)
        exact_cost = ((train - centroids[exact_labels(train, centroids)]) ** 2).sum(axis=1).mean()
        grouped_cost = (
            ((train - centroids[grouped_labels(train, centroids, helper(train, seed))]) ** 2).sum(axis=1).mean()
        )
        static.append(float(grouped_cost / exact_cost))
    print(
        f"static penalty: grouped p{PROBES} on fixed exact e-means centroids costs {statistics.median(static):.4f}x\n"
    )
    print(
        "| training | exact test MSE (median) | seed spread of exact | grouped test MSE / exact (median) | per seed |"
    )
    print("|---|---|---|---|---|")
    for name, train_with in ((f"Lloyd {LLOYD_ITERATIONS} it, full batch", lloyd), (f"e-means T={EPOCHS}", emeans)):
        exact = [mse(test, train_with(train, seed, False)) for seed in SEEDS]
        ratios = [mse(test, train_with(train, seed, True)) / error for seed, error in zip(SEEDS, exact, strict=True)]
        spread = (max(exact) - min(exact)) / statistics.median(exact)
        print(
            f"| {name} | {statistics.median(exact):.1f} | {spread:.4f} | {statistics.median(ratios):.4f} | "
            f"{', '.join(f'{r:.4f}' for r in ratios)} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
