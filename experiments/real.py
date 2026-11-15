import statistics

import faiss
import numpy as np

from ezmeans.config import SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import SWAP_RATIO, Emeans, Metric, Revival, law_recipe

TRAIN_ROWS = 100_000
TEST_ROWS = 100_000
CLUSTER_COUNTS = (1024, 4096)
SEEDS = range(5)
EPOCHS = (2, 3, 5)
LLOYD_ITERATIONS = 25
ARMS = ((Revival.MASS, SWAP_RATIO), (Revival.VERIFY, 1.0), (Revival.VERIFY, 4.0))


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def lloyd(x: Matrix, k: int, seed: int) -> Matrix:
    kmeans = faiss.Kmeans(x.shape[1], k, niter=LLOYD_ITERATIONS, seed=seed)
    kmeans.train(x)
    return np.asarray(kmeans.centroids, dtype=np.float32)


def main() -> None:
    train = read_fvecs(SIFT_LEARN, TRAIN_ROWS)
    test = read_fvecs(SIFT_BASE, TEST_ROWS)
    print(f"SIFT: train {train.shape}, test {test.shape}, {len(SEEDS)} seeds\n")
    print("| k | epochs | revival | ratio | test / Lloyd-25, median | stdev over seeds | train / Lloyd-25 | swaps |")
    print("|---|---|---|---|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        references = [lloyd(train, k, seed) for seed in SEEDS]
        test_reference = statistics.median(mse(test, centroids) for centroids in references)
        train_reference = statistics.median(mse(train, centroids) for centroids in references)
        for epochs in EPOCHS:
            for revival, ratio in ARMS:
                fits, tests, swaps = [], [], []
                for seed in SEEDS:
                    recipe = law_recipe(len(train), k, epochs, seed)
                    model = Emeans(k, recipe, revival, ratio, Metric.EUCLIDEAN)
                    centroids = model.train(train)
                    fits.append(mse(train, centroids) / train_reference)
                    tests.append(mse(test, centroids) / test_reference)
                    swaps.append(model.swaps.done)
                print(
                    f"| {k} | {epochs} | {revival.value} | {ratio:g} | {statistics.median(tests):.4f} | "
                    f"{statistics.stdev(tests):.4f} | {statistics.median(fits):.4f} | {statistics.median(swaps):g} |"
                )


if __name__ == "__main__":
    main()
