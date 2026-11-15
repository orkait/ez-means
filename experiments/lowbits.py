import statistics

import faiss
import numpy as np

from ezmeans.config import DEEP_BASE, SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fbin, read_fvecs
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe

ROWS = 100_000
DEEP_TEST_START = 900_000
CLUSTERS = 1024
EPOCHS = (2, 3)
SEEDS = (0, 1)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids), 1)
    return float(distances.astype(np.float64).mean())


def main() -> None:
    datasets = {
        "SIFT (integer values)": (read_fvecs(SIFT_LEARN, ROWS), read_fvecs(SIFT_BASE, ROWS)),
        "DEEP* (float embeddings)": (read_fbin(DEEP_BASE, 0, ROWS), read_fbin(DEEP_BASE, DEEP_TEST_START, ROWS)),
    }
    print(f"e-means assignment in bfloat16 against float32, k {CLUSTERS}, {len(SEEDS)} seeds\n")
    print("| dataset | epochs | test MSE float32 | test MSE bfloat16 | bfloat16 / float32 |")
    print("|---|---|---|---|---|")
    for name, (train, test) in datasets.items():
        for epochs in EPOCHS:
            errors = {}
            for assignment in (Assignment.EXACT, Assignment.BFLOAT16):
                errors[assignment] = statistics.median(
                    mse(
                        test,
                        Emeans(
                            CLUSTERS,
                            law_recipe(len(train), CLUSTERS, epochs, seed),
                            Revival.MASS,
                            metric=Metric.EUCLIDEAN,
                            assignment=assignment,
                        ).train(train),
                    )
                    for seed in SEEDS
                )
            exact, low = errors[Assignment.EXACT], errors[Assignment.BFLOAT16]
            print(f"| {name} | {epochs} | {exact:.6g} | {low:.6g} | {low / exact:.5f} |", flush=True)


if __name__ == "__main__":
    main()
