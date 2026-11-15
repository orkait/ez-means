import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from ezmeans.config import CLUSTERS, DIM, SIFT_LEARN
from ezmeans.dataset import Matrix, generate, read_fvecs
from ezmeans.reference import Emeans, Metric, Revival, law_recipe

SEEDS = range(5)
SPHERICAL_EPOCHS = 25
SIFT_ROWS = 20000
SIFT_CLUSTERS = 256
SIFT_EPOCHS = 3


def identical(x: Matrix, k: int, epochs: int, metric: Metric) -> int:
    spherical = metric is Metric.SPHERICAL
    matches = 0
    for seed in SEEDS:
        reference, _ = NumpyEmeans(x.shape[1], k, nepochs=epochs, spherical=spherical, seed=seed).train(x)
        ours = Emeans(k, law_recipe(len(x), k, epochs, seed), Revival.MASS, metric=metric).train(x)
        matches += int(np.array_equal(np.asarray(reference, dtype=np.float32), ours))
    return matches


def main() -> None:
    kepler = generate().points
    sift = read_fvecs(SIFT_LEARN, SIFT_ROWS)
    print(f"spherical, kepler: {identical(kepler, CLUSTERS, SPHERICAL_EPOCHS, Metric.SPHERICAL)} of {len(SEEDS)}")
    print(f"euclidean, sift: {identical(sift, SIFT_CLUSTERS, SIFT_EPOCHS, Metric.EUCLIDEAN)} of {len(SEEDS)}")
    print(f"dims: kepler {kepler.shape}, sift {sift.shape}, sift range {sift.min():g}..{sift.max():g}, dim {DIM}")


if __name__ == "__main__":
    main()
