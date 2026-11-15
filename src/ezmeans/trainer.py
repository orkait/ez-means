from ezmeans import kernel
from ezmeans.dataset import Matrix
from ezmeans.reference import (
    ALPHA_MIN_FACTOR,
    CHECK_EVERY,
    GROUP_ITERATIONS,
    INIT_USAGE,
    SPLIT_EPSILON,
    Floats,
    Recipe,
    epoch_rows,
)


def ez_means(x: Floats, k: int, recipe: Recipe, probes: int, sampling: float | None = None) -> tuple[Matrix, float]:
    settings = kernel.Settings(
        batch=recipe.batch,
        probes=probes,
        check_every=CHECK_EVERY,
        group_iterations=GROUP_ITERATIONS,
        seed=recipe.seed,
        alpha0=recipe.alpha0,
        alpha_min_factor=ALPHA_MIN_FACTOR,
        tau=recipe.tau,
        init_usage=INIT_USAGE,
        split_epsilon=SPLIT_EPSILON,
        ad_epsilon=sampling or 0.0,
    )
    centroids, work = kernel.train(x, k, epoch_rows(len(x), recipe.epochs), settings)
    return centroids, work / x.shape[1]
