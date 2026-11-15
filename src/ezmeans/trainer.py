from ezmeans import kernel
from ezmeans.kernel import Matrix
from ezmeans.recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_EVERY,
    GROUP_ITERATIONS,
    INIT_USAGE,
    SPLIT_EPSILON,
    Recipe,
    epoch_rows,
)


def ez_means(x: Matrix, k: int, recipe: Recipe, probes: int, init: Matrix | None = None) -> tuple[Matrix, float]:
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
    )
    centroids, work = kernel.train(x, k, epoch_rows(len(x), recipe.epochs), settings, init)
    return centroids, work / x.shape[1]
