from ezmeans import kernel
from ezmeans.kernel import Matrix
from ezmeans.recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_EVERY,
    GROUP_ITERATIONS,
    INIT_USAGE,
    SPLIT_EPSILON,
    Recipe,
    default_probes,
    epoch_rows,
)


def ez_means(x: Matrix, k: int, recipe: Recipe, probes: int | None = None, init: Matrix | None = None) -> Matrix:
    settings = kernel.Settings(
        batch=recipe.batch,
        probes=default_probes(k) if probes is None else probes,
        check_every=CHECK_EVERY,
        group_iterations=GROUP_ITERATIONS,
        seed=recipe.seed,
        alpha0=recipe.alpha0,
        alpha_min_factor=ALPHA_MIN_FACTOR,
        tau=recipe.tau,
        init_usage=INIT_USAGE,
        split_epsilon=SPLIT_EPSILON,
    )
    return kernel.train(x, k, epoch_rows(len(x), recipe.epochs), settings, init)
