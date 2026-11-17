from ezmeans.cpu import native
from ezmeans.cpu.native import Matrix
from ezmeans.layout import Layout, Regroup, default_layout
from ezmeans.recipe import ALPHA_MIN_FACTOR, CHECK_EVERY, INIT_USAGE, SPLIT_EPSILON, Recipe, epoch_rows


def train(x: Matrix, k: int, recipe: Recipe, layout: Layout | None = None, init: Matrix | None = None) -> Matrix:
    chosen = default_layout(k) if layout is None else layout
    settings = native.Settings(
        batch=recipe.batch,
        probes=chosen.probes,
        check_every=CHECK_EVERY,
        group_iterations=chosen.regroup.value,
        seed=recipe.seed,
        alpha0=recipe.alpha0,
        alpha_min_factor=ALPHA_MIN_FACTOR,
        tau=recipe.tau,
        init_usage=INIT_USAGE,
        split_epsilon=SPLIT_EPSILON,
        groups=chosen.groups,
        warm_groups=chosen.regroup is Regroup.WARM,
    )
    return native.train(x, k, epoch_rows(len(x), recipe.epochs), settings, init)
