import dataclasses
from collections.abc import Callable

import pytest

from ezmeans.recipe import law_recipe

ROWS = 4096
CLUSTERS = 256
EPOCHS = 2
SEED = 0
BATCH_LAW = "throughput"


@pytest.mark.parametrize(
    "call",
    [
        lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), epochs=0.0),
        lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), epochs=-0.5),
        lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), alpha0=0.0),
        lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), alpha0=1.5),
        lambda: law_recipe(0, CLUSTERS, EPOCHS, SEED),
        lambda: law_recipe(ROWS, 0, EPOCHS, SEED),
        lambda: law_recipe(ROWS, CLUSTERS, -0.5, SEED),
    ],
    ids=[
        "zero epochs",
        "negative epochs",
        "zero step",
        "step above one",
        "no rows",
        "no clusters",
        "negative law epochs",
    ],
)
def test_recipe_rejects_impossible_values(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    ("n", "k", "epochs"),
    [(2_176, 40, 25), (5_000, 4_096, 0.6), (100_000, 1_024, 2), (1_000_000, 16_384, 1.4), (10_000_000, 65_536, 2.2)],
)
def test_law_matches_faiss(n: int, k: int, epochs: float) -> None:
    base = pytest.importorskip("faiss.contrib.e_means.base")
    recipe = law_recipe(n, k, epochs, SEED)
    assert (recipe.batch, recipe.alpha0) == base._resolve_recipe(n, k, epochs, BATCH_LAW, None, None)
