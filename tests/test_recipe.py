import dataclasses
from collections.abc import Callable

import pytest

from ezmeans.recipe import RecipeError, RecipeFault, default_probes, law_recipe

ROWS = 4096
CLUSTERS = 256
EPOCHS = 2
SEED = 0
BATCH_LAW = "throughput"


@pytest.mark.parametrize(
    ("call", "fault"),
    [
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), epochs=0.0), RecipeFault.EPOCHS),
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), epochs=-0.5), RecipeFault.EPOCHS),
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), alpha0=0.0), RecipeFault.STEP),
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), alpha0=1.5), RecipeFault.STEP),
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), batch=0), RecipeFault.BATCH),
        (lambda: dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), batch=-1), RecipeFault.BATCH),
        (lambda: law_recipe(0, CLUSTERS, EPOCHS, SEED), RecipeFault.LAW_DOMAIN),
        (lambda: law_recipe(ROWS, 0, EPOCHS, SEED), RecipeFault.LAW_DOMAIN),
        (lambda: law_recipe(ROWS, CLUSTERS, -0.5, SEED), RecipeFault.LAW_DOMAIN),
    ],
    ids=[
        "zero epochs",
        "negative epochs",
        "zero step",
        "step above one",
        "zero batch",
        "negative batch",
        "no rows",
        "no clusters",
        "negative law epochs",
    ],
)
def test_recipe_rejects_impossible_values(call: Callable[[], object], fault: RecipeFault) -> None:
    with pytest.raises(RecipeError) as raised:
        call()
    assert raised.value.fault is fault


@pytest.mark.parametrize(
    ("n", "k", "epochs"),
    [(2_176, 40, 25), (5_000, 4_096, 0.6), (100_000, 1_024, 2), (1_000_000, 16_384, 1.4), (10_000_000, 65_536, 2.2)],
)
def test_law_matches_faiss(n: int, k: int, epochs: float) -> None:
    base = pytest.importorskip("faiss.contrib.e_means.base")
    recipe = law_recipe(n, k, epochs, SEED)
    assert (recipe.batch, recipe.alpha0) == base._resolve_recipe(n, k, epochs, BATCH_LAW, None, None)


@pytest.mark.parametrize(
    ("k", "probes"), [(4, 2), (16, 4), (1_024, 4), (4_096, 4), (16_384, 8), (65_536, 16), (262_144, 32)]
)
def test_default_probes_cover_a_sixteenth_of_the_groups(k: int, probes: int) -> None:
    assert default_probes(k) == probes
