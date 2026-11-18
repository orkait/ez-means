from dataclasses import replace
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ezmeans import Layout, Recipe, Regroup, law_recipe
from ezmeans.recipe import CHECK_EVERY, MAX_ALPHA0

LEFT = 1.0
RIGHT = 3.0
LONE = 14.0
CROWD = 10
STARTS = (LEFT, LONE)
REVIVAL_CHECKS = 2
SINGLE_GROUP = Layout(1, 1, Regroup.COLD)


class LoneCluster(NamedTuple):
    rows: npt.NDArray[np.float32]
    start: npt.NDArray[np.float32]
    recipe: Recipe


def lone_cluster(dims: int, seed: int) -> LoneCluster:
    values = [LEFT] * CROWD + [RIGHT] * CROWD + [LONE]
    rows = np.zeros((len(values), dims), dtype=np.float32)
    rows[:, 0] = values
    start = np.zeros((len(STARTS), dims), dtype=np.float32)
    start[:, 0] = STARTS
    epochs = REVIVAL_CHECKS * CHECK_EVERY
    recipe = replace(law_recipe(len(rows), len(STARTS), epochs, seed), batch=len(rows), alpha0=MAX_ALPHA0)
    return LoneCluster(rows, start, recipe)
