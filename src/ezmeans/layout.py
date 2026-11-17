import math
from dataclasses import dataclass
from enum import Enum

from ezmeans.recipe import (
    CHECK_EVERY,
    GROUP_ITERATIONS,
    GROUPS_PER_PROBE,
    Recipe,
    default_probes,
    epoch_rows,
    group_count,
)

WARM_ITERATIONS = 2
MIN_PROBES = 6
PROBE_SPREAD = 0.625
MAX_GROUPS = 512


class Regroup(Enum):
    COLD = GROUP_ITERATIONS
    WARM = WARM_ITERATIONS


@dataclass(frozen=True)
class Layout:
    groups: int
    probes: int
    regroup: Regroup


def law_probes(groups: int) -> int:
    return min(groups, max(MIN_PROBES, math.ceil(PROBE_SPREAD * math.sqrt(groups))))


def work_layout(n: int, k: int, recipe: Recipe) -> Layout:
    plan = epoch_rows(n, recipe.epochs)
    rows = sum(plan)
    regroups = math.ceil(sum(math.ceil(part / recipe.batch) for part in plan) / CHECK_EVERY)
    passes = Regroup.WARM.value + 1

    def work(groups: int) -> float:
        return rows * (groups + law_probes(groups) * k / groups) + regroups * passes * k * groups

    groups = min((1 << power for power in range(max(1, min(k, MAX_GROUPS)).bit_length())), key=work)
    return Layout(groups, law_probes(groups), Regroup.WARM)


def default_layout(k: int) -> Layout:
    groups = group_count(k)
    return Layout(groups, min(groups, max(MIN_PROBES, math.ceil(groups / GROUPS_PER_PROBE))), Regroup.COLD)


def paper_layout(k: int) -> Layout:
    return Layout(group_count(k), default_probes(k), Regroup.COLD)
