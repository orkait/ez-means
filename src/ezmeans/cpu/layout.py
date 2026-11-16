import math
from dataclasses import dataclass
from enum import Enum

from ezmeans.recipe import GROUP_ITERATIONS, GROUPS_PER_PROBE, group_count

WARM_ITERATIONS = 2
MIN_PROBES = 6
WIDE_CLUSTERS = 65_536
WIDE_MIN_DIMS = 64
WIDE_GROUP_FACTOR = 2
WIDE_PROBE_SPREAD = 0.625


class Regroup(Enum):
    COLD = GROUP_ITERATIONS
    WARM = WARM_ITERATIONS


@dataclass(frozen=True)
class Layout:
    groups: int
    probes: int
    regroup: Regroup


def cpu_layout(k: int, dims: int) -> Layout:
    if k < WIDE_CLUSTERS or dims < WIDE_MIN_DIMS:
        groups = group_count(k)
        return Layout(groups, min(groups, max(MIN_PROBES, math.ceil(groups / GROUPS_PER_PROBE))), Regroup.COLD)
    groups = 1 << (math.isqrt(WIDE_GROUP_FACTOR**2 * k).bit_length() - 1)
    return Layout(groups, min(groups, math.ceil(WIDE_PROBE_SPREAD * math.sqrt(groups))), Regroup.WARM)
