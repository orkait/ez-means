import math
from dataclasses import dataclass
from enum import Enum

REVIVAL_THRESHOLD = 0.1
INIT_USAGE = 2.0**-16
CHECK_EVERY = 5
ALPHA_MIN_FACTOR = 0.1
SPLIT_EPSILON = 1e-4
GROUP_ITERATIONS = 10
BATCH_LAW = (0.035134, 0.134749, 0.876473, 0.813344)
STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
MIN_ALPHA0 = 1e-4
MAX_ALPHA0 = 1.0
MIN_PROBES = 4
GROUPS_PER_PROBE = 16


class RecipeFault(Enum):
    EPOCHS = "epochs must be positive"
    BATCH = "batch must be at least 1"
    STEP = "alpha0 must be in (0, 1]"
    LAW_DOMAIN = "the scaling law needs n >= 1, k >= 1 and epochs > 0"


class RecipeError(ValueError):
    def __init__(self, fault: RecipeFault) -> None:
        super().__init__(fault.value)
        self.fault = fault


@dataclass(frozen=True)
class Recipe:
    epochs: float
    batch: int
    alpha0: float
    tau: float
    seed: int

    def __post_init__(self) -> None:
        if not self.epochs > 0:
            raise RecipeError(RecipeFault.EPOCHS)
        if self.batch < 1:
            raise RecipeError(RecipeFault.BATCH)
        if not 0 < self.alpha0 <= MAX_ALPHA0:
            raise RecipeError(RecipeFault.STEP)


def group_count(k: int) -> int:
    return max(1, round(math.sqrt(k)))


def default_probes(k: int) -> int:
    groups = group_count(k)
    return min(groups, max(MIN_PROBES, math.ceil(groups / GROUPS_PER_PROBE)))


def epoch_rows(n: int, epochs: float) -> list[int]:
    whole = math.floor(epochs)
    remainder = round((epochs - whole) * n)
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def nearest_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    low = 1 << math.floor(math.log2(value))
    high = 2 * low
    return low if value / low < high / value else high


def law_recipe(n: int, k: int, epochs: float, seed: int) -> Recipe:
    if n < 1 or k < 1 or not epochs > 0:
        raise RecipeError(RecipeFault.LAW_DOMAIN)
    rate, k_power, n_power, epochs_power = BATCH_LAW
    law_batch = nearest_power_of_two(rate * k**k_power * n**n_power * epochs**epochs_power)
    batch = max(1, min(law_batch, nearest_power_of_two(n / 2), n))
    step, k_power, batch_power, n_power, epochs_power = STEP_LAW
    alpha0 = step * k**k_power * batch**batch_power * n**n_power * epochs**epochs_power
    return Recipe(epochs, batch, min(max(alpha0, MIN_ALPHA0), MAX_ALPHA0), REVIVAL_THRESHOLD, seed)
