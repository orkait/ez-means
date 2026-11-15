import os
import sys
import time

from ezmeans.config import SIFT_BASE
from ezmeans.dataset import read_fvecs
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

ROWS = 1_000_000
CLUSTERS = 4096
PROBES = 4
SEED = 0
EPOCHS = (2, 3)
REPEATS = 3


def main() -> None:
    label = sys.argv[1]
    x = read_fvecs(SIFT_BASE, ROWS)
    cells = []
    for epochs in EPOCHS:
        recipe = law_recipe(ROWS, CLUSTERS, epochs, SEED)
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            ez_means(x, CLUSTERS, recipe, PROBES)
            times.append(time.perf_counter() - start)
        cells.append(f"{min(times):.2f}")
    wait = os.environ.get("OMP_WAIT_POLICY", "default")
    print(f"| {label} | {os.environ.get('OMP_NUM_THREADS')} | {wait} | {' | '.join(cells)} |", flush=True)


if __name__ == "__main__":
    main()
