import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ezmeans.config import PAPER_BENCH_DIR, PAPER_TARGETS
from ezmeans.dataset import Matrix
from ezmeans.reference import Assignment, Emeans, Metric, Recipe, Revival, law_recipe
from ezmeans.trainer import ez_means

sys.path.insert(0, str(PAPER_BENCH_DIR))
import paper_data

CLUSTERS = {"deep1b-1M": 16384, "spacev-1M": 16384, "fb-ssnpp-1M": 16384, "deep1b-10M": 65536}
MILLION = ("deep1b-1M", "spacev-1M", "fb-ssnpp-1M")
PAPER_CROSSING = {"deep1b-1M": 2.0, "spacev-1M": 2.4, "fb-ssnpp-1M": 1.4}
SEEDS = (10, 11, 12)
BUDGETS = tuple(round(1.0 + 0.2 * step, 1) for step in range(11))
CHUNK_ROWS = 32768
GROUPED_PROBES = 4
WIDE_PROBES = 8
VERIFY_RATIO = 1.0


Fit = Callable[[Matrix, Recipe, int], Matrix]


@dataclass(frozen=True)
class Arm:
    name: str
    fit: Fit


def python_arm(revival: Revival, assignment: Assignment) -> Fit:
    def fit(train: Matrix, recipe: Recipe, k: int) -> Matrix:
        model = Emeans(k, recipe, revival, VERIFY_RATIO, Metric.EUCLIDEAN, assignment, GROUPED_PROBES)
        return model.train(train)

    return fit


def ez_arm(probes: int) -> Fit:
    def fit(train: Matrix, recipe: Recipe, k: int) -> Matrix:
        centroids, _ = ez_means(train, k, recipe, probes)
        return centroids

    return fit


ARMS = {
    arm.name: arm
    for arm in (
        Arm("e-means", python_arm(Revival.MASS, Assignment.EXACT)),
        Arm("e-means + verify swap", python_arm(Revival.VERIFY, Assignment.EXACT)),
        Arm("e-means + grouped p4", python_arm(Revival.MASS, Assignment.GROUPED)),
        Arm("ez-means", ez_arm(GROUPED_PROBES)),
        Arm("ez-means p8", ez_arm(WIDE_PROBES)),
    )
}


def test_error(centroids: Matrix, test: Matrix) -> float:
    norms = (centroids.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    total = 0.0
    for start in range(0, len(test), CHUNK_ROWS):
        rows = np.ascontiguousarray(test[start : start + CHUNK_ROWS], dtype=np.float32)
        nearest = (norms[None, :] - np.float32(2.0) * (rows @ centroids.T)).argmin(axis=1)
        gap = rows.astype(np.float64) - centroids[nearest].astype(np.float64)
        total += float((gap * gap).sum())
    return total / len(test)


def crossing(train: Matrix, test: Matrix, arm: Arm, k: int, seed: int, target: float) -> tuple[float, float, float]:
    budget, error, seconds = BUDGETS[-1], float("inf"), 0.0
    for budget in BUDGETS:
        recipe = law_recipe(len(train), k, budget, seed)
        start = time.perf_counter()
        centroids = arm.fit(train, recipe, k)
        seconds = time.perf_counter() - start
        error = test_error(centroids, test)
        if error <= target:
            return budget, error, seconds
    return float("inf"), error, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(CLUSTERS), default=list(MILLION))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    options = parser.parse_args()
    chosen = [ARMS[name] for name in options.arms]
    targets = json.loads(PAPER_TARGETS.read_text())["table3"]
    print(
        "| dataset | arm | crossings per seed | crossing, upper median | paper crossing | error / target at crossing "
        "| CPU s at crossing, median | CPU s per seed |"
    )
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for name in options.datasets:
        raw, test = paper_data.load(name)
        train = np.ascontiguousarray(paper_data.rows(raw, name))
        for arm in chosen:
            crossings, ratios, times = [], [], []
            for seed in options.seeds:
                target = float(targets[name]["per_seed"][str(seed)]["test_mse"])
                budget, error, seconds = crossing(train, test, arm, CLUSTERS[name], seed, target)
                crossings.append(budget)
                ratios.append(error / target)
                times.append(seconds)
            upper = sorted(crossings)[len(crossings) // 2]
            print(
                f"| {name} | {arm.name} | {', '.join(f'{c:g}' for c in crossings)} | {upper:g} | "
                f"{PAPER_CROSSING.get(name, '-')} | {statistics.median(ratios):.4f} | {statistics.median(times):.1f} | "
                f"{', '.join(f'{t:.1f}' for t in times)} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
