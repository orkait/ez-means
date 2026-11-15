import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import faiss
import numpy as np
from scipy.linalg import hadamard

from ezmeans import kernel
from ezmeans.config import DINO_BASE, SIFT_BASE, SIFT_LEARN, SSNPP_BASE
from ezmeans.dataset import Matrix, read_bvecs, read_fvecs, read_u8bin
from ezmeans.reference import Assignment, Emeans, Metric, Revival, law_recipe, rotation
from ezmeans.trainer import ez_means

CLUSTERS = 1024
EPOCHS = 2
PROBES = 4
SEED = 0
CHECKPOINT = 16
WORK_ROWS = 20_000
PRUNE_ROWS = 2_000
AD_EPSILONS = (1.0, 2.1)
SHRINKS = (4, 2)


RANDOM_SPACE = "random Hadamard"


@dataclass(frozen=True)
class Rule:
    name: str
    epsilon: float | None


@dataclass(frozen=True)
class Source:
    train: Callable[[], Matrix]
    test: Callable[[], Matrix]


SOURCES = {
    "sift": Source(lambda: read_fvecs(SIFT_BASE, 1_000_000), lambda: read_fvecs(SIFT_LEARN, 100_000)),
    "ssnpp": Source(lambda: read_u8bin(SSNPP_BASE, 0, 500_000), lambda: read_u8bin(SSNPP_BASE, 900_000, 100_000)),
    "dino": Source(lambda: read_bvecs(DINO_BASE, 0, 200_000), lambda: read_bvecs(DINO_BASE, 900_000, 100_000)),
}


def hadamard_rotation(d: int) -> Matrix:
    signs = np.random.default_rng(SEED).choice((-1.0, 1.0), d)
    walsh = np.asarray(hadamard(d), dtype=np.float64)
    return (signs[:, None] * walsh / math.sqrt(d)).astype(np.float32)


def mse(x: Matrix, centroids: Matrix) -> float:
    distances, _ = faiss.knn(x, np.ascontiguousarray(centroids, dtype=np.float32), 1)
    return float(distances.astype(np.float64).mean())


def train_ez(x: Matrix) -> Matrix:
    centroids, _ = ez_means(x, CLUSTERS, law_recipe(len(x), CLUSTERS, EPOCHS, SEED), PROBES)
    return centroids


def rotation_work(rows: Matrix, centroids: Matrix, spaces: dict[str, Matrix]) -> None:
    d = rows.shape[1]
    print("\nIdea 3: rotation before exact early abandon (grouped p4)\n")
    print("| space | centroid-dims per row | share of brute force k*d | kernel ms |")
    print("|---|---|---|---|")
    for name, basis in spaces.items():
        x, c = np.ascontiguousarray(rows @ basis), np.ascontiguousarray(centroids @ basis)
        model = Emeans(
            CLUSTERS,
            law_recipe(len(x), CLUSTERS, 1, SEED),
            Revival.MASS,
            metric=Metric.EUCLIDEAN,
            assignment=Assignment.FUSED,
            probes=PROBES,
        )
        model.regroup(c, faiss.METRIC_L2)
        offsets = np.cumsum([0, *map(len, model.blocks)])
        start = time.perf_counter()
        _, _, work = kernel.assign(x, model.groups, c, np.concatenate(model.blocks), offsets, PROBES)
        seconds = time.perf_counter() - start
        print(f"| {name} | {work / len(x):.0f} | {work / len(x) / (CLUSTERS * d):.4f} | {seconds * 1e3:.1f} |")


def pruning(rows: Matrix, centroids: Matrix, spaces: dict[str, Matrix]) -> None:
    d = rows.shape[1]
    print("\nIdea 2: exact against probabilistic (ADSampling) pruning, all k centroids, oracle bound\n")
    print("| space | rule | dims computed / brute force | true nearest kept |")
    print("|---|---|---|---|")
    for name, basis in spaces.items():
        x, c = (rows @ basis).astype(np.float64), (centroids @ basis).astype(np.float64)
        _, nearest = faiss.knn(x.astype(np.float32), c.astype(np.float32), 1)
        best = ((x - c[nearest[:, 0]]) ** 2).sum(axis=1)[:, None]
        rules = [Rule("exact", None)]
        if name == RANDOM_SPACE:
            rules += [Rule(f"ADSampling eps {eps}", eps) for eps in AD_EPSILONS]
        alive = {rule: np.ones((len(x), len(c)), dtype=bool) for rule in rules}
        work = dict.fromkeys(rules, 0)
        partial = np.zeros((len(x), len(c)))
        for m in range(CHECKPOINT, d + 1, CHECKPOINT):
            partial += ((x[:, None, m - CHECKPOINT : m] - c[None, :, m - CHECKPOINT : m]) ** 2).sum(axis=2)
            for rule in rules:
                work[rule] += int(alive[rule].sum()) * CHECKPOINT
                if rule.epsilon is None:
                    alive[rule] &= partial <= best
                else:
                    alive[rule] &= partial * d / m <= best * (1.0 + rule.epsilon / math.sqrt(m)) ** 2
        for rule in rules:
            kept = alive[rule][np.arange(len(x)), nearest[:, 0]].mean()
            print(f"| {name} | {rule.name} | {work[rule] / (len(x) * len(c) * d):.4f} | {kept:.4f} |")


def projection(train: Matrix, test: Matrix, full: Matrix, full_seconds: float) -> None:
    d = train.shape[1]
    turn = hadamard_rotation(d)
    base = mse(test, full)
    print("\nIdea 1: random Hadamard projection to d', ez-means there, one full-dim lift\n")
    print("| dims | test MSE / full | project s | ez-means s | lift s | total s | total / full |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {d} (full) | 1.0000 | 0 | {full_seconds:.2f} | 0 | {full_seconds:.2f} | 1.00 |")
    for shrink in SHRINKS:
        kept = d // shrink
        basis = np.ascontiguousarray(turn[:, :kept])
        start = time.perf_counter()
        small = np.ascontiguousarray(train @ basis)
        projected = time.perf_counter()
        centres = train_ez(small)
        trained = time.perf_counter()
        _, labels = faiss.knn(small, centres, 1)
        sums, counts = kernel.accumulate(train, labels[:, 0], CLUSTERS)
        lifted = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centres @ basis.T)
        done = time.perf_counter()
        total = done - start
        print(
            f"| {kept} | {mse(test, lifted) / base:.4f} | {projected - start:.2f} | {trained - projected:.2f} | "
            f"{done - trained:.2f} | {total:.2f} | {total / full_seconds:.2f} |",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=sorted(SOURCES), required=True)
    source = SOURCES[parser.parse_args().data]
    train, test = source.train(), source.test()
    d = train.shape[1]
    print(f"train {train.shape}, test {test.shape}, k {CLUSTERS}, T={EPOCHS}, p{PROBES}")
    start = time.perf_counter()
    full = train_ez(train)
    full_seconds = time.perf_counter() - start
    spaces = {"raw": np.eye(d, dtype=np.float32), "PCA": rotation(train, SEED), RANDOM_SPACE: hadamard_rotation(d)}
    rotation_work(train[:WORK_ROWS], full, spaces)
    pruning(train[:PRUNE_ROWS], full, spaces)
    projection(train, test, full, full_seconds)


if __name__ == "__main__":
    main()
