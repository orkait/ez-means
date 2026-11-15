import math
from dataclasses import dataclass

import faiss
import numpy as np
import numpy.typing as npt

from ezmeans.config import SIFT_BASE
from ezmeans.dataset import Labels, Matrix, read_fvecs
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

ROWS = 1_000_000
SAMPLE = 2_000
EPOCHS = 2
PROBES = 4
SEED = 0
GROUP_ITERATIONS = 10
CLUSTER_COUNTS = (16384, 65536)
TWO_LEVEL_PROBES = (4, 8)
BEAMS = ((2, 4), (3, 6), (4, 8))


@dataclass(frozen=True)
class Outcome:
    evaluations: float
    found: float
    excess: float


def kmeans(points: Matrix, count: int) -> tuple[Matrix, Labels]:
    model = faiss.Kmeans(points.shape[1], count, niter=GROUP_ITERATIONS, seed=SEED, min_points_per_centroid=1)
    model.train(points)
    means = np.asarray(model.centroids, dtype=np.float32)
    _, home = faiss.knn(points, means, 1)
    return means, np.asarray(home[:, 0], dtype=np.int64)


def score(full: npt.NDArray[np.float32], candidates: list[Labels], evaluations: npt.NDArray[np.float64]) -> Outcome:
    truth = full.min(axis=1)
    reached = np.array([full[i, chosen].min() for i, chosen in enumerate(candidates)])
    return Outcome(
        float(evaluations.mean()),
        float((reached <= truth).mean()),
        float(((reached - truth) / np.maximum(truth, 1e-9)).mean()),
    )


def two_level(points: Matrix, centroids: Matrix, full: npt.NDArray[np.float32], probes: int) -> Outcome:
    groups, home = kmeans(centroids, round(math.sqrt(len(centroids))))
    members = [np.flatnonzero(home == g) for g in range(len(groups))]
    _, probed = faiss.knn(points, groups, probes)
    candidates = [np.concatenate([members[g] for g in row]) for row in probed]
    evaluations = np.array([len(groups) + len(c) for c in candidates], dtype=np.float64)
    return score(full, candidates, evaluations)


def three_level(points: Matrix, centroids: Matrix, full: npt.NDArray[np.float32], beams: tuple[int, int]) -> Outcome:
    top_count = round(len(centroids) ** (1 / 3))
    tops, top_home = kmeans(centroids, top_count)
    subs: list[Matrix] = []
    sub_members: list[list[Labels]] = []
    for t in range(top_count):
        inside = np.flatnonzero(top_home == t)
        means, home = kmeans(np.ascontiguousarray(centroids[inside]), max(1, min(len(inside), top_count)))
        subs.append(means)
        sub_members.append([inside[home == s] for s in range(len(means))])
    _, chosen_tops = faiss.knn(points, tops, beams[0])
    candidates, evaluations = [], []
    for i, row in enumerate(chosen_tops):
        options = [(t, s) for t in row for s in range(len(subs[t]))]
        means = np.stack([subs[t][s] for t, s in options])
        gaps = ((means - points[i]) ** 2).sum(axis=1)
        picked = [options[j] for j in np.argsort(gaps)[: beams[1]]]
        chosen = np.concatenate([sub_members[t][s] for t, s in picked])
        candidates.append(chosen)
        evaluations.append(top_count + len(options) + len(chosen))
    return score(full, candidates, np.array(evaluations, dtype=np.float64))


def main() -> None:
    x = read_fvecs(SIFT_BASE, ROWS)
    points = np.ascontiguousarray(x[np.random.default_rng(SEED).choice(ROWS, SAMPLE, replace=False)])
    print(f"SIFT {x.shape}, {SAMPLE} sample points, centroids from ez-means T={EPOCHS}\n")
    print("| k | routing | distance evaluations per point | share of k | true nearest found | mean excess cost |")
    print("|---|---|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        centroids, _ = ez_means(x, k, law_recipe(ROWS, k, EPOCHS, SEED), PROBES)
        full = (
            (points**2).sum(axis=1)[:, None] + (centroids**2).sum(axis=1)[None, :] - 2.0 * points @ centroids.T
        ).astype(np.float32)
        rows = [(f"2-level p{p}", two_level(points, centroids, full, p)) for p in TWO_LEVEL_PROBES]
        rows += [(f"3-level beams {b}", three_level(points, centroids, full, b)) for b in BEAMS]
        for name, outcome in rows:
            print(
                f"| {k} | {name} | {outcome.evaluations:.0f} | {outcome.evaluations / k:.4f} | "
                f"{outcome.found:.4f} | {outcome.excess:.4%} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
