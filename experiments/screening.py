import math
from dataclasses import dataclass

import faiss
import numpy as np
import numpy.typing as npt
from scipy.linalg import hadamard

from ezmeans.config import DEEP_BASE, SIFT_BASE
from ezmeans.dataset import Matrix, read_fbin, read_fvecs
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

ROWS = 1_000_000
SAMPLE = 2_000
CLUSTERS = 4096
EPOCHS = 2
PROBES = 4
SEED = 0
GROUP_ITERATIONS = 10
CHUNK = 16
BLOCKS = (4, 8)
EPSILONS = (1.9, 3.0)
QUERY_BITS: tuple[int | None, ...] = (None, 8, 4)

Wide = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Tally:
    screened: float
    kept: float
    float_dims: float


def chunked_dims(gaps: Wide, best: float) -> Wide:
    count, d = gaps.shape
    partial = np.cumsum(gaps.reshape(count, d // CHUNK, CHUNK).sum(axis=2), axis=1)
    beyond = partial > best
    first = np.where(beyond.any(axis=1), beyond.argmax(axis=1) + 1, d // CHUNK)
    return np.asarray(first * CHUNK, dtype=np.float64)


def walsh(d: int) -> Wide:
    signs: Wide = np.asarray(np.random.default_rng(SEED).choice((-1.0, 1.0), d), dtype=np.float64)
    walsh_matrix: Wide = np.asarray(hadamard(d), dtype=np.float64)
    turn: Wide = walsh_matrix * signs[None, :] / math.sqrt(d)
    return turn


def study(name: str, x: Matrix) -> None:
    d = x.shape[1]
    width = 1 << (d - 1).bit_length()
    centroids, _ = ez_means(x, CLUSTERS, law_recipe(len(x), CLUSTERS, EPOCHS, SEED), PROBES)
    groups_model = faiss.Kmeans(d, round(math.sqrt(CLUSTERS)), niter=GROUP_ITERATIONS, seed=SEED)
    groups_model.train(centroids)
    groups = np.asarray(groups_model.centroids, dtype=np.float32)
    _, home = faiss.knn(centroids, groups, 1)
    home = home[:, 0]
    points = x[np.random.default_rng(SEED).choice(len(x), SAMPLE, replace=False)]
    _, probed = faiss.knn(points, groups, PROBES)
    turn = walsh(width)
    padded = np.zeros((len(centroids), width))
    padded[:, :d] = centroids - groups[home]
    residual_norm = np.linalg.norm(padded, axis=1)
    rotated = padded @ turn.T / np.maximum(residual_norm, 1e-12)[:, None]
    codes = np.sign(rotated) / math.sqrt(width)
    alignment = (codes * rotated).sum(axis=1)
    totals: dict[str, list[Tally]] = {}
    for i, row in enumerate(points.astype(np.float64)):
        chosen = np.flatnonzero(np.isin(home, probed[i]))
        cand = centroids[chosen].astype(np.float64)
        gaps = (row - cand) ** 2
        full = gaps.sum(axis=1)
        best = float(full.min())
        nearest = full <= best
        base = chunked_dims(gaps, best)
        totals.setdefault("exact early abandon (today)", []).append(Tally(0.0, 1.0, float(base.mean())))
        for block in BLOCKS:
            means_x = row.reshape(d // block, block).mean(axis=1)
            means_c = cand.reshape(len(cand), d // block, block).mean(axis=2)
            bound = block * ((means_c - means_x) ** 2).sum(axis=1)
            alive = bound <= best
            dims = d / block + np.where(alive, base, 0.0)
            totals.setdefault(f"PAA bound, block {block}", []).append(
                Tally(float(1 - alive.mean()), float(alive[nearest].all()), float(dims.mean()))
            )
        query = np.zeros((len(cand), width))
        query[:, :d] = row - groups[home[chosen]]
        query_norm = np.linalg.norm(query, axis=1)
        turned = query @ turn.T
        spread = np.sqrt(np.maximum(1 - alignment[chosen] ** 2, 0) / alignment[chosen] ** 2) / math.sqrt(width - 1)
        for bits in QUERY_BITS:
            if bits is None:
                shown = turned
            else:
                low = turned.min(axis=1, keepdims=True)
                step = (turned.max(axis=1, keepdims=True) - low) / (2**bits - 1)
                shown = low + step * np.round((turned - low) / np.maximum(step, 1e-30))
            estimate = (shown * codes[chosen]).sum(axis=1) / np.maximum(query_norm, 1e-12) / alignment[chosen]
            for epsilon in EPSILONS:
                inner = np.minimum(estimate + epsilon * spread, 1.0)
                lower = query_norm**2 + residual_norm[chosen] ** 2 - 2 * query_norm * residual_norm[chosen] * inner
                alive = lower <= best
                dims = np.where(alive, base, 0.0)
                query_label = "float query" if bits is None else f"{bits}-bit query"
                totals.setdefault(f"RaBitQ 1-bit, {query_label}, eps {epsilon}", []).append(
                    Tally(float(1 - alive.mean()), float(alive[nearest].all()), float(dims.mean()))
                )
    print(f"\n{name}: {x.shape}, k {CLUSTERS}, grouped p{PROBES}, {SAMPLE} points, oracle best distance\n")
    print("| screen | candidates screened out | true nearest kept | float dims per candidate | vs today |")
    print("|---|---|---|---|---|")
    today = np.mean([t.float_dims for t in totals["exact early abandon (today)"]])
    for label, tallies in totals.items():
        dims = float(np.mean([t.float_dims for t in tallies]))
        print(
            f"| {label} | {np.mean([t.screened for t in tallies]):.3f} | {np.mean([t.kept for t in tallies]):.4f} | "
            f"{dims:.1f} | {dims / today:.2f} |",
            flush=True,
        )


def main() -> None:
    study("SIFT", read_fvecs(SIFT_BASE, ROWS))
    study("DEEP", read_fbin(DEEP_BASE, 0, ROWS))


if __name__ == "__main__":
    main()
