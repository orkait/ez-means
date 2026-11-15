import math
from dataclasses import dataclass
from enum import Enum

import faiss
import numpy as np
import numpy.typing as npt
from faiss.contrib.e_means.base import _resolve_recipe
from faiss.contrib.e_means.e_means_spherical import normalize_into

from ezmeans import kernel
from ezmeans.dataset import Matrix

BATCH_LAW = "throughput"
REVIVAL_THRESHOLD = 0.1
INIT_USAGE = 2.0**-16
CHECK_EVERY = 5
ALPHA_MIN_FACTOR = 0.1
SPLIT_EPSILON = 1e-4
EPOCH_SEED_MULTIPLIER = 1000003
EPOCH_SEED_MODULUS = 2**32
SWAP_RATIO = 4.0
TOP_TWO = 2

Floats = npt.NDArray[np.float32]
Wide = npt.NDArray[np.float64]
Indices = npt.NDArray[np.int64]


class Revival(Enum):
    MASS = "mass"
    SWAP = "swap"
    MULTI = "multi"
    GAIN = "gain"
    VERIFY = "verify"


class Metric(Enum):
    SPHERICAL = "spherical"
    EUCLIDEAN = "euclidean"


class Assignment(Enum):
    EXACT = "exact"
    INDEXED = "indexed"
    GROUPED = "grouped"
    FUSED = "fused"
    BFLOAT16 = "bfloat16"


GROUP_ITERATIONS = 10
ROTATION_SAMPLE = 100_000
BFLOAT16_ROUNDING = 0x7FFF
BFLOAT16_MASK = 0xFFFF0000


def group_count(k: int) -> int:
    return max(1, round(math.sqrt(k)))


def bfloat16(values: Floats) -> Floats:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + (BFLOAT16_ROUNDING + ((bits >> 16) & 1))) & BFLOAT16_MASK
    return rounded.astype(np.uint32).view(np.float32)


def rotation(x: Floats, seed: int) -> Floats:
    sample = x[np.random.default_rng(seed).choice(len(x), min(len(x), ROTATION_SAMPLE), replace=False)]
    centered = sample.astype(np.float64) - sample.mean(axis=0, dtype=np.float64)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return np.ascontiguousarray(vectors[:, ::-1], dtype=np.float32)


SWAPPING = frozenset({Revival.SWAP, Revival.MULTI, Revival.GAIN, Revival.VERIFY})


@dataclass(frozen=True)
class Recipe:
    epochs: float
    batch: int
    alpha0: float
    tau: float
    seed: int


def epoch_rows(n: int, epochs: float) -> list[int]:
    whole = math.floor(epochs)
    remainder = round((epochs - whole) * n)
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def law_recipe(n: int, k: int, epochs: float, seed: int) -> Recipe:
    batch, alpha0 = _resolve_recipe(n, k, epochs, BATCH_LAW, None, None)
    return Recipe(epochs, int(batch), float(alpha0), REVIVAL_THRESHOLD, seed)


@dataclass
class Swaps:
    distortion: Wide
    removal: Wide
    worst_cost: Wide
    worst_row: Indices
    last_rows: Indices
    last_labels: Indices
    last_cost: Wide
    probe_row: Indices
    probe_gain: Wide
    probe_batches: int = 0
    done: int = 0


def scheduled_alpha(alpha0: float, step: int, total: int) -> float:
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(1, total)))
    return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


def unit(rows: Floats) -> Floats:
    out = np.empty_like(rows)
    normalize_into(rows, out)
    return out


class Emeans:
    def __init__(
        self,
        k: int,
        recipe: Recipe,
        revival: Revival,
        ratio: float = SWAP_RATIO,
        metric: Metric = Metric.SPHERICAL,
        assignment: Assignment = Assignment.EXACT,
        probes: int = 1,
    ) -> None:
        self.k = k
        self.recipe = recipe
        self.revival = revival
        self.ratio = ratio
        self.metric = metric
        self.assignment = assignment
        self.probes = probes
        self.distances = 0.0
        self.searches = 0
        self.groups = np.zeros((0, 0), dtype=np.float32)
        self.blocks: list[Indices] = []
        self.rng = np.random.RandomState(recipe.seed)
        nothing = np.zeros(0, dtype=np.int64)
        self.swaps = Swaps(
            np.zeros(k),
            np.zeros(k),
            np.full(k, -np.inf),
            np.zeros(k, dtype=np.int64),
            nothing,
            nothing,
            np.zeros(0),
            np.full(k, -1, dtype=np.int64),
            np.zeros(k),
        )

    def readout(self, centroids: Floats) -> Floats:
        match self.metric:
            case Metric.SPHERICAL:
                return unit(centroids)
            case Metric.EUCLIDEAN:
                return centroids

    def search(self, rows: Floats, centroids: Floats, depth: int, kind: int) -> tuple[Floats, Indices]:
        match self.assignment:
            case Assignment.EXACT:
                self.distances += len(rows) * self.k
                found, labels = faiss.knn(rows, centroids, depth, metric=kind)
                return found, labels
            case Assignment.INDEXED:
                d = centroids.shape[1]
                lists = group_count(self.k)
                coarse = faiss.IndexFlatIP(d) if kind == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(coarse, d, lists, kind)
                index.cp.min_points_per_centroid = 1
                index.train(centroids)
                index.add(centroids)
                index.nprobe = self.probes
                found, labels = index.search(rows, depth)
                _, probed = index.quantizer.search(rows, self.probes)
                _, homes = index.quantizer.search(centroids, 1)
                sizes = np.bincount(homes[:, 0], minlength=lists)
                self.distances += len(rows) * lists + int(sizes[probed].sum())
                return found, labels
            case Assignment.GROUPED | Assignment.FUSED:
                return self.grouped(rows, centroids, depth, kind)
            case Assignment.BFLOAT16:
                self.distances += len(rows) * self.k
                found, labels = faiss.knn(bfloat16(rows), bfloat16(centroids), depth, metric=kind)
                return found, labels

    def fused(self, rows: Floats, centroids: Floats) -> tuple[Floats, Indices]:
        offsets = np.cumsum([0, *map(len, self.blocks)])
        best, labels, work = kernel.assign(
            rows, self.groups, centroids, np.concatenate(self.blocks), offsets, min(self.probes, len(self.groups))
        )
        self.distances += work / rows.shape[1]
        return best[:, None], labels[:, None]

    def regroup(self, centroids: Floats, kind: int) -> None:
        lists = group_count(self.k)
        kmeans = faiss.Kmeans(
            centroids.shape[1],
            lists,
            niter=GROUP_ITERATIONS,
            seed=self.recipe.seed,
            spherical=kind == faiss.METRIC_INNER_PRODUCT,
            min_points_per_centroid=1,
        )
        kmeans.train(centroids)
        self.groups = np.asarray(kmeans.centroids, dtype=np.float32)
        _, homes = faiss.knn(centroids, self.groups, 1, metric=kind)
        self.blocks = [np.flatnonzero(homes[:, 0] == group) for group in range(lists)]
        self.distances += len(centroids) * lists * GROUP_ITERATIONS

    def grouped(self, rows: Floats, centroids: Floats, depth: int, kind: int) -> tuple[Floats, Indices]:
        if self.searches % CHECK_EVERY == 0:
            self.regroup(centroids, kind)
        self.searches += 1
        lowest = kind == faiss.METRIC_L2
        if self.assignment is Assignment.FUSED and depth == 1 and lowest:
            return self.fused(rows, centroids)
        probes = min(self.probes, len(self.groups))
        _, probed = faiss.knn(rows, self.groups, probes, metric=kind)
        self.distances += len(rows) * len(self.groups)
        worst = np.float32(np.inf if lowest else -np.inf)
        values = np.full((len(rows), probes * depth), worst, dtype=np.float32)
        labels = np.full((len(rows), probes * depth), -1, dtype=np.int64)
        for slot in range(probes):
            for group, block in enumerate(self.blocks):
                members = np.flatnonzero(probed[:, slot] == group)
                if len(members) == 0 or len(block) == 0:
                    continue
                reach = min(depth, len(block))
                found, local = faiss.knn(rows[members], centroids[block], reach, metric=kind)
                self.distances += len(members) * len(block)
                columns = slice(slot * depth, slot * depth + reach)
                values[members, columns] = found
                labels[members, columns] = block[local]
        order = np.argsort(values if lowest else -values, axis=1, kind="stable")[:, :depth]
        return np.take_along_axis(values, order, axis=1), np.take_along_axis(labels, order, axis=1)

    def assign(self, rows: Floats) -> tuple[Wide, Indices, Wide]:
        depth = TOP_TWO if self.revival in SWAPPING else 1
        centroids = self.readout(self.centroids)
        match self.metric:
            case Metric.SPHERICAL:
                scores, labels = self.search(rows, centroids, depth, faiss.METRIC_INNER_PRODUCT)
                costs = 1.0 - scores.astype(np.float64)
            case Metric.EUCLIDEAN:
                distances, labels = self.search(rows, centroids, depth, faiss.METRIC_L2)
                costs = distances.astype(np.float64)
        return costs[:, 0], labels[:, 0].astype(np.int64), costs[:, depth - 1]

    def probe_costs(self, rows: Floats, probes: Floats) -> Wide:
        wide_rows, wide_probes = rows.astype(np.float64), probes.astype(np.float64)
        match self.metric:
            case Metric.SPHERICAL:
                return 1.0 - (wide_rows * wide_probes).sum(axis=1)
            case Metric.EUCLIDEAN:
                gap = wide_rows - wide_probes
                return (gap * gap).sum(axis=1)

    def train(self, x: Floats, init: Floats | None = None) -> Matrix:
        if self.assignment is not Assignment.FUSED:
            return self.fit(x, init)
        turn = rotation(x, self.recipe.seed)
        start = None if init is None else np.ascontiguousarray(init @ turn)
        return np.ascontiguousarray(self.fit(np.ascontiguousarray(x @ turn), start) @ turn.T)

    def fit(self, x: Floats, init: Floats | None) -> Matrix:
        n, d = x.shape
        k, recipe = self.k, self.recipe
        start = x[np.random.RandomState(recipe.seed).permutation(n)[:k]] if init is None else init
        self.centroids = self.readout(start.copy())
        self.usage = np.full(k, INIT_USAGE, dtype=np.float32)
        plan = epoch_rows(n, recipe.epochs)
        total = sum((rows + recipe.batch - 1) // recipe.batch for rows in plan)
        pass_counts = np.zeros(k)
        pass_sums = np.zeros((k, d), dtype=np.float32)
        step = 0
        for epoch, length in enumerate(plan):
            last = epoch == len(plan) - 1
            order = np.random.RandomState(
                (recipe.seed * EPOCH_SEED_MULTIPLIER + epoch) % EPOCH_SEED_MODULUS
            ).permutation(n)
            for begin in range(0, length, recipe.batch):
                batch = order[begin : min(begin + recipe.batch, length)]
                rows = x[batch]
                best, labels, second = self.assign(rows)
                sums, counts = kernel.accumulate(rows, labels, k)
                alpha = scheduled_alpha(recipe.alpha0, step, total)
                self.update(sums, counts, alpha)
                if self.revival in SWAPPING:
                    self.observe(x, batch, labels, best, second, alpha)
                if last:
                    pass_counts += counts
                    pass_sums += sums
                if (step + 1) % CHECK_EVERY == 0:
                    self.revive()
                    if self.revival in SWAPPING and not last:
                        self.swap(x)
                step += 1
        nonempty = pass_counts > 0
        means = np.where(
            nonempty[:, None], pass_sums / np.maximum(pass_counts, 1)[:, None].astype(np.float32), self.centroids
        )
        return self.readout(means.astype(np.float32))

    def update(self, sums: Floats, counts: Floats, alpha: float) -> None:
        touched = counts > 0
        if alpha != 1.0:
            self.usage *= 1.0 - alpha
        retained = self.usage[touched] if alpha != 1.0 else np.zeros_like(counts[touched])
        mass = retained + alpha * counts[touched]
        mean = sums[touched] / counts[touched, None]
        weight = alpha * counts[touched] / mass
        position = self.centroids[touched] + weight[:, None] * (mean - self.centroids[touched])
        position[weight == 1] = mean[weight == 1]
        self.centroids[touched] = position
        self.usage[touched] = mass

    def revive(self) -> None:
        k = self.k
        total = self.usage.sum(dtype="float32")
        threshold = np.divide(
            np.multiply(total, np.float32(self.recipe.tau), dtype="float32"), np.float32(k), dtype="float32"
        )
        unused = self.usage < threshold
        donors = (~unused) & (self.usage > 0)
        count = min(int(unused.sum()), int(donors.sum()))
        if count == 0:
            return
        uniform = np.maximum(self.rng.random_sample(k), np.finfo("float64").tiny)
        keys = np.full(k, -np.inf)
        keys[donors] = np.log(self.usage[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys)[-count:][::-1]
        target = np.flatnonzero(unused)[:count]
        sign = np.ones(self.centroids.shape[1], dtype="float32")
        sign[1::2] = -1.0
        offset = SPLIT_EPSILON * sign
        picked = self.centroids[source]
        half = self.usage[source] / 2.0
        self.centroids[source] = picked * (1.0 - offset)
        self.centroids[target] = picked * (1.0 + offset)
        self.usage[source] = half
        self.usage[target] = half

    def observe(self, x: Floats, batch: Indices, labels: Indices, best: Wide, second: Wide, alpha: float) -> None:
        swaps = self.swaps
        if self.revival is Revival.VERIFY:
            probes = swaps.probe_row[labels]
            fresh = (probes >= 0) & (probes != batch)
            probe = self.probe_costs(x[batch[fresh]], x[probes[fresh]])
            np.add.at(swaps.probe_gain, labels[fresh], np.maximum(0.0, best[fresh] - probe))
            swaps.probe_batches += 1
        decay = 1.0 - alpha
        swaps.distortion *= decay
        swaps.removal *= decay
        np.add.at(swaps.distortion, labels, alpha * best)
        np.add.at(swaps.removal, labels, alpha * (second - best))
        order = np.lexsort((-best, labels))
        ranked = labels[order]
        heads = order[np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])]
        owners = labels[heads]
        worse = best[heads] > swaps.worst_cost[owners]
        swaps.worst_cost[owners[worse]] = best[heads[worse]]
        swaps.worst_row[owners[worse]] = batch[heads[worse]]
        swaps.last_rows, swaps.last_labels, swaps.last_cost = batch, labels, best

    def gains(self, x: Floats) -> Wide:
        swaps = self.swaps
        probe = self.probe_costs(x[swaps.last_rows], x[swaps.worst_row[swaps.last_labels]])
        total = np.zeros(self.k)
        np.add.at(total, swaps.last_labels, np.maximum(0.0, swaps.last_cost - probe))
        return np.where(np.isfinite(swaps.worst_cost), total, 0.0)

    def candidates(self, x: Floats) -> tuple[Wide, Indices, npt.NDArray[np.bool_]]:
        swaps = self.swaps
        found = np.isfinite(swaps.worst_cost)
        match self.revival:
            case Revival.VERIFY:
                proposed = swaps.probe_row >= 0
                merit = np.where(proposed, swaps.probe_gain / max(1, swaps.probe_batches), 0.0)
                return merit, swaps.probe_row, proposed
            case Revival.GAIN:
                return self.gains(x), swaps.worst_row, found
            case Revival.SWAP | Revival.MULTI | Revival.MASS:
                return swaps.distortion, swaps.worst_row, found

    def swap(self, x: Floats) -> None:
        swaps = self.swaps
        limit = 1 if self.revival is Revival.SWAP else self.k // 2
        merit, seeds, valid = self.candidates(x)
        ceiling = swaps.removal.max()
        removals = iter(np.argsort(swaps.removal))
        used: set[int] = set()
        made = 0
        for split in (int(i) for i in np.argsort(-merit)):
            if made == limit:
                break
            if split in used or not valid[split]:
                continue
            removed = next((int(j) for j in removals if int(j) != split and int(j) not in used), None)
            if removed is None or merit[split] <= self.ratio * swaps.removal[removed]:
                break
            self.centroids[removed] = x[seeds[split]]
            half = self.usage[split] / 2.0
            self.usage[split] = half
            self.usage[removed] = half
            for slot in (split, removed):
                swaps.distortion[slot] = 0.0
                swaps.removal[slot] = ceiling
                used.add(slot)
            made += 1
        swaps.done += made
        if self.revival is Revival.VERIFY:
            swaps.probe_row = np.where(np.isfinite(swaps.worst_cost), swaps.worst_row, -1)
            swaps.probe_row[list(used)] = -1
            swaps.probe_gain[:] = 0.0
            swaps.probe_batches = 0
        swaps.worst_cost[:] = -np.inf
