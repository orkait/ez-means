import argparse
import json
import math
import statistics
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl
from faiss.contrib.e_means.e_means_torch import Emeans as TorchEmeans
from faiss.contrib.e_means.torch_backend import TritonBackend, select_backend
from paper_cpu import BUDGETS, CLUSTERS, PAPER_CROSSING, Arm, Fit, crossing, test_error
from race import lloyd, mse

from ezmeans.config import PAPER_BENCH_DIR, PAPER_TARGETS, SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import CHECK_EVERY, GROUP_ITERATIONS, Recipe, law_recipe

sys.path.insert(0, str(PAPER_BENCH_DIR))
import paper_data

VRAM_FRACTION = 0.5
BLOCK_M = 64
BLOCK_N = 64
SELECT_ROWS = 32
MIN_TILE = 16
WARM_SEED = 99
AUTOMATIC = "auto"
CURVE_SEED = 10
CURVE_EPOCHS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CHUNK_ROWS = 8192
SIFT_ROWS = 1_000_000
SIFT_TEST_ROWS = 100_000
SIFT_SEEDS = (0, 1)
SIFT_BUDGETS = (2.0, 2.2, 2.4, 3.0)
LLOYD_ITERATIONS = 25
device = torch.device("cuda")

Targets = dict[str, dict[str, float]]


@triton.jit
def select_kernel(
    rows,
    row_stride,
    centers,
    center_norms,
    out_top,
    out_norm,
    n,
    D: tl.constexpr,
    DP: tl.constexpr,
    GP: tl.constexpr,
    BM: tl.constexpr,
    P: tl.constexpr,
):
    offs = tl.program_id(0) * BM + tl.arange(0, BM)
    live = offs < n
    cols = tl.arange(0, DP)
    a = tl.load(
        rows + offs[:, None].to(tl.int64) * row_stride + cols[None, :],
        mask=live[:, None] & (cols[None, :] < D),
        other=0.0,
    )
    groups = tl.arange(0, GP)
    score = tl.load(center_norms + groups)[None, :] - 2.0 * tl.dot(
        a, tl.load(centers + groups[None, :] * DP + cols[:, None])
    )
    for j in tl.static_range(P):
        pick = tl.argmin(score, axis=1)
        tl.store(out_top + offs * P + j, pick, mask=live)
        score = tl.where(groups[None, :] == pick[:, None], float("inf"), score)
    wide = a.to(tl.float32)
    tl.store(out_norm + offs, tl.sum(wide * wide, axis=1), mask=live)


@triton.jit
def member_kernel(
    rows,
    row_stride,
    members,
    norms,
    ids,
    pair_point,
    pair_slot,
    block_group,
    block_start,
    group_end,
    group_size,
    out_value,
    out_id,
    width,
    D: tl.constexpr,
    DP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.load(block_group + pid).to(tl.int64)
    start = tl.load(block_start + pid)
    end = tl.load(group_end + g)
    if start >= end:
        return
    offs = start + tl.arange(0, BM)
    live = offs < end
    points = tl.load(pair_point + offs, mask=live, other=0).to(tl.int64)
    cols = tl.arange(0, DP)
    a = tl.load(
        rows + points[:, None] * row_stride + cols[None, :], mask=live[:, None] & (cols[None, :] < D), other=0.0
    )
    best = tl.full((BM,), float("inf"), tl.float32)
    best_id = tl.zeros((BM,), tl.int32)
    base = g * width
    for first in range(0, tl.load(group_size + g), BN):
        lanes = base + first + tl.arange(0, BN)
        score = tl.load(norms + lanes)[None, :] - 2.0 * tl.dot(
            a, tl.load(members + lanes[None, :] * DP + cols[:, None])
        )
        low = tl.min(score, axis=1)
        chosen = tl.load(ids + base + first + tl.argmin(score, axis=1))
        take = low < best
        best = tl.where(take, low, best)
        best_id = tl.where(take, chosen, best_id)
    slot = tl.load(pair_slot + offs, mask=live, other=0)
    tl.store(out_value + slot, best, mask=live)
    tl.store(out_id + slot, best_id, mask=live)


def nearest(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return ((centers**2).sum(1)[None, :] - 2.0 * (points @ centers.T)).argmin(dim=1)


class GroupedBackend(TritonBackend):
    def __init__(self, chunk: int, k: int, d: int, probes: int, seed: int) -> None:
        super().__init__(chunk)
        self.groups = round(math.sqrt(k))
        self.probes = probes
        self.d = d
        self.padded = triton.next_power_of_2(max(d, MIN_TILE))
        self.span = triton.next_power_of_2(max(self.groups, MIN_TILE))
        self.seed = seed
        self.batches = 0

    def regroup(self, c: torch.Tensor) -> None:
        generator = torch.Generator(device=device).manual_seed(self.seed)
        g = c[torch.randperm(len(c), device=device, generator=generator)[: self.groups]].clone()
        for _ in range(GROUP_ITERATIONS):
            owner = nearest(c, g)
            sums = torch.zeros_like(g).index_add_(0, owner, c)
            counts = torch.bincount(owner, minlength=self.groups)
            g = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], g)
        owner = nearest(c, g)
        order = torch.argsort(owner)
        sizes = torch.bincount(owner, minlength=self.groups)
        self.sizes = sizes.int()
        self.width = triton.cdiv(int(sizes.max()), BLOCK_N) * BLOCK_N
        slot = torch.arange(len(c), device=device) - (torch.cumsum(sizes, 0) - sizes)[owner[order]]
        self.ids = torch.zeros((self.groups, self.width), device=device, dtype=torch.int32)
        self.ids[owner[order], slot] = order.int()
        self.valid = torch.zeros((self.groups, self.width), device=device, dtype=torch.bool)
        self.valid[owner[order], slot] = True
        self.gather = self.ids.long()
        self.centers = torch.zeros((self.span, self.padded), device=device, dtype=torch.bfloat16)
        self.centers[: self.groups, : self.d] = g
        self.center_norms = torch.full((self.span,), float("inf"), device=device)
        self.center_norms[: self.groups] = (self.centers[: self.groups].float() ** 2).sum(1)
        self.members = torch.zeros((self.groups, self.width, self.padded), device=device, dtype=torch.bfloat16)

    def prepare_centroids(
        self, centroids: torch.Tensor, dtype: torch.dtype, spherical: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = centroids.float()
        if self.batches % CHECK_EVERY == 0:
            self.regroup(c)
        self.batches += 1
        self.members[:, :, : self.d] = c[self.gather]
        self.norms = torch.where(self.valid, (self.members.float() ** 2).sum(-1), float("inf"))
        prepared: tuple[torch.Tensor, torch.Tensor] = super().prepare_centroids(centroids, dtype, spherical)
        return prepared

    def assign(
        self,
        x: torch.Tensor,
        centroids: torch.Tensor,
        centroid_norm: torch.Tensor | None = None,
        spherical: bool = False,
        distance_out: torch.Tensor | None = None,
        labels_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, p, groups = len(x), self.probes, self.groups
        top = torch.empty(n * p, device=device, dtype=torch.int32)
        row_norm = torch.empty(n, device=device)
        select_kernel[(triton.cdiv(n, SELECT_ROWS),)](
            x,
            x.stride(0),
            self.centers,
            self.center_norms,
            top,
            row_norm,
            n,
            D=self.d,
            DP=self.padded,
            GP=self.span,
            BM=SELECT_ROWS,
            P=p,
        )
        pair_group, order = torch.sort(top.long())
        counts = torch.bincount(pair_group, minlength=groups)
        ends = torch.cumsum(counts, 0)
        blocks = triton.cdiv(counts, BLOCK_M)
        reach = torch.cumsum(blocks, 0)
        bound = triton.cdiv(n * p, BLOCK_M) + groups
        slots = torch.arange(bound, device=device)
        block_group = torch.searchsorted(reach, slots, right=True).clamp(max=groups - 1)
        block_start = (
            ends[block_group] - counts[block_group] + (slots - reach[block_group] + blocks[block_group]) * BLOCK_M
        )
        value = torch.empty(n * p, device=device)
        ident = torch.empty(n * p, device=device, dtype=torch.int32)
        member_kernel[(bound,)](
            x,
            x.stride(0),
            self.members,
            self.norms,
            self.ids,
            (order // p).int(),
            order.int(),
            block_group.int(),
            block_start.int(),
            ends.int(),
            self.sizes,
            value,
            ident,
            self.width,
            D=self.d,
            DP=self.padded,
            BM=BLOCK_M,
            BN=BLOCK_N,
        )
        value, ident = value.view(n, p), ident.view(n, p)
        pick = value.argmin(dim=1, keepdim=True)
        return value.gather(1, pick).squeeze(1) + row_norm, ident.gather(1, pick).squeeze(1).long()


def gpu_fit(xg: torch.Tensor, probes: int) -> Fit:
    def fit(train: Matrix, recipe: Recipe, k: int) -> Matrix:
        model = TorchEmeans(
            xg.shape[1],
            k,
            nepochs=recipe.epochs,
            seed=recipe.seed,
            capture_assignments=False,
            assignment_backend=TritonBackend.name if probes else AUTOMATIC,
        )
        if probes:
            model._execution_backend = GroupedBackend(model.centroid_chunk_size, k, xg.shape[1], probes, recipe.seed)
        centroids, _ = model.train(xg)
        torch.cuda.synchronize()
        result: Matrix = np.ascontiguousarray(centroids.float().cpu().numpy())
        return result

    return fit


def empty_clusters(xg: torch.Tensor, centroids: Matrix) -> int:
    c = torch.from_numpy(centroids).to(device)
    counts = torch.zeros(len(c), device=device, dtype=torch.long)
    for start in range(0, len(xg), CHUNK_ROWS):
        counts += torch.bincount(nearest(xg[start : start + CHUNK_ROWS].float(), c), minlength=len(c))
    return int((counts == 0).sum())


def curve(name: str, train: Matrix, test: Matrix, xg: torch.Tensor, arms: list[Arm], target: float) -> None:
    for arm in arms:
        for epochs in CURVE_EPOCHS:
            start = time.perf_counter()
            centroids = arm.fit(train, law_recipe(len(train), CLUSTERS[name], epochs, CURVE_SEED), CLUSTERS[name])
            seconds = time.perf_counter() - start
            print(
                f"| {name} | {arm.name} | {epochs:g} | {test_error(centroids, test) / target:.4f} | {seconds:.2f} | "
                f"{empty_clusters(xg, centroids)} |",
                flush=True,
            )


def warm(fit: Fit, train: Matrix, k: int, budgets: tuple[float, ...]) -> None:
    for budget in budgets:
        fit(train, law_recipe(len(train), k, budget, WARM_SEED), k)


def sift(k: int, probes: list[int]) -> None:
    train, test = read_fvecs(SIFT_BASE, SIFT_ROWS), read_fvecs(SIFT_LEARN, SIFT_TEST_ROWS)
    reference = {seed: mse(test, lloyd(LLOYD_ITERATIONS)(train, k, seed)) for seed in SIFT_SEEDS}
    xg = torch.from_numpy(train).to(device)
    print("| arm | epochs | test MSE / Lloyd-25, per seed | GPU s, per seed |")
    print("|---|---|---|---|", flush=True)
    for p in probes:
        fit = gpu_fit(xg, p)
        warm(fit, train, k, SIFT_BUDGETS)
        for epochs in SIFT_BUDGETS:
            ratios, times = [], []
            for seed in SIFT_SEEDS:
                start = time.perf_counter()
                centroids = fit(train, law_recipe(len(train), k, epochs, seed), k)
                times.append(time.perf_counter() - start)
                ratios.append(mse(test, centroids) / reference[seed])
            print(
                f"| {f'ez-means GPU p{p}' if p else 'faiss GPU e-means'} | {epochs:g} | "
                f"{', '.join(f'{r:.4f}' for r in ratios)} | {', '.join(f'{t:.3f}' for t in times)} |",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(CLUSTERS), default=["deep1b-1M"])
    parser.add_argument("--probes", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sift", type=int, metavar="K")
    options = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    if options.sift:
        sift(options.sift, options.probes)
        return
    targets = json.loads(PAPER_TARGETS.read_text())["table3"]
    if options.curve:
        print("| dataset | arm | epochs | test MSE / seed-10 target | GPU s | empty clusters |")
        print("|---|---|---|---|---|---|", flush=True)
    else:
        print(
            "| dataset | arm | crossings per seed | crossing, upper median | paper crossing | error / target "
            "| GPU s at crossing, median | GPU s per seed |"
        )
        print("|---|---|---|---|---|---|---|---|", flush=True)
    for name in options.datasets:
        raw, test = paper_data.load(name)
        train = np.ascontiguousarray(paper_data.rows(raw, name))
        xg = torch.from_numpy(train).to(device)
        if options.bf16:
            xg = xg.to(torch.bfloat16)
        k = CLUSTERS[name]
        baseline = f"faiss GPU e-means, {select_backend(xg, xg.dtype, k, xg.shape[1], AUTOMATIC).name}"
        arms = [Arm(f"ez-means GPU p{p}" if p else baseline, gpu_fit(xg, p)) for p in options.probes]
        for arm in arms:
            warm(arm.fit, train, k, CURVE_EPOCHS if options.curve else BUDGETS)
        if options.curve:
            curve(name, train, test, xg, arms, float(targets[name]["per_seed"][str(CURVE_SEED)]["test_mse"]))
        else:
            race(name, train, test, arms, options.seeds, targets[name]["per_seed"])
        del xg
        torch.cuda.empty_cache()


def race(name: str, train: Matrix, test: Matrix, arms: list[Arm], seeds: list[int], targets: Targets) -> None:
    for arm in arms:
        crossings, ratios, times = [], [], []
        for seed in seeds:
            target = float(targets[str(seed)]["test_mse"])
            budget, error, seconds = crossing(train, test, arm, CLUSTERS[name], seed, target)
            crossings.append(budget)
            ratios.append(error / target)
            times.append(seconds)
        upper = sorted(crossings)[len(crossings) // 2]
        print(
            f"| {name} | {arm.name} | {', '.join(f'{c:g}' for c in crossings)} | {upper:g} | "
            f"{PAPER_CROSSING.get(name, '-')} | {statistics.median(ratios):.4f} | {statistics.median(times):.2f} | "
            f"{', '.join(f'{t:.2f}' for t in times)} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
