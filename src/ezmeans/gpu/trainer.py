import math
from dataclasses import dataclass, field

import torch

from ezmeans.gpu import kernels
from ezmeans.gpu.constants import MAX_ROW_NORM
from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.gpu.kernels import Schedule
from ezmeans.layout import Layout, default_layout
from ezmeans.recipe import (
    ALPHA_MIN_FACTOR,
    CHECK_EVERY,
    INIT_USAGE,
    MAX_ALPHA0,
    SPLIT_EPSILON,
    Recipe,
    epoch_rows,
    revival_threshold,
)


@dataclass(frozen=True)
class StepKind:
    regroup: bool
    revive: bool


def scheduled(recipe: Recipe, step: int, total: int) -> float:
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))
    return recipe.alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


def schedule(recipe: Recipe, plan: list[int], device: torch.device) -> Schedule:
    lasts = [epoch == len(plan) - 1 for epoch, rows in enumerate(plan) for _ in range(0, rows, recipe.batch)]
    alphas = [scheduled(recipe, step, len(lasts)) for step in range(len(lasts))]

    def column(values: list[float] | list[bool], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    carries = [0.0 if alpha == MAX_ALPHA0 else 1.0 - alpha for alpha in alphas]
    fades = [1.0 if alpha == MAX_ALPHA0 else 1.0 - alpha for alpha in alphas]
    return (
        column(alphas, torch.float32),
        column(carries, torch.float32),
        column(fades, torch.float32),
        column(lasts, torch.int32),
    )


@dataclass
class Steps:
    x: torch.Tensor
    index: GroupIndex
    recipe: Recipe
    generator: torch.Generator
    centroids: torch.Tensor
    usage: torch.Tensor
    sums: torch.Tensor
    counts: torch.Tensor
    passes: tuple[torch.Tensor, torch.Tensor]
    schedule: Schedule
    clock: torch.Tensor
    picked: torch.Tensor
    batch: torch.Tensor
    graphs: dict[StepKind, torch.cuda.CUDAGraph] = field(default_factory=dict)
    seen: set[StepKind] = field(default_factory=set)

    def run(self, rows: torch.Tensor, kind: StepKind) -> None:
        if kind.regroup:
            self.index.regroup(self.centroids)
        else:
            self.index.refresh(self.centroids)
        kernels.accumulate(rows, self.index.assign(rows), self.sums, self.counts)
        kernels.ema(self.centroids, self.usage, self.sums, self.counts, self.passes, self.schedule, self.clock)
        self.clock.add_(1)
        if kind.revive:
            revive(self.centroids, self.usage, self.recipe, self.generator)

    def gathered(self, kind: StepKind) -> None:
        torch.index_select(self.x, 0, self.picked, out=self.batch)
        self.run(self.batch, kind)

    def full(self, indices: torch.Tensor, kind: StepKind) -> None:
        self.picked.copy_(indices)
        graph = self.graphs.get(kind)
        if graph is not None:
            graph.replay()
        elif kind in self.seen:
            self.graphs[kind] = self.capture(kind)
        else:
            self.gathered(kind)
            self.seen.add(kind)

    def capture(self, kind: StepKind) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        stream = torch.Stream(self.x.device)
        stream.wait_stream(torch.cuda.current_stream())
        with stream:
            graph.capture_begin()
            self.gathered(kind)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        return graph


def revive(centroids: torch.Tensor, usage: torch.Tensor, recipe: Recipe, generator: torch.Generator) -> None:
    k = len(usage)
    unused = usage < usage.sum() * revival_threshold(recipe.tau, centroids.shape[1]) / k
    donors = ~unused & (usage > 0)
    chosen = torch.arange(k, device=usage.device) < torch.minimum(unused.sum(), donors.sum())
    draw = torch.rand(k, generator=generator, device=usage.device).clamp(min=torch.finfo(torch.float32).tiny)
    keys = torch.where(donors, usage.log() - (-draw.log()).log(), -torch.inf)
    sources = torch.where(chosen, keys.sort(descending=True).indices, k)
    targets = torch.where(chosen, (~unused).to(torch.uint8).argsort(stable=True), k)
    origin = torch.arange(k + 1, device=usage.device)
    origin[targets] = sources
    sign = torch.zeros(k + 1, device=usage.device)
    sign[sources] = -1.0
    sign[targets] = 1.0
    split = SPLIT_EPSILON * (1 - 2 * (torch.arange(centroids.shape[1], device=centroids.device) % 2))
    centroids.copy_(centroids[origin[:k]] * (1 + sign[:k, None] * split))
    usage.copy_(usage[origin[:k]] / torch.where(sign[:k] != 0, 2.0, 1.0))


def bounded(rows: torch.Tensor) -> bool:
    return bool(torch.linalg.vector_norm(rows, dim=1, dtype=torch.float32).max() < MAX_ROW_NORM)


def train(
    x: torch.Tensor, k: int, recipe: Recipe, layout: Layout | None = None, init: torch.Tensor | None = None
) -> torch.Tensor:
    n = len(x)
    plan = epoch_rows(n, recipe.epochs)
    if not 1 <= k <= n or min(plan) < 1:
        raise GpuInputError(Fault.BAD_SIZE)
    chosen = default_layout(k) if layout is None else layout
    index = GroupIndex(k, x.shape[-1], chosen, recipe.seed, x.device)
    index.check(x, torch.bfloat16, None)
    if not bounded(x):
        raise GpuInputError(Fault.NOT_FINITE)
    generator = torch.Generator(index.device).manual_seed(recipe.seed)
    if init is None:
        centroids = x[torch.randperm(n, generator=generator, device=index.device)[:k]].float()
    else:
        index.check(init, torch.float32, k)
        if init.shape[1] != index.dims:
            raise GpuInputError(Fault.WRONG_SHAPE)
        if not bounded(init):
            raise GpuInputError(Fault.NOT_FINITE)
        centroids = init.clone(memory_format=torch.contiguous_format)
    usage = torch.full((k,), INIT_USAGE, device=index.device)
    pass_sums, pass_counts = torch.zeros_like(centroids), torch.zeros_like(usage)
    steps = Steps(
        x,
        index,
        recipe,
        generator,
        centroids,
        usage,
        torch.zeros_like(centroids),
        torch.zeros_like(usage),
        (pass_sums, pass_counts),
        schedule(recipe, plan, index.device),
        torch.zeros(1, dtype=torch.int32, device=index.device),
        torch.empty(recipe.batch, dtype=torch.int64, device=index.device),
        torch.empty((recipe.batch, x.shape[1]), dtype=x.dtype, device=index.device),
    )
    step = 0
    with torch.cuda.device(index.device):
        for rows in plan:
            order = torch.randperm(n, generator=generator, device=index.device)
            for begin in range(0, rows, recipe.batch):
                indices = order[begin : min(begin + recipe.batch, rows)]
                kind = StepKind(step % CHECK_EVERY == 0, (step + 1) % CHECK_EVERY == 0)
                if len(indices) == recipe.batch:
                    steps.full(indices, kind)
                else:
                    steps.run(x[indices], kind)
                step += 1
    return torch.where((pass_counts > 0)[:, None], pass_sums / pass_counts.clamp(min=1)[:, None], centroids)
