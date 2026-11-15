import math

import torch

from ezmeans.gpu import kernels
from ezmeans.gpu.constants import MAX_ROW_NORM
from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.recipe import ALPHA_MIN_FACTOR, CHECK_EVERY, INIT_USAGE, SPLIT_EPSILON, Recipe, default_probes, epoch_rows


def scheduled(recipe: Recipe, step: int, total: int) -> float:
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))
    return recipe.alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


def revive(centroids: torch.Tensor, usage: torch.Tensor, recipe: Recipe, generator: torch.Generator) -> None:
    unused = usage < usage.sum() * recipe.tau / len(usage)
    donors = ~unused & (usage > 0)
    count = int(torch.minimum(unused.sum(), donors.sum()))
    if count == 0:
        return
    draw = torch.rand(len(usage), generator=generator, device=usage.device).clamp(min=torch.finfo(torch.float32).tiny)
    keys = torch.where(donors, usage.log() - (-draw.log()).log(), -torch.inf)
    sources = keys.topk(count).indices
    targets = (~unused).to(torch.uint8).argsort(stable=True)[:count]
    split = SPLIT_EPSILON * (1 - 2 * (torch.arange(centroids.shape[1], device=centroids.device) % 2))
    picked = centroids[sources]
    centroids[sources] = picked * (1 - split)
    centroids[targets] = picked * (1 + split)
    usage[sources] /= 2
    usage[targets] = usage[sources]


def bounded(rows: torch.Tensor) -> bool:
    return bool(torch.linalg.vector_norm(rows, dim=1, dtype=torch.float32).max() < MAX_ROW_NORM)


def train(
    x: torch.Tensor, k: int, recipe: Recipe, probes: int | None = None, init: torch.Tensor | None = None
) -> torch.Tensor:
    index = GroupIndex(k, x.shape[-1], default_probes(k) if probes is None else probes, recipe.seed, x.device)
    index.check(x, torch.bfloat16, None)
    n = len(x)
    plan = epoch_rows(n, recipe.epochs)
    if not 1 <= k <= n or min(plan) < 1:
        raise GpuInputError(Fault.BAD_SIZE)
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
    sums, counts = torch.zeros_like(centroids), torch.zeros_like(usage)
    pass_sums, pass_counts = torch.zeros_like(centroids), torch.zeros_like(usage)
    total = sum(math.ceil(rows / recipe.batch) for rows in plan)
    step = 0
    with torch.cuda.device(index.device):
        for epoch, rows in enumerate(plan):
            order = torch.randperm(n, generator=generator, device=index.device)
            for begin in range(0, rows, recipe.batch):
                batch = x[order[begin : min(begin + recipe.batch, rows)]]
                if step % CHECK_EVERY == 0:
                    index.regroup(centroids)
                else:
                    index.refresh(centroids)
                kernels.accumulate(batch, index.assign(batch), sums, counts)
                alpha = scheduled(recipe, step, total)
                kernels.ema(centroids, usage, sums, counts, (pass_sums, pass_counts), alpha, epoch == len(plan) - 1)
                if (step + 1) % CHECK_EVERY == 0:
                    revive(centroids, usage, recipe, generator)
                step += 1
    return torch.where((pass_counts > 0)[:, None], pass_sums / pass_counts.clamp(min=1)[:, None], centroids)
