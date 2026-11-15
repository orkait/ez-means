from typing import Annotated, NamedTuple

import torch
import triton
import triton.language as tl

from ezmeans.gpu.constants import (
    ACCUMULATE_ROWS,
    BUCKET_BLOCK,
    DIM_CHUNK,
    EMA_ROWS,
    GROUP_BLOCK,
    LLOYD_ROWS,
    MEMBER_TILE,
    MIN_TILE,
    PAIR_ROWS,
    REDUCE_BLOCK,
    REFRESH_ROWS,
    RELAXED,
    SELECT_ROWS,
    UNREACHABLE,
)
from ezmeans.recipe import group_count


class Geometry(NamedTuple):
    dims: int
    padded: int
    chunk: int
    clusters: int
    groups: int
    span: int
    probes: int
    tile: int


class Matrix[T](NamedTuple):
    data: T
    stride: int
    rows: int


class Vectors[T](NamedTuple):
    data: T
    norms: T


class Members[T](NamedTuple):
    data: T
    norms: T
    order: T
    ends: T
    sizes: T


class Lloyd[T](NamedTuple):
    dots: T
    owner: T
    sums: T
    sizes: T


class Pairs[T](NamedTuple):
    top: T
    items: T
    counts: T
    ends: T
    blocks: T
    fill: T
    value: T
    ident: T
    total: int


class State[T](NamedTuple):
    centroids: T
    usage: T
    sums: T
    counts: T
    pass_sums: T
    pass_counts: T


class Rate(NamedTuple):
    alpha: float
    keep: float


class Mode(NamedTuple):
    decays: bool
    last: bool


@triton.jit
def _select(
    rows: Matrix[tl.tensor],
    centers: Vectors[tl.tensor],
    pairs: Pairs[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < rows.rows
    groups = tl.arange(0, geometry.span)
    dots = tl.zeros((block, geometry.span), tl.float32)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        a = tl.load(
            rows.data + offs[:, None].to(tl.int64) * rows.stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < geometry.dims),
            other=0.0,
        )
        dots += tl.dot(a, tl.load(centers.data + groups[None, :] * geometry.padded + cols[:, None]))
    score = tl.load(centers.norms + groups)[None, :] - 2.0 * dots
    for j in tl.static_range(geometry.probes):
        pick = tl.argmin(score, axis=1)
        tl.store(pairs.top + offs * geometry.probes + j, pick, mask=live)
        tl.atomic_add(pairs.counts + pick, 1, mask=live, sem=RELAXED)
        picked = (groups[None, :] <= pick[:, None]) & (groups[None, :] >= pick[:, None])
        score = tl.where(picked, float("inf"), score)


@triton.jit
def _plan(
    pairs: Pairs[tl.tensor], geometry: "Annotated[Geometry, tl.constexpr]", block: "Annotated[int, tl.constexpr]"
) -> None:
    groups = tl.arange(0, geometry.span)
    live = groups < geometry.groups
    size = tl.load(pairs.counts + groups, mask=live, other=0)
    tl.store(pairs.ends + groups, tl.cumsum(size, axis=0), mask=live)
    tl.store(pairs.blocks + groups, tl.cumsum((size + block - 1) // block, axis=0), mask=live)


@triton.jit
def _bucket(pairs: Pairs[tl.tensor], block: "Annotated[int, tl.constexpr]") -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < pairs.total
    key = tl.load(pairs.top + offs, mask=live, other=0)
    start = tl.load(pairs.ends + key, mask=live, other=0) - tl.load(pairs.counts + key, mask=live, other=0)
    tl.store(pairs.items + start + tl.atomic_add(pairs.fill + key, 1, mask=live, sem=RELAXED), offs, mask=live)


@triton.jit
def _member(
    rows: Matrix[tl.tensor],
    pairs: Pairs[tl.tensor],
    members: Members[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    pid = tl.program_id(0)
    groups = tl.arange(0, geometry.span)
    reach = tl.load(pairs.blocks + groups, mask=groups < geometry.groups, other=UNREACHABLE)
    group = tl.sum((reach <= pid).to(tl.int32), axis=0)
    if group >= geometry.groups:
        return
    count = tl.load(pairs.counts + group)
    first_block = tl.load(pairs.blocks + group) - (count + block - 1) // block
    end = tl.load(pairs.ends + group)
    offs = end - count + (pid - first_block) * block + tl.arange(0, block)
    live = offs < end
    item = tl.load(pairs.items + offs, mask=live, other=0)
    point = (item // geometry.probes).to(tl.int64)
    hi = tl.load(members.ends + group)
    lo = hi - tl.load(members.sizes + group)
    best = tl.full((block,), float("inf"), tl.float32)
    best_id = tl.zeros((block,), tl.int32)
    for first in tl.range(lo, hi, geometry.tile):
        lanes = first + tl.arange(0, geometry.tile)
        inside = lanes < hi
        dots = tl.zeros((block, geometry.tile), tl.float32)
        for base in tl.static_range(0, geometry.padded, geometry.chunk):
            cols = base + tl.arange(0, geometry.chunk)
            a = tl.load(
                rows.data + point[:, None] * rows.stride + cols[None, :],
                mask=live[:, None] & (cols[None, :] < geometry.dims),
                other=0.0,
            )
            b = tl.load(
                members.data + lanes[None, :].to(tl.int64) * geometry.padded + cols[:, None],
                mask=inside[None, :],
                other=0.0,
            )
            dots += tl.dot(a, b)
        score = tl.load(members.norms + lanes, mask=inside, other=float("inf"))[None, :] - 2.0 * dots
        low = tl.min(score, axis=1)
        lane = first + tl.argmin(score, axis=1)
        take = low < best
        best = tl.where(take, low, best)
        best_id = tl.where(take, tl.load(members.order + lane, mask=lane < hi, other=0), best_id)
    tl.store(pairs.value + item, best, mask=live)
    tl.store(pairs.ident + item, best_id, mask=live)


@triton.jit
def _reduce(
    pairs: Pairs[tl.tensor],
    labels: tl.tensor,
    count: int,
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < count
    best = tl.full((block,), float("inf"), tl.float32)
    best_id = tl.zeros((block,), tl.int32)
    for j in tl.static_range(geometry.probes):
        candidate = tl.load(pairs.value + offs * geometry.probes + j, mask=live, other=float("inf"))
        take = candidate < best
        best = tl.where(take, candidate, best)
        best_id = tl.where(take, tl.load(pairs.ident + offs * geometry.probes + j, mask=live, other=0), best_id)
    tl.store(labels + offs, best_id.to(tl.int64), mask=live)


@triton.jit
def _refresh(
    centroids: Matrix[tl.tensor],
    members: Members[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < geometry.clusters
    source = tl.load(members.order + offs, mask=live, other=0).to(tl.int64)
    norms = tl.zeros((block,), tl.float32)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        rounded = tl.load(
            centroids.data + source[:, None] * centroids.stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < geometry.dims),
            other=0.0,
        ).to(tl.bfloat16)
        tl.store(
            members.data + offs[:, None].to(tl.int64) * geometry.padded + cols[None, :], rounded, mask=live[:, None]
        )
        wide = rounded.to(tl.float32)
        norms += tl.sum(wide * wide, axis=1)
    tl.store(members.norms + offs, norms, mask=live)


@triton.jit
def _seed(
    centroids: Matrix[tl.tensor],
    picks: tl.tensor,
    centers: Vectors[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    groups = tl.program_id(0) * block + tl.arange(0, block)
    live = groups < geometry.groups
    source = tl.load(picks + groups, mask=live, other=0).to(tl.int64)
    norms = tl.zeros((block,), tl.float32)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        center = tl.load(
            centroids.data + source[:, None] * centroids.stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < geometry.dims),
            other=0.0,
        )
        tl.store(centers.data + groups[:, None] * geometry.padded + cols[None, :], center, mask=live[:, None])
        norms += tl.sum(center * center, axis=1)
    tl.store(centers.norms + groups, norms, mask=live)


@triton.jit
def _home(
    centroids: Matrix[tl.tensor],
    centers: Vectors[tl.tensor],
    lloyd: Lloyd[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < geometry.clusters
    groups = tl.arange(0, geometry.span)
    product = tl.load(
        lloyd.dots + offs[:, None].to(tl.int64) * geometry.span + groups[None, :], mask=live[:, None], other=0.0
    )
    home = tl.argmin(tl.load(centers.norms + groups)[None, :] - 2.0 * product, axis=1)
    tl.store(lloyd.owner + offs, home, mask=live)
    tl.atomic_add(lloyd.sizes + home, 1, mask=live, sem=RELAXED)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        inside = live[:, None] & (cols[None, :] < geometry.dims)
        a = tl.load(
            centroids.data + offs[:, None].to(tl.int64) * centroids.stride + cols[None, :], mask=inside, other=0.0
        )
        tl.atomic_add(lloyd.sums + home[:, None] * geometry.padded + cols[None, :], a, mask=inside, sem=RELAXED)


@triton.jit
def _update(
    centers: Vectors[tl.tensor],
    lloyd: Lloyd[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    groups = tl.program_id(0) * block + tl.arange(0, block)
    live = groups < geometry.groups
    size = tl.load(lloyd.sizes + groups, mask=live, other=0)
    occupied = size[:, None] > 0
    scale = tl.maximum(size, 1).to(tl.float32)[:, None]
    norms = tl.zeros((block,), tl.float32)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cells = groups[:, None] * geometry.padded + base + tl.arange(0, geometry.chunk)[None, :]
        total = tl.load(lloyd.sums + cells, mask=live[:, None], other=0.0)
        center = tl.where(occupied, total / scale, tl.load(centers.data + cells, mask=live[:, None], other=0.0))
        tl.store(centers.data + cells, center, mask=live[:, None])
        tl.store(lloyd.sums + cells, tl.zeros((block, geometry.chunk), tl.float32), mask=live[:, None])
        norms += tl.sum(center * center, axis=1)
    tl.store(centers.norms + groups, norms, mask=live)
    tl.store(lloyd.sizes + groups, tl.zeros((block,), tl.int32), mask=live)


@triton.jit
def _publish(
    centers: Vectors[tl.tensor],
    lloyd: Lloyd[tl.tensor],
    rounded: Vectors[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    groups = tl.program_id(0) * block + tl.arange(0, block)
    live = groups < geometry.groups
    norms = tl.zeros((block,), tl.float32)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cells = groups[:, None] * geometry.padded + base + tl.arange(0, geometry.chunk)[None, :]
        center = tl.load(centers.data + cells, mask=live[:, None], other=0.0).to(tl.bfloat16)
        tl.store(rounded.data + cells, center, mask=live[:, None])
        wide = center.to(tl.float32)
        norms += tl.sum(wide * wide, axis=1)
    occupied = tl.load(lloyd.sizes + groups, mask=live, other=0) > 0
    tl.store(rounded.norms + groups, tl.where(occupied, norms, float("inf")), mask=live)


@triton.jit
def _accumulate(
    rows: Matrix[tl.tensor],
    labels: tl.tensor,
    state: State[tl.tensor],
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < rows.rows
    label = tl.load(labels + offs, mask=live, other=0)
    tl.atomic_add(state.counts + label, 1.0, mask=live, sem=RELAXED)
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        inside = live[:, None] & (cols[None, :] < geometry.dims)
        values = tl.load(rows.data + offs[:, None].to(tl.int64) * rows.stride + cols[None, :], mask=inside, other=0.0)
        cells = label[:, None] * geometry.dims + cols[None, :]
        tl.atomic_add(state.sums + cells, values.to(tl.float32), mask=inside, sem=RELAXED)


@triton.jit
def _ema(
    state: State[tl.tensor],
    rate: Rate,
    mode: "Annotated[Mode, tl.constexpr]",
    geometry: "Annotated[Geometry, tl.constexpr]",
    block: "Annotated[int, tl.constexpr]",
) -> None:
    offs = tl.program_id(0) * block + tl.arange(0, block)
    live = offs < geometry.clusters
    count = tl.load(state.counts + offs, mask=live, other=0.0)
    old = tl.load(state.usage + offs, mask=live, other=0.0)
    mass = rate.alpha * count
    if mode.decays:
        old = old * rate.keep
        mass += old
    touched = count > 0
    weight = tl.where(touched, rate.alpha * count / tl.where(touched, mass, 1.0), 0.0)[:, None]
    divisor = tl.where(touched, count, 1.0)[:, None]
    for base in tl.static_range(0, geometry.padded, geometry.chunk):
        cols = base + tl.arange(0, geometry.chunk)
        inside = live[:, None] & (cols[None, :] < geometry.dims)
        cells = offs[:, None].to(tl.int64) * geometry.dims + cols[None, :]
        total = tl.load(state.sums + cells, mask=inside, other=0.0)
        center = tl.load(state.centroids + cells, mask=inside, other=0.0)
        mean = total / divisor
        moved = tl.where(weight >= 1.0, mean, center + weight * (mean - center))
        tl.store(state.centroids + cells, tl.where(touched[:, None], moved, center), mask=inside)
        if mode.last:
            tl.store(
                state.pass_sums + cells, tl.load(state.pass_sums + cells, mask=inside, other=0.0) + total, mask=inside
            )
        tl.store(state.sums + cells, tl.zeros((block, geometry.chunk), tl.float32), mask=inside)
    tl.store(state.usage + offs, tl.where(touched, mass, old), mask=live)
    if mode.last:
        tl.store(state.pass_counts + offs, tl.load(state.pass_counts + offs, mask=live, other=0.0) + count, mask=live)
    tl.store(state.counts + offs, tl.zeros((block,), tl.float32), mask=live)


def tile(width: int) -> int:
    return triton.next_power_of_2(max(width, MIN_TILE))


def layout(k: int, dims: int, probes: int) -> Geometry:
    padded, groups = tile(dims), group_count(k)
    return Geometry(dims, padded, min(DIM_CHUNK, padded), k, groups, tile(groups), probes, MEMBER_TILE)


def matrix(data: torch.Tensor) -> Matrix[torch.Tensor]:
    return Matrix(data, data.stride(0), len(data))


def pairs(count: int, geometry: Geometry, device: torch.device) -> Pairs[torch.Tensor]:
    total = count * geometry.probes
    flat, span = torch.empty(total, dtype=torch.int32, device=device), geometry.span
    return Pairs(
        top=flat,
        items=torch.empty_like(flat),
        counts=torch.zeros(span, dtype=torch.int32, device=device),
        ends=torch.empty(span, dtype=torch.int32, device=device),
        blocks=torch.empty(span, dtype=torch.int32, device=device),
        fill=torch.zeros(span, dtype=torch.int32, device=device),
        value=torch.empty(total, dtype=torch.float32, device=device),
        ident=torch.empty_like(flat),
        total=total,
    )


def select(
    rows: Matrix[torch.Tensor], centers: Vectors[torch.Tensor], plan: Pairs[torch.Tensor], geometry: Geometry
) -> None:
    _select[(triton.cdiv(rows.rows, SELECT_ROWS),)](rows, centers, plan, geometry, SELECT_ROWS)


def bucket(plan: Pairs[torch.Tensor], geometry: Geometry) -> None:
    _plan[(1,)](plan, geometry, PAIR_ROWS)
    _bucket[(triton.cdiv(plan.total, BUCKET_BLOCK),)](plan, BUCKET_BLOCK)


def member(
    rows: Matrix[torch.Tensor], plan: Pairs[torch.Tensor], members: Members[torch.Tensor], geometry: Geometry
) -> None:
    _member[(triton.cdiv(plan.total, PAIR_ROWS) + geometry.groups,)](rows, plan, members, geometry, PAIR_ROWS)


def reduce(plan: Pairs[torch.Tensor], labels: torch.Tensor, geometry: Geometry) -> None:
    count = len(labels)
    _reduce[(triton.cdiv(count, REDUCE_BLOCK),)](plan, labels, count, geometry, REDUCE_BLOCK)


def refresh(centroids: Matrix[torch.Tensor], members: Members[torch.Tensor], geometry: Geometry) -> None:
    _refresh[(triton.cdiv(geometry.clusters, REFRESH_ROWS),)](centroids, members, geometry, REFRESH_ROWS)


def seed(
    centroids: Matrix[torch.Tensor], picks: torch.Tensor, centers: Vectors[torch.Tensor], geometry: Geometry
) -> None:
    _seed[(triton.cdiv(geometry.groups, GROUP_BLOCK),)](centroids, picks, centers, geometry, GROUP_BLOCK)


def home(
    centroids: Matrix[torch.Tensor], centers: Vectors[torch.Tensor], lloyd: Lloyd[torch.Tensor], geometry: Geometry
) -> None:
    dims = geometry.dims
    torch.mm(centroids.data[:, :dims], centers.data[:, :dims].T, out=lloyd.dots)
    _home[(triton.cdiv(geometry.clusters, LLOYD_ROWS),)](centroids, centers, lloyd, geometry, LLOYD_ROWS)


def update(centers: Vectors[torch.Tensor], lloyd: Lloyd[torch.Tensor], geometry: Geometry) -> None:
    _update[(triton.cdiv(geometry.groups, GROUP_BLOCK),)](centers, lloyd, geometry, GROUP_BLOCK)


def publish(
    centers: Vectors[torch.Tensor], lloyd: Lloyd[torch.Tensor], rounded: Vectors[torch.Tensor], geometry: Geometry
) -> None:
    _publish[(triton.cdiv(geometry.groups, GROUP_BLOCK),)](centers, lloyd, rounded, geometry, GROUP_BLOCK)


def accumulate(
    rows: Matrix[torch.Tensor], labels: torch.Tensor, state: State[torch.Tensor], geometry: Geometry
) -> None:
    _accumulate[(triton.cdiv(rows.rows, ACCUMULATE_ROWS),)](rows, labels, state, geometry, ACCUMULATE_ROWS)


def ema(state: State[torch.Tensor], rate: Rate, mode: Mode, geometry: Geometry) -> None:
    _ema[(triton.cdiv(geometry.clusters, EMA_ROWS),)](state, rate, mode, geometry, EMA_ROWS)
