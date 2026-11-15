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


@triton.jit
def _select(
    rows,
    row_stride,
    n,
    centers,
    center_norms,
    top,
    counts,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    GP: tl.constexpr,
    BM: tl.constexpr,
    P: tl.constexpr,
):
    offs = tl.program_id(0) * BM + tl.arange(0, BM)
    live = offs < n
    groups = tl.arange(0, GP)
    dots = tl.zeros((BM, GP), tl.float32)
    for base in range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        a = tl.load(
            rows + offs[:, None].to(tl.int64) * row_stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < D),
            other=0.0,
        )
        dots += tl.dot(a, tl.load(centers + groups[None, :] * DP + cols[:, None]))
    score = tl.load(center_norms + groups)[None, :] - 2.0 * dots
    for j in tl.static_range(P):
        pick = tl.argmin(score, axis=1)
        tl.store(top + offs * P + j, pick, mask=live)
        tl.atomic_add(counts + pick, 1, mask=live, sem=RELAXED)
        score = tl.where(groups[None, :] == pick[:, None], float("inf"), score)


@triton.jit
def _plan(counts, ends, blocks, G, GP: tl.constexpr, BM: tl.constexpr):
    groups = tl.arange(0, GP)
    live = groups < G
    size = tl.load(counts + groups, mask=live, other=0)
    tl.store(ends + groups, tl.cumsum(size, axis=0), mask=live)
    tl.store(blocks + groups, tl.cumsum((size + BM - 1) // BM, axis=0), mask=live)


@triton.jit
def _bucket(keys, m, ends, counts, fill, out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < m
    key = tl.load(keys + offs, mask=live, other=0)
    start = tl.load(ends + key, mask=live, other=0) - tl.load(counts + key, mask=live, other=0)
    tl.store(out + start + tl.atomic_add(fill + key, 1, mask=live, sem=RELAXED), offs, mask=live)


@triton.jit
def _member(
    rows,
    row_stride,
    items,
    members,
    member_norms,
    order,
    group_ends,
    group_sizes,
    pair_ends,
    pair_counts,
    pair_blocks,
    G,
    value,
    ident,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    GP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.arange(0, GP)
    reach = tl.load(pair_blocks + groups, mask=groups < G, other=UNREACHABLE)
    g = tl.sum((reach <= pid).to(tl.int32), axis=0)
    if g >= G:
        return
    count = tl.load(pair_counts + g)
    first_block = tl.load(pair_blocks + g) - (count + BM - 1) // BM
    end = tl.load(pair_ends + g)
    offs = end - count + (pid - first_block) * BM + tl.arange(0, BM)
    live = offs < end
    item = tl.load(items + offs, mask=live, other=0)
    point = (item // P).to(tl.int64)
    hi = tl.load(group_ends + g)
    lo = hi - tl.load(group_sizes + g)
    best = tl.full((BM,), float("inf"), tl.float32)
    best_id = tl.zeros((BM,), tl.int32)
    for first in range(lo, hi, BN):
        lanes = first + tl.arange(0, BN)
        inside = lanes < hi
        dots = tl.zeros((BM, BN), tl.float32)
        for base in range(0, DP, DK):
            cols = base + tl.arange(0, DK)
            a = tl.load(
                rows + point[:, None] * row_stride + cols[None, :], mask=live[:, None] & (cols[None, :] < D), other=0.0
            )
            b = tl.load(members + lanes[None, :].to(tl.int64) * DP + cols[:, None], mask=inside[None, :], other=0.0)
            dots += tl.dot(a, b)
        score = tl.load(member_norms + lanes, mask=inside, other=float("inf"))[None, :] - 2.0 * dots
        low = tl.min(score, axis=1)
        lane = first + tl.argmin(score, axis=1)
        take = low < best
        best = tl.where(take, low, best)
        best_id = tl.where(take, tl.load(order + lane, mask=lane < hi, other=0), best_id)
    tl.store(value + item, best, mask=live)
    tl.store(ident + item, best_id, mask=live)


@triton.jit
def _reduce(value, ident, n, labels, P: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < n
    best = tl.full((BLOCK,), float("inf"), tl.float32)
    best_id = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(P):
        candidate = tl.load(value + offs * P + j, mask=live, other=float("inf"))
        take = candidate < best
        best = tl.where(take, candidate, best)
        best_id = tl.where(take, tl.load(ident + offs * P + j, mask=live, other=0), best_id)
    tl.store(labels + offs, best_id.to(tl.int64), mask=live)


@triton.jit
def _refresh(
    centroids,
    centroid_stride,
    order,
    members,
    member_norms,
    k,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < k
    source = tl.load(order + offs, mask=live, other=0).to(tl.int64)
    norms = tl.zeros((BLOCK,), tl.float32)
    for base in tl.static_range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        rounded = tl.load(
            centroids + source[:, None] * centroid_stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < D),
            other=0.0,
        ).to(tl.bfloat16)
        tl.store(members + offs[:, None].to(tl.int64) * DP + cols[None, :], rounded, mask=live[:, None])
        wide = rounded.to(tl.float32)
        norms += tl.sum(wide * wide, axis=1)
    tl.store(member_norms + offs, norms, mask=live)


@triton.jit
def _seed(
    centroids,
    centroid_stride,
    picks,
    centers,
    center_norms,
    G,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    BG: tl.constexpr,
):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    live = groups < G
    source = tl.load(picks + groups, mask=live, other=0).to(tl.int64)
    norms = tl.zeros((BG,), tl.float32)
    for base in tl.static_range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        center = tl.load(
            centroids + source[:, None] * centroid_stride + cols[None, :],
            mask=live[:, None] & (cols[None, :] < D),
            other=0.0,
        )
        tl.store(centers + groups[:, None] * DP + cols[None, :], center, mask=live[:, None])
        norms += tl.sum(center * center, axis=1)
    tl.store(center_norms + groups, norms, mask=live)


@triton.jit
def _home(
    centroids,
    centroid_stride,
    k,
    dots,
    center_norms,
    owner,
    sums,
    sizes,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    GP: tl.constexpr,
    BM: tl.constexpr,
):
    offs = tl.program_id(0) * BM + tl.arange(0, BM)
    live = offs < k
    groups = tl.arange(0, GP)
    product = tl.load(dots + offs[:, None].to(tl.int64) * GP + groups[None, :], mask=live[:, None], other=0.0)
    home = tl.argmin(tl.load(center_norms + groups)[None, :] - 2.0 * product, axis=1)
    tl.store(owner + offs, home, mask=live)
    tl.atomic_add(sizes + home, 1, mask=live, sem=RELAXED)
    for base in tl.static_range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        inside = live[:, None] & (cols[None, :] < D)
        a = tl.load(centroids + offs[:, None].to(tl.int64) * centroid_stride + cols[None, :], mask=inside, other=0.0)
        tl.atomic_add(sums + home[:, None] * DP + cols[None, :], a, mask=inside, sem=RELAXED)


@triton.jit
def _update(centers, center_norms, sums, sizes, G, DP: tl.constexpr, DK: tl.constexpr, BG: tl.constexpr):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    live = groups < G
    size = tl.load(sizes + groups, mask=live, other=0)
    occupied = size[:, None] > 0
    scale = tl.maximum(size, 1).to(tl.float32)[:, None]
    norms = tl.zeros((BG,), tl.float32)
    for base in tl.static_range(0, DP, DK):
        cells = groups[:, None] * DP + base + tl.arange(0, DK)[None, :]
        total = tl.load(sums + cells, mask=live[:, None], other=0.0)
        center = tl.where(occupied, total / scale, tl.load(centers + cells, mask=live[:, None], other=0.0))
        tl.store(centers + cells, center, mask=live[:, None])
        tl.store(sums + cells, tl.zeros((BG, DK), tl.float32), mask=live[:, None])
        norms += tl.sum(center * center, axis=1)
    tl.store(center_norms + groups, norms, mask=live)
    tl.store(sizes + groups, tl.zeros((BG,), tl.int32), mask=live)


@triton.jit
def _publish(centers, sizes, rounded, rounded_norms, G, DP: tl.constexpr, DK: tl.constexpr, BG: tl.constexpr):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    live = groups < G
    norms = tl.zeros((BG,), tl.float32)
    for base in tl.static_range(0, DP, DK):
        cells = groups[:, None] * DP + base + tl.arange(0, DK)[None, :]
        center = tl.load(centers + cells, mask=live[:, None], other=0.0).to(tl.bfloat16)
        tl.store(rounded + cells, center, mask=live[:, None])
        wide = center.to(tl.float32)
        norms += tl.sum(wide * wide, axis=1)
    occupied = tl.load(sizes + groups, mask=live, other=0) > 0
    tl.store(rounded_norms + groups, tl.where(occupied, norms, float("inf")), mask=live)


@triton.jit
def _accumulate(
    rows, row_stride, labels, n, sums, counts, D: tl.constexpr, DP: tl.constexpr, DK: tl.constexpr, BM: tl.constexpr
):
    offs = tl.program_id(0) * BM + tl.arange(0, BM)
    live = offs < n
    label = tl.load(labels + offs, mask=live, other=0)
    tl.atomic_add(counts + label, 1.0, mask=live, sem=RELAXED)
    for base in tl.static_range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        inside = live[:, None] & (cols[None, :] < D)
        values = tl.load(rows + offs[:, None].to(tl.int64) * row_stride + cols[None, :], mask=inside, other=0.0)
        tl.atomic_add(sums + label[:, None] * D + cols[None, :], values.to(tl.float32), mask=inside, sem=RELAXED)


@triton.jit
def _ema(
    centroids,
    usage,
    sums,
    counts,
    pass_sums,
    pass_counts,
    k,
    alpha,
    keep,
    D: tl.constexpr,
    DP: tl.constexpr,
    DK: tl.constexpr,
    BM: tl.constexpr,
    DECAYS: tl.constexpr,
    LAST: tl.constexpr,
):
    offs = tl.program_id(0) * BM + tl.arange(0, BM)
    live = offs < k
    count = tl.load(counts + offs, mask=live, other=0.0)
    old = tl.load(usage + offs, mask=live, other=0.0)
    mass = alpha * count
    if DECAYS:
        old = old * keep
        mass += old
    touched = count > 0
    weight = tl.where(touched, alpha * count / tl.where(touched, mass, 1.0), 0.0)[:, None]
    divisor = tl.where(touched, count, 1.0)[:, None]
    for base in tl.static_range(0, DP, DK):
        cols = base + tl.arange(0, DK)
        inside = live[:, None] & (cols[None, :] < D)
        cells = offs[:, None].to(tl.int64) * D + cols[None, :]
        total = tl.load(sums + cells, mask=inside, other=0.0)
        center = tl.load(centroids + cells, mask=inside, other=0.0)
        mean = total / divisor
        moved = tl.where(weight == 1.0, mean, center + weight * (mean - center))
        tl.store(centroids + cells, tl.where(touched[:, None], moved, center), mask=inside)
        if LAST:
            tl.store(pass_sums + cells, tl.load(pass_sums + cells, mask=inside, other=0.0) + total, mask=inside)
        tl.store(sums + cells, tl.zeros((BM, DK), tl.float32), mask=inside)
    tl.store(usage + offs, tl.where(touched, mass, old), mask=live)
    if LAST:
        tl.store(pass_counts + offs, tl.load(pass_counts + offs, mask=live, other=0.0) + count, mask=live)
    tl.store(counts + offs, tl.zeros((BM,), tl.float32), mask=live)


def tile(width: int) -> int:
    return int(triton.next_power_of_2(max(width, MIN_TILE)))


def chunk(padded: int) -> int:
    return min(DIM_CHUNK, padded)


def select(
    rows: torch.Tensor,
    centers: torch.Tensor,
    center_norms: torch.Tensor,
    top: torch.Tensor,
    counts: torch.Tensor,
    dims: int,
    probes: int,
) -> None:
    n, padded = len(rows), centers.shape[1]
    _select[(triton.cdiv(n, SELECT_ROWS),)](
        rows,
        rows.stride(0),
        n,
        centers,
        center_norms,
        top,
        counts,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        GP=centers.shape[0],
        BM=SELECT_ROWS,
        P=probes,
    )


def plan(counts: torch.Tensor, ends: torch.Tensor, blocks: torch.Tensor, groups: int, block: int) -> None:
    _plan[(1,)](counts, ends, blocks, groups, GP=tile(groups), BM=block)


def bucket(keys: torch.Tensor, ends: torch.Tensor, counts: torch.Tensor, fill: torch.Tensor, out: torch.Tensor) -> None:
    m = len(keys)
    _bucket[(triton.cdiv(m, BUCKET_BLOCK),)](keys, m, ends, counts, fill, out, BLOCK=BUCKET_BLOCK)


def member(
    rows: torch.Tensor,
    items: torch.Tensor,
    members: torch.Tensor,
    member_norms: torch.Tensor,
    order: torch.Tensor,
    group_ends: torch.Tensor,
    group_sizes: torch.Tensor,
    pair_ends: torch.Tensor,
    pair_counts: torch.Tensor,
    pair_blocks: torch.Tensor,
    value: torch.Tensor,
    ident: torch.Tensor,
    dims: int,
    groups: int,
    probes: int,
) -> None:
    padded = members.shape[1]
    _member[(triton.cdiv(len(items), PAIR_ROWS) + groups,)](
        rows,
        rows.stride(0),
        items,
        members,
        member_norms,
        order,
        group_ends,
        group_sizes,
        pair_ends,
        pair_counts,
        pair_blocks,
        groups,
        value,
        ident,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        GP=tile(groups),
        BM=PAIR_ROWS,
        BN=MEMBER_TILE,
        P=probes,
    )


def reduce(value: torch.Tensor, ident: torch.Tensor, labels: torch.Tensor, probes: int) -> None:
    n = len(labels)
    _reduce[(triton.cdiv(n, REDUCE_BLOCK),)](value, ident, n, labels, P=probes, BLOCK=REDUCE_BLOCK)


def refresh(
    centroids: torch.Tensor, order: torch.Tensor, members: torch.Tensor, member_norms: torch.Tensor, dims: int
) -> None:
    k, padded = len(order), members.shape[1]
    _refresh[(triton.cdiv(k, REFRESH_ROWS),)](
        centroids,
        centroids.stride(0),
        order,
        members,
        member_norms,
        k,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        BLOCK=REFRESH_ROWS,
    )


def seed(
    centroids: torch.Tensor, picks: torch.Tensor, centers: torch.Tensor, center_norms: torch.Tensor, dims: int
) -> None:
    groups, padded = len(picks), centers.shape[1]
    _seed[(triton.cdiv(groups, GROUP_BLOCK),)](
        centroids,
        centroids.stride(0),
        picks,
        centers,
        center_norms,
        groups,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        BG=GROUP_BLOCK,
    )


def lloyd(
    centroids: torch.Tensor,
    centers: torch.Tensor,
    center_norms: torch.Tensor,
    dots: torch.Tensor,
    owner: torch.Tensor,
    sums: torch.Tensor,
    sizes: torch.Tensor,
    dims: int,
) -> None:
    k, padded, span = len(centroids), centers.shape[1], centers.shape[0]
    torch.mm(centroids[:, :dims], centers[:, :dims].T, out=dots)
    _home[(triton.cdiv(k, LLOYD_ROWS),)](
        centroids,
        centroids.stride(0),
        k,
        dots,
        center_norms,
        owner,
        sums,
        sizes,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        GP=span,
        BM=LLOYD_ROWS,
    )


def update(
    centers: torch.Tensor, center_norms: torch.Tensor, sums: torch.Tensor, sizes: torch.Tensor, groups: int
) -> None:
    padded = centers.shape[1]
    _update[(triton.cdiv(groups, GROUP_BLOCK),)](
        centers,
        center_norms,
        sums,
        sizes,
        groups,
        DP=padded,
        DK=chunk(padded),
        BG=GROUP_BLOCK,
    )


def publish(
    centers: torch.Tensor, sizes: torch.Tensor, rounded: torch.Tensor, rounded_norms: torch.Tensor, groups: int
) -> None:
    padded = centers.shape[1]
    _publish[(triton.cdiv(groups, GROUP_BLOCK),)](
        centers,
        sizes,
        rounded,
        rounded_norms,
        groups,
        DP=padded,
        DK=chunk(padded),
        BG=GROUP_BLOCK,
    )


def accumulate(rows: torch.Tensor, labels: torch.Tensor, sums: torch.Tensor, counts: torch.Tensor) -> None:
    n, dims = len(rows), sums.shape[1]
    padded = tile(dims)
    _accumulate[(triton.cdiv(n, ACCUMULATE_ROWS),)](
        rows, rows.stride(0), labels, n, sums, counts, D=dims, DP=padded, DK=chunk(padded), BM=ACCUMULATE_ROWS
    )


def ema(
    centroids: torch.Tensor,
    usage: torch.Tensor,
    sums: torch.Tensor,
    counts: torch.Tensor,
    passes: tuple[torch.Tensor, torch.Tensor],
    alpha: float,
    last: bool,
) -> None:
    k, dims = centroids.shape
    padded = tile(dims)
    _ema[(triton.cdiv(k, EMA_ROWS),)](
        centroids,
        usage,
        sums,
        counts,
        *passes,
        k,
        alpha,
        1.0 - alpha,
        D=dims,
        DP=padded,
        DK=chunk(padded),
        BM=EMA_ROWS,
        DECAYS=alpha != 1.0,
        LAST=last,
    )
