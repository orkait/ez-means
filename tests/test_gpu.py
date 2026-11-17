import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.profiler import DeviceType, ProfilerActivity, profile

from ezmeans import Layout, Recipe, Regroup, cpu, law_recipe
from ezmeans.gpu import Fault, GpuInputError, GroupIndex, train
from ezmeans.gpu.constants import INT32_MAX
from ezmeans.layout import MAX_GROUPS
from ezmeans.recipe import group_count

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

ROWS = 20_000
CLUSTERS = 1024
PROBES = 4
SEED = 0
EPOCHS = 2
DIMS = 32
SPAN_ROWS = 256
POISONED_ROW = 7
OVERFLOW_SCALE = 1e20
TINY_EPOCHS = 1e-5
QUALITY_TOLERANCE = 0.01
SCORE_TOLERANCE = 1e-5
TF32_TOLERANCE = 2.0**-9
DRIFT = 0.05
SYNC_ERROR = "error"
SYNC_WARN = "warn"
SYNC_DEFAULT = "default"
SYNC_MESSAGE = "called a synchronizing CUDA operation"
SHORT_EPOCHS = 2
LONG_EPOCHS = 6
EVEN_BATCH = 1000
HOST_LAUNCHES = frozenset(
    {"cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaMemcpyAsync", "cudaMemsetAsync", "cudaGraphLaunch"}
)
DEVICE_OPS_PER_HOST_LAUNCH = 2


def layout(probes: int, regroup: Regroup = Regroup.COLD, clusters: int = CLUSTERS) -> Layout:
    return Layout(group_count(clusters), probes, regroup)


def sample(dims: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    rows = torch.randn(ROWS, dims, generator=generator)
    centroids = rows[torch.randperm(ROWS, generator=generator)[:CLUSTERS]].clone()
    return rows.cuda().to(torch.bfloat16), centroids.cuda()


def poisoned(value: float) -> torch.Tensor:
    rows, _ = sample(DIMS)
    rows[POISONED_ROW] = value
    return rows


def poisoned_start() -> torch.Tensor:
    _, start = sample(DIMS)
    start[POISONED_ROW] = float("nan")
    return start


def overflowing_assign() -> torch.Tensor:
    rows, centroids = sample(DIMS)
    probes = group_count(CLUSTERS)
    index = GroupIndex(CLUSTERS, DIMS, layout(probes), SEED)
    index.regroup(centroids)
    return index.assign(rows[:1].expand(INT32_MAX // probes + 1, DIMS))


def squared_error(rows: np.ndarray, centroids: np.ndarray) -> float:
    wide, centers = rows.astype(np.float64), centroids.astype(np.float64)
    squared = (wide**2).sum(axis=1)[:, None] - 2.0 * wide @ centers.T + (centers**2).sum(axis=1)[None, :]
    return float(squared.min(axis=1).mean())


def probing_every_group_is_exact(rows: torch.Tensor, centroids: torch.Tensor) -> bool:
    clusters, dims = centroids.shape
    index = GroupIndex(clusters, dims, layout(group_count(clusters), clusters=clusters), SEED)
    index.regroup(centroids)
    labels = index.assign(rows)
    wide, rounded = rows.float(), centroids.to(torch.bfloat16).float()
    scores = (rounded**2).sum(1)[None, :] - 2.0 * (wide @ rounded.T)
    chosen = scores.gather(1, labels[:, None]).squeeze(1)
    slack = SCORE_TOLERANCE * ((wide**2).sum(1) + (rounded**2).sum(1).max())
    return bool((chosen <= scores.min(dim=1).values + slack).all())


@pytest.mark.parametrize("dims", [96, 100, 128, 256, 1024, 3072])
def test_probing_every_group_is_brute_force(dims: int) -> None:
    assert probing_every_group_is_exact(*sample(dims))


def test_probing_every_group_is_brute_force_at_the_largest_group_span() -> None:
    generator = torch.Generator().manual_seed(SEED)
    centroids = torch.randn(MAX_GROUPS**2, DIMS, generator=generator)
    rows = torch.randn(SPAN_ROWS, DIMS, generator=generator)
    assert probing_every_group_is_exact(rows.cuda().to(torch.bfloat16), centroids.cuda())


@pytest.mark.parametrize("regroup", list(Regroup))
@pytest.mark.parametrize("dims", [DIMS, 96, 256])
def test_regroup_homes_every_centroid_at_its_nearest_center(dims: int, regroup: Regroup) -> None:
    _, start = sample(dims)
    moved = start + DRIFT * torch.randn(start.shape, generator=torch.Generator().manual_seed(SEED)).cuda()
    index = GroupIndex(CLUSTERS, dims, layout(PROBES, regroup), SEED)
    for centroids in (start, moved):
        index.regroup(centroids)
    wide, centers = moved.double(), index.centers[: index.groups, :dims].double()
    distances = torch.cdist(wide, centers).square()
    chosen = distances.gather(1, index.owner.long()[:, None]).squeeze(1)
    slack = TF32_TOLERANCE * ((wide**2).sum(1) + (centers**2).sum(1).max())
    assert bool((chosen <= distances.min(dim=1).values + slack).all())


@pytest.mark.parametrize("regroup", list(Regroup))
def test_index_never_synchronizes_the_host(regroup: Regroup) -> None:
    rows, centroids = sample(DIMS)
    index = GroupIndex(CLUSTERS, DIMS, layout(PROBES, regroup), SEED)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(SYNC_ERROR)
    try:
        for _ in range(2):
            index.regroup(centroids)
        index.refresh(centroids)
        index.assign(rows)
    finally:
        torch.cuda.set_sync_debug_mode(SYNC_DEFAULT)


def even_recipe(epochs: int) -> Recipe:
    return replace(law_recipe(ROWS, CLUSTERS, epochs, SEED), batch=EVEN_BATCH)


def host_waits(epochs: int) -> int:
    rows, start = sample(DIMS)
    recipe = even_recipe(epochs)
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(SYNC_WARN)
        try:
            train(rows, CLUSTERS, recipe, layout(PROBES), init=start)
        finally:
            torch.cuda.set_sync_debug_mode(SYNC_DEFAULT)
    return sum(str(warning.message).startswith(SYNC_MESSAGE) for warning in caught)


def test_training_waits_on_the_host_the_same_number_of_times_at_any_length() -> None:
    assert host_waits(SHORT_EPOCHS) == host_waits(LONG_EPOCHS)


def test_training_launches_most_device_work_from_captured_steps() -> None:
    rows, start = sample(DIMS)
    recipe = even_recipe(LONG_EPOCHS)
    train(rows, CLUSTERS, recipe, layout(PROBES), init=start)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train(rows, CLUSTERS, recipe, layout(PROBES), init=start)
        torch.cuda.synchronize()
    events = prof.events()
    device = sum(event.device_type == DeviceType.CUDA for event in events)
    host = sum(event.name in HOST_LAUNCHES for event in events)
    assert host * DEVICE_OPS_PER_HOST_LAUNCH < device


@pytest.mark.parametrize("regroup", list(Regroup))
def test_gpu_training_from_a_shared_start_matches_cpu_quality(regroup: Regroup) -> None:
    rows, start = sample(DIMS)
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    data = rows.float().cpu().numpy()
    shared = layout(PROBES, regroup)
    gpu = squared_error(data, train(rows, CLUSTERS, recipe, shared, init=start).cpu().numpy())
    reference = squared_error(data, cpu.train(data, CLUSTERS, recipe, shared, init=start.cpu().numpy()))
    assert gpu <= reference * (1 + QUALITY_TOLERANCE)


@pytest.mark.parametrize(
    ("call", "fault"),
    [
        (lambda: train(poisoned(float("nan")), CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)), Fault.NOT_FINITE),
        (lambda: train(poisoned(float("inf")), CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)), Fault.NOT_FINITE),
        (lambda: train(poisoned(OVERFLOW_SCALE), CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)), Fault.NOT_FINITE),
        (lambda: train(sample(DIMS)[0].float(), CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)), Fault.WRONG_DTYPE),
        (
            lambda: train(sample(DIMS)[0][: CLUSTERS - 1], CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)),
            Fault.BAD_SIZE,
        ),
        (lambda: train(sample(DIMS)[0], CLUSTERS, law_recipe(ROWS, CLUSTERS, TINY_EPOCHS, SEED)), Fault.BAD_SIZE),
        (overflowing_assign, Fault.TOO_MANY_PAIRS),
        (
            lambda: train(sample(DIMS)[0], CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), init=poisoned_start()),
            Fault.NOT_FINITE,
        ),
        (
            lambda: train(
                sample(DIMS)[0], CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), init=sample(DIMS + 1)[1]
            ),
            Fault.WRONG_SHAPE,
        ),
        (lambda: GroupIndex(CLUSTERS, DIMS, Layout(CLUSTERS + 1, PROBES, Regroup.COLD), SEED), Fault.BAD_GROUPS),
        (lambda: GroupIndex(CLUSTERS, DIMS, Layout(PROBES, PROBES + 1, Regroup.WARM), SEED), Fault.BAD_PROBES),
    ],
    ids=[
        "nan",
        "inf",
        "overflow",
        "float32 rows",
        "fewer rows than clusters",
        "empty plan",
        "pairs past int32",
        "nan start",
        "start wider than rows",
        "more groups than clusters",
        "more probes than groups",
    ],
)
def test_bad_input_raises_gpu_input_error(call: Callable[[], object], fault: Fault) -> None:
    with pytest.raises(GpuInputError) as raised:
        call()
    assert raised.value.fault is fault
