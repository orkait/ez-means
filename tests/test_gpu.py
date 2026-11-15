from collections.abc import Callable

import numpy as np
import pytest
import torch

from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.gpu.trainer import train
from ezmeans.recipe import group_count, law_recipe
from ezmeans.trainer import ez_means

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

ROWS = 20_000
CLUSTERS = 1024
PROBES = 4
SEED = 0
EPOCHS = 2
DIMS = 32
POISONED_ROW = 7
OVERFLOW_SCALE = 1e20
TINY_EPOCHS = 1e-5
QUALITY_TOLERANCE = 0.01
SCORE_TOLERANCE = 1e-5
SYNC_ERROR = "error"
SYNC_DEFAULT = "default"


def sample(dims: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    rows = torch.randn(ROWS, dims, generator=generator)
    centroids = rows[torch.randperm(ROWS, generator=generator)[:CLUSTERS]].clone()
    return rows.cuda().to(torch.bfloat16), centroids.cuda()


def poisoned(value: float) -> torch.Tensor:
    rows, _ = sample(DIMS)
    rows[POISONED_ROW] = value
    return rows


def squared_error(rows: np.ndarray, centroids: np.ndarray) -> float:
    wide, centers = rows.astype(np.float64), centroids.astype(np.float64)
    squared = (wide**2).sum(axis=1)[:, None] - 2.0 * wide @ centers.T + (centers**2).sum(axis=1)[None, :]
    return float(squared.min(axis=1).mean())


@pytest.mark.parametrize("dims", [96, 100, 128, 256])
def test_probing_every_group_is_brute_force(dims: int) -> None:
    rows, centroids = sample(dims)
    index = GroupIndex(CLUSTERS, dims, group_count(CLUSTERS), SEED)
    index.regroup(centroids)
    index.refresh(centroids)
    labels = index.assign(rows)
    wide, rounded = rows.float(), centroids.to(torch.bfloat16).float()
    scores = (rounded**2).sum(1)[None, :] - 2.0 * (wide @ rounded.T)
    chosen = scores.gather(1, labels[:, None]).squeeze(1)
    slack = SCORE_TOLERANCE * ((wide**2).sum(1) + (rounded**2).sum(1).max())
    assert bool((chosen <= scores.min(dim=1).values + slack).all())


def test_index_never_synchronizes_the_host() -> None:
    rows, centroids = sample(DIMS)
    index = GroupIndex(CLUSTERS, DIMS, PROBES, SEED)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(SYNC_ERROR)
    try:
        index.regroup(centroids)
        index.refresh(centroids)
        index.assign(rows)
    finally:
        torch.cuda.set_sync_debug_mode(SYNC_DEFAULT)


def test_gpu_training_matches_cpu_quality() -> None:
    rows, _ = sample(DIMS)
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    data = rows.float().cpu().numpy()
    gpu = squared_error(data, train(rows, CLUSTERS, recipe, PROBES).cpu().numpy())
    cpu = squared_error(data, ez_means(data, CLUSTERS, recipe, PROBES))
    assert gpu <= cpu * (1 + QUALITY_TOLERANCE)


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
    ],
    ids=["nan", "inf", "overflow", "float32 rows", "fewer rows than clusters", "empty plan"],
)
def test_bad_input_raises_gpu_input_error(call: Callable[[], object], fault: Fault) -> None:
    with pytest.raises(GpuInputError) as raised:
        call()
    assert raised.value.fault is fault
