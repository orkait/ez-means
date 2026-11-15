import pytest
import torch

from ezmeans.gpu.index import GroupIndex
from ezmeans.reference import group_count

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

ROWS = 20_000
CLUSTERS = 1024
PROBES = 4
SEED = 0
SYNC_ERROR = "error"
SYNC_DEFAULT = "default"
SCORE_TOLERANCE = 1e-5


def sample(dims: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    rows = torch.randn(ROWS, dims, generator=generator)
    centroids = rows[torch.randperm(ROWS, generator=generator)[:CLUSTERS]].clone()
    return rows.cuda().to(torch.bfloat16), centroids.cuda()


@pytest.mark.parametrize("dims", [96, 100, 128, 256])
def test_probing_every_group_is_brute_force(dims: int) -> None:
    rows, centroids = sample(dims)
    index = GroupIndex(CLUSTERS, dims, group_count(CLUSTERS), SEED)
    index.regroup(centroids)
    index.refresh(centroids)
    distance, labels = index.assign(rows)
    wide, rounded = rows.float(), centroids.to(torch.bfloat16).float()
    row_norms = (wide**2).sum(1)
    scores = (rounded**2).sum(1)[None, :] - 2.0 * (wide @ rounded.T)
    chosen = scores.gather(1, labels[:, None]).squeeze(1)
    slack = SCORE_TOLERANCE * (row_norms + (rounded**2).sum(1).max())
    assert bool((chosen <= scores.min(dim=1).values + slack).all())
    assert bool(((distance - chosen - row_norms).abs() <= slack).all())


def test_index_never_synchronizes_the_host() -> None:
    rows, centroids = sample(128)
    index = GroupIndex(CLUSTERS, 128, PROBES, SEED)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(SYNC_ERROR)
    try:
        index.regroup(centroids)
        index.refresh(centroids)
        index.assign(rows)
    finally:
        torch.cuda.set_sync_debug_mode(SYNC_DEFAULT)
