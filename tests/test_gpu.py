import pytest
import torch

from ezmeans.gpu.backend import GroupedBackend, emeans, train
from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.recipe import group_count, law_recipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

ROWS = 20_000
CLUSTERS = 1024
PROBES = 4
SEED = 0
EPOCHS = 1
QUALITY_LAW_DIMS = 8
POISONED_ROW = 7
FOREIGN_BACKEND = "foreign"
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


def test_training_follows_the_recipe() -> None:
    rows, _ = sample(QUALITY_LAW_DIMS)
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    model, _ = emeans(QUALITY_LAW_DIMS, CLUSTERS, recipe, PROBES, rows.device)
    model.train(rows)
    assert (model.recipe["batch_size"], model.recipe["alpha0"]) == (recipe.batch, recipe.alpha0)


def test_training_rejects_non_finite_rows() -> None:
    rows, _ = sample(QUALITY_LAW_DIMS)
    rows[POISONED_ROW] = float("nan")
    with pytest.raises(GpuInputError) as raised:
        train(rows, CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), PROBES)
    assert raised.value.fault is Fault.NOT_FINITE


def test_training_fails_when_faiss_bypasses_the_grouped_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GroupedBackend, "name", FOREIGN_BACKEND)
    rows, _ = sample(QUALITY_LAW_DIMS)
    with pytest.raises(GpuInputError) as raised:
        train(rows, CLUSTERS, law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), PROBES)
    assert raised.value.fault is Fault.UNSUPPORTED_FAISS
