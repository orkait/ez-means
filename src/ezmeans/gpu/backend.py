import torch
from faiss.contrib.e_means.e_means_torch import Emeans as TorchEmeans
from faiss.contrib.e_means.torch_backend import TritonBackend

from ezmeans.gpu.constants import MAX_ROW_NORM
from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.recipe import CHECK_EVERY, Recipe, default_probes


class GroupedBackend(TritonBackend):
    def __init__(self, chunk: int, index: GroupIndex) -> None:
        super().__init__(chunk)
        self.index = index
        self.batches = 0

    def prepare_centroids(
        self, centroids: torch.Tensor, dtype: torch.dtype, spherical: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spherical:
            raise GpuInputError(Fault.SPHERICAL)
        if self.batches % CHECK_EVERY == 0:
            self.index.regroup(centroids)
        self.batches += 1
        self.index.refresh(centroids)
        return self.index.members, self.index.member_norms

    def assign(
        self,
        x: torch.Tensor,
        centroids: torch.Tensor,
        centroid_norm: torch.Tensor | None = None,
        spherical: bool = False,
        distance_out: torch.Tensor | None = None,
        labels_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spherical:
            raise GpuInputError(Fault.SPHERICAL)
        distance, labels = self.index.assign(x)
        if distance_out is not None:
            distance = distance_out.copy_(distance)
        if labels_out is not None:
            labels = labels_out.copy_(labels)
        return distance, labels


def emeans(dims: int, k: int, recipe: Recipe, probes: int, device: torch.device) -> tuple[TorchEmeans, GroupedBackend]:
    model = TorchEmeans(
        dims,
        k,
        nepochs=recipe.epochs,
        batch_size=recipe.batch,
        alpha0=recipe.alpha0,
        seed=recipe.seed,
        capture_assignments=False,
        assignment_backend=TritonBackend.name,
        unused_threshold=recipe.tau,
    )
    backend = GroupedBackend(model.centroid_chunk_size, GroupIndex(k, dims, probes, recipe.seed, device))
    model._execution_backend = backend
    return model, backend


def train(x: torch.Tensor, k: int, recipe: Recipe, probes: int | None = None) -> torch.Tensor:
    if not bool(torch.linalg.vector_norm(x, dim=1, dtype=torch.float32).max() < MAX_ROW_NORM):
        raise GpuInputError(Fault.NOT_FINITE)
    model, backend = emeans(x.shape[1], k, recipe, default_probes(k) if probes is None else probes, x.device)
    centroids, _ = model.train(x)
    if backend.batches == 0:
        raise GpuInputError(Fault.UNSUPPORTED_FAISS)
    return torch.as_tensor(centroids)
