import torch
from faiss.contrib.e_means.e_means_torch import Emeans as TorchEmeans
from faiss.contrib.e_means.torch_backend import TritonBackend

from ezmeans.gpu.index import Fault, GpuInputError, GroupIndex
from ezmeans.recipe import CHECK_EVERY, Recipe


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


def emeans(dims: int, k: int, recipe: Recipe, probes: int, device: torch.device) -> TorchEmeans:
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
    model._execution_backend = GroupedBackend(
        model.centroid_chunk_size, GroupIndex(k, dims, probes, recipe.seed, device)
    )
    return model


def train(x: torch.Tensor, k: int, recipe: Recipe, probes: int) -> torch.Tensor:
    if not bool(torch.isfinite(x).all()):
        raise GpuInputError(Fault.NOT_FINITE)
    centroids, _ = emeans(x.shape[1], k, recipe, probes, x.device).train(x)
    return torch.as_tensor(centroids)
