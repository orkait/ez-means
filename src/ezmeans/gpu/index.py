from enum import Enum

import torch

from ezmeans.gpu import kernels
from ezmeans.gpu.constants import CUDA, MATRIX_RANK, MAX_GROUP_SPAN
from ezmeans.gpu.kernels import Lloyd, Members, Vectors
from ezmeans.recipe import GROUP_ITERATIONS


class Fault(Enum):
    NO_CUDA = "CUDA is not available"
    NOT_CUDA = "the index runs on CUDA devices only"
    WRONG_DEVICE = "the tensor is not on the index's device"
    WRONG_DTYPE = "the tensor has the wrong dtype"
    WRONG_SHAPE = "the tensor has the wrong shape"
    NOT_ROW_MAJOR = "the tensor's rows must be contiguous"
    BAD_PROBES = "probes must be between 1 and the number of groups"
    TOO_MANY_GROUPS = "k needs more groups than the kernels support"
    NOT_GROUPED = "regroup must run before refresh or assign"
    NOT_FINITE = "every row must be finite, with a norm small enough for float32 squared distances"
    BAD_SIZE = "k must be between 1 and the number of rows, and the epochs must cover at least one row"


class GpuInputError(ValueError):
    def __init__(self, fault: Fault) -> None:
        super().__init__(fault.value)
        self.fault = fault


def resolve(device: torch.device | None) -> torch.device:
    if not torch.cuda.is_available():
        raise GpuInputError(Fault.NO_CUDA)
    chosen = torch.device(CUDA) if device is None else device
    if chosen.type != CUDA:
        raise GpuInputError(Fault.NOT_CUDA)
    return torch.device(CUDA, torch.cuda.current_device() if chosen.index is None else chosen.index)


class GroupIndex:
    def __init__(self, k: int, dims: int, probes: int, seed: int, device: torch.device | None = None) -> None:
        self.device = resolve(device)
        self.geometry = kernels.layout(k, dims, probes)
        groups, span, padded = self.geometry.groups, self.geometry.span, self.geometry.padded
        if span > MAX_GROUP_SPAN:
            raise GpuInputError(Fault.TOO_MANY_GROUPS)
        if not 1 <= probes <= groups:
            raise GpuInputError(Fault.BAD_PROBES)
        picks = torch.randperm(k, generator=torch.Generator().manual_seed(seed))[:groups]
        self.picks = picks.to(device=self.device, dtype=torch.int32)
        self.centers = self.vectors(span, padded, torch.float32)
        self.rounded = self.vectors(span, padded, torch.bfloat16)
        sizes = torch.zeros(span, dtype=torch.int32, device=self.device)
        self.lloyd = Lloyd(
            dots=torch.zeros((k, span), dtype=torch.float32, device=self.device),
            owner=torch.zeros(k, dtype=torch.int32, device=self.device),
            sums=torch.zeros((span, padded), dtype=torch.float32, device=self.device),
            sizes=sizes,
        )
        self.members = Members(
            data=torch.zeros((k, padded), dtype=torch.bfloat16, device=self.device),
            norms=torch.zeros(k, dtype=torch.float32, device=self.device),
            order=torch.zeros(k, dtype=torch.int32, device=self.device),
            ends=torch.zeros(span, dtype=torch.int32, device=self.device),
            sizes=sizes,
        )
        self.grouped = False

    def vectors(self, count: int, width: int, dtype: torch.dtype) -> Vectors[torch.Tensor]:
        return Vectors(
            torch.zeros((count, width), dtype=dtype, device=self.device),
            torch.full((count,), float("inf"), dtype=torch.float32, device=self.device),
        )

    def regroup(self, centroids: torch.Tensor) -> None:
        self.check(centroids, torch.float32, self.geometry.clusters)
        matrix = kernels.matrix(centroids)
        with torch.cuda.device(self.device):
            self.lloyd.sums.zero_()
            self.lloyd.sizes.zero_()
            kernels.seed(matrix, self.picks, self.centers, self.geometry)
            for iteration in range(GROUP_ITERATIONS + 1):
                kernels.home(matrix, self.centers, self.lloyd, self.geometry)
                if iteration < GROUP_ITERATIONS:
                    kernels.update(self.centers, self.lloyd, self.geometry)
            torch.cumsum(self.lloyd.sizes, 0, dtype=torch.int32, out=self.members.ends)
            self.members.order.copy_(torch.argsort(self.lloyd.owner, stable=True))
            kernels.publish(self.centers, self.lloyd, self.rounded, self.geometry)
        self.grouped = True

    def refresh(self, centroids: torch.Tensor) -> None:
        self.check(centroids, torch.float32, self.geometry.clusters)
        self.require_groups()
        with torch.cuda.device(self.device):
            kernels.refresh(kernels.matrix(centroids), self.members, self.geometry)

    def assign(self, rows: torch.Tensor) -> torch.Tensor:
        self.check(rows, torch.bfloat16, None)
        self.require_groups()
        labels = torch.empty(len(rows), dtype=torch.int64, device=self.device)
        if len(rows) == 0:
            return labels
        matrix, plan = kernels.matrix(rows), kernels.pairs(len(rows), self.geometry, self.device)
        with torch.cuda.device(self.device):
            kernels.select(matrix, self.rounded, plan, self.geometry)
            kernels.bucket(plan, self.geometry)
            kernels.member(matrix, plan, self.members, self.geometry)
            kernels.reduce(plan, labels, self.geometry)
        return labels

    def check(self, tensor: torch.Tensor, dtype: torch.dtype, rows: int | None) -> None:
        if not tensor.is_cuda:
            raise GpuInputError(Fault.NOT_CUDA)
        if tensor.device != self.device:
            raise GpuInputError(Fault.WRONG_DEVICE)
        if tensor.dtype != dtype:
            raise GpuInputError(Fault.WRONG_DTYPE)
        if tensor.dim() != MATRIX_RANK or tensor.shape[1] < self.geometry.dims:
            raise GpuInputError(Fault.WRONG_SHAPE)
        if rows is not None and tensor.shape[0] != rows:
            raise GpuInputError(Fault.WRONG_SHAPE)
        if tensor.stride(1) != 1:
            raise GpuInputError(Fault.NOT_ROW_MAJOR)

    def require_groups(self) -> None:
        if not self.grouped:
            raise GpuInputError(Fault.NOT_GROUPED)
