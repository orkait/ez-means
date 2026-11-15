from enum import Enum

import torch

from ezmeans.gpu import kernels
from ezmeans.gpu.constants import CUDA, MATRIX_RANK, MAX_GROUP_SPAN, PAIR_ROWS
from ezmeans.recipe import GROUP_ITERATIONS, group_count


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
        self.k = k
        self.dims = dims
        self.groups = group_count(k)
        self.probes = probes
        span = kernels.tile(self.groups)
        if span > MAX_GROUP_SPAN:
            raise GpuInputError(Fault.TOO_MANY_GROUPS)
        if not 1 <= probes <= self.groups:
            raise GpuInputError(Fault.BAD_PROBES)
        padded = kernels.tile(dims)
        picks = torch.randperm(k, generator=torch.Generator().manual_seed(seed))[: self.groups]
        self.picks = picks.to(device=self.device, dtype=torch.int32)
        self.centers = torch.zeros((span, padded), dtype=torch.float32, device=self.device)
        self.center_norms = torch.full((span,), float("inf"), dtype=torch.float32, device=self.device)
        self.sums = torch.zeros((span, padded), dtype=torch.float32, device=self.device)
        self.sizes = torch.zeros(span, dtype=torch.int32, device=self.device)
        self.owner = torch.zeros(k, dtype=torch.int32, device=self.device)
        self.dots = torch.zeros((k, span), dtype=torch.float32, device=self.device)
        self.rounded = torch.zeros((span, padded), dtype=torch.bfloat16, device=self.device)
        self.rounded_norms = torch.full((span,), float("inf"), dtype=torch.float32, device=self.device)
        self.group_ends = torch.zeros(span, dtype=torch.int32, device=self.device)
        self.fill = torch.zeros(span, dtype=torch.int32, device=self.device)
        self.order = torch.zeros(k, dtype=torch.int32, device=self.device)
        self.members = torch.zeros((k, padded), dtype=torch.bfloat16, device=self.device)
        self.member_norms = torch.zeros(k, dtype=torch.float32, device=self.device)
        self.grouped = False

    def regroup(self, centroids: torch.Tensor) -> None:
        self.check(centroids, torch.float32, self.k)
        with torch.cuda.device(self.device):
            self.sums.zero_()
            self.sizes.zero_()
            kernels.seed(centroids, self.picks, self.centers, self.center_norms, self.dims)
            for iteration in range(GROUP_ITERATIONS + 1):
                kernels.lloyd(
                    centroids, self.centers, self.center_norms, self.dots, self.owner, self.sums, self.sizes, self.dims
                )
                if iteration < GROUP_ITERATIONS:
                    kernels.update(self.centers, self.center_norms, self.sums, self.sizes, self.groups)
            torch.cumsum(self.sizes, 0, dtype=torch.int32, out=self.group_ends)
            self.fill.zero_()
            kernels.bucket(self.owner, self.group_ends, self.sizes, self.fill, self.order)
            kernels.publish(self.centers, self.sizes, self.rounded, self.rounded_norms, self.groups)
        self.grouped = True

    def refresh(self, centroids: torch.Tensor) -> None:
        self.check(centroids, torch.float32, self.k)
        self.require_groups()
        with torch.cuda.device(self.device):
            kernels.refresh(centroids, self.order, self.members, self.member_norms, self.dims)

    def assign(self, rows: torch.Tensor) -> torch.Tensor:
        self.check(rows, torch.bfloat16, None)
        self.require_groups()
        n, span = len(rows), len(self.sizes)
        pairs = n * self.probes
        labels = torch.empty(n, dtype=torch.int64, device=self.device)
        if n == 0:
            return labels
        counts = torch.zeros(span, dtype=torch.int32, device=self.device)
        fill = torch.zeros(span, dtype=torch.int32, device=self.device)
        ends = torch.empty(span, dtype=torch.int32, device=self.device)
        blocks = torch.empty(span, dtype=torch.int32, device=self.device)
        top = torch.empty(pairs, dtype=torch.int32, device=self.device)
        items = torch.empty(pairs, dtype=torch.int32, device=self.device)
        value = torch.empty(pairs, dtype=torch.float32, device=self.device)
        ident = torch.empty(pairs, dtype=torch.int32, device=self.device)
        with torch.cuda.device(self.device):
            kernels.select(rows, self.rounded, self.rounded_norms, top, counts, self.dims, self.probes)
            kernels.plan(counts, ends, blocks, self.groups, PAIR_ROWS)
            kernels.bucket(top, ends, counts, fill, items)
            kernels.member(
                rows,
                items,
                self.members,
                self.member_norms,
                self.order,
                self.group_ends,
                self.sizes,
                ends,
                counts,
                blocks,
                value,
                ident,
                self.dims,
                self.groups,
                self.probes,
            )
            kernels.reduce(value, ident, labels, self.probes)
        return labels

    def check(self, tensor: torch.Tensor, dtype: torch.dtype, rows: int | None) -> None:
        if not tensor.is_cuda:
            raise GpuInputError(Fault.NOT_CUDA)
        if tensor.device != self.device:
            raise GpuInputError(Fault.WRONG_DEVICE)
        if tensor.dtype != dtype:
            raise GpuInputError(Fault.WRONG_DTYPE)
        if tensor.dim() != MATRIX_RANK or tensor.shape[1] < self.dims:
            raise GpuInputError(Fault.WRONG_SHAPE)
        if rows is not None and tensor.shape[0] != rows:
            raise GpuInputError(Fault.WRONG_SHAPE)
        if tensor.stride(1) != 1:
            raise GpuInputError(Fault.NOT_ROW_MAJOR)

    def require_groups(self) -> None:
        if not self.grouped:
            raise GpuInputError(Fault.NOT_GROUPED)
