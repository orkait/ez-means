import ctypes
import functools
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from ezmeans.cpu.build import build

Matrix = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int64]

FLOATS = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
INDICES = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
SIZE = ctypes.c_int64


class KernelStatus(IntEnum):
    DONE = 0
    BAD_PROBES = 1
    NO_MEMORY = 2
    BAD_LABEL = 3
    BAD_DIMS = 4
    BAD_SIZE = 5
    NOT_FINITE = 6


class KernelError(RuntimeError):
    def __init__(self, status: KernelStatus) -> None:
        super().__init__(f"kernel failed: {status.name}")
        self.status = status


@functools.cache
def library() -> ctypes.CDLL:
    loaded = ctypes.CDLL(str(build()))
    loaded.assign.argtypes = [
        FLOATS,
        SIZE,
        SIZE,
        FLOATS,
        SIZE,
        FLOATS,
        INDICES,
        INDICES,
        SIZE,
        INDICES,
    ]
    loaded.accumulate.argtypes = [FLOATS, SIZE, SIZE, INDICES, SIZE, FLOATS, FLOATS]
    loaded.train.argtypes = [
        FLOATS,
        SIZE,
        SIZE,
        SIZE,
        INDICES,
        SIZE,
        ctypes.POINTER(Settings),
        FLOATS,
        SIZE,
        FLOATS,
    ]
    for function in (loaded.assign, loaded.accumulate, loaded.train):
        function.restype = ctypes.c_int
    return loaded


class Settings(ctypes.Structure):
    _fields_ = [
        ("batch", ctypes.c_int64),
        ("probes", ctypes.c_int64),
        ("check_every", ctypes.c_int64),
        ("group_iterations", ctypes.c_int64),
        ("seed", ctypes.c_uint64),
        ("alpha0", ctypes.c_double),
        ("alpha_min_factor", ctypes.c_double),
        ("tau", ctypes.c_double),
        ("init_usage", ctypes.c_double),
        ("split_epsilon", ctypes.c_double),
    ]


def check(status: int) -> None:
    code = KernelStatus(status)
    if code is not KernelStatus.DONE:
        raise KernelError(code)


def assign(rows: Matrix, groups: Matrix, centroids: Matrix, homes: Labels, probes: int) -> Labels:
    count, dims = rows.shape
    if groups.shape[1:] != (dims,) or centroids.shape[1:] != (dims,) or homes.shape != (len(centroids),):
        raise KernelError(KernelStatus.BAD_SIZE)
    if np.any((homes < 0) | (homes >= len(groups))):
        raise KernelError(KernelStatus.BAD_LABEL)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(homes, minlength=len(groups)))))
    labels = np.empty(count, dtype=np.int64)
    check(
        library().assign(
            np.ascontiguousarray(rows),
            count,
            dims,
            np.ascontiguousarray(groups),
            len(groups),
            np.ascontiguousarray(centroids),
            np.ascontiguousarray(np.argsort(homes, kind="stable"), dtype=np.int64),
            np.ascontiguousarray(offsets, dtype=np.int64),
            probes,
            labels,
        )
    )
    return labels


def accumulate(rows: Matrix, labels: Labels, k: int) -> tuple[Matrix, Matrix]:
    count, dims = rows.shape
    if labels.shape != (count,):
        raise KernelError(KernelStatus.BAD_SIZE)
    sums = np.empty((k, dims), dtype=np.float32)
    counts = np.empty(k, dtype=np.float32)
    check(
        library().accumulate(
            np.ascontiguousarray(rows), count, dims, np.ascontiguousarray(labels, dtype=np.int64), k, sums, counts
        )
    )
    return sums, counts


def train(x: Matrix, k: int, plan: list[int], settings: Settings, init: Matrix | None) -> Matrix:
    n, dims = x.shape
    if init is not None and init.shape != (k, dims):
        raise KernelError(KernelStatus.BAD_SIZE)
    centroids = np.empty((k, dims), dtype=np.float32)
    plan_rows = np.asarray(plan, dtype=np.int64)
    start = np.empty(0, dtype=np.float32) if init is None else np.ascontiguousarray(init, dtype=np.float32)
    check(
        library().train(
            np.ascontiguousarray(x),
            n,
            dims,
            k,
            plan_rows,
            len(plan_rows),
            ctypes.byref(settings),
            start,
            0 if init is None else len(start),
            centroids,
        )
    )
    return centroids
