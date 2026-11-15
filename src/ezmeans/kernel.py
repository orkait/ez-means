import ctypes
import functools
import hashlib
import os
import subprocess
import tempfile
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ezmeans.config import CACHE, KERNEL_COMPILE, KERNEL_SOURCE

Matrix = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int64]

FLOATS = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
INDICES = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
SIZE = ctypes.c_int64
LIBRARY_STEM = "libkernel"
LIBRARY_SUFFIX = ".so"
KEY_LENGTH = 16
TARGET_QUERY = ("-Q", "--help=target")


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


def build_key() -> str:
    target = subprocess.run([*KERNEL_COMPILE, *TARGET_QUERY], capture_output=True, text=True, check=True).stdout
    digest = hashlib.sha256(KERNEL_SOURCE.read_bytes())
    digest.update("\0".join(KERNEL_COMPILE).encode())
    digest.update(target.encode())
    return digest.hexdigest()[:KEY_LENGTH]


def build() -> Path:
    path = CACHE / f"{LIBRARY_STEM}-{build_key()}{LIBRARY_SUFFIX}"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=CACHE, suffix=LIBRARY_SUFFIX)
        os.close(handle)
        try:
            subprocess.run([*KERNEL_COMPILE, str(KERNEL_SOURCE), "-o", temporary], check=True)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


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
        FLOATS,
        INDICES,
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
        INDICES,
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


def assign(rows: Matrix, groups: Matrix, centroids: Matrix, homes: Labels, probes: int) -> tuple[Matrix, Labels, int]:
    count, dims = rows.shape
    if groups.shape[1:] != (dims,) or centroids.shape[1:] != (dims,) or homes.shape != (len(centroids),):
        raise KernelError(KernelStatus.BAD_SIZE)
    if np.any((homes < 0) | (homes >= len(groups))):
        raise KernelError(KernelStatus.BAD_LABEL)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(homes, minlength=len(groups)))))
    best = np.empty(count, dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    work = np.zeros(1, dtype=np.int64)
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
            best,
            labels,
            work,
        )
    )
    return best, labels, int(work[0])


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


def train(x: Matrix, k: int, plan: list[int], settings: Settings, init: Matrix | None) -> tuple[Matrix, int]:
    n, dims = x.shape
    if init is not None and init.shape != (k, dims):
        raise KernelError(KernelStatus.BAD_SIZE)
    centroids = np.empty((k, dims), dtype=np.float32)
    work = np.zeros(1, dtype=np.int64)
    rows = np.asarray(plan, dtype=np.int64)
    start = np.empty(0, dtype=np.float32) if init is None else np.ascontiguousarray(init, dtype=np.float32)
    check(
        library().train(
            np.ascontiguousarray(x),
            n,
            dims,
            k,
            rows,
            len(rows),
            ctypes.byref(settings),
            start,
            0 if init is None else len(start),
            centroids,
            work,
        )
    )
    return centroids, int(work[0])
