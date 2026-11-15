import ctypes
import functools
import subprocess

import numpy as np

from ezmeans.config import KERNEL_COMPILE, KERNEL_LIBRARY, KERNEL_SOURCE
from ezmeans.dataset import Labels, Matrix

DONE = 0
FLOATS = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
INDICES = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
SIZE = ctypes.c_int64


@functools.cache
def library() -> ctypes.CDLL:
    if not KERNEL_LIBRARY.exists() or KERNEL_LIBRARY.stat().st_mtime < KERNEL_SOURCE.stat().st_mtime:
        KERNEL_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*KERNEL_COMPILE, str(KERNEL_SOURCE), "-o", str(KERNEL_LIBRARY)], check=True)
    loaded = ctypes.CDLL(str(KERNEL_LIBRARY))
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
        ("ad_epsilon", ctypes.c_double),
        ("screen_epsilon", ctypes.c_double),
    ]


def check(status: int) -> None:
    if status != DONE:
        raise RuntimeError(f"kernel failed with status {status}")


def assign(
    rows: Matrix, groups: Matrix, centroids: Matrix, order: Labels, offsets: Labels, probes: int
) -> tuple[Matrix, Labels, int]:
    count, dims = rows.shape
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
            np.ascontiguousarray(order, dtype=np.int64),
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
