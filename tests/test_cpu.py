import dataclasses
import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ezmeans import law_recipe
from ezmeans.config import KERNEL_COMPILE
from ezmeans.cpu import KernelError, KernelStatus, native, train
from ezmeans.cpu.build import KERNEL_SOURCE
from ezmeans.cpu.native import Labels, Matrix
from ezmeans.recipe import group_count

ROWS = 4096
CLUSTERS = 256
DIMS = 32
KERNEL_CHUNK = 16
EPOCHS = 2
PROBES = 4
SEED = 0
GROUPS = group_count(CLUSTERS)
MANY_GROUPS = 80
HUGE_BATCH = 2**62
POISONED_ROW = 7
OVERFLOW_SCALE = np.float32(1e20)
TINY_EPOCHS = 1e-4
THREAD_COUNTS = ("1", "4")
TRAIN_SCRIPT = f"""
import hashlib, sys
import numpy as np
from ezmeans import law_recipe
from ezmeans.cpu import train
x = np.random.default_rng({SEED}).standard_normal(({ROWS}, {DIMS})).astype(np.float32)
centroids = train(x, {CLUSTERS}, law_recipe(len(x), {CLUSTERS}, {EPOCHS}, {SEED}), {PROBES})
sys.stdout.write(hashlib.sha256(centroids.tobytes()).hexdigest())
"""
BLOCK_FAISS = 'import sys\nsys.modules["faiss"] = None\n'
STRICT_WARNINGS = (
    "-Wpedantic",
    "-Wshadow",
    "-Wconversion",
    "-Wsign-conversion",
    "-Wcast-qual",
    "-Wdouble-promotion",
    "-Werror",
    "-c",
)
OBJECT_SUFFIX = ".o"


def points(dims: int) -> Matrix:
    return np.random.default_rng(SEED).standard_normal((ROWS, dims)).astype(np.float32)


def poisoned(value: float) -> Matrix:
    x = points(DIMS)
    x[POISONED_ROW] = value
    return x


def round_robin() -> Labels:
    return np.arange(CLUSTERS, dtype=np.int64) % GROUPS


def fit(x: Matrix, init: Matrix | None = None, epochs: float = EPOCHS) -> Matrix:
    return train(x, CLUSTERS, law_recipe(ROWS, CLUSTERS, epochs, SEED), PROBES, init=init)


def search(x: Matrix, homes: Labels, probes: int = GROUPS, groups: int = GROUPS) -> Labels:
    centroids = np.ascontiguousarray(x[:CLUSTERS])
    return native.assign(x, centroids[:groups], centroids, homes, probes)


@pytest.mark.parametrize(("groups", "occupied"), [(GROUPS, GROUPS), (GROUPS, GROUPS - 1), (MANY_GROUPS, MANY_GROUPS)])
@pytest.mark.parametrize("dims", [16, 64, 128])
def test_probing_every_occupied_group_is_brute_force(dims: int, groups: int, occupied: int) -> None:
    x = points(dims)
    labels = search(x, np.arange(CLUSTERS, dtype=np.int64) % occupied, occupied, groups)
    wide, centers = x.astype(np.float64), x[:CLUSTERS].astype(np.float64)
    squared = (wide**2).sum(axis=1)[:, None] - 2.0 * wide @ centers.T + (centers**2).sum(axis=1)[None, :]
    assert np.array_equal(labels, squared.argmin(axis=1))


@pytest.mark.parametrize(
    ("call", "status"),
    [
        (lambda: fit(poisoned(np.nan)), KernelStatus.NOT_FINITE),
        (lambda: fit(poisoned(np.inf)), KernelStatus.NOT_FINITE),
        (lambda: fit(points(DIMS) * OVERFLOW_SCALE), KernelStatus.NOT_FINITE),
        (lambda: fit(points(DIMS), init=poisoned(np.nan)[:CLUSTERS]), KernelStatus.NOT_FINITE),
        (lambda: fit(points(DIMS), init=points(DIMS)[: CLUSTERS - 1]), KernelStatus.BAD_SIZE),
        (lambda: fit(points(DIMS), init=points(DIMS // 2)[:CLUSTERS]), KernelStatus.BAD_SIZE),
        (lambda: fit(points(DIMS), epochs=TINY_EPOCHS), KernelStatus.BAD_SIZE),
        (lambda: search(poisoned(np.nan), round_robin()), KernelStatus.NOT_FINITE),
        (lambda: search(points(DIMS), round_robin() + GROUPS), KernelStatus.BAD_LABEL),
        (lambda: search(points(DIMS), round_robin()[1:]), KernelStatus.BAD_SIZE),
        (lambda: native.accumulate(points(DIMS), round_robin(), CLUSTERS), KernelStatus.BAD_SIZE),
    ],
    ids=[
        "nan row",
        "inf row",
        "overflow",
        "nan init",
        "short init",
        "narrow init",
        "empty plan",
        "nan search",
        "home out of range",
        "short homes",
        "short labels",
    ],
)
def test_bad_input_raises_kernel_error(call: Callable[[], object], status: KernelStatus) -> None:
    with pytest.raises(KernelError) as raised:
        call()
    assert raised.value.status is status


@pytest.mark.parametrize("dims", [20, 100])
def test_zero_padding_does_not_change_centroids(dims: int) -> None:
    x = points(dims)
    width = -(-dims // KERNEL_CHUNK) * KERNEL_CHUNK
    padded = np.zeros((ROWS, width), dtype=np.float32)
    padded[:, :dims] = x
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    centroids = train(x, CLUSTERS, recipe, PROBES)
    reference = train(padded, CLUSTERS, recipe, PROBES)
    assert np.array_equal(centroids, reference[:, :dims])


def test_a_batch_past_the_rows_is_one_batch_per_epoch() -> None:
    x = points(DIMS)
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    whole = train(x, CLUSTERS, dataclasses.replace(recipe, batch=ROWS), PROBES)
    assert np.array_equal(whole, train(x, CLUSTERS, dataclasses.replace(recipe, batch=HUGE_BATCH), PROBES))


def test_accumulate_matches_numpy_add_at() -> None:
    x = points(DIMS)
    labels = np.random.default_rng(SEED).integers(0, CLUSTERS, ROWS)
    sums, counts = native.accumulate(x, labels, CLUSTERS)
    expected = np.zeros((CLUSTERS, DIMS), dtype=np.float32)
    np.add.at(expected, labels, x)
    assert np.array_equal(sums, expected)
    assert np.array_equal(counts, np.bincount(labels, minlength=CLUSTERS).astype(np.float32))


def test_centroids_do_not_depend_on_thread_count() -> None:
    digests = set()
    for threads in THREAD_COUNTS:
        environment = os.environ | {"OMP_NUM_THREADS": threads}
        result = subprocess.run(
            [sys.executable, "-c", TRAIN_SCRIPT], env=environment, capture_output=True, text=True, check=True
        )
        digests.add(result.stdout)
    assert len(digests) == 1
    assert hashlib.sha256(b"").hexdigest() not in digests


def test_cpu_training_runs_without_faiss() -> None:
    subprocess.run([sys.executable, "-c", BLOCK_FAISS + TRAIN_SCRIPT], capture_output=True, check=True)


def test_kernel_compiles_without_strict_warnings(tmp_path: Path) -> None:
    output = tmp_path / KERNEL_SOURCE.with_suffix(OBJECT_SUFFIX).name
    command = [*KERNEL_COMPILE, *STRICT_WARNINGS, str(KERNEL_SOURCE), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
