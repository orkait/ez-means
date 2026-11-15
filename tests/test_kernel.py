import dataclasses
import hashlib
import os
import subprocess
import sys
from collections.abc import Callable

import faiss
import numpy as np
import pytest

from ezmeans import kernel
from ezmeans.kernel import KernelError, KernelStatus, Labels, Matrix
from ezmeans.recipe import group_count, law_recipe
from ezmeans.trainer import ez_means

ROWS = 4096
CLUSTERS = 256
DIMS = 32
EPOCHS = 2
PROBES = 4
SEED = 0
GROUPS = group_count(CLUSTERS)
POISONED_ROW = 7
OVERFLOW_SCALE = np.float32(1e20)
TINY_EPOCHS = 1e-4
THREAD_COUNTS = ("1", "4")
DETERMINISM_SCRIPT = """
import hashlib, sys
import numpy as np
from ezmeans.recipe import law_recipe
from ezmeans.trainer import ez_means
x = np.random.default_rng(0).standard_normal((4096, 48)).astype(np.float32)
centroids, _ = ez_means(x, 256, law_recipe(len(x), 256, 2, 0), 4)
sys.stdout.write(hashlib.sha256(centroids.tobytes()).hexdigest())
"""


def points(dims: int) -> Matrix:
    return np.random.default_rng(SEED).standard_normal((ROWS, dims)).astype(np.float32)


def poisoned(value: float) -> Matrix:
    x = points(DIMS)
    x[POISONED_ROW] = value
    return x


def round_robin() -> Labels:
    return np.arange(CLUSTERS, dtype=np.int64) % GROUPS


def fit(x: Matrix, init: Matrix | None = None, epochs: float = EPOCHS) -> tuple[Matrix, float]:
    return ez_means(x, CLUSTERS, law_recipe(ROWS, CLUSTERS, epochs, SEED), PROBES, init=init)


def search(x: Matrix, homes: Labels) -> tuple[Matrix, Labels, int]:
    centroids = np.ascontiguousarray(x[:CLUSTERS])
    return kernel.assign(x, centroids[:GROUPS], centroids, homes, GROUPS)


@pytest.mark.parametrize("dims", [16, 64, 128])
def test_probing_every_group_is_brute_force(dims: int) -> None:
    x = points(dims)
    _, labels, _ = search(x, round_robin())
    _, truth = faiss.knn(x, x[:CLUSTERS], 1)
    assert np.array_equal(labels, truth[:, 0])


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
        (lambda: kernel.accumulate(points(DIMS), round_robin(), CLUSTERS), KernelStatus.BAD_SIZE),
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


@pytest.mark.parametrize(("epochs", "alpha0"), [(0.0, 1.0), (-0.5, 1.0), (EPOCHS, 0.0), (EPOCHS, 1.5)])
def test_recipe_rejects_impossible_values(epochs: float, alpha0: float) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(law_recipe(ROWS, CLUSTERS, EPOCHS, SEED), epochs=epochs, alpha0=alpha0)


@pytest.mark.parametrize("dims", [20, 100])
def test_zero_padding_does_not_change_centroids(dims: int) -> None:
    x = points(dims)
    width = -(-dims // 16) * 16
    padded = np.zeros((ROWS, width), dtype=np.float32)
    padded[:, :dims] = x
    recipe = law_recipe(ROWS, CLUSTERS, EPOCHS, SEED)
    centroids, _ = ez_means(x, CLUSTERS, recipe, PROBES)
    reference, _ = ez_means(padded, CLUSTERS, recipe, PROBES)
    assert np.array_equal(centroids, reference[:, :dims])


def test_accumulate_matches_numpy_add_at() -> None:
    x = points(64)
    labels = np.random.default_rng(SEED).integers(0, CLUSTERS, ROWS)
    sums, counts = kernel.accumulate(x, labels, CLUSTERS)
    expected = np.zeros((CLUSTERS, 64), dtype=np.float32)
    np.add.at(expected, labels, x)
    assert np.array_equal(sums, expected)
    assert np.array_equal(counts, np.bincount(labels, minlength=CLUSTERS).astype(np.float32))


def test_centroids_do_not_depend_on_thread_count() -> None:
    digests = set()
    for threads in THREAD_COUNTS:
        environment = os.environ | {"OMP_NUM_THREADS": threads}
        result = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SCRIPT], env=environment, capture_output=True, text=True, check=True
        )
        digests.add(result.stdout)
    assert len(digests) == 1
    assert hashlib.sha256(b"").hexdigest() not in digests
