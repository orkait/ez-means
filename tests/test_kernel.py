import hashlib
import os
import subprocess
import sys

import faiss
import numpy as np
import pytest

from ezmeans import kernel
from ezmeans.kernel import Matrix
from ezmeans.recipe import group_count, law_recipe
from ezmeans.trainer import ez_means

ROWS = 4096
CLUSTERS = 256
EPOCHS = 2
PROBES = 4
SEED = 0
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


@pytest.mark.parametrize("dims", [16, 64, 128])
def test_probing_every_group_is_brute_force(dims: int) -> None:
    x = points(dims)
    centroids = np.ascontiguousarray(x[:CLUSTERS])
    groups = group_count(CLUSTERS)
    homes = np.arange(CLUSTERS) % groups
    offsets = np.cumsum([0, *np.bincount(homes)])
    _, labels, _ = kernel.assign(x, centroids[:groups], centroids, np.argsort(homes, kind="stable"), offsets, groups)
    _, truth = faiss.knn(x, centroids, 1)
    assert np.array_equal(labels, truth[:, 0])


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


def test_init_must_have_one_row_per_cluster() -> None:
    x = points(32)
    with pytest.raises(kernel.KernelError) as raised:
        ez_means(x, CLUSTERS, law_recipe(ROWS, CLUSTERS, 1, SEED), PROBES, init=x[: CLUSTERS - 1])
    assert raised.value.status is kernel.KernelStatus.BAD_SIZE


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
