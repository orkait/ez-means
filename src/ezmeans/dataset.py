from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ezmeans.config import CLUSTERS, DATA_SEED, DIM, NOISE, POINTS

Matrix = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Dataset:
    points: Matrix
    labels: Labels
    classes: int


@dataclass(frozen=True)
class Spec:
    points: int
    classes: int
    noise: float
    skew: float


def unit_rows(rows: npt.NDArray[np.float64]) -> Matrix:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def generate() -> Dataset:
    rng = np.random.default_rng(DATA_SEED)
    directions = unit_rows(rng.standard_normal((CLUSTERS, DIM)))
    labels = rng.integers(0, CLUSTERS, POINTS)
    noise = NOISE * rng.standard_normal((POINTS, DIM))
    return Dataset(unit_rows(directions[labels] + noise), labels, CLUSTERS)


def sample(spec: Spec, seed: int) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    directions = unit_rows(rng.standard_normal((spec.classes, DIM)))
    weights = 1.0 / np.arange(1, spec.classes + 1) ** spec.skew
    weights /= weights.sum()

    def draw() -> Dataset:
        labels = rng.choice(spec.classes, spec.points, p=weights)
        noise = spec.noise * rng.standard_normal((spec.points, DIM))
        return Dataset(unit_rows(directions[labels] + noise), labels, spec.classes)

    return draw(), draw()


def read_fvecs(path: Path, rows: int) -> Matrix:
    width = int(np.fromfile(path, dtype="<i4", count=1)[0])
    raw = np.fromfile(path, dtype="<i4", count=rows * (width + 1)).reshape(rows, width + 1)
    return np.ascontiguousarray(raw[:, 1:].view("<f4"), dtype=np.float32)


def read_fbin(path: Path, start: int, rows: int) -> Matrix:
    width = int(np.fromfile(path, dtype="<i4", count=2)[1])
    header, item = 8, 4
    raw = np.fromfile(path, dtype="<f4", count=rows * width, offset=header + start * width * item)
    return np.ascontiguousarray(raw.reshape(rows, width), dtype=np.float32)


def read_u8bin(path: Path, start: int, rows: int) -> Matrix:
    width = int(np.fromfile(path, dtype="<i4", count=2)[1])
    header = 8
    raw = np.fromfile(path, dtype=np.uint8, count=rows * width, offset=header + start * width)
    return raw.reshape(rows, width).astype(np.float32)


def read_bvecs(path: Path, start: int, rows: int) -> Matrix:
    width = int(np.fromfile(path, dtype="<i4", count=1)[0])
    stride = width + 4
    raw = np.fromfile(path, dtype=np.uint8, count=rows * stride, offset=start * stride)
    return raw.reshape(rows, stride)[:, 4:].astype(np.float32)


def class_means(data: Dataset) -> Matrix:
    sums = np.zeros((data.classes, DIM))
    np.add.at(sums, data.labels, data.points)
    present = np.bincount(data.labels, minlength=data.classes) > 0
    return unit_rows(sums[present])
