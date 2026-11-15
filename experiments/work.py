import sys
from dataclasses import dataclass

import numpy as np
import torch

from ezmeans.config import PAPER_BENCH_DIR, SIFT_BASE
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.gpu.backend import train
from ezmeans.gpu.constants import CUDA, MEMBER_TILE
from ezmeans.gpu.index import GroupIndex
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

sys.path.insert(0, str(PAPER_BENCH_DIR))
import paper_data

SEED = 10
LLOYD_ITERATIONS = 25
SAMPLE = 50_000
SIFT = "sift1M"
SIFT_ROWS = 1_000_000


@dataclass(frozen=True)
class Setting:
    name: str
    k: int
    epochs: float
    probes: int
    emeans_epochs: float


SETTINGS = (
    Setting(SIFT, 4096, 3.0, 4, 2.0),
    Setting("deep1b-1M", 16384, 2.0, 4, 1.8),
    Setting("fb-ssnpp-1M", 16384, 1.4, 4, 1.4),
    Setting("spacev-1M", 16384, 2.6, 8, 2.2),
)


def load(name: str) -> Matrix:
    if name == SIFT:
        return read_fvecs(SIFT_BASE, SIFT_ROWS)
    raw, _ = paper_data.load(name)
    rows: Matrix = np.ascontiguousarray(paper_data.rows(raw, name))
    return rows


def gpu_fraction(x: Matrix, setting: Setting) -> float:
    xg = torch.from_numpy(x).to(torch.device(CUDA))
    recipe = law_recipe(len(x), setting.k, setting.epochs, SEED)
    centroids = train(xg, setting.k, recipe, setting.probes).float().contiguous()
    index = GroupIndex(setting.k, x.shape[1], setting.probes, SEED, xg.device)
    index.regroup(centroids)
    centers = index.centers[: index.groups, : x.shape[1]]
    scanned = torch.ceil(index.sizes[: index.groups].float() / MEMBER_TILE) * MEMBER_TILE
    generator = torch.Generator(device=xg.device).manual_seed(SEED)
    rows = xg[torch.randperm(len(x), device=xg.device, generator=generator)[:SAMPLE]]
    near = index.center_norms[None, : index.groups] - 2.0 * rows @ centers.T
    probed = near.topk(setting.probes, dim=1, largest=False).indices
    return float((len(index.sizes) + scanned[probed].sum(1)).mean()) / setting.k


def main() -> None:
    print(
        "| data | k | ez-means T, p | ez-means work / (T n k) | e-means T | fewer distances than e-means | than Lloyd-25 | GPU work / (n k), one epoch |"
    )
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for setting in SETTINGS:
        x = load(setting.name)
        n, k = len(x), setting.k
        _, work = ez_means(x, k, law_recipe(n, k, setting.epochs, SEED), setting.probes)
        print(
            f"| {setting.name} | {k} | {setting.epochs:g}, p{setting.probes} | {work / (setting.epochs * n * k):.4f} | "
            f"{setting.emeans_epochs:g} | {setting.emeans_epochs * n * k / work:.1f}x | {LLOYD_ITERATIONS * n * k / work:.0f}x | "
            f"{gpu_fraction(x, setting):.4f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
