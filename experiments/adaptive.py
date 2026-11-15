import sys

import numpy as np
import torch

from ezmeans.config import PAPER_BENCH_DIR
from ezmeans.gpu.backend import train
from ezmeans.gpu.constants import CUDA
from ezmeans.gpu.index import GroupIndex
from ezmeans.reference import law_recipe

sys.path.insert(0, str(PAPER_BENCH_DIR))
import paper_data

K = 16384
SAMPLE = 50_000
CHUNK = 4096
SEED = 10
TRAIN_PROBES = 8
FIXED = (2, 4, 6, 8)
SLACK = (0.1, 0.2)
REACH = (0.2, 0.3, 0.4, 0.5)
CAP = 32
DATASETS = (("deep1b-1M", 2.0), ("fb-ssnpp-1M", 1.4), ("spacev-1M", 2.6))


def policies(
    near: torch.Tensor, rank: torch.Tensor, seed_best: torch.Tensor, radius: torch.Tensor
) -> dict[str, torch.Tensor]:
    closest = near.min(1, keepdim=True).values
    gap = near.clamp(min=0).sqrt()
    first = rank == 0
    masks = {f"fixed p{p}": rank < p for p in FIXED}
    masks |= {f"slack {s:g}": (near <= closest * (1 + s)) & (rank < CAP) for s in SLACK}
    masks |= {f"bound {a:g}": first | ((gap - a * radius[None, :] < seed_best) & (rank < CAP)) for a in REACH}
    return masks


def main() -> None:
    device = torch.device(CUDA)
    print("| data | rule | mean probes | excess over exact | exact label |")
    print("|---|---|---|---|---|", flush=True)
    for name, epochs in DATASETS:
        raw, _ = paper_data.load(name)
        xg = torch.from_numpy(np.ascontiguousarray(paper_data.rows(raw, name))).to(device)
        centroids = train(xg, K, law_recipe(len(xg), K, epochs, SEED), TRAIN_PROBES).float().contiguous()
        index = GroupIndex(K, centroids.shape[1], TRAIN_PROBES, SEED, device)
        index.regroup(centroids)
        owner = index.owner.long()
        centers = index.centers[: index.groups, : centroids.shape[1]]
        spread = (centroids - centers[owner]).pow(2).sum(1).sqrt()
        radius = torch.zeros(index.groups, device=device).scatter_reduce(0, owner, spread, reduce="amax")
        generator = torch.Generator(device=device).manual_seed(SEED)
        sample = xg[torch.randperm(len(xg), device=device, generator=generator)[:SAMPLE]]
        probes: dict[str, float] = {}
        found: dict[str, float] = {}
        hits: dict[str, float] = {}
        exact_total = 0.0
        for chunk in sample.split(CHUNK):
            row_norms = (chunk**2).sum(1, keepdim=True)
            full = (row_norms - 2.0 * chunk @ centroids.T + (centroids**2).sum(1)).clamp(min=0)
            exact, exact_id = full.min(1)
            exact_total += float(exact.sum())
            near = row_norms - 2.0 * chunk @ centers.T + (centers**2).sum(1)
            rank = near.argsort(1).argsort(1)
            seed_best = full.masked_fill(~(rank == 0)[:, owner], float("inf")).min(1).values.sqrt()[:, None]
            for rule, probed in policies(near, rank, seed_best, radius).items():
                best, best_id = full.masked_fill(~probed[:, owner], float("inf")).min(1)
                probes[rule] = probes.get(rule, 0.0) + float(probed.sum())
                found[rule] = found.get(rule, 0.0) + float(best.sum())
                hits[rule] = hits.get(rule, 0.0) + float((best_id == exact_id).sum())
        for rule, count in probes.items():
            print(
                f"| {name} | {rule} | {count / SAMPLE:.2f} | {found[rule] / exact_total - 1:.4%} | "
                f"{hits[rule] / SAMPLE:.4f} |",
                flush=True,
            )
        del xg, sample
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
