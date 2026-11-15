import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch
from faiss.contrib.e_means.e_means_torch import Emeans as TorchEmeans
from faiss.contrib.e_means.torch_backend import select_backend
from paper_cpu import BUDGETS, CLUSTERS, PAPER_CROSSING, Arm, Fit, crossing, test_error
from race import lloyd, mse

from ezmeans.config import PAPER_BENCH_DIR, PAPER_TARGETS, SIFT_BASE, SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.gpu.backend import train as ez_train
from ezmeans.gpu.constants import CUDA
from ezmeans.reference import Recipe, law_recipe

sys.path.insert(0, str(PAPER_BENCH_DIR))
import paper_data

VRAM_FRACTION = 0.5
BASELINE_PROBES = 0
WARM_SEED = 99
AUTOMATIC = "auto"
CURVE_SEED = 10
CURVE_EPOCHS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CHUNK_ROWS = 8192
SIFT_ROWS = 1_000_000
SIFT_TEST_ROWS = 100_000
SIFT_SEEDS = (0, 1)
SIFT_BUDGETS = (2.0, 2.2, 2.4, 3.0)
LLOYD_ITERATIONS = 25

Targets = dict[str, dict[str, float]]


def baseline_train(x: torch.Tensor, k: int, recipe: Recipe) -> torch.Tensor:
    model = TorchEmeans(x.shape[1], k, nepochs=recipe.epochs, seed=recipe.seed, capture_assignments=False)
    centroids, _ = model.train(x)
    return torch.as_tensor(centroids)


def gpu_fit(xg: torch.Tensor, probes: int) -> Fit:
    def fit(train: Matrix, recipe: Recipe, k: int) -> Matrix:
        trained = baseline_train(xg, k, recipe) if probes == BASELINE_PROBES else ez_train(xg, k, recipe, probes)
        torch.cuda.synchronize()
        result: Matrix = np.ascontiguousarray(trained.float().cpu().numpy())
        return result

    return fit


def arm_name(xg: torch.Tensor, k: int, probes: int) -> str:
    if probes != BASELINE_PROBES:
        return f"ez-means GPU p{probes}"
    return f"faiss GPU e-means, {select_backend(xg, xg.dtype, k, xg.shape[1], AUTOMATIC).name}"


def warm(fit: Fit, train: Matrix, k: int, budgets: tuple[float, ...]) -> None:
    for budget in budgets:
        fit(train, law_recipe(len(train), k, budget, WARM_SEED), k)


def empty_clusters(xg: torch.Tensor, centroids: Matrix) -> int:
    c = torch.from_numpy(centroids).to(xg.device)
    norms = (c**2).sum(1)
    counts = torch.zeros(len(c), device=xg.device, dtype=torch.long)
    for start in range(0, len(xg), CHUNK_ROWS):
        nearest = (norms[None, :] - 2.0 * (xg[start : start + CHUNK_ROWS].float() @ c.T)).argmin(dim=1)
        counts += torch.bincount(nearest, minlength=len(c))
    return int((counts == 0).sum())


def race(name: str, train: Matrix, test: Matrix, arms: list[Arm], seeds: list[int], targets: Targets) -> None:
    for arm in arms:
        crossings, ratios, times = [], [], []
        for seed in seeds:
            target = float(targets[str(seed)]["test_mse"])
            budget, error, seconds = crossing(train, test, arm, CLUSTERS[name], seed, target)
            crossings.append(budget)
            ratios.append(error / target)
            times.append(seconds)
        upper = sorted(crossings)[len(crossings) // 2]
        print(
            f"| {name} | {arm.name} | {', '.join(f'{c:g}' for c in crossings)} | {upper:g} | "
            f"{PAPER_CROSSING.get(name, '-')} | {statistics.median(ratios):.4f} | {statistics.median(times):.2f} | "
            f"{', '.join(f'{t:.2f}' for t in times)} |",
            flush=True,
        )


def curve(name: str, train: Matrix, test: Matrix, xg: torch.Tensor, arms: list[Arm], target: float) -> None:
    for arm in arms:
        for epochs in CURVE_EPOCHS:
            start = time.perf_counter()
            centroids = arm.fit(train, law_recipe(len(train), CLUSTERS[name], epochs, CURVE_SEED), CLUSTERS[name])
            seconds = time.perf_counter() - start
            print(
                f"| {name} | {arm.name} | {epochs:g} | {test_error(centroids, test) / target:.4f} | {seconds:.2f} | "
                f"{empty_clusters(xg, centroids)} |",
                flush=True,
            )


def sift(k: int, probes: list[int]) -> None:
    train, test = read_fvecs(SIFT_BASE, SIFT_ROWS), read_fvecs(SIFT_LEARN, SIFT_TEST_ROWS)
    reference = {seed: mse(test, lloyd(LLOYD_ITERATIONS)(train, k, seed)) for seed in SIFT_SEEDS}
    xg = torch.from_numpy(train).to(torch.device(CUDA))
    print("| arm | epochs | test MSE / Lloyd-25, per seed | GPU s, per seed |")
    print("|---|---|---|---|", flush=True)
    for p in probes:
        fit = gpu_fit(xg, p)
        warm(fit, train, k, SIFT_BUDGETS)
        for epochs in SIFT_BUDGETS:
            ratios, times = [], []
            for seed in SIFT_SEEDS:
                start = time.perf_counter()
                centroids = fit(train, law_recipe(len(train), k, epochs, seed), k)
                times.append(time.perf_counter() - start)
                ratios.append(mse(test, centroids) / reference[seed])
            print(
                f"| {arm_name(xg, k, p)} | {epochs:g} | {', '.join(f'{r:.4f}' for r in ratios)} | "
                f"{', '.join(f'{t:.3f}' for t in times)} |",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(CLUSTERS), default=["deep1b-1M"])
    parser.add_argument("--probes", type=int, nargs="+", default=[BASELINE_PROBES, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sift", type=int, metavar="K")
    options = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    if options.sift:
        sift(options.sift, options.probes)
        return
    targets = json.loads(PAPER_TARGETS.read_text())["table3"]
    if options.curve:
        print("| dataset | arm | epochs | test MSE / seed-10 target | GPU s | empty clusters |")
        print("|---|---|---|---|---|---|", flush=True)
    else:
        print(
            "| dataset | arm | crossings per seed | crossing, upper median | paper crossing | error / target "
            "| GPU s at crossing, median | GPU s per seed |"
        )
        print("|---|---|---|---|---|---|---|---|", flush=True)
    for name in options.datasets:
        raw, test = paper_data.load(name)
        train = np.ascontiguousarray(paper_data.rows(raw, name))
        xg = torch.from_numpy(train).to(torch.device(CUDA))
        if options.bf16:
            xg = xg.to(torch.bfloat16)
        k = CLUSTERS[name]
        arms = [Arm(arm_name(xg, k, p), gpu_fit(xg, p)) for p in options.probes]
        for arm in arms:
            warm(arm.fit, train, k, CURVE_EPOCHS if options.curve else BUDGETS)
        if options.curve:
            curve(name, train, test, xg, arms, float(targets[name]["per_seed"][str(CURVE_SEED)]["test_mse"]))
        else:
            race(name, train, test, arms, options.seeds, targets[name]["per_seed"])
        del xg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
