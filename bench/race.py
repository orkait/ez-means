import argparse
import importlib
import json
import math
import random
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from ezmeans import BatchLaw, Layout, Recipe, Regroup, default_layout, law_recipe, work_layout
from ezmeans.gpu import train as ez_train
from ezmeans.recipe import group_count

SEEDS = tuple(range(10, 20))
BUDGETS = tuple(round(1.0 + 0.2 * step, 1) for step in range(11))
PROBES = (4, 8, 16)
UPLOAD_ROWS = 1 << 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
TARGETS_FILE = "paper_targets.json"
TARGET_SPLIT = "table3"
BACKEND_KEY = "assignment_backend"
CUDA = "cuda"
DERIVED = "derived layout"


class Scale(StrEnum):
    MILLION = "1M"
    TEN_MILLION = "10M"


class Rows(StrEnum):
    BF16 = "BF16"
    FP32 = "FP32"


class Backend(StrEnum):
    AUTO = "auto"
    CUTE = "cute"
    TRITON = "triton"


@dataclass(frozen=True)
class ScaleSpec:
    datasets: tuple[str, ...]
    k: int
    law: BatchLaw


SCALES = {
    Scale.MILLION: ScaleSpec(("deep1b-1M", "fb-ssnpp-1M", "spacev-1M"), 16_384, BatchLaw.THROUGHPUT),
    Scale.TEN_MILLION: ScaleSpec(("deep1b-10M",), 65_536, BatchLaw.QUALITY),
}
FAISS_LAW = {BatchLaw.THROUGHPUT: "throughput", BatchLaw.QUALITY: "quality"}


@dataclass(frozen=True)
class Faiss:
    root: Path
    data: ModuleType
    xb: ModuleType
    emeans: ModuleType
    quantizer: ModuleType


@dataclass(frozen=True)
class Inputs:
    bf16: torch.Tensor
    fp32: torch.Tensor


@dataclass(frozen=True)
class Context:
    faiss: Faiss
    inputs: Inputs
    spec: ScaleSpec
    backend: Backend


@dataclass(frozen=True)
class FaissArm:
    rows: Rows

    @property
    def name(self) -> str:
        return f"e-means, {self.rows.value} rows"

    def call(self, context: Context, recipe: Recipe, start: torch.Tensor) -> Callable[[], torch.Tensor]:
        d = context.inputs.bf16.shape[1]
        options: dict[str, object] = {
            "seed": recipe.seed,
            "batch_law": FAISS_LAW[context.spec.law],
            BACKEND_KEY: context.backend.value,
        }
        if self.rows is Rows.FP32:
            model, x = context.faiss.emeans.QuantizedEmeans, context.inputs.fp32
            options["quantizer"] = context.faiss.quantizer.CastQuantizer(d, torch.bfloat16)
        else:
            model, x = context.faiss.emeans.Emeans, context.inputs.bf16

        def run() -> torch.Tensor:
            trained = model(d, context.spec.k, nepochs=recipe.epochs, **options)
            trained.train(x, init_centroids=start)
            return torch.as_tensor(trained.centroids)

        return run


@dataclass(frozen=True)
class EzArm:
    probes: int | None

    @property
    def name(self) -> str:
        return f"ez-means, {DERIVED}" if self.probes is None else f"ez-means, p={self.probes}"

    def call(self, context: Context, recipe: Recipe, start: torch.Tensor) -> Callable[[], torch.Tensor]:
        k, n = context.spec.k, len(context.inputs.bf16)
        layout = work_layout(n, k, recipe) if self.probes is None else Layout(group_count(k), self.probes, Regroup.COLD)
        return lambda: ez_train(context.inputs.bf16, k, recipe, layout, init=start)


Arm = FaissArm | EzArm
ARMS: tuple[Arm, ...] = (FaissArm(Rows.BF16), FaissArm(Rows.FP32), *(EzArm(p) for p in PROBES), EzArm(None))


@dataclass(frozen=True)
class Record:
    dataset: str
    arm: str
    k: int
    seeds: list[int]
    crossing: list[float | None]
    error_ratio: list[float]
    seconds: list[float]


def load_faiss(path: Path) -> Faiss:
    sys.path.insert(0, str(path))
    return Faiss(
        path,
        importlib.import_module("paper_data"),
        importlib.import_module("paper_xb"),
        importlib.import_module("faiss.contrib.e_means"),
        importlib.import_module("faiss.contrib.e_means.quantizer"),
    )


def upload(faiss: Faiss, raw: np.ndarray, name: str, dtype: torch.dtype) -> torch.Tensor:
    x = torch.empty((len(raw), raw.shape[1]), dtype=dtype, device=CUDA)
    for begin in range(0, len(raw), UPLOAD_ROWS):
        chunk = np.asarray(faiss.data.rows(raw, name, begin, begin + UPLOAD_ROWS), dtype=np.float32)
        x[begin : begin + UPLOAD_ROWS] = torch.from_numpy(chunk).cuda().to(dtype)
    return x


def start_rows(faiss: Faiss, raw: np.ndarray, name: str, k: int, seed: int) -> torch.Tensor:
    picks = np.asarray(faiss.xb.starts(len(raw), k, seed))
    rows = np.ascontiguousarray(faiss.data.rows(raw, name, index=picks), dtype=np.float32)
    return torch.from_numpy(rows).cuda()


def seed_targets(faiss: Faiss, name: str) -> dict[int, float]:
    split = json.loads((faiss.root / TARGETS_FILE).read_text())[TARGET_SPLIT]
    return {int(seed): float(entry["test_mse"]) for seed, entry in split[name]["per_seed"].items()}


def crossing(
    context: Context, arm: Arm, seed: int, start: torch.Tensor, test: np.ndarray, target: float
) -> tuple[float | None, float, float]:
    n = len(context.inputs.bf16)
    error, seconds = math.inf, 0.0
    for budget in BUDGETS:
        call = arm.call(context, law_recipe(n, context.spec.k, budget, seed, context.spec.law), start)
        centroids, median, _ = context.faiss.xb.timed(call, warmup=call)
        error, seconds = float(context.faiss.xb.test_error(centroids, test)), float(median)
        if error <= target:
            return budget, error / target, seconds
    return None, error / target, seconds


def finished(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["dataset"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())}


def race(faiss: Faiss, scale: Scale, seeds: tuple[int, ...], backend: Backend, path: Path) -> None:
    spec, done = SCALES[scale], finished(path)
    for dataset in spec.datasets:
        raw, test = faiss.data.load(dataset)
        context = Context(
            faiss,
            Inputs(upload(faiss, raw, dataset, torch.bfloat16), upload(faiss, raw, dataset, torch.float32)),
            spec,
            backend,
        )
        targets = seed_targets(faiss, dataset)
        starts = {seed: start_rows(faiss, raw, dataset, spec.k, seed) for seed in seeds}
        for arm in ARMS:
            if (dataset, arm.name) in done:
                continue
            runs = [crossing(context, arm, seed, starts[seed], test, targets[seed]) for seed in seeds]
            record = Record(
                dataset, arm.name, spec.k, list(seeds), [r[0] for r in runs], [r[1] for r in runs], [r[2] for r in runs]
            )
            with path.open("a") as out:
                out.write(json.dumps(asdict(record)) + "\n")
            print(f"{dataset}: {arm.name} done", flush=True)
        del context, starts
        faiss.xb.free()


def bootstrap(ratios: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(BOOTSTRAP_RESAMPLES))
    return medians[int(LOWER_QUANTILE * BOOTSTRAP_RESAMPLES)], medians[int(UPPER_QUANTILE * BOOTSTRAP_RESAMPLES) - 1]


def sign_test(ratios: list[float]) -> float:
    wins, losses = sum(r > 1 for r in ratios), sum(r < 1 for r in ratios)
    total = wins + losses
    if total == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(total, i) for i in range(min(wins, losses) + 1)) / (1 << total))


def crossed_seconds(record: Record) -> list[float]:
    return [s for s, c in zip(record.seconds, record.crossing, strict=True) if c is not None]


def paired(base: Record, arm: Record) -> list[float]:
    return [
        b / s
        for b, s, cb, cs in zip(base.seconds, arm.seconds, base.crossing, arm.crossing, strict=True)
        if cb is not None and cs is not None
    ]


def summarize(path: Path) -> None:
    records = [Record(**raw) for raw in map(json.loads, path.read_text().splitlines())]
    for dataset in dict.fromkeys(r.dataset for r in records):
        arms = [r for r in records if r.dataset == dataset]
        faiss = [r for r in arms if r.arm.startswith("e-means")]
        base = min(faiss, key=lambda r: statistics.median(crossed_seconds(r)))
        print(
            f"\n{dataset}, baseline {base.arm}\n| Arm | Epochs | Time (s) | Speed-up | 95% CI | Sign test p |\n|---|---|---|---|---|---|"
        )
        for r in arms:
            hits = [c for c in r.crossing if c is not None]
            misses = len(r.crossing) - len(hits)
            row = f"| {r.arm} | {statistics.fmean(hits):.2f} | {statistics.median(crossed_seconds(r)):.4f} |"
            if r is base:
                print(row + " | | |")
                continue
            ratios = paired(base, r)
            low, high = bootstrap(ratios)
            note = f" (misses {misses} of {len(r.crossing)})" if misses else ""
            print(row + f" {statistics.median(ratios):.2f}{note} | [{low:.2f}, {high:.2f}] | {sign_test(ratios):.3f} |")
        by_name = {r.arm: r for r in arms}
        default = by_name.get(EzArm(default_layout(base.k).probes).name)
        derived = by_name.get(EzArm(None).name)
        if default is not None and derived is not None:
            ratios = paired(default, derived)
            low, high = bootstrap(ratios)
            print(
                f"\n{derived.arm} over {default.arm} (default_layout): {statistics.median(ratios):.2f} "
                f"[{low:.2f}, {high:.2f}], sign test p {sign_test(ratios):.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Race ez-means against e-means to the e-means paper's per-seed targets."
    )
    parser.add_argument(
        "--faiss-bench", type=Path, required=True, help="faiss benchs/e_means, with its datasets prepared"
    )
    parser.add_argument("--scale", type=Scale, choices=tuple(Scale), default=Scale.MILLION)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--backend", type=Backend, choices=tuple(Backend), default=Backend.AUTO)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true")
    options = parser.parse_args()
    if not options.summary_only:
        race(load_faiss(options.faiss_bench), options.scale, tuple(options.seeds), options.backend, options.results)
    summarize(options.results)


if __name__ == "__main__":
    main()
