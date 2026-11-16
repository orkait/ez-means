<div align="center">

# ez-means

**A faster e-means for k-means clustering on CPU and GPU**

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![C11](https://img.shields.io/badge/kernel-C11%20%2B%20OpenMP-555555?logo=c&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-%E2%89%A52.1-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-pre--release%200.0.1-orange)

</div>

ez-means is a Python package for training k-means centroids on large vector sets, with a portable C kernel on CPU and Triton kernels on GPU. It keeps the training recipe of [e-means](https://arxiv.org/abs/2611.09501) (Meta FAIR): the batch law, the EMA update, mass revival and the last-epoch mean. It changes how points are assigned:

| Change | What it does |
|---|---|
| Grouped probing | the centroids are regrouped into groups, and each point searches only its `probes` nearest groups |
| Exact early abandon | a centroid is dropped as soon as its running squared distance passes the best so far, so the probed groups are searched exactly |
| Deterministic CPU loop | training runs in C11 with OpenMP, and the centroids are bit-identical at any thread count |

It is for anyone who trains many centroids on millions of vectors, for example IVF index training or codebooks.

## Quickstart

```bash
uv sync
```

```python
import numpy as np
from ezmeans import law_recipe
from ezmeans.cpu import train

x = np.random.default_rng(0).standard_normal((100_000, 128)).astype(np.float32)
centroids = train(x, 1024, law_recipe(len(x), 1024, 2, 0))
```

| Call | Returns |
|---|---|
| `ezmeans.law_recipe(n, k, epochs, seed, law=BatchLaw.THROUGHPUT)` | the batch size and step size from the e-means scaling law; `BatchLaw.QUALITY` is the batch eight times smaller that the e-means paper uses above 2M rows |
| `ezmeans.cpu.train(x, k, recipe, layout=None, init=None)` | the `k × d` centroids; `init` is an optional `k × d` start |
| `ezmeans.cpu.cpu_layout(k, d)` | the `Layout(groups, probes, regroup)` used when `layout` is `None` (below) |
| `ezmeans.cpu.paper_layout(k)` | the layout of the ez-means paper's experiments: round(√k) groups, `default_probes(k)` probes, `Regroup.COLD`; pass it as `layout` to reproduce them |
| `ezmeans.default_probes(k)` | the probes the GPU path uses when `probes` is `None`: 1/16 of the √k groups, at least 4 |

The first call compiles `src/ezmeans/cpu/kernel.c` with `gcc` into `~/.cache/ezmeans/`. Fewer probes is faster and can miss a point's nearest centroid; probing every group is exact.

| CPU layout | Groups | Probes | Regroup | k=4096 · 16384 · 65536, d ≥ 64 |
|---|---|---|---|---|
| `k < 65536` or `d < 64` | round(√k) | 1/16 of the groups, at least 6, at most the groups | `Regroup.COLD`: 10 Lloyd iterations from random centroids | 64/6 · 128/8 · - |
| `k ≥ 65536` and `d ≥ 64` | the largest power of two up to 2√k | ⌈0.625 √groups⌉ | `Regroup.WARM`: 2 Lloyd iterations from the previous groups | - · - · 512/15 |

| Error | Raised when |
|---|---|
| `KernelError(NOT_FINITE)` | a row or centroid has no finite squared distance: a NaN, an infinity, or values large enough to overflow `float32` |
| `KernelError(BAD_SIZE)` | `k` is below 1 or above the rows, the layout's groups are below 1 or above `k`, `init` is not `k × d`, or the epochs cover no rows |
| `KernelError(BAD_PROBES)` | the layout's probes are below 1 or above its groups |
| `RecipeError` (a `ValueError`) | a `Recipe` with `epochs <= 0` (`EPOCHS`), `batch < 1` (`BATCH`) or `alpha0` outside (0, 1] (`STEP`), or `law_recipe` with `n < 1`, `k < 1` or `epochs <= 0` (`LAW_DOMAIN`) |

## GPU

```bash
uv sync --extra gpu
```

```python
import torch
from ezmeans import law_recipe
from ezmeans.gpu import train

x = torch.randn(1_000_000, 96, device="cuda").to(torch.bfloat16)
centroids = train(x, 16384, law_recipe(len(x), 16384, 2, 0))
```

`ezmeans.gpu` runs the same e-means loop as the CPU path (EMA update, mass revival, cosine step, last-epoch mean). The grouped search, the batch sums and the EMA update are Triton kernels that never synchronize the host; the loop synchronizes once per revival check. `x` is a CUDA `bfloat16` tensor; distances accumulate in `float32`. `train(x, k, recipe, probes=None, init=None)` takes an optional `k × d` CUDA `float32` start, as the CPU path does. Rows that are not finite, or whose norm passes about 9.2e18 (half the square root of the `float32` maximum, where squared distances could overflow), raise `GpuInputError(NOT_FINITE)`.

## Installation

| Requirement | Why |
|---|---|
| Python 3.13 and numpy 2.1 or later | the only runtime dependency of the CPU path |
| `gcc` with OpenMP | the kernel is compiled at first use (`KERNEL_COMPILE` in `src/ezmeans/config.py`) |
| CUDA and the `gpu` extra (torch 2.15+, triton 3.9+) | only for `ezmeans.gpu` |
| Linux x86-64 | the only platform tested so far |

| Variable | Default | Used for |
|---|---|---|
| `EZMEANS_CACHE` | `~/.cache/ezmeans` | the compiled kernel |

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
```

| Path | Holds |
|---|---|
| `src/ezmeans/recipe.py` | the recipe both backends train with: scaling laws, epoch plan, default probes, `RecipeError` |
| `src/ezmeans/config.py` | the one config: cache directory and gcc flags |
| `src/ezmeans/cpu/` | `kernel.c` (C11 + OpenMP), `build.py` (hash-keyed gcc build), `native.py` (ctypes binding and input checks), `layout.py` (groups, probes and regroup per k and d), `trainer.py` |
| `src/ezmeans/gpu/` | `kernels.py` (Triton), `index.py` (`GroupIndex`), `trainer.py` (the e-means loop), `constants.py` |
| `tests/` | `test_recipe.py`, `test_cpu.py`, `test_gpu.py` |

`tests/test_gpu.py` runs only where `torch` and `triton` are installed and a CUDA device is present. The tests that hold the vendored scaling law equal to faiss run only where faiss is installed, as it is in the `dev` group.

## Limitations

| Limitation | Where |
|---|---|
| Both paths assign by Euclidean distance only | `src/ezmeans/cpu/kernel.c`, `src/ezmeans/gpu/kernels.py` |
| GPU results are not bit-identical run to run: the kernels and the batch sums accumulate with float atomics | `src/ezmeans/gpu/kernels.py`, `src/ezmeans/gpu/trainer.py` |
| The GPU path holds at most 512 groups, so `k` up to 262,656; a larger `k` raises `GpuInputError(TOO_MANY_GROUPS)` | `MAX_GROUP_SPAN` in `src/ezmeans/gpu/constants.py` |
| Tested only on Linux x86-64 with gcc; ARM, macOS and Windows are untested | none yet |
| The CPU layout was measured only at k = 4096, 16384 and 65536; the GPU path keeps √k groups and its own probes | `src/ezmeans/cpu/layout.py` |

## Credits

ez-means builds on [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., Meta FAIR) and on [faiss](https://github.com/facebookresearch/faiss) (MIT). `src/ezmeans/recipe.py` carries the e-means batch and step laws from `faiss.contrib.e_means` 1.16 (notice in [`NOTICE`](NOTICE)).

## License

Apache-2.0, Copyright 2026 Orkait. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE), which also carries the MIT notice for the vendored faiss scaling laws.
