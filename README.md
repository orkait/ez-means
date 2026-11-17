<div align="center">

# ez-means

**k-means with the e-means recipe and a grouped nearest-centroid search, on CPU and GPU**

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![C11](https://img.shields.io/badge/CPU-C11%20%2B%20OpenMP-555555?logo=c&logoColor=white)
![Triton](https://img.shields.io/badge/GPU-Triton-76B900?logo=nvidia&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-%E2%89%A52.1-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

**[Playground](https://ez-means.pages.dev/playground)** · [Blog post](https://ez-means.pages.dev/) · [Standings](https://ez-means.pages.dev/standings) · [Results map](https://ez-means.pages.dev/results-map)

</div>

ez-means is a small Python package for training k-means centroids on large sets of vectors. It uses the training recipe of [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., 2026) as it is, and changes only one step. Instead of comparing each point with all k centroids, it first compares the point with about √k group centres, and then searches only the centroids of its nearest few groups. It runs as a portable C kernel on the CPU and as Triton kernels on the GPU.

It may be useful if you train many centroids on millions of vectors, for example for an IVF index or a codebook. The idea is simple, and it stands on the work of the e-means authors and of faiss. You can race both methods in your browser in the [playground](https://ez-means.pages.dev/playground).

## 🔍 How it works

| Step | What happens | Distances per point at k = 16,384 |
|---|---|---:|
| 1 | Every five batches, the current centroids are clustered into G = round(√k) groups, with a few Lloyd iterations on the centroids | shared by the batch |
| 2 | The point is measured against the G group centres | 128 |
| 3 | Its p nearest groups are kept (the probes) | 0 |
| 4 | Only the members of those groups are searched, exactly | about 8 × 128 = 1,024 |
| | **Total, against 16,384 for e-means** | **about 1,152, or 7%** |

On the CPU, each distance computation also stops as soon as its partial sum passes the best distance found so far, so the search inside the chosen groups stays exact. When every group is searched, the assignment is the same as the brute-force one. Everything else, namely the batch-size and step-size laws, the moving-average update, the revival of unused clusters and the mean over the last epoch, is the e-means recipe, unchanged.

## 📊 Results

We follow the 1M protocol of the e-means paper: k = 16,384, training budgets from 1.0 to 3.0 epochs in steps of 0.2, and the per-seed target test MSE that comes with the e-means benchmark, for seeds 10 to 19. Each cell is the median over seeds of e-means' time divided by ez-means' time on the same seed, with a 95% bootstrap interval. The baseline is the e-means of faiss 1.16: its NumPy code on the CPU, its Triton kernels on the RTX 3060 and its CuTe kernels on the H100, with the faster of BF16 and FP32 input.

| Machine | Code | Layout | DEEP | SSNPP | SPACEV |
|---|---|---|---|---|---|
| 16 Zen 3 cores | paper, CPU kernel before 0.1.0 | `default_layout`: 128 groups, 8 probes | 12.7x [10.8, 16.6] | 4.6x [4.4, 4.9] | 9.7x [9.2, 10.4] |
| RTX 3060 | paper, tag `paper-gpu` | `default_layout` | 5.1x [4.8, 5.6] | 4.3x [4.2, 4.5] | 4.2x [4.1, 4.6] |
| H100 | paper, tag `paper-gpu` | `default_layout` | 1.0x [0.9, 1.1] | 1.0x [1.0, 1.2] | 0.9x [0.8, 1.1] |
| H100 | tag `v0.1.0` | `default_layout` | 1.4x [1.2, 1.5] | 1.2x [0.9, 1.3] | 1.1x [1.0, 1.2] |
| H100 | tag `v0.1.0` | `work_layout` | 1.3x [1.2, 1.5] | 1.2x [1.2, 1.3] | 1.1x [1.1, 1.2] |

DEEP with 10M rows and k = 65,536 on the H100, with the quality batch law that the e-means paper uses at this size:

| Layout | paper, tag `paper-gpu` | tag `v0.1.0` |
|---|---|---|
| `default_layout`: 256 groups, 16 probes | 1.1x [1.0, 1.1] | 1.3x [1.2, 1.4] |
| 256 groups, 8 probes | 1.6x [1.2, 1.6] | 1.8x [1.7, 2.0] |
| 256 groups, 4 probes | 1.8x [1.6, 2.1] | 2.6x [2.2, 3.1] |
| `work_layout` | not measured | 2.0x [1.8, 2.2] |

Every row in both tables reaches the target on all 10 seeds. The tag `paper-gpu` marks the code behind the paper's GPU tables, and `v0.1.0` this release. Each H100 column is a separate run, and the speed-ups are paired within that run. The times include the extra training that ez-means sometimes needs, at most 0.4 epochs on average at the default layout. The ez-means paper (arXiv link to follow) has the full tables, a comparison with Lloyd, SuperKMeans and scikit-learn on SIFT1M, and an analysis of why the approximate search costs so little quality.

| Setting | What we measured |
|---|---|
| k = 16,384, CPU or consumer GPU | the largest gains, in the first table |
| H100 at 1M points | 0.9x to 1.4x, close to e-means with the CuTe kernels that faiss selects on Hopper GPUs |
| H100 at 10M points, k = 65,536 | 1.1x to 1.3x at the default layout, up to 2.6x with 4 probes |
| k = 256 on a GPU | e-means is faster here: the grouping costs more than it saves |

## 🚀 Quickstart

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

The first call compiles `src/ezmeans/cpu/kernel.c` with `gcc` into `~/.cache/ezmeans/`. On the GPU, install the `gpu` extra and pass a CUDA `bfloat16` tensor:

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

## 🧩 API

| Call | Returns |
|---|---|
| `ezmeans.law_recipe(n, k, epochs, seed, law=BatchLaw.THROUGHPUT)` | the batch size and step size from the e-means scaling laws; `BatchLaw.QUALITY` gives the batch eight times smaller that the e-means paper uses above 2M rows |
| `ezmeans.cpu.train(x, k, recipe, layout=None, init=None)` | the `k × d` centroids, from a `float32` numpy array; `init` is an optional `k × d` start |
| `ezmeans.gpu.train(x, k, recipe, layout=None, init=None)` | the same on a CUDA `bfloat16` tensor, with distances accumulated in `float32`; `init` is an optional `k × d` CUDA `float32` start |
| `ezmeans.default_layout(k)` | the layout both paths use when `layout` is `None` |
| `ezmeans.work_layout(n, k, recipe)` | a layout that computes fewer distances, for when reaching a target quality sooner matters more than the last fraction of a percent at a fixed number of epochs |
| `ezmeans.paper_layout(k)` | the layout of the ez-means paper's experiments, to reproduce them |
| `ezmeans.default_probes(k)` | the probes of `paper_layout`: 1/16 of the √k groups, at least 4 |

## 🗂️ Layouts

A layout is the number of groups, the number of probes and how the groups are rebuilt. More probes are slower and closer to exact; searching every group is exact.

| Layout | Groups | Probes | Regroup | k = 4,096 · 16,384 · 65,536, 1M rows, 2 epochs |
|---|---|---|---|---|
| `default_layout(k)` | round(√k) | 1/16 of the groups, at least 6, at most the groups | `Regroup.COLD`: 10 Lloyd iterations from random centroids | 64/6 · 128/8 · 256/16 |
| `work_layout(n, k, recipe)` | the power of two, at most 512 and at most k, that minimises the distances of the run: rows × (groups + probes × k / groups), plus regroups × 3 × k × groups | ⌈0.625 √groups⌉, at least 6, at most the groups | `Regroup.WARM`: 2 Lloyd iterations, from random centroids at the first regroup and from the previous groups after that | 128/8 · 256/10 · 512/15 |
| `paper_layout(k)` | round(√k) | 1/16 of the groups, at least 4 | `Regroup.COLD` | 64/4 · 128/8 · 256/16 |

Every layout depends only on k, the rows and the recipe, so the same call gives the same layout on every CPU and GPU. `work_layout` does less distance work than the default, but at a fixed number of epochs it can end slightly higher in test MSE (by 0.05% to 0.11% on SSNPP and SPACEV in the paper's GPU measurements), so it is opt-in.

## ⚙️ Behaviour

| Path | Detail |
|---|---|
| CPU | The training loop is C11 with OpenMP, without intrinsics and with floating-point contraction disabled, so the centroids are the same bits at any thread count and on every compiler and `-march` target tested |
| GPU | The search, the batch sums and the EMA update are Triton kernels, and the revival of unused clusters runs on the device without a host round trip. Every full batch replays one of three CUDA graphs (a plain step, a step that regroups, a step that revives), each captured the second time its step runs, so the number of host waits does not grow with the training length |

<details>
<summary>Main errors</summary>

| Error | Raised when |
|---|---|
| `KernelError(NOT_FINITE)` | a row or centroid has no finite squared distance: a NaN, an infinity, or values large enough to overflow `float32` |
| `KernelError(BAD_SIZE)` | `k` is below 1 or above the rows, the layout's groups are below 1 or above `k`, `init` is not `k × d`, or the epochs cover no rows |
| `KernelError(BAD_PROBES)` | the layout's probes are below 1 or above its groups |
| `GpuInputError(NOT_FINITE)` | a GPU row is not finite, or its norm passes about 9.2e18, half the square root of the `float32` maximum, where squared distances could overflow |
| `GpuInputError(TOO_MANY_GROUPS)` | a GPU layout needs more than 512 groups |
| `RecipeError` (a `ValueError`) | a `Recipe` with `epochs <= 0` (`EPOCHS`), `batch < 1` (`BATCH`) or `alpha0` outside (0, 1] (`STEP`), or `law_recipe` with `n < 1`, `k < 1` or `epochs <= 0` (`LAW_DOMAIN`) |

</details>

## 📦 Installation

| Requirement | Why |
|---|---|
| Python 3.13 and numpy 2.1 or later | the only runtime dependencies of the CPU path |
| `gcc`, or `clang` with libomp, with OpenMP | the kernel is compiled at first use (`KERNEL_COMPILE` in `src/ezmeans/config.py`) |
| CUDA and the `gpu` extra (torch 2.15+, triton 3.9+) | only for `ezmeans.gpu` |
| Linux on x86-64 or aarch64 | the platforms tested so far |

| Variable | Default | Used for |
|---|---|---|
| `EZMEANS_CACHE` | `~/.cache/ezmeans` | the compiled kernel |
| `EZMEANS_CC` | `gcc` | the C compiler that builds the kernel |
| `EZMEANS_MARCH` | `native` | the `-march` target, for example `x86-64-v3` to build one kernel for a mixed cluster |

## 🔁 Reproducing the GPU results

`bench/race.py` runs the e-means paper's protocol against the e-means of faiss and prints the tables above. It needs the faiss e-means benchmark and its data; [`bench/README.md`](bench/README.md) has the setup and the commands.

## 🛠️ Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests bench
```

| Path | Holds |
|---|---|
| `src/ezmeans/recipe.py` | the recipe both paths train with: scaling laws, epoch plan, default probes, `RecipeError` |
| `src/ezmeans/layout.py` | the layout both paths search with: `Layout`, `Regroup`, `default_layout`, `work_layout`, `paper_layout` |
| `src/ezmeans/config.py` | the one config: cache directory and compiler flags |
| `src/ezmeans/cpu/` | `kernel.c` (C11 and OpenMP), `build.py` (hash-keyed build), `native.py` (ctypes binding and input checks), `trainer.py` |
| `src/ezmeans/gpu/` | `kernels.py` (Triton), `index.py` (`GroupIndex`), `trainer.py` (the e-means loop), `constants.py` |
| `tests/` | `test_recipe.py`, `test_layout.py`, `test_cpu.py`, `test_gpu.py` |
| `bench/` | `race.py`, which reproduces the GPU results against e-means |

`tests/test_gpu.py` runs only where `torch` and `triton` are installed and a CUDA device is present. The tests that hold the vendored scaling laws equal to faiss run only where faiss is installed, as it is in the `dev` group.

## 📏 Limitations

| Limitation | Where |
|---|---|
| Both paths assign by Euclidean distance only | `src/ezmeans/cpu/kernel.c`, `src/ezmeans/gpu/kernels.py` |
| GPU runs are not bit-identical from run to run, because the kernels and the batch sums accumulate with floating-point atomics; the CPU kernel is | `src/ezmeans/gpu/kernels.py`, `src/ezmeans/gpu/trainer.py` |
| The GPU kernels hold at most 512 groups, so `default_layout` fits k up to 262,656 on the GPU, and `work_layout` caps its groups at 512 | `MAX_GROUPS` in `src/ezmeans/layout.py` |
| The same bits are tested on Linux only: x86-64 with gcc 14 and 15 and clang 19 at the `x86-64`, `x86-64-v3` and AVX-512 `native` targets, and aarch64 (NVIDIA Grace) with gcc 13 | `test_centroids_do_not_depend_on_compiler_or_target` in `tests/test_cpu.py` |
| `work_layout` minimises a count of distances, not time: it does not model how much a member distance costs against a group-centre distance, or how unevenly the groups fill | `src/ezmeans/layout.py` |

## 🙏 Credits

ez-means builds on [e-means](https://arxiv.org/abs/2611.09501) by Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard, Matthijs Douze and Hervé Jégou, and on [faiss](https://github.com/facebookresearch/faiss) (MIT). `src/ezmeans/recipe.py` carries the e-means batch and step laws from `faiss.contrib.e_means` 1.16, with their notice in [`NOTICE`](NOTICE). If you use ez-means, please also cite e-means:

```bibtex
@misc{bessa2026emeansoutperformingkmeansin,
  title={e-means: outperforming k-means (in one-tenth the epochs)},
  author={Swann Bessa and Pierre-Emmanuel Mazaré and Gergely Szilvasy and Hugo Touvron and David Picard and Matthijs Douze and Hervé Jégou},
  year={2026},
  eprint={2611.09501},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.09501},
}
```

## 📄 Licence

Apache-2.0, Copyright 2026 Orkait. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE), which also carries the MIT notice for the vendored faiss scaling laws.
