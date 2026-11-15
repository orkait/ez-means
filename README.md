<div align="center">

# ez-means

**A faster e-means for k-means clustering on CPU and GPU**

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![C11](https://img.shields.io/badge/kernel-C11%20%2B%20OpenMP-555555?logo=c&logoColor=white)
![faiss](https://img.shields.io/badge/faiss--cpu-1.16.0-0467DF)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-pre--release%200.0.1-orange)

</div>

ez-means is a Python package for training k-means centroids on large vector sets, with a portable C kernel on CPU and Triton kernels on GPU. It keeps the training recipe of [e-means](https://arxiv.org/abs/2611.09501) (Meta FAIR): the batch law, the EMA update, mass revival and the last-epoch mean. It changes how points are assigned:

| Change | What it does |
|---|---|
| Grouped probing | the centroids are regrouped into √k groups, and each point searches only its `probes` nearest groups |
| Exact early abandon | a centroid is dropped as soon as its running squared distance passes the best so far, so the probed groups are searched exactly |
| Deterministic CPU loop | training runs in C11 with OpenMP, and the centroids are bit-identical at any thread count |

It is for anyone who trains many centroids on millions of vectors, for example IVF index training or codebooks.

## 🚀 Quickstart

```bash
uv sync
```

```python
import numpy as np
from ezmeans.recipe import law_recipe
from ezmeans.trainer import ez_means

x = np.random.default_rng(0).standard_normal((100_000, 128)).astype(np.float32)
centroids, work = ez_means(x, 1024, law_recipe(len(x), 1024, 2, 0), 4)
```

| Call | Returns |
|---|---|
| `law_recipe(n, k, epochs, seed)` | the batch size and step size from the e-means scaling law |
| `ez_means(x, k, recipe, probes, init=None)` | the `k × d` centroids and the distance work in full-distance units; `init` is an optional `k × d` start |

The first call compiles `src/ezmeans/kernel.c` with `gcc` into `~/.cache/ezmeans/`. Fewer probes is faster and can miss a point's nearest centroid; probing every group is exact.

## ⚡ GPU

```bash
uv sync --group gpu
```

```python
import torch
from ezmeans.gpu.backend import train
from ezmeans.recipe import law_recipe

x = torch.randn(1_000_000, 96, device="cuda").to(torch.bfloat16)
centroids = train(x, 16384, law_recipe(len(x), 16384, 2, 0), 4)
```

`ezmeans.gpu` runs faiss' GPU e-means (`TorchEmeans`) with its assignment swapped for a grouped search in Triton kernels that never synchronize the host. `x` is a CUDA `bfloat16` tensor; distances accumulate in `float32`.

## 📦 Installation

| Requirement | Why |
|---|---|
| Python 3.13 and [uv](https://docs.astral.sh/uv/) | `pyproject.toml` pins the dependencies |
| `gcc` with OpenMP | the kernel is compiled at first use (`KERNEL_COMPILE` in `src/ezmeans/config.py`) |
| CUDA and the `gpu` dependency group | only for `ezmeans.gpu` |
| Linux x86-64 | the only platform tested so far |

| Variable | Default | Used for |
|---|---|---|
| `EZMEANS_CACHE` | `~/.cache/ezmeans` | the compiled kernel |

## 🧪 Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
```

`tests/test_gpu.py` runs only where `torch` and `triton` are installed and a CUDA device is present.

## ⚠️ Limitations

| Limitation | Where |
|---|---|
| The GPU path assigns by Euclidean distance only; spherical training raises `GpuInputError` | `src/ezmeans/gpu/backend.py` |
| The GPU path sets a private attribute of faiss' `TorchEmeans` (faiss 1.16) to swap the assignment backend | `src/ezmeans/gpu/backend.py` |
| GPU results are not bit-identical run to run: the kernels accumulate with float atomics | `src/ezmeans/gpu/kernels.py` |
| Tested only on Linux x86-64 with gcc; ARM, macOS and Windows are untested | none yet |

## 🙏 Credits

ez-means builds on [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., Meta FAIR) and on [faiss](https://github.com/facebookresearch/faiss) (MIT): it uses the e-means scaling law from `faiss.contrib.e_means`, faiss' GPU e-means training loop, and faiss for search in the tests.

## 📄 License

MIT, Copyright (c) 2026 Orkait. See [`LICENSE`](LICENSE).
