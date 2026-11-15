<div align="center">

# ez-means

**A faster e-means for k-means clustering on CPU**

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![C11](https://img.shields.io/badge/kernel-C11%20%2B%20OpenMP-555555?logo=c&logoColor=white)
![faiss](https://img.shields.io/badge/faiss--cpu-1.16.0-0467DF)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-pre--release%200.0.1-orange)

</div>

ez-means is a Python package with a portable C kernel for training k-means centroids on large vector sets. It keeps the training recipe of [e-means](https://arxiv.org/abs/2611.09501) (Meta FAIR): the batch law, the EMA update, mass revival and the last-epoch mean. It changes how points are assigned and how the loop runs:

| Change | What it does |
|---|---|
| Grouped probing | the centroids are regrouped into √k groups, and each point searches only its 4 nearest groups |
| Exact early abandon | a centroid is dropped as soon as its running squared distance passes the best so far, so the probed groups are searched exactly |
| Portable C loop | training runs in C11 with OpenMP, and gives the same bits on every compiler and SIMD level we tested |

It is for anyone who trains many centroids on millions of vectors, for example IVF index training or codebooks, and wants that on a CPU.

## 🚀 Quickstart

```bash
uv sync
```

```python
import numpy as np
from ezmeans.reference import law_recipe
from ezmeans.trainer import ez_means

x = np.random.default_rng(0).standard_normal((100_000, 128)).astype(np.float32)
centroids, work = ez_means(x, 1024, law_recipe(len(x), 1024, 2, 0), 4)
```

`ez_means(x, k, recipe, probes)` returns the `k × d` centroids and the distance work in full-distance units. `law_recipe(n, k, epochs, seed)` picks the batch size and step size from the e-means scaling law. The first call compiles `src/ezmeans/kernel.c` with `gcc` into `~/.cache/ezmeans/`.

## 📊 Results

Details, every table and the dead ends are in [`results/RESULTS.md`](results/RESULTS.md).

The race measures time to reach Lloyd-25 test MSE within 0.1% (SIFT1M, k=4096, median of 2 seeds, 4 threads):

| Method | Zen 5 (Ryzen 9700X) | Zen 3 (EPYC, Modal) | Broadwell (Xeon 2.2 GHz, Kaggle) |
|---|---:|---:|---:|
| faiss Lloyd, 25 iterations | 109.0 s | 150.8 s | 281.1 s |
| faiss SuperKMeans, 25 iterations | 85.2 s | 184.0 s | 285.7 s |
| scikit-learn KMeans, 25 iterations | 43.6 s | 83.3 s | 302.6 s |
| e-means, the faiss implementation | 10.9 s | 15.2 s | 28.5 s |
| **ez-means** | **1.57 s** | **3.22 s** | **8.63 s** |

The paper's own 1M protocol uses its test rows and per-seed targets, with k=16384 and the paper's 10 seeds (10-19), on 16 Zen 3 cores:

| Dataset | e-means: mean crossing, median seconds | ez-means: mean crossing, median seconds | Paired speedup, median (min-max) |
|---|---|---|---:|
| DEEP 1M | 1.94 epochs, 18.9 s | 2.14 epochs, 1.4 s | 13.8x (10.5-20.6) |
| SSNPP 1M | 1.40 epochs, 29.8 s | 1.42 epochs, 3.3 s | 9.0x (7.4-10.5) |
| SPACEV 1M | 2.30 epochs, 30.1 s | 2.56 epochs, 2.8 s (8 probes) | 11.1x (8.9-15.8) |

On GPU, `experiments/gpu.py` swaps faiss' brute-force assignment for two Triton kernels that do the grouped search, and keeps the rest of faiss' GPU e-means:

| Setting | faiss GPU e-means | ez-means GPU | Speedup |
|---|---:|---:|---:|
| DEEP 1M, k=16384, RTX 3060, 10 seeds | 0.46 s (Triton) | 0.17 s | 2.8x |
| DEEP 10M, k=65536, RTX 3060, 3 seeds | 25.1 s (Triton) | 2.9 s | 8.6x |
| DEEP 10M, k=65536, H100, 3 seeds | 2.60 s (CuTe) | 0.47 s | 5.5x |
| 1M, k=16384, H100 | 0.04-0.06 s (CuTe) | 0.12-0.16 s | slower |

## 📦 Installation

| Requirement | Why |
|---|---|
| Python 3.13 and [uv](https://docs.astral.sh/uv/) | `pyproject.toml` pins the dependencies |
| `gcc` with OpenMP | the kernel is compiled at first use (`KERNEL_COMPILE` in `src/ezmeans/config.py`) |
| Linux x86-64 | the only platform tested so far |

| Variable | Default | Used for |
|---|---|---|
| `EZMEANS_DATA` | `~/datasets/emeans` | datasets read by the experiments |
| `EZMEANS_CACHE` | `~/.cache/ezmeans` | the compiled kernel |

## 🧪 Reproducing the experiments

Each script in `experiments/` prints the tables in `results/RESULTS.md`. They expect SIFT1M under `$EZMEANS_DATA/sift/sift/`. The paper-protocol scripts also need the e-means benchmark folder from faiss (`benchs/e_means`) under `$EZMEANS_DATA/faiss-bench/` and its data, fetched with that folder's `paper_download.py`.

```bash
uv run python experiments/race.py --round large
uv run python experiments/paper_cpu.py --arms e-means ez-means
uv run pytest
```

<details>
<summary>What each experiment measures</summary>

| Script | Measures |
|---|---|
| `race.py` | time to Lloyd-25 quality against faiss, scikit-learn and e-means |
| `paper_cpu.py` | epoch crossings on the paper's 1M (and 10M) protocol |
| `gpu.py` | the same protocol on GPU, faiss GPU e-means against ez-means' Triton kernels (`uv sync --group gpu`, CUDA) |
| `fused_time.py` | exact, grouped and ez-means wall time and distance work |
| `approximate.py` | whether approximate assignment hurts Lloyd or e-means after training |
| `adsampling.py` | ADSampling pruning with a fast Hadamard rotation |
| `projection.py` | random rotation, ADSampling and JL projection, tested separately |
| `routing.py` | 2-level against 3-level centroid routing |
| `kernel_probe.py` | threads, wait policy and FMA contraction |
| `validate.py` | the reference e-means is bit-identical to faiss `NumpyEmeans` |
| `swap.py`, `generality.py`, `tuning.py`, `real.py`, `anatomy.py`, `baseline.py`, `compare.py` | revival variants and the failure anatomy on clustered data |
| `bounds.py`, `margins.py`, `prefix.py`, `stack.py`, `budget.py`, `ivf.py`, `hierarchy.py`, `grouped.py`, `lowbits.py` | speed ideas, including the ones that did not pay |

`cloud/modal_jobs.py` runs the race, a CPU probe, the paper protocol and the H100 GPU protocol on [Modal](https://modal.com): `uv run --group cloud modal run cloud/modal_jobs.py --job race` (or `--job paper --dataset deep1b-1M --arms e-means,ez-means`, or `--job gpu --dataset deep1b-10M --arms 0,16`).

</details>

## ⚠️ Limitations

| Limitation | Evidence |
|---|---|
| The race covers one dataset (SIFT1M) at one k | `results/RESULTS.md` sections 5, 7, 12 and 15 |
| A fixed 4 probes is too few for SPACEV at k=16384, and k=65536 needs 16 | sections 10, 16 and 17 |
| The lead over e-means at the race target depends on the CPU: 6.9x on Zen 5, 4.7x on Zen 3, 3.3x on Broadwell | sections 7, 12 and 15 |
| On an H100 at 1M, faiss' CuTe kernels are faster than ez-means' GPU path | section 17 |
| The GPU path lives in `experiments/gpu.py` and sets a private attribute of faiss' `TorchEmeans` to swap the assignment backend | section 17 |
| Tested only on Linux x86-64 with gcc. ARM, macOS and Windows are untested | none yet |

## 🙏 Credits

ez-means builds on [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., Meta FAIR) and on [faiss](https://github.com/facebookresearch/faiss) (MIT): it uses the e-means scaling law from `faiss.contrib.e_means`, and faiss for the baselines.

## 📄 License

MIT, Copyright (c) 2026 Orkait. See [`LICENSE`](LICENSE).
