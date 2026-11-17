# Paper race

`bench/race.py` reproduces the GPU tables of the ez-means paper. For each dataset and seed it trains ez-means and e-means (the faiss implementation) at budgets of 1.0 to 3.0 epochs in steps of 0.2, and records the first budget at which the test MSE reaches the e-means paper's per-seed target. It then reports the paired speed-up over e-means: the median over seeds, a 95% bootstrap interval and a two-sided sign test.

It follows the e-means benchmark's own protocol (`benchs/e_means/paper_xb.py` in faiss):

| Step | How |
|---|---|
| Start | both methods start from the same rows, the first k of a seeded CUDA permutation (`paper_xb.starts`) |
| Timing | an untimed warm-up with the identical call, then the call timed between two synchronisations; three times and the median when it takes under 2 s (`paper_xb.timed`) |
| Score | one exact scorer for both, with TF32 off (`paper_xb.test_error`) |
| e-means input | BF16 rows, and FP32 rows cast batch by batch with `CastQuantizer`; the faster of the two is the baseline |
| Batch law | throughput at 1M rows, quality at 10M, as in the e-means paper |

ez-means runs `p = 4, 8, 16` with √k groups and cold regroups, and `work_layout`. Under each dataset, the summary also prints `work_layout` over `default_layout`, paired per seed, which is the GPU comparison of layouts in the paper.

## Setup

The script needs the faiss e-means benchmark and its data. The benchmark is part of the faiss repository, not of this package.

```bash
uv sync --extra gpu
git clone https://github.com/facebookresearch/faiss
export EMEANS_DATA=/path/to/data
export EMEANS_CACHE=/path/to/cache
cd faiss/benchs/e_means && python paper_download.py --results 1M && cd -
```

`--results 10M` also fetches DEEP 10M. `paper_download.py --dry-run` prints the download size first.

## Run

```bash
uv run python bench/race.py --faiss-bench faiss/benchs/e_means --scale 1M --results race-1M.jsonl
uv run python bench/race.py --faiss-bench faiss/benchs/e_means --scale 10M --results race-10M.jsonl
```

| Option | Meaning |
|---|---|
| `--scale` | `1M` (DEEP, SSNPP and SPACEV at k = 16,384) or `10M` (DEEP at k = 65,536) |
| `--seeds` | seeds to run, 10 to 19 by default, as in the paper |
| `--backend` | e-means assignment kernels: `auto` (faiss' choice), `cute` (what faiss picks on Hopper GPUs, as in the paper's H100 runs) or `triton` |
| `--results` | a JSONL file with one line per dataset and arm; a run that stops resumes from it |
| `--summary-only` | print the tables from an existing results file |

Every run is in-process, unlike the e-means benchmark, which starts a fresh process per timed run.
