# ez-means: e-means made faster

ez-means is our faster version of the e-means clustering algorithm (Bessa et al., arXiv 2611.09501). It keeps e-means' training recipe: batch law, EMA step, mass revival and last-epoch mean. It changes how points are assigned and how the loop runs:

| Change | What it does |
|---|---|
| Grouped probing | centroids are regrouped into √k groups, and each point searches only its 4 nearest groups |
| Early abandon | a partial distance search drops a centroid as soon as its running sum passes the best so far |
| Accumulation | a counting sort replaces `np.add.at` |
| The loop | the whole training loop is portable C11, with no hardware-specific code |

On SIFT1M, at k=4096 on 4 cores, ez-means reaches Lloyd-25 quality 128x faster than faiss Lloyd, 50x faster than sklearn KMeans and 11.7x faster than the paper's e-means (section 5, AVX-512 build). The portable build gives the same bits on every compiler and SIMD level, and costs 16-22% against intrinsics (section 6). These numbers come from one machine, one dataset and one k.

A second axis is reliability. A verified swap revival takes cluster-structured data to oracle quality in 2-5 epochs, but on SIFT it does not pay (section 2).

Every number below comes from the scripts in this folder. The harness `ours.py` is bit-identical to faiss `NumpyEmeans` in both metrics (`validate.py`: 5 of 5 seeds spherical, 5 of 5 Euclidean).

## 1. e-means against k-means and k-means++ (`compare.py`)

| Method | Kepler-like: test cosine | Kepler-like: classes recovered, min / median of 40 | Kepler-like: ms | SIFT k=1024: test MSE / k-means | SIFT k=1024: s |
|---|---:|---:|---:|---:|---:|
| k-means (Lloyd, random init, 25 it) | 0.91732 | 20 / 26 | 0.8 | 1.0000 | 2.75 |
| k-means++ (Lloyd, 25 it) | 0.97319 | 34 / 36 | 3.2 | 0.9997 | 14.92 (numpy ++ init) |
| e-means T=2 | 0.94425 | 25 / 30 | 3.4 | 0.9997 | 0.38 |
| e-means T=5 | 0.94976 | 26 / 31.5 | 5.5 | 0.9965 | 1.03 |
| e-means T=25 | 0.97176 | 35 / 36 | 17.3 | 0.9954 | 4.76 |
| e-means + verified swap T=2 | **0.99025** | 38 / 40 | 5.2 | 1.0004 | 0.52 |
| e-means + verified swap T=5 | **0.99028** | **40 / 40** | 8.1 | 0.9966 | 1.51 |

The kepler-like data is 40 unit directions in 32 dimensions, 2,176 points, noise 0.025, 10 seeds. SIFT is trained on `sift_learn` (100k) and tested on 100k of `sift_base`, with 3 seeds. Times come from the Python harness and are indicative only.

## 2. Reliability: the verified swap (`generality.py`, `tuning.py`, `real.py`)

Every failing seed of e-means on clustered data has the same shape: m centroids each straddle two classes while m classes hold two centroids each. Mass-based revival never fires on this, because neither centroid is dead.

The verified swap fixes it in four steps:
1. Keep streaming per-cluster statistics from the assignment: distortion, and removal cost from the top-2 margin.
2. Propose the farthest point of a cluster as a new centroid.
3. Measure that probe's gain on fresh points over the next window.
4. Swap when gain > κ × the cheapest removal cost (κ=1).

Synthetic results, test cosine, 10 seeds:

| Case | e-means (mass revival) | + verified swap | Swaps |
|---|---:|---:|---:|
| A clustered (k=40) | 0.95357 | **0.99028** (40/40) | 7 |
| B overlapping clusters | 0.74570 | **0.75928** (40/40) | 4 |
| C no cluster structure | 0.38273 | 0.38273 | 0 |
| D skewed cluster sizes | 0.94949 | **0.99027** | 20.5 |
| E large (n=65k, k=256) | 0.94853 | **0.99040** (256/256) | 47.5 |

On SIFT, with 5 seeds, the swap does not pay:

| k | T | e-means test / Lloyd-25 | + verified swap, κ=1 | + verified swap, κ=4 |
|---:|---:|---:|---:|---:|
| 1024 | 5 | 0.9964 | 0.9965 | 0.9968 |
| 4096 | 5 | 0.9936 | 0.9954 | 0.9945 |

The 0.04-0.2% loss at k=4096 is 2-3 stdev. On continuous data the acceptance test needs stronger evidence.

Two earlier variants failed and are kept for the record:
- the unverified gain (selection bias when a cluster has 1-2 members per batch);
- multi-swap on raw distortion (thrashing).

## 3. Speed levers on SIFT1M (`budget.py`, `ivf.py`, `hierarchy.py`, `grouped.py`)

Each lever is measured against exact e-means at the same epochs, on 1M `sift_base` points, tested on `sift_learn`.

| Lever | k | Test MSE / exact | Distance computations / exact | Wall time / exact |
|---|---:|---:|---:|---:|
| Train on 256 points per cluster | 1024 | 1.0079 | 0.26 | 0.26 |
| Train on 64 points per cluster | 1024 | 1.0290 | 0.07 | 0.07 |
| IVF over centroids, rebuilt each batch, p=4 | 1024 | 1.0002 | 0.167 | 0.80 |
| Grouped (block-sparse GEMM), p=4 | 1024 | 1.0004 | 0.176 | 0.84 |
| Coarse √k + per-cell fine | 1024 | 1.0128 | 0.048 | 0.51 |
| Coarse + fine + IVF refine | 1024 | 1.0012 | 0.133 | 0.90 |
| IVF, p=4 | 4096 | 1.0001 | 0.083 | 0.38 |
| **Grouped, p=4** | 4096 | **1.0001** | 0.088 | **0.34** |
| Grouped, p=2 | 4096 | 1.0021 | 0.055 | 0.26 |

What the table says:

| Finding | Evidence |
|---|---|
| Training on a subsample is not free | 0.8% worse at 256 points per cluster, which is faiss Lloyd's default cap |
| Approximate assignment costs almost nothing in quality at p=4 | ≤ 0.04% |
| The FLOP cut grows with k | 6x at k=1024, 11x at k=4096 |
| Wall time follows only when the work stays in large GEMMs | grouped reaches 2.9x at k=4096 and 1.2x at k=1024, where Python overhead and small blocks dominate |

## 4. Fused C kernel (`kernel.c`, `fused_time.py`, `lowbits.py`)

`kernel.c` replaces two pieces of the Python loop:

| Piece | What it does |
|---|---|
| `assign` | probes the √k groups, then scans the probed groups' centroids with partial distance search. The data is PCA-rotated first. Centroids are laid out 16 per AVX-512 register, and a tile is dropped once its running sum passes the best distance so far. The result is exact over the probed groups |
| `accumulate` | counting-sorts the batch by label, then sums each cluster's rows in batch order, so the sums are bit-identical to `np.add.at` |

SIFT1M train, SIFT learn 100k test, k=4096, T=2, seed 0, 4 cores:

| Assignment | Test MSE / exact | Distance work / exact | Seconds | Speedup |
|---|---|---|---|---|
| exact (C accumulate) | 1.0000 | 1.0000 | 8.99 | 1.0x |
| grouped p4, faiss per block | 1.0004 | 0.0885 | 2.66 | 3.4x |
| fused p4 | 1.0004 | 0.0532 | 1.40 | 6.4x |
| ez-means p4 (`train` in `kernel.c`) | 0.9999 | 0.0760 | 0.85 | 11.1x (rerun: exact 9.46 s) |

Before `accumulate`, the exact arm took 10.15 s, and `np.add.at` alone was 1.34 s of the fused run (now 0.04 s).

ez-means runs the whole training in C:

| Stage | How |
|---|---|
| Init and epoch order | splitmix64 Fisher-Yates |
| Rows | read by index, with no gather copy |
| Regroup | Lloyd into √k groups, 10 rounds, PDS nearest |
| Batch steps | the same assign, accumulate, EMA update, mass revival and last-epoch mean |

It differs from the Python arms in two ways:

| Difference | Why |
|---|---|
| Raw coordinates, no PCA rotation | rotation saves 0.14 s of kernel time at T=2 but costs about 0.24 s |
| Its own RNG streams | so it is not bit-identical to faiss |

Checks:

| Check | Result |
|---|---|
| SIFT 100k, k=1024, T=2 | test MSE ratio to exact e-means 0.9994 (seed 0) and 1.0006 (seed 1); 0.05 s against 0.54-0.73 s |
| Determinism | centroids are bit-identical at 1 and 4 threads (same sha256) |

| Finding | Evidence |
|---|---|
| The kernel's labels equal grouped's | 100% equal at p4 and at p32. At p32, where every group is probed, they also equal exact `faiss.knn` |
| Prefix pruning written in numpy loses to its own overhead | 93.95 s against 9.97 s for exact |
| Prefix pruning stacks weakly on grouping | within a probed group, a 16-dimension prefix drops only about 28% of candidates, because grouping already spends the leading PCA directions |
| bf16 assignment keeps quality | test MSE ratio 0.9999 to 1.0004 on SIFT and DEEP, k=1024, T=2 and 3. Speed is untested: inputs were rounded and the compute stayed in fp32 |
| The kernel runs at about 5 centroid-dims per cycle against a load limit near 16 | the cost per centroid-dim is the same at k=1024 and k=4096. perf shows 1.3 L1 misses per 16-lane step. Sorting rows by group gains 14% |

Where the fused run's 1.33 s goes: kernel 0.62, regroup k-means 0.19, PCA rotation 0.24, batch gather 0.17, accumulate 0.04.

## 5. Race (`race.py --round large`)

The race measures time to reach Lloyd-25 test MSE within 0.1% on SIFT1M train and 100k test, k=4096, with the median of seeds 0 and 1 on 4 cores. ez-means ran with the AVX-512 kernel that section 6 replaces.

| Method | Budget | Test MSE / Lloyd-25 | Seconds | ez-means is faster by |
|---|---|---|---|---|
| faiss Lloyd | 25 it | 1.0000 | 110.21 | 128x |
| sklearn KMeans, random init | 25 it | 1.0004 | 43.19 | 50x |
| faiss e-means (paper) | T=2 | 1.0009 | 10.08 | 11.7x |
| sklearn MiniBatchKMeans | 10 epochs | 1.0068, never reaches | 37.54 | |
| ez-means prototype, Python loop | T=3 | 0.9996 | 3.88 | 4.5x |
| **ez-means** | T=2 | 1.0010 | 0.86 | |

T=2 sits on the edge of the target. At T=3 (1.28 s, 0.9995) ez-means is 86x faster than Lloyd-25 and 34x faster than sklearn. At equal epochs it is 11.7x faster than paper e-means at both T=2 and T=3. Of the 128x over Lloyd, about 11x comes from e-means itself (2 passes instead of 25), and about 12x from this work.

## 6. A portable kernel (`kernel.c`)

The AVX-512 intrinsics are gone. The kernel is now plain C11:

| Construct | Why it works everywhere |
|---|---|
| 16-lane `float` arrays with `#pragma omp simd` on the lane loops | the compiler picks the vector width |
| `malloc` and a modulo RNG | no `aligned_alloc` and no `__int128` |
| `-ffp-contract=off` | no FMA, so every build computes the same bits |

SIFT1M, k=4096, T=2, seed 0, best of 3:

| Build | 4 threads, s | 1 thread, s | Centroids sha256 |
|---|---|---|---|
| AVX-512 intrinsics (old) | 0.81 | 2.89 | `858583d2` (uses FMA) |
| portable, gcc, native | 0.95 | 3.54 | `397e427a` |
| portable, gcc, x86-64-v3 (AVX2) | 1.06 | | `397e427a` |
| portable, gcc, x86-64 (SSE2) | 1.80 | 6.73 | `397e427a` |
| portable, clang, native | | 3.34 | `397e427a` |
| portable, clang, x86-64 (SSE2) | | 6.64 | `397e427a` |

| Finding | Evidence |
|---|---|
| Every portable build gives the same bits | one sha256 across gcc and clang, from SSE2 to AVX-512 |
| The portable kernel costs 16-22% against intrinsics | 1 thread: 3.34-3.54 s against 2.89 s, mostly the FMA that `-ffp-contract=off` gives up |
| Plain loops without `omp simd` are compiler-fragile | clang vectorized the 4-stream loop with gathers and scatters, 56.7 s against 1.51 s for gcc. Named accumulators plus `omp simd` fixed it |
| The clang 4-thread numbers are single-threaded | this host has no `libomp`, and clang with `-fopenmp=libgomp` emits no parallel calls (`nm` shows no `GOMP_parallel`). Clang is compared at 1 thread |
| A fixed 16-lane vector-extension type is worse than `omp simd` | 18.4 s on AVX2 and 14.2 s on SSE2 |

On the installed portable kernel (`fused_time.py`):

| Arm | Seconds | Test MSE / exact |
|---|---|---|
| exact | 9.53 | 1.0000 |
| ez-means | 1.03 | 1.0001 |

That is 9.3x. Centroids are bit-identical at 1 and 4 threads, and `validate.py` is still 5/5 for both metrics.

## 7. Race on the portable kernel, and against faiss SuperKMeans (`race.py --round large`, `--round super`)

The setup is the same as section 5. faiss 1.16 also ships `SuperKMeans` (Kuffo, Hepkema and Boncz 2026, arXiv 2603.20009). It runs a Lloyd iteration with a 16-dimension partial GEMM against all k centroids, then ADSampling pruning over the remaining dimensions in a PDX layout.

Time to Lloyd-25 test MSE within 0.1%, medians:

| Method | Budget | Seconds | ez-means is faster by |
|---|---|---|---|
| faiss Lloyd | 25 it | 109.04 to 115.38 | 69x to 76x |
| faiss SuperKMeans | 25 it (10 it: 1.0033, 15 it: 1.0014) | 85.24 | 56x |
| sklearn KMeans | 25 it | 43.64 | 28x |
| faiss e-means (paper) | T=2 | 10.85 | 6.9x |
| **ez-means, portable** | T=3 | 1.51 to 1.57 | |

At T=2 the portable ez-means reaches 1.0011, just outside the target; the AVX-512 build reached 1.0010, just inside. At equal epochs ez-means is 10.3x faster than paper e-means at T=2 and 9.9x at T=3. SuperKMeans makes each iteration cheaper (85 s against 115 s for 25), but it still needs 25 iterations, and every point still meets all k centroids at the GEMM boundary.

## 8. FFT and Hadamard ideas (`projection.py`)

Each idea was tested separately, at k=1024 and T=2:

| Dataset | Dimension | Train rows |
|---|---|---|
| SIFT | 128 | 1M |
| SSNPP | 256 | 500k |
| DINO ViT-L | 1024 | 200k |

The random rotation is a sign flip followed by a Walsh-Hadamard transform.

| Idea | SIFT 128 | SSNPP 256 | DINO 1024 | Verdict |
|---|---|---|---|---|
| A random rotation before exact early abandon, in centroid-dims per row (raw, PCA, Hadamard) | 20677, 14465, 20497 | 52783, 52145, 52798 | 184650, 121801, 184715 | dead: a random rotation equals raw. PCA helps SIFT and DINO by 30-34%, but a dense PCA rotation costs about one brute-force pass at d=1024 |
| ADSampling probabilistic pruning, eps 2.1, against exact pruning on raw (share of brute-force dims; nearest kept) | 0.187 against 0.379; 1.0000 | 0.936 against 0.888; 0.9985 | **0.053 against 0.497; 0.9925** | alive at high d: 9.4x less distance work on DINO, losing 0.75% of nearest labels. No gain on SSNPP |
| Project to d' with Hadamard, run ez-means there, then lift once to full d (test MSE / full; total time / full) | d'=32: 1.0548, 0.89; d'=64: 1.0128, 1.92 | d'=64: 1.0223, 1.03; d'=128: 1.0109, 1.75 | d'=256: 1.0027, 1.31; d'=512: 1.0005, 2.57 | dead at 128-256. On DINO the quality cost is small and ez-means itself runs 2-3.8x faster, but the dense projection and the knn lift eat the gain |

The ADSampling numbers use an oracle bound (the true nearest distance) for every rule. "Nearest kept" below 1 for the exact rule is an artifact of the harness: it sums chunks in float64 in a different order from the reference distance, so the last check can fail by one ulp. On integer data (SSNPP and DINO raw) the sums are exact, and it reads 1.0000.

FFT's only real role is to make the random rotation O(d log d), which is what makes ADSampling cheap at high d. With a dense rotation, rotating DINO costs as much as a brute-force pass.

## 9. ADSampling inside ez-means (`adsampling.py`)

`train` takes `ad_epsilon` in `Settings`. When it is above 0:

| Step | What happens |
|---|---|
| Rotation | the data is rotated once by random signs and a fast Walsh-Hadamard transform in C, O(d log d) per row, and the centroids are rotated back at the end |
| Pruning | every 16-dimension checkpoint scales the early-abandon bound by (m/d)(1+ε/√m)² |
| Best update | still compares exact distances |

At ε=0 every factor is 1, and the centroids are bit-identical to before (same sha256 at seeds 0 and 1). T=2, p4, 4 cores:

| Data | k | Pruning | Seconds | Test MSE / exact | Distance work / exact |
|---|---|---|---|---|---|
| SIFT 1M, d=128 | 4096 | exact | 1.09 | 1.0000 | 1.0000 |
| | | ε 2.1 | 1.13 | 0.9997 | 0.94 |
| | | ε 1.0 | 0.96 | 0.9998 | 0.71 |
| DINO 300k, d=1024 | 1024 | exact | 1.26 | 1.0000 | 1.0000 |
| | | ε 2.1 | 0.97 | 0.9992 | 0.61 |
| | | ε 1.0 | 0.88 | 1.0003 | 0.46 |
| | 4096 | exact | 2.28 | 1.0000 | 1.0000 |
| | | ε 2.1 | 1.79 | 1.0001 | 0.61 |
| | | ε 1.0 | 1.58 | 1.0003 | 0.45 |

Live, the 9.4x work cut from section 8 shrinks to 1.6-2.2x, which is 1.3-1.44x in wall time at d=1024 with no quality loss. There are two reasons. Grouped probing already cuts each point's candidates to about 5%. And the running bound is looser than the oracle bound. The oracle number should not be quoted as a result.

## 10. The paper's 1M protocol at k=16384 (`paper_cpu.py`)

This uses the paper's data, test rows and per-seed Flash@25 targets (seeds 10-12). The crossing is the smallest epoch budget, in steps of 0.2, whose test MSE is at or below the target. The time is the CPU seconds of that run, on 4 cores.

| Dataset | Arm | Crossings per seed | Crossing, upper median | Paper | Seconds at crossing | Speedup |
|---|---|---|---|---|---|---|
| DEEP 1M (d 96) | e-means | 2, 1.8, 2 | 2 | 2 | 27.0 | |
| | ez-means p4 | 2, 2, 2.2 | 2 | | 1.8 | 15x |
| SPACEV 1M (d 100) | e-means | 2, 2.6, 2.2 | 2.2 | 2.4 | 29.9 | |
| | ez-means p4 | 3, never by 3.0, 2.8 | 3 | | 2.7 | fails one seed |
| | ez-means p8 | 2.6, 2.6, 2.6 | 2.6 | | 3.7 | 8x |
| SSNPP 1M (d 256) | e-means | 1.4, 1.4, 1.4 | 1.4 | 1.4 | 46.1 | |
| | ez-means p4 | 1.4, 1.4, 1.4 | 1.4 | | 4.4 | 10.5x |

At k=16384 there are 128 groups, so p4 searches about 3% of the centroids. That is enough for DEEP and SSNPP but not for SPACEV, which needs p8 (about 6%). SPACEV has d=100, which exposed that the kernel required d to be a multiple of 16. `train` now zero-pads to the next multiple of 16 (the next power of two under ADSampling); results at d=128 are bit-identical to before.

## 11. Approximate assignment does not hurt Lloyd either (`approximate.py`)

SIFT 100k, k=1024, grouped p4, 5 seeds:

| Training | Grouped test MSE / exact, median | Per seed |
|---|---|---|
| fixed exact e-means centroids, assignment only | 1.0037 | |
| Lloyd, 25 full-batch iterations | 0.9994 | 0.9994, 0.9998, 0.9988, 0.9992, 0.9994 |
| e-means T=2 | 1.0000 | 1.0000, 1.0016, 0.9994, 0.9995, 1.0005 |

The seed spread of the exact runs is 0.12-0.13%. On fixed centroids, approximate assignment costs 0.37%. Train with the same approximate assignment, and that cost disappears for both Lloyd and e-means. So it is not the EMA averaging out misses. The centroids co-adapt to the grouped partition, and they stay good under exact assignment at test time.

## 12. Second machine: Modal, AMD EPYC Zen 3 (`modal_jobs.py`)

The second machine is a Modal container with 4 cores and 16 GB. It reports AMD family 25 model 1 at about 3.1 GHz, with AVX2 and no AVX-512, and gcc `-march=native` picks znver3. The race is the same as section 7 (`race.py --round remote`), against Lloyd-25:

| Method | Budget | Test MSE / Lloyd-25 | Modal s | Local s | ez-means faster by, Modal |
|---|---|---|---|---|---|
| faiss Lloyd | 25 it | 1.0000 | 150.78 | 109.04 | 47x |
| faiss SuperKMeans | 25 it | 1.0001 | 184.01 | 85.24 | 57x |
| sklearn KMeans | 25 it | 1.0004 | 83.25 | 43.64 | 26x |
| faiss e-means (paper) | T=2 | 1.0003 | 15.19 | 10.85 | 4.7x |
| ez-means | T=3 | 1.0000 | 3.22 | 1.57 | |

The lead over classical k-means holds on the second machine. The lead over paper e-means falls from about 10x to 5.5-6.9x at equal epochs. Moving from local to Modal slows ez-means by 2.0-2.6x but slows paper e-means by only 1.4x. The cause is not yet measured. The candidates are the lower clock, AVX2 without FMA, and OpenMP spin-waits under Modal's CPU quota. SuperKMeans is slower than Lloyd on this CPU.

### Why ez-means slows more than paper e-means on Zen 3 (`kernel_probe.py`)

ez-means alone on SIFT1M, k=4096, best of 3 seconds, on both machines, with FMA contraction off and on:

| Contraction | Threads | Wait | Zen 5 T=2 | Zen 5 T=3 | Zen 3 T=2 | Zen 3 T=3 |
|---|---|---|---|---|---|---|
| off | 4 | default | 1.02 | 1.50 | 2.31 | 3.48 |
| off | 4 | passive | 1.02 | 1.49 | 2.48 | 3.64 |
| off | 1 | default | 3.73 | 5.42 | 8.61 | 12.85 |
| fast (FMA) | 4 | default | 1.02 | 1.49 | 2.30 | 3.26 |
| fast (FMA) | 1 | default | 3.66 | 5.35 | 8.17 | 12.25 |

| Candidate cause | Verdict |
|---|---|
| OpenMP spin-waits under the CPU quota | no: passive waiting is slightly slower, and 4 threads scale 3.7x on both machines |
| Giving up FMA | small: 2% on Zen 5, 5-6% on Zen 3 |
| Per-core speed | yes: one thread is 2.3x slower on Zen 3 (3.1 GHz, AVX2) than on Zen 5 |

The paper's e-means runs in a BLAS GEMM, which slows only 1.4x on Zen 3. So the lead over paper e-means depends on the hardware: about 10x at equal epochs on Zen 5, and 5.5-6.9x on Zen 3. The lead over classical k-means, 26-57x, holds on both.

## 13. Next algorithmic levers (`warm_start.py`, `routing.py`, perf)

ez-means on SIFT1M at T=2, 4 cores. `perf` samples per phase, as a share of the whole run, including Python and data loading:

| k | Total | Assignment | Regroup | Accumulate |
|---|---|---|---|---|
| 4096 | 1.04 s | 81% | 2.4% | 4.2% |
| 16384 | 2.26 s | 71% | 11% | 2.5% |
| 65536 | 7.33 s | 72% | 16% | 1.0% |

Each lever was then tested on its own:

| Lever | Result | Verdict |
|---|---|---|
| Start each point's search from its previous-epoch centroid | 0.7-1.5% less distance work at k 4096 and 16384. The scan already visits the nearest group first, so its bound is tight early | dead, code removed |
| 3-level routing with beams, against 2-level, on fixed trained centroids (2000 points) | at equal true-nearest rate it needs about 1.5-1.7x fewer distance evaluations, not the 3.5x the candidate count suggests. Example at k=16384: 2-level p4 uses 691 evaluations, finds the true nearest 93.9% of the time, and costs 0.95% extra. 3-level with beams (4, 8) uses 370, 92.6% and 1.17% | modest; only for k ≥ 65536 |
| Fixed p4 as k grows | excess cost on fixed centroids is 0.37% at k=1024, 0.95% at 16384 and 1.74% at 65536 | probes should grow with k |
| Initialize from ez-means on a 10% or 25% subsample, then train for fewer epochs, with default or trusted initial usage | every arm lies on the plain time-quality curve. The best, 10% subsample then T=2, reaches 1.0003 against plain T=3 in 1.16 s, while plain reaches about 1.0008 at the same time | dead: e-means already uses its first pass well |

The race budgets T=1, 2, 3, 4 also overstate ez-means' time to target: plain T=2 lands at 1.0011, so a fractional T of about 2.2 (about 1.15 s) would cross. The paper protocol, with 0.2-epoch steps, does not have this problem.

## Next

| # | Step | Why |
|---|---|---|
| 1 | Probes as a function of k (and adaptive per point) | the fixed-p4 excess grows with k; SPACEV at k=16384 needs p8 |
| 2 | 10 seeds and error bars on the paper protocol | the paper uses seeds 10-19 |
| 3 | The 10M / k=65536 protocol, and a third CPU (Intel AVX-512 on Modal) | scale and hardware coverage |

## Verification

| Command | Result |
|---|---|
| `.venv/bin/python validate.py` | bit-identical to faiss NumpyEmeans: spherical 5/5, Euclidean 5/5 |
| `ruff check`, `ruff format --check`, `mypy --strict` over this folder | clean |
| `compare.py`, `generality.py`, `tuning.py`, `real.py`, `budget.py`, `ivf.py`, `hierarchy.py`, `grouped.py`, `fused_time.py`, `lowbits.py` | the tables above. All runs used 4 cores and 8 GB, with `OMP_NUM_THREADS=4` |
| `gcc -std=c11 -O3 -march=native -fopenmp -ffp-contract=off -Wall -Wextra -Wpedantic` on `kernel.c` | no warnings |
| `rg "immintrin\|__m512\|__int128\|aligned_alloc\|__builtin\|__attribute__" kernel.c` | no matches |
| `race.py --round large` | section 5 (AVX-512: 647 s wall, 1.27 GB peak); section 7 (portable: 655 s, 1.27 GB) |
| `race.py --round super` | section 7, 558 s wall, 1.63 GB peak |
| `projection.py --data sift / ssnpp / dino` | section 8; 5.0, 7.0 and 25.8 s wall; peak 3.3 GB on DINO |
