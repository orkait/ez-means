# ez-means: e-means made faster

ez-means is our faster version of the e-means clustering algorithm (Bessa et al., arXiv 2611.09501). It keeps e-means' training recipe: batch law, EMA step, mass revival and last-epoch mean. It changes how points are assigned and how the loop runs:

| Change | What it does |
|---|---|
| Grouped probing | centroids are regrouped into √k groups, and each point searches only its 4 nearest groups |
| Early abandon | a partial distance search drops a centroid as soon as its running sum passes the best so far |
| Accumulation | a counting sort replaces `np.add.at` |
| The loop | the whole training loop is portable C11, with no hardware-specific code |

On SIFT1M, at k=4096 on 4 cores, ez-means reaches Lloyd-25 quality 128x faster than faiss Lloyd, 50x faster than sklearn KMeans and 11.7x faster than the paper's e-means (section 5, AVX-512 build). The portable build gives the same bits on every compiler and SIMD level, and costs 16-22% against intrinsics (section 6).

The lead holds across machines, datasets and seeds:

| Setting | ez-means against | Speedup | Section |
|---|---|---|---|
| SIFT1M k=4096, three CPUs (Zen 5, Zen 3, Broadwell), portable build | faiss Lloyd, SuperKMeans, sklearn | 26-76x | 7, 12, 15 |
| same | paper e-means | 3.3-6.9x | 7, 12, 15 |
| paper protocol, DEEP / SSNPP / SPACEV 1M, k=16384, 10 seeds, 16 Zen 3 cores | paper e-means | 13.8x / 9.0x / 11.1x (paired medians) | 16 |
| same protocol, RTX 3060, 10 seeds | faiss GPU e-means (Triton) | 2.9x / 2.4x / 2.2x | 17 |
| DEEP 10M, k=65536, RTX 3060 (3 seeds) / H100 (10 seeds) | faiss GPU e-means (Triton / CuTe) | 11.2x / 10.0x (2.3x against CuTe's best seed) | 17 |
| 1M, k=16384, H100, 10 seeds | faiss GPU e-means (CuTe) | 0.3-0.5x: ez-means is slower | 17 |

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

## 14. Screening bounds from other fields (`screening.py`, `screened_time.py`)

These are fixed trained centroids, k=4096, grouped p4 candidates, 2000 points, and an oracle best distance. "Float dims" counts the float work per candidate that is still scanned.

| Screen | Field | SIFT: screened out, nearest kept, float dims | DEEP: screened out, nearest kept, float dims |
|---|---|---|---|
| exact early abandon (today) | | 0, 1.0000, 65.2 | 0, 1.0000, 50.2 |
| PAA block-mean lower bound, block 4 | time-series search | 0.159, 1.0000, 90.5 (1.39x more) | 0.135, 1.0000, 70.4 (1.40x more) |
| RaBitQ 1-bit code, high-probability bound, eps 3.0 | vector databases | 0.964, 1.0000, 3.8 (17x less) | 0.965, 1.0000, 3.0 (17x less) |
| RaBitQ, eps 1.9 | | 0.982, 0.9855, 2.1 | 0.982, 0.9875, 1.6 |

A 4-bit or 8-bit query gives the same numbers as a float query. The RaBitQ bound is probabilistic, not deterministic.

In the C kernel (`ez_means(..., screen=eps)`, off by default) the screening does not yet pay. On SIFT1M, k=4096, T=2:

| Version | Seconds | Against exact (1.02 s) |
|---|---|---|
| v1: bit planes built with a branch per bit | 4.91 | 4.8x slower |
| v2: bit planes built in registers, branch-free | 2.00 | 2.0x slower |

Quality holds: test MSE is within 0.04% of exact at k 4096 and 16384. perf shows the per-candidate estimate (scalar popcounts over 4 planes and 2 words, a divide, branches) costs more than the 16-lane float scan it replaces.

| Build | Cost per candidate estimate (microbenchmark, 4-bit query) |
|---|---|
| clang, native (vectorized popcount) | 0.26 ns |
| gcc, native (scalar `popcnt`) | 1.24 ns |
| gcc, x86-64 baseline (no `popcnt`) | 4.92 ns |
| reference: the float scan today | about 2.4 ns |

Making it pay needs group-ordered codes and lane-parallel estimates, and the vector popcount that only AVX-512 VPOPCNTDQ and ARM NEON provide. It stays experimental.

## 15. Third machine: Kaggle, Intel Broadwell (`race.py --round remote`)

A Kaggle CPU notebook reports an Intel Xeon at 2.2 GHz, family 6 model 79 (Broadwell), 2 cores with 2 threads each, AVX2 and FMA, no AVX-512. It runs the same race as section 12 with 4 threads, faiss 1.16.0, numpy 2.1.3 and sklearn 1.6.1 from the notebook image:

| Method | Budget | Test MSE / Lloyd-25 | Seconds | ez-means faster by |
|---|---|---|---|---|
| faiss Lloyd | 25 it | 1.0000 | 281.10 | 32.6x |
| faiss SuperKMeans | 25 it | 1.0001 | 285.67 | 33.1x |
| sklearn KMeans | 25 it | 1.0004 | 302.61 | 35.1x |
| faiss e-means (paper) | T=2 | 1.0003 | 28.50 | 3.3x |
| faiss e-means (paper) | T=3 | 0.9996 | 40.51 | 4.7x at equal epochs |
| ez-means | T=3 | 1.0000 | 8.63 | |

With the portable kernel, the lead over classical k-means holds on all three machines: 28-76x on Zen 5 (section 7), 26-57x on Zen 3 (section 12) and 33-35x on Broadwell. The lead over paper e-means at the time-to-target shrinks as the CPU gets older and narrower: 6.9x on Zen 5, 4.7x on Zen 3 and 3.3x on Broadwell. ez-means T=2 (7.69 s) is not comparable here: its median includes the first-call kernel compile.

## 16. The paper protocol with 10 seeds (`paper_cpu.py --seeds`, `modal_jobs.py --job paper`)

Section 10 used 3 seeds. The paper uses seeds 10-19, so this repeats the protocol on all 10, on a Modal container with 16 Zen 3 cores (AMD family 25 model 1) and 16 OpenMP threads, k=16384. Each dataset ran in its own container. Seconds are wall seconds of the crossing run. The speedup is paired: the same seed's e-means time over its ez-means time.

| Dataset | Arm | Crossings, seeds 10-19 | Crossing mean ± sd | Seconds, median (mean ± sd) | Paired speedup, median (min-max) |
|---|---|---|---|---|---|
| DEEP 1M | e-means | 1.8, 1.8, 2, 2.8, 1.8, 2, 1.8, 1.6, 2.2, 1.6 | 1.94 ± 0.35 | 18.9 (20.7 ± 4.9) | |
| | ez-means p4 | 2, 2, 2.2, 2.8, 1.8, 2, 2.2, 1.8, 2.6, 2 | 2.14 ± 0.33 | 1.4 (1.45 ± 0.13) | 13.8x (10.5-20.6) |
| SSNPP 1M | e-means | 1.4 on all 10 | 1.40 ± 0 | 29.8 (29.5 ± 2.8) | |
| | ez-means p4 | 1.4 on 9, 1.6 on 1 | 1.42 ± 0.06 | 3.3 (3.31 ± 0.19) | 9.0x (7.4-10.5) |
| SPACEV 1M | e-means | 2.2, 2.6, 2.2, 2.6, 2.2, 2, 2.2, 2, 2.4, 2.6 | 2.30 ± 0.24 | 30.1 (31.5 ± 5.6) | |
| | ez-means p8 | 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.8, 2, 2.4, 2.8 | 2.56 ± 0.23 | 2.8 (2.74 ± 0.21) | 11.1x (8.9-15.8) |

Over 10 seeds ez-means is 9-14x faster than e-means at the paper's targets on this CPU, and never slower than 7.4x on any seed. It needs 0.02-0.26 more epochs on average, and the extra epochs cost less than the faster assignment saves. The ez-means time also varies less across seeds (sd 0.13-0.21 s against 2.8-5.6 s).

## 17. ez-means on GPU (`gpu.py`)

faiss runs e-means on GPU through PyTorch with its own kernels: Triton on A100-class and newer cards, CuTe on H100. Both compute every point against every centroid in BF16 with FP32 accumulation. Its CuTe "grouped" path is still exact: it finds each row's best centroid tile in a full scan, then recovers the id inside that tile. So the question is whether ez-means' approximate grouped assignment also pays on a GPU, where the brute-force GEMM runs near the hardware's peak.

`gpu.py` keeps faiss' training loop (`TorchEmeans`: batch law, EMA, revival, last-epoch mean) and replaces only its assignment backend with two Triton kernels:

| Kernel | Work |
|---|---|
| select | for a tile of rows: BF16 dots against the √k group centers with FP32 accumulation, the top p groups, and the row norms, in one pass |
| member | (row, group) pairs are sorted by group; each program takes up to 64 pairs of one group and scans that group's members in 64-centroid tiles with `tl.dot` |

Groups are rebuilt every 5 batches with 10 Lloyd iterations over the centroids, as in the C loop. The plumbing is on the GPU, with no host synchronization per batch: block tables come from `searchsorted` over a fixed upper bound of blocks, and empty blocks return at once. On fixed random centroids, probing all groups matches exact FP32 labels 1.0000 (50k rows), so the kernels are exact when nothing is pruned.

On an RTX 3060 (12 GB, BF16 dense peak with FP32 accumulation about 25 TFLOPS), faiss' Triton brute force runs at about 19 TFLOPS at k=4096. A perfect brute-force kernel could therefore be at most about 1.35x faster than this baseline. The paper protocol with 10 seeds at k=16384, in GPU seconds of the crossing run:

| Dataset | Arm | Crossing mean ± sd | Seconds, median | Paired speedup, median (min-max) |
|---|---|---|---|---|
| DEEP 1M | faiss GPU e-means (Triton) | 1.92 ± 0.27 | 0.46 | |
| | ez-means GPU p4 | 2.10 ± 0.38 | 0.16 | 2.9x (2.4-4.2) |
| | ez-means GPU p8 | 1.94 ± 0.21 | 0.20 | 2.3x (2.1-2.9) |
| SSNPP 1M | faiss GPU e-means (Triton) | 1.40 ± 0 | 0.71 | |
| | ez-means GPU p4 | 1.44 ± 0.08 | 0.29 | 2.4x (2.2-2.7) |
| | ez-means GPU p8 | 1.40 ± 0 | 0.36 | 2.0x (1.9-2.1) |
| SPACEV 1M | faiss GPU e-means (Triton) | 2.30 ± 0.19 | 0.58 | |
| | ez-means GPU p4 | never by 3.0 on 4 of 10 seeds | | fails |
| | ez-means GPU p8 | 2.66 ± 0.21 | 0.27 | 2.2x (1.8-2.4) |

Every speedup is above the 1.35x a perfect brute-force kernel could reach. The epoch costs match the CPU loop: SSNPP p4 crosses at 1.44 on GPU and 1.42 on CPU (section 16), and SPACEV p8 at 2.66 and 2.56. Between identical reruns, some ez-means seeds cross one 0.2 step earlier or later. The GPU loop is not bit-deterministic: faiss accumulates the batch sums with `index_add_`, which sums in a different order from run to run on CUDA. On SPACEV both need about 0.3-0.4 more epochs than e-means, so that cost belongs to the pruning, not to the GPU.

A first run of this table crossed SSNPP p4 at 1.78. The cause was a precision bug: the kernels took FP32 norms of the centroids but dotted against their BF16 copies, so the score error scaled with |x|·|c| instead of |x − c|. faiss avoids this by taking norms of the BF16 copies, and so does `gpu.py` now. With the bug, probing all 128 groups still crossed one SSNPP seed late (1.4, 1.6, 1.4 against faiss' 1.4 on all three), so the lag was not only from pruning.

A crossing protocol can flatter a method that drops fast and then stalls. `gpu.py --curve` trains seed 10 to 1, 1.5, 2, 3, 4 and 6 epochs and also counts empty clusters (no training point under exact FP32 assignment). Test MSE over the seed-10 target, and GPU seconds:

| Dataset | Epochs | faiss GPU e-means | ez-means GPU p4 | ez-means GPU p8 |
|---|---|---|---|---|
| DEEP | 1 | 1.0126, 0.24 s | 1.0131, 0.10 s | 1.0114, 0.12 s |
| | 2 | 0.9989, 0.52 s | 1.0002, 0.19 s | 0.9965, 0.23 s |
| | 3 | 0.9976, 0.71 s | 0.9966, 0.20 s | 0.9951, 0.29 s |
| | 6 | 0.9965, 1.39 s | 0.9924, 0.31 s | 0.9928, 0.46 s |
| SSNPP | 1 | 1.0037, 0.53 s | 1.0040, 0.26 s | 1.0036, 0.37 s |
| | 2 | 0.9973, 1.04 s | 0.9984, 0.48 s | 0.9979, 0.62 s |
| | 3 | 0.9953, 1.41 s | 0.9967, 0.50 s | 0.9966, 0.79 s |
| | 6 | 0.9945, 2.77 s | 0.9954, 0.74 s | 0.9942, 1.14 s |
| SPACEV | 1 | 1.0200, 0.25 s | 1.0250, 0.12 s | 1.0217, 0.15 s |
| | 2 | 1.0001, 0.53 s | 1.0038, 0.23 s | 1.0022, 0.28 s |
| | 3 | 0.9957, 0.76 s | 1.0001, 0.25 s | 0.9977, 0.36 s |
| | 6 | 0.9903, 1.48 s | 0.9938, 0.38 s | 0.9932, 0.58 s |

ez-means keeps improving out to 6 epochs. Within the range it was run, it is ahead at equal GPU seconds on all three datasets: on SPACEV, p8 reaches 0.9932 in 0.58 s while faiss reaches 1.0001 in 0.53 s. At equal epochs it is within 0.15% of faiss on DEEP and SSNPP, often ahead, and up to 0.5% behind on SPACEV. Its floor is not measured: faiss at 6 epochs on SPACEV (0.9903, 1.48 s) is better than anything ez-means reached by 6 epochs. Empty clusters are the same for both: 5-37 after one epoch, falling to 0-3 from 2 epochs on, so the revival keeps working with pruned assignment.

### Scale: DEEP 10M, k=65536, and the H100

k=65536 has 256 groups. On the RTX 3060, 3 seeds (10-12), against the seeds' targets; "never" means not by 3.0 epochs, and its time is the 3.0-epoch run:

| Arm | Crossings | Seconds per seed | Paired speedup, median |
|---|---|---|---|
| faiss GPU e-means (Triton) | 2.6, never, 2.8 | 23.2, 26.8, 25.1 | |
| ez-means GPU p8 | 1.8, 3, 2.8 | 1.7, 2.4, 2.3 | 11.2x (10.9-13.4) |
| ez-means GPU p16 | 2, 2.6, never | 2.6, 3.4, 3.9 | 8.0x, one seed misses |

With 3 seeds the crossings at this scale are noisy for both methods: faiss misses one seed here, and p16 misses one seed on the 3060 but none on the H100 below.

Then on a Modal H100 80GB, with BF16 input so that faiss selects its CuTe kernels, the fastest path it has (`modal_jobs.py --job gpu`, seeds 10-19):

| Dataset | faiss GPU e-means (CuTe): crossing mean ± sd, s median (range) | ez-means GPU: arm, crossing mean ± sd, s median (range) | ez-means faster by, paired median (min-max) |
|---|---|---|---|
| DEEP 1M, k=16384 | 2.00 ± 0.30, 0.04 (0.03-0.04) | p8: 1.92 ± 0.36, 0.10 (0.08-0.11) | 0.3x: slower |
| SSNPP 1M | 1.40 ± 0, 0.04 (0.04-0.07) | p8: 1.40 ± 0, 0.09 (0.08-0.09) | 0.4x: slower |
| SPACEV 1M | 2.32 ± 0.21, 0.04 (0.03-0.06) | p16: 2.42 ± 0.20, 0.09 (0.08-0.12) | 0.5x: slower |
| DEEP 10M, k=65536 | 2.46 ± 0.44, 2.30 (0.63-5.38) | p8: 2.44 ± 0.48, 0.27 (0.21-0.32) | 10.0x (2.2-17.7) |

At 1M the H100 does a full k=16384 epoch in a few milliseconds, faiss finishes in 0.03-0.07 s, and ez-means sits at 0.08-0.12 s whatever p is (p4 to p16 all take about 0.09-0.10 s). Since its time does not grow with p, it is not arithmetic; the likely cost is the per-batch kernel launches and PyTorch bookkeeping (not profiled on the H100). So on a fast GPU at this size ez-means is 2-3x slower than faiss.

At 10M and k=65536 the arithmetic dominates, and ez-means p8 crosses on all 10 seeds in 0.21-0.32 s. faiss' CuTe time depends on the seed far more than its crossing does: 0.63 s on seed 13 (crossing 3.0) but 5.38 s on seed 17 (crossing 1.8). The same seeds are fast or slow in a second run, and every budget was trained once before timing, so it is not kernel compilation; the cause inside faiss is not known. The paired median is 10.0x, and against faiss' best seed ez-means' median is 2.3x faster. p16 crosses earlier (2.12 ± 0.40) in 0.35 s (7.4x), and p4 misses 2 of 10 seeds. Cutting the per-batch launches (CUDA graphs or one persistent kernel) is the open lever for small problems on fast GPUs.

The CPU version of this 10M protocol (e-means against ez-means on 16 Zen 3 cores, `modal_jobs.py --job scale`) hit Modal's 3-hour limit before the e-means arm finished its 3 seeds, so there is no CPU number at 10M.

### Where the GPU gain starts

`gpu.py --sift 4096` runs the race setting of section 5 on the RTX 3060: SIFT1M, k=4096, seeds 0 and 1, against faiss Lloyd-25 computed on the CPU in the same run.

| Arm | Smallest budget within 0.1% of Lloyd-25 (median of 2 seeds) | GPU s |
|---|---|---|
| faiss GPU e-means (Triton) | T=2, 1.0009 | 0.148 |
| ez-means GPU p4 | T=2.2, 1.0006 | 0.113 |
| ez-means GPU p8 | T=2, 1.0007 | 0.138 |

At k=4096 the gain over faiss on the same GPU is only 1.3x. Brute force there costs 4096 dots per point, and the grouped search still pays for 64 group dots, about 256 member dots and its own bookkeeping. The gain grows with k: 1.3x at 4096, 2.2-2.9x at 16384 and about 11x at 65536 on the RTX 3060.

### How the GPU arm got here

The first version, with the group selection in PyTorch, was only 1.1-1.35x faster. v1-v3 were scratch versions and are not in the repo; v6 is `gpu.py`. The `--curve` table above was measured with v5; v6 changes no arithmetic, only how far each group is scanned. DEEP p4 at the crossing, 3 seeds, median:

| Version | Seconds | Change |
|---|---|---|
| v1 | 0.33 | PyTorch `topk` selection, `cdist` regroup, one host sync per batch |
| v3 | 0.28 | no host sync, group centers fixed between regroups as in the C loop |
| v4 | 0.17 | fused select kernel; regroup by GEMM and `argmin`. A profile of v3 had `topk` at 20% of GPU time and `cdist` at 18% |
| v5 | 0.20 | norms of the BF16 copies; the time moves with the crossings (2.2, 1.6, 2 here), and SSNPP p4 crosses at 1.4 on all 3 seeds |
| v6 (above) | 0.16 | each group is scanned to its own size, not to the largest group's. At k=4096 groups hold 20-176 centroids and the padded width was 192. At the same crossings p8 got 10-18% faster at 1M and about 30% faster at 10M |

v2 computed the group scores with a BF16-output GEMM, which rounds them to 8 mantissa bits. SSNPP (d=256, large norms) then never reached its target (1.0040 at 3.0 epochs). v3 restored FP32 scores, and the v4 select kernel keeps FP32 accumulation.

## Next

| # | Step | Why |
|---|---|---|
| 1 | Probes as a function of k (and adaptive per point) | the fixed-p4 excess grows with k; SPACEV at k=16384 needs p8 on CPU and GPU |
| 2 | Probe 1/16 of the groups by default (p4 at k=4096, p8 at 16384, p16 at 65536) | the smallest setting that crossed on every seed in sections 7, 16 and 17; not yet tested as a rule |
| 3 | ez-means' floor past 6 epochs on SPACEV | faiss at 6 epochs is 0.2% better than ez-means at 6 (section 17) |
| 4 | Fewer launches per batch on GPU (CUDA graphs or a persistent kernel) | on an H100 at 1M faiss CuTe is 2-3x faster, and ez-means' time does not move with p |
| 5 | Why faiss CuTe's time at 10M varies 8x by seed | it decides how large the H100 lead at 10M really is (2.3x against the best seed, 10x paired) |
| 6 | An Intel AVX-512 CPU | every CPU so far is AMD or pre-AVX-512 Intel |

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
| `race.py --round remote` in a Kaggle CPU notebook | section 15 |
| `modal run cloud/modal_jobs.py --job paper --dataset <name> --arms <arms>`, one container per dataset | section 16 |
| `gpu.py --datasets deep1b-1M fb-ssnpp-1M spacev-1M --seeds 10 ... 19`, `gpu.py --curve` and `gpu.py --datasets deep1b-10M --probes 0 8` on an RTX 3060 (torch 2.15.0, Triton 3.9.0, VRAM capped at 50%) | section 17 |
| `modal run cloud/modal_jobs.py --job gpu --dataset deep1b-1M,fb-ssnpp-1M,spacev-1M,deep1b-10M --arms 0,4,8,16` (1M rows) and `--dataset deep1b-10M --arms 0,8,16` with every budget trained once before timing (10M row); H100 80GB HBM3, CUTLASS DSL 4.7.1, QuACK 0.6.5, seeds 10-19 | section 17 |
| `gpu.py --sift 4096 --probes 0 4 8` on the RTX 3060 | section 17, where the GPU gain starts |
