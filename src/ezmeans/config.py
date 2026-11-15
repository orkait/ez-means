import os
from pathlib import Path

DIM = 32
CLUSTERS = 40
POINTS = 2176
NOISE = 0.025
DATA_SEED = 0

RUN_SEEDS = range(20)
EPOCH_BUDGETS = (1, 2, 3, 5, 10, 25)
TIMING_SEED = 0
TIMING_REPEATS = 5
THREAD_COUNTS = (1, 4)

RECOVERY_COSINE = 0.99
RECOVERY_PASS = 38

KEPLER_EPOCHS = 25

DATA = Path(os.environ.get("EZMEANS_DATA", Path.home() / "datasets" / "emeans"))
CACHE = Path(os.environ.get("EZMEANS_CACHE", Path.home() / ".cache" / "ezmeans"))

KERNEL_SOURCE = Path(__file__).parent / "kernel.c"
KERNEL_LIBRARY = CACHE / "libkernel.so"
KERNEL_COMPILE = (
    "gcc",
    "-std=c11",
    "-O3",
    "-march=native",
    "-fopenmp",
    "-ffp-contract=off",
    "-Wall",
    "-Wextra",
    "-shared",
    "-fPIC",
)

PAPER_BENCH_DIR = DATA / "faiss-bench" / "benchs" / "e_means"
PAPER_TARGETS = PAPER_BENCH_DIR / "paper_targets.json"

PAPER_DATA = DATA / "paper" / "data"
DEEP_BASE = PAPER_DATA / "deep1b" / "base.1B.fbin"
SSNPP_BASE = PAPER_DATA / "fb_ssnpp" / "FB_ssnpp_database.u8bin"
DINO_BASE = PAPER_DATA / "dino_vitl_10B" / "chunked_base_10B" / "chunk_0000.bvecs"

SIFT_DIR = DATA / "sift" / "sift"
SIFT_LEARN = SIFT_DIR / "sift_learn.fvecs"
SIFT_BASE = SIFT_DIR / "sift_base.fvecs"
