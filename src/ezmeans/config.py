import os
from pathlib import Path

CACHE = Path(os.environ.get("EZMEANS_CACHE", Path.home() / ".cache" / "ezmeans"))
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
