import os
from pathlib import Path

CACHE_VARIABLE = "EZMEANS_CACHE"
COMPILER_VARIABLE = "EZMEANS_CC"
TARGET_VARIABLE = "EZMEANS_MARCH"

CACHE = Path(os.environ.get(CACHE_VARIABLE, Path.home() / ".cache" / "ezmeans"))
COMPILER = os.environ.get(COMPILER_VARIABLE, "gcc")
TARGET = os.environ.get(TARGET_VARIABLE, "native")
KERNEL_COMPILE = (
    COMPILER,
    "-std=c11",
    "-O3",
    f"-march={TARGET}",
    "-fopenmp",
    "-ffp-contract=off",
    "-Wall",
    "-Wextra",
    "-shared",
    "-fPIC",
)
