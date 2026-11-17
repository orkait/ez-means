import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from ezmeans.config import CACHE, KERNEL_COMPILE

KERNEL_SOURCE = Path(__file__).with_name("kernel.c")
LIBRARY_STEM = "libkernel"
LIBRARY_SUFFIX = ".so"
KEY_LENGTH = 16
TARGET_QUERY = ("-dM", "-E", "-x", "c", os.devnull)


def build_key() -> str:
    target = subprocess.run([*KERNEL_COMPILE, *TARGET_QUERY], capture_output=True, text=True, check=True).stdout
    digest = hashlib.sha256(KERNEL_SOURCE.read_bytes())
    digest.update("\0".join(KERNEL_COMPILE).encode())
    digest.update(target.encode())
    return digest.hexdigest()[:KEY_LENGTH]


def build() -> Path:
    path = CACHE / f"{LIBRARY_STEM}-{build_key()}{LIBRARY_SUFFIX}"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=CACHE, suffix=LIBRARY_SUFFIX)
        os.close(handle)
        try:
            subprocess.run([*KERNEL_COMPILE, str(KERNEL_SOURCE), "-o", temporary], check=True)
            Path(temporary).replace(path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path
