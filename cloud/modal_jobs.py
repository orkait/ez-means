import os
import subprocess
import sys
from pathlib import Path

import modal

from ezmeans.config import DATA, KERNEL_COMPILE, KERNEL_LIBRARY, KERNEL_SOURCE, PAPER_BENCH_DIR, PAPER_DATA, SIFT_DIR

REPO = Path(__file__).parents[1]
REMOTE_REPO = Path("/root/ez-means")
REMOTE_SRC = REMOTE_REPO / "src"
REMOTE_EXPERIMENTS = REMOTE_REPO / "experiments"
REMOTE_DATA = Path("/root/datasets/emeans")
REMOTE_BENCH = REMOTE_DATA / PAPER_BENCH_DIR.relative_to(DATA)
REMOTE_SIFT = REMOTE_DATA / SIFT_DIR.relative_to(DATA).parent
REMOTE_PAPER_DATA = REMOTE_DATA / PAPER_DATA.relative_to(DATA)
REMOTE_PAPER_CACHE = REMOTE_PAPER_DATA.parent / "cache"
SKIPPED = [".venv", "build", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".git"]
RACE_CORES = 4
SCALE_CORES = 16
RACE_MEMORY_MB = 16384
SCALE_MEMORY_MB = 32768
HOUR = 3600
DOWNLOAD_THREADS = 16
SCALE_DATASET = "deep1b-10M"
PAPER_SEEDS = tuple(range(10, 20))
CPU_FIELDS = ("Vendor ID", "CPU family", "Model:", "Model name", "CPU(s)", "Thread", "CPU max MHz", "Flags")
GPU_KIND = "H100"
GPU_CORES = 8
GPU_MEMORY_MB = 65536
CONTRACTIONS = ("off", "fast")
PROBE_SETTINGS = (("4", "default"), ("4", "PASSIVE"), ("1", "default"))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "util-linux")
    .pip_install("numpy==2.5.4", "faiss-cpu==1.16.0", "scipy==1.18.1", "scikit-learn==1.9.1", "h5py==3.16.0")
    .env({"PYTHONPATH": str(REMOTE_SRC)})
    .add_local_dir(PAPER_BENCH_DIR, str(REMOTE_BENCH), ignore=SKIPPED)
    .add_local_dir(REPO, str(REMOTE_REPO), ignore=SKIPPED)
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "util-linux")
    .pip_install(
        "numpy==2.5.4",
        "faiss-cpu==1.16.0",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "h5py==3.16.0",
        "torch==2.15.0",
        "triton==3.9.0",
        "nvidia-cutlass-dsl[cu13]==4.7.1",
        "quack-kernels[cu13]==0.6.5",
        "cuda-python==13.5.0",
        "cuda-cccl[cu13]==1.3.0",
    )
    .env({"PYTHONPATH": str(REMOTE_SRC)})
    .add_local_dir(PAPER_BENCH_DIR, str(REMOTE_BENCH), ignore=SKIPPED)
    .add_local_dir(REPO, str(REMOTE_REPO), ignore=SKIPPED)
)
sift = modal.Volume.from_name("ez-means-data", create_if_missing=True)
app = modal.App("ez-means", image=image)


def run(command: list[str], threads: int, extra: dict[str, str]) -> str:
    machine = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
    lines = [line[:160] for line in machine.splitlines() if line.startswith(CPU_FIELDS)]
    lines.append(f"usable CPUs: {len(os.sched_getaffinity(0))}, OMP threads: {threads}")
    print("\n".join(lines), flush=True)
    environment = os.environ | extra | {"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
    with subprocess.Popen(
        command, cwd=REMOTE_EXPERIMENTS, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        for line in process.stdout or ():
            print(line, end="", flush=True)
            lines.append(line.rstrip())
    lines.append(f"exit {process.returncode}")
    return "\n".join(lines)


def fetch(datasets: list[str]) -> dict[str, str]:
    paths = {"EMEANS_DATA": str(REMOTE_PAPER_DATA), "EMEANS_CACHE": str(REMOTE_PAPER_CACHE)}
    os.environ.update(paths)
    sys.path.insert(0, str(REMOTE_BENCH))
    import paper_download

    for dataset in datasets:
        paper_download.download(paths["EMEANS_DATA"], paper_download.needs(dataset), DOWNLOAD_THREADS)
        held = paper_download.heldout_needs(dataset)
        if held is not None:
            paper_download.download_heldout(paths["EMEANS_DATA"], dict([held]), DOWNLOAD_THREADS)
    return paths


def paper_run(dataset: str, arms: list[str], seeds: list[str], cores: int) -> str:
    paths = fetch([dataset])
    command = [sys.executable, "paper_cpu.py", "--arms", *arms, "--datasets", dataset, "--seeds", *seeds]
    return run(command, cores, paths)


@app.function(cpu=RACE_CORES, memory=RACE_MEMORY_MB, timeout=HOUR, volumes={str(REMOTE_SIFT): sift})
def race() -> str:
    return run([sys.executable, "race.py", "--round", "remote"], RACE_CORES, {})


@app.function(cpu=SCALE_CORES, memory=SCALE_MEMORY_MB, timeout=3 * HOUR)
def scale() -> str:
    return paper_run(SCALE_DATASET, ["e-means", "ez-means"], [str(seed) for seed in PAPER_SEEDS[:3]], SCALE_CORES)


@app.function(cpu=SCALE_CORES, memory=SCALE_MEMORY_MB, timeout=3 * HOUR)
def paper(dataset: str, arms: list[str]) -> str:
    return paper_run(dataset, arms, [str(seed) for seed in PAPER_SEEDS], SCALE_CORES)


@app.function(image=gpu_image, gpu=GPU_KIND, cpu=GPU_CORES, memory=GPU_MEMORY_MB, timeout=HOUR)
def gpu(datasets: list[str], probes: list[str], seeds: list[str]) -> str:
    paths = fetch(datasets)
    card = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    command = [sys.executable, "gpu.py", "--bf16", "--datasets", *datasets, "--probes", *probes, "--seeds", *seeds]
    return card + run(command, GPU_CORES, paths)


@app.function(cpu=RACE_CORES, memory=RACE_MEMORY_MB, timeout=HOUR, volumes={str(REMOTE_SIFT): sift})
def probe() -> str:
    rows = []
    for contraction in CONTRACTIONS:
        flags = [flag for flag in KERNEL_COMPILE if not flag.startswith("-ffp-contract")]
        KERNEL_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        compile_command = [*flags, f"-ffp-contract={contraction}", str(KERNEL_SOURCE), "-o", str(KERNEL_LIBRARY)]
        subprocess.run(compile_command, check=True)
        for threads, wait in PROBE_SETTINGS:
            environment = os.environ | {"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": "1"}
            if wait != "default":
                environment["OMP_WAIT_POLICY"] = wait
            command = [sys.executable, "kernel_probe.py", f"contract {contraction}"]
            result = subprocess.run(
                command, cwd=REMOTE_EXPERIMENTS, env=environment, capture_output=True, text=True, check=False
            )
            rows.append(result.stdout.strip() or result.stderr[-2000:])
    return "\n".join(rows)


@app.function(cpu=RACE_CORES, memory=RACE_MEMORY_MB, timeout=HOUR)
def machine() -> str:
    flags = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    target = subprocess.run(["gcc", "-march=native", "-Q", "--help=target"], capture_output=True, text=True, check=True)
    chosen = [line.strip() for line in target.stdout.splitlines() if line.strip().startswith(("-march=", "-mtune="))]
    return "\n".join([flags, *chosen])


@app.local_entrypoint()
def main(job: str, dataset: str = "", arms: str = "") -> None:
    if job == "paper":
        print(paper.remote(dataset, arms.split(",")))
        return
    if job == "gpu":
        print(gpu.remote(dataset.split(","), arms.split(","), [str(seed) for seed in PAPER_SEEDS]))
        return
    jobs = {"race": race, "scale": scale, "machine": machine, "probe": probe}
    print(jobs[job].remote())
