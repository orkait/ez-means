from importlib.util import find_spec

GPU_MODULES = ("torch", "triton")

collect_ignore = [] if all(find_spec(name) for name in GPU_MODULES) else ["test_gpu.py"]
