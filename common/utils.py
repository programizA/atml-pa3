"""Small helpers shared by all three tasks: seeding, JSON/JSONL files, timing, GPU memory, sizes."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SEED = 6304


def seed_everything(seed: int = SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_default) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def dir_size_bytes(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def mb(n_bytes) -> float:
    return round(n_bytes / 2**20, 2)


class Timer:
    """Wall-clock timer that synchronises CUDA so GPU work is included."""

    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.elapsed = time.perf_counter() - self.t0


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_peak_memory():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {"peak_allocated_mb": mb(torch.cuda.max_memory_allocated()),
            "peak_reserved_mb": mb(torch.cuda.max_memory_reserved())}


def environment_info() -> dict:
    import platform

    info = {"python": platform.python_version()}
    for mod in ["torch", "transformers", "peft", "lm_eval", "datasets", "numpy", "math_verify"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info
