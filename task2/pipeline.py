"""Task 2 GPU jobs: feature caching -> training -> final evaluation (+ optional 49-token run).

    python -m task2.pipeline --gpus 2 [--optional]
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from common.lanes import Job, run_jobs


def build_jobs(a):
    py = shlex.quote(sys.executable)
    root = Path(a.root)
    data, cache = root / "data", Path(a.cache)
    common = f"--lm {a.lm} --data {data} --cache {cache}"
    jobs = [Job("features", f"{py} -m task2.features --data {data} --images {a.images} --cache {cache} {a.vision_arg}",
                done=lambda: (cache / "keys.txt").exists())]
    runs = [("main", root / "run", "")]
    if a.optional:
        runs.append(("pool49", root / "run_pool49", "--pool"))
    for name, rdir, flag in runs:
        jobs.append(Job(f"train_{name}", f"{py} -m task2.train {common} --out {rdir} {flag}",
                        done=lambda r=rdir: (r / "DONE").exists(), deps=["features"],
                        gpu_pref=0 if name == "main" else 1))
        jobs.append(Job(f"eval_{name}", f"{py} -m task2.evaluate {common} --run {rdir} {flag}",
                        done=lambda r=rdir: (r / "test_summary.json").exists(), deps=[f"train_{name}"],
                        gpu_pref=0 if name == "main" else 1))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task2")
    ap.add_argument("--lm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vision_arg", default="")
    ap.add_argument("--images", default="/tmp/pa3_images")
    ap.add_argument("--cache", default="results/task2/cache")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--optional", action="store_true")
    ap.add_argument("--deadline_hours", type=float, default=None)
    a = ap.parse_args()
    state = run_jobs(build_jobs(a), a.gpus, Path(a.root) / "logs", deadline_hours=a.deadline_hours)
    print(state)
    return state


if __name__ == "__main__":
    main()
