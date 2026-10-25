"""A tiny two-GPU job runner for Kaggle's "GPU T4 x2" machines.

Each job is a shell command plus a done-check and a list of jobs it depends on. One worker per GPU
repeatedly takes the first job (in list order) whose dependencies are finished, runs it with
CUDA_VISIBLE_DEVICES set to its GPU, and logs output to a file. Finished jobs are detected from
their output files, so re-running the notebook (for example in a new session with earlier results
restored) skips everything already done.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Job:
    name: str
    cmd: str
    done: Callable[[], bool]
    deps: list = field(default_factory=list)
    est_hours: float | None = None      # skip starting if it cannot finish before the deadline
    gpu_pref: int | None = None          # optional: only run on this GPU


def run_jobs(jobs, n_gpus, log_dir, deadline_hours=None, cwd=None, poll=20, echo=True):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    lock = threading.Lock()
    state = {j.name: ("done" if j.done() else "todo") for j in jobs}
    by_name = {j.name: j for j in jobs}

    def pick(gpu):
        with lock:
            for j in jobs:
                if state[j.name] != "todo" or (j.gpu_pref is not None and j.gpu_pref != gpu and n_gpus > 1):
                    continue
                if any(state[d] == "failed" for d in j.deps):
                    state[j.name] = "failed"
                    continue
                if all(state[d] == "done" for d in j.deps):
                    if deadline_hours and j.est_hours:
                        left = deadline_hours - (time.time() - t_start) / 3600
                        if j.est_hours > left:
                            state[j.name] = "deferred"
                            print(f"[lanes] deferring {j.name}: needs ~{j.est_hours:.1f}h, {left:.1f}h left", flush=True)
                            continue
                    state[j.name] = "running"
                    return j
        return None

    def worker(gpu):
        while True:
            j = pick(gpu)
            if j is None:
                with lock:
                    pending = [n for n, s in state.items() if s in ("todo", "running")]
                if not pending:
                    return
                time.sleep(poll)
                continue
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            log = log_dir / f"{j.name}.log"
            t0 = time.time()
            if echo:
                print(f"[lanes] GPU{gpu} start {j.name}", flush=True)
            with open(log, "a") as f:
                f.write(f"\n=== {time.ctime()} GPU{gpu}: {j.cmd}\n")
                f.flush()
                rc = subprocess.run(j.cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode
            ok = rc == 0 and j.done()
            with lock:
                state[j.name] = "done" if ok else "failed"
            if echo:
                tail = "" if ok else " -- see " + str(log) + "\n" + _tail(log)
                print(f"[lanes] GPU{gpu} {'done' if ok else 'FAILED'} {j.name} in {(time.time()-t0)/60:.1f} min{tail}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(max(1, n_gpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return state


def _tail(path, n=15):
    try:
        return "".join(open(path).readlines()[-n:])
    except OSError:
        return ""


def progress(log_dir, pattern="update"):
    """Last progress line of every job log (for a quick status cell)."""
    out = {}
    for f in sorted(Path(log_dir).glob("*.log")):
        lines = [l.strip() for l in open(f, errors="ignore") if l.strip()]
        hit = [l for l in lines if pattern in l]
        out[f.stem] = (hit or lines or [""])[-1][:160]
    return out
