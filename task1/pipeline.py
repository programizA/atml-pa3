"""All Task 1 GPU work as one job list, run across the available GPUs by common.lanes.

    python -m task1.pipeline --gpus 2 --deadline_hours 11.3

Jobs: five training runs, seven evaluation conditions (benchmarks + qualitative generation),
then the weight/activation diagnostics. Each job is skipped if its outputs already exist.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from common.lanes import Job, run_jobs
from common.utils import read_json
from task1 import config as C


def eval_done(eval_dir, qual_dir, cond):
    f = Path(eval_dir) / cond / "summary.json"
    if not f.exists() or not (Path(qual_dir) / f"{cond}.jsonl").exists():
        return False
    s = read_json(f)
    return "gsm8k" in s and (Path(eval_dir) / cond / "likelihood_samples.jsonl").exists()


def estimates(root):
    """Hours per job from the speed probe / finished runs, if known."""
    est = {}
    sp = Path(root) / "speed_probe.json"
    if sp.exists():
        for m, secs in read_json(sp).get("sec_per_update", {}).items():
            est[f"train_{m}"] = secs * C.TOTAL_UPDATES / 3600 * 1.05 + 0.05
    if "train_full" in est:
        est.setdefault("train_l2sp", est["train_full"] * 1.1)
    for m in C.METHODS:
        r = Path(root) / "runs" / m / "resources.json"
        if r.exists():
            s = read_json(r)
            if s.get("updates_done"):
                est[f"train_{m}"] = s["train_seconds"] / s["updates_done"] * C.TOTAL_UPDATES / 3600 * 1.05 + 0.05
    secs = []
    for c, _, _ in C.CONDITIONS:
        f = Path(root) / "eval" / c / "summary.json"
        if f.exists():
            s = read_json(f)
            t = s.get("gsm8k", {}).get("seconds", 0) + s.get("likelihood_seconds", 0) + s.get("qualitative", {}).get("seconds", 0)
            if t:
                secs.append(t)
    for c, _, _ in C.CONDITIONS:
        # unmeasured: assume ~0.9 h (GSM8K generation + 3 likelihood benchmarks + 24 responses on a T4)
        est.setdefault(f"eval_{c}", max(secs) / 3600 * 1.1 + 0.1 if secs else 0.9)
    est.setdefault("diagnostics", 0.25)
    return est


def build_jobs(a):
    py = shlex.quote(sys.executable)
    root = Path(a.root)
    runs, ev, qual, data = root / "runs", root / "eval", root / "qualitative", root / "data"
    est = estimates(root)
    jobs = []
    order = ["oplora", "full", "lora16", "l2sp", "lora4"]   # roughly longest first (speed probe)
    for m in order:
        jobs.append(Job(f"train_{m}",
                        f"{py} -m task1.train --method {m} --model {a.model} --data {data} --out {runs} "
                        f"--ckpt_dir {a.ckpt_dir} {a.train_extra}",
                        done=lambda m=m: (runs / m / "DONE").exists(), est_hours=est.get(f"train_{m}")))
    for c, run, _ in C.CONDITIONS:
        jobs.append(Job(f"eval_{c}",
                        f"{py} -m task1.evaluate --condition {c} --model {a.model} --runs {runs} --out {ev} "
                        f"--probe {data / 'probe_set.json'} --qual_yaml {data / 'first_plot_questions.yaml'} "
                        f"--qual_out {qual} --tmp {a.tmp} {a.eval_extra}",
                        done=lambda c=c: eval_done(ev, qual, c),
                        deps=[f"train_{run}"] if run else [], est_hours=est.get(f"eval_{c}")))
    jobs.append(Job("diagnostics",
                    f"{py} -m task1.diagnostics --model {a.model} --runs {runs} --probe {data / 'probe_set.json'} "
                    f"--out {root / 'diagnostics'} {a.diag_extra}",
                    done=lambda: (root / "diagnostics" / "diagnostics.json").exists()
                    and len(read_json(root / "diagnostics" / "diagnostics.json")["conditions"]) == 6,
                    deps=[f"train_{m}" for m in order], est_hours=est.get("diagnostics")))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task1")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--ckpt_dir", default="/tmp/pa3_recovery")
    ap.add_argument("--tmp", default="/tmp/pa3_merged")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--only", default=None, help="comma-separated job names")
    ap.add_argument("--train_extra", default="")
    ap.add_argument("--eval_extra", default="")
    ap.add_argument("--diag_extra", default="")
    a = ap.parse_args()
    jobs = build_jobs(a)
    if a.only:
        keep = set(a.only.split(","))
        for j in jobs:
            j.deps = [d for d in j.deps if d in keep]
        jobs = [j for j in jobs if j.name in keep]
    state = run_jobs(jobs, a.gpus, Path(a.root) / "logs", deadline_hours=a.deadline_hours)
    print({k: v for k, v in state.items()})
    return state


if __name__ == "__main__":
    main()
