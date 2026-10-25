"""Time a few real optimizer updates per method before committing GPU hours.

    python -m task1.speed_probe --updates 12

Runs each method for a few updates into a scratch folder (deleted afterwards), and records seconds
per update and peak GPU memory. The pipeline uses these numbers to decide whether a training job
can still finish before the Kaggle session ends. The probe's updates use the real first batches
and settings, so the memory peak is representative of training.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.utils import read_json, read_jsonl, write_json
from task1 import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task1")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--methods", default="full,l2sp,lora16,oplora")
    ap.add_argument("--updates", type=int, default=12)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    root = Path(a.root)
    scratch = Path("/tmp/pa3_speed_probe")
    out = {"updates": a.updates, "sec_per_update": {}, "peak_allocated_gb": {}, "notes": {}}
    for m in a.methods.split(","):
        shutil.rmtree(scratch, ignore_errors=True)
        cmd = [sys.executable, "-m", "task1.train", "--method", m, "--model", a.model, "--data",
               str(root / "data"), "--out", str(scratch), "--stop_after", str(a.updates)] + a.extra.split()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            out["notes"][m] = "FAILED: " + r.stdout[-1500:] + r.stderr[-1500:]
            print(m, "FAILED\n", r.stderr[-3000:], flush=True)
            continue
        log = read_jsonl(scratch / m / "train_log.jsonl")
        secs = [x["seconds"] for x in log[2:]] or [x["seconds"] for x in log]
        res = read_json(scratch / m / "resources.json")
        out["sec_per_update"][m] = float(np.median(secs))
        pk = res.get("peak_memory_training", {}).get("peak_allocated_mb")
        out["peak_allocated_gb"][m] = pk / 1024 if pk else None
        out["notes"][m] = {"skipped": res.get("skipped_updates"), "first_losses": [round(x["sft_loss"], 4) for x in log[:3]]}
        print(f"{m}: {out['sec_per_update'][m]:.2f} s/update -> ~{out['sec_per_update'][m]*C.TOTAL_UPDATES/3600:.2f} h "
              f"for 1,250 updates; peak {out['peak_allocated_gb'][m]} GB", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    if "lora16" in out["sec_per_update"]:
        out["sec_per_update"].setdefault("lora4", out["sec_per_update"]["lora16"])
    write_json(root / "speed_probe.json", out)


if __name__ == "__main__":
    main()
