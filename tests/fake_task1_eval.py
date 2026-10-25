"""Synthetic per-example eval outputs for all seven conditions, to exercise task1.analyze."""
import sys
from pathlib import Path
import numpy as np
from common.utils import write_json, write_jsonl
from task1 import config as C

root = Path(sys.argv[1])
rng = np.random.default_rng(0)
N = {"gsm8k": 1319, "hellaswag": 1000, "boolq": 600, "arc_easy": 500}
base_p = {"gsm8k": .55, "hellaswag": .47, "boolq": .64, "arc_easy": .56}
shift = {"baseline": (0, 0), "full_mid": (.08, -.01), "full": (.10, -.02), "l2sp": (.10, -.015),
         "lora4": (.05, -.003), "lora16": (.07, -.006), "oplora": (.065, -.004)}
lat = {b: rng.random(N[b]) for b in N}
for cond, _, _ in C.CONDITIONS:
    d = root / "eval" / cond; d.mkdir(parents=True, exist_ok=True)
    gm, rt = shift[cond]
    corr = {b: (lat[b] < base_p[b] + (gm if b == "gsm8k" else rt)).astype(float) for b in N}
    write_jsonl(d / "gsm8k_samples.jsonl", [{"doc_id": i, "correct_flexible-extract": c} for i, c in enumerate(corr["gsm8k"])])
    rows = []
    for b in ("hellaswag", "boolq", "arc_easy"):
        rows += [{"task": b, "doc_id": i, "acc": c, "acc_norm": c} for i, c in enumerate(corr[b])]
    write_jsonl(d / "likelihood_samples.jsonl", rows)
    s = {"condition": cond, "gsm8k": {"flexible": corr["gsm8k"].mean(), "strict": corr["gsm8k"].mean() - .1,
         "truncated_rate": .01, "unextractable_flexible_rate": .02, "mean_gen_tokens": 210}}
    for b in ("hellaswag", "boolq", "arc_easy"):
        s[b] = {"acc": corr[b].mean() - .02, "acc_norm": corr[b].mean()}
    if cond in ("lora4", "lora16", "oplora"):
        s["merge_seconds"] = 12.0
    write_json(d / "summary.json", s)
