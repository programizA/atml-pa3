"""GSM8K extraction audit: separate changes in answer *format* from changes in answer *correctness*.

The assignment's primary GSM8K metric is lm-eval's "flexible-extract": the last number-like
string in the response, matched by the regex  (-?[$0-9.,]{2,})|(-?[0-9]+). That regex also matches
"$$", the LaTeX display-maths delimiter, and keeps "$12.00" as written. The untuned Qwen3 answers in
LaTeX and often ends with  \\boxed{90} $$ , so its correct answers are frequently read as "$$".
After MetaMathQA fine-tuning every model ends with MetaMath's "The answer is: 90", which the regex
reads correctly.

As a diagnostic (not a replacement for the prescribed metric) we re-read every saved response with
an answer-aware rule: the last \\boxed{...} if present, else the last "The answer is: X", else the
last plain number; commas and "$" dropped; compared numerically with the gold answer. The gap
between the two scores is what formatting alone does to the measured accuracy.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import read_jsonl
from task1 import config as C


def _last_number(s):
    if s is None:
        return None
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", ""))
    return m[-1] if m else None


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def answer_aware(text):
    b = last_boxed(text)
    if b is not None:
        return _last_number(b)
    m = re.findall(r"The answer is:?\s*\$?(-?[\d,\.]+)", text)
    return _last_number(m[-1]) if m else _last_number(text)


def same_number(a, gold):
    try:
        return a is not None and abs(float(a) - float(str(gold).replace(",", ""))) < 1e-6
    except ValueError:
        return False


def audit(eval_dir, out_dir, n_boot=4000, seed=C.SEED):
    per, rows = {}, []
    for cond, _, _ in C.CONDITIONS:
        f = Path(eval_dir) / cond / "gsm8k_samples.jsonl"
        if not f.exists():
            continue
        r = pd.DataFrame(read_jsonl(f)).sort_values("doc_id")
        aware = r.response.map(answer_aware)
        ok = np.array([same_number(a, g) for a, g in zip(aware, r.target)], dtype=float)
        flex = r["correct_flexible-extract"].to_numpy(dtype=float)
        per[cond] = (flex, ok)
        fe = r["answer_flexible-extract"].astype(str)
        rows.append({"condition": cond, "label": C.LABELS[cond], "flexible": flex.mean(), "answer_aware": ok.mean(),
                     "format_gap_pp": (ok.mean() - flex.mean()) * 100,
                     "aware_right_flex_wrong": int(((ok == 1) & (flex == 0)).sum()),
                     "flex_right_aware_wrong": int(((ok == 0) & (flex == 1)).sum()),
                     "flex_read_$$": float((fe == "$$").mean()),
                     "uses_boxed": float(r.response.map(lambda t: last_boxed(t) is not None).mean()),
                     "uses_answer_is": float(r.response.str.contains(r"The answer is:?", regex=True).mean()),
                     "mean_gen_tokens": float(r.gen_tokens.mean()), "truncated": float(r.truncated.mean())})
    df = pd.DataFrame(rows)
    if "baseline" in per:
        rng = np.random.default_rng(seed)
        n = len(per["baseline"][0])
        idx = rng.integers(0, n, size=(n_boot, n))
        for col, k in (("flexible", 0), ("answer_aware", 1)):
            lo, hi, mid = [], [], []
            for cond in df.condition:
                d = (per[cond][k] - per["baseline"][k]) * 100
                bs = d[idx].mean(1)
                mid.append(d.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            df[f"{col}_dpp"], df[f"{col}_lo"], df[f"{col}_hi"] = mid, lo, hi
    df.to_csv(Path(out_dir) / "gsm8k_extraction_audit.csv", index=False)
    return df
