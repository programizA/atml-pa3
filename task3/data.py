"""Load, join and verify the supplied rollouts; build answer groups, correctness labels, per-rollout
scores, splits and the ten candidate orders.

    python -m task3.data --raw <ATML-PA3-Data folder> --meta <MathVista testmini.json> --out results/task3

Per rollout i (T_i retained tokens; stop tokens are already excluded):
    L_i = mean_t logp_t                 mean log-probability of the sampled tokens
    K_i = mean_t kl_u_p_t               self-certainty: mean KL(uniform || p_t)
    V_i = mean_t v_t (MathVista)        visual engagement, v_t = A_img / (A_img + A_prompt + A_gen)
Token-level kl and v are kept for VisConf, which must be calibrated on each candidate subset.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common.utils import read_json, write_json
from task3 import config as C
from task3.answers import (NO_ANSWER, gold_mathvista, grade_math, group_math, missing, normalize_mathvista,
                           scorer_version)

KEYS = ["dataset", "sample_id", "rollout_id"]


def load_joined(folder, vlm):
    folder = Path(folder)
    g = pq.read_table(folder / "generations.parquet").to_pandas()
    p = pq.read_table(folder / "probability_metrics.parquet").to_pandas()
    df = g.merge(p, on=KEYS, how="inner", validate="one_to_one")
    if vlm:
        a = pq.read_table(folder / "attention_metrics.parquet").to_pandas()
        a = a.rename(columns={"n_generated_text_tokens": "attn_prev_counts"})
        df = df.merge(a, on=KEYS, how="inner", validate="one_to_one")
    assert len(df) == len(g) == len(p), "join lost rows"
    df = df.sort_values(["sample_id", "rollout_id"], kind="stable").reset_index(drop=True)
    return df


def verify_alignment(df, vlm):
    """Every per-token list must have one entry per retained generated token."""
    n = df["n_generated_text_tokens"].to_numpy()
    rep = {"rows": len(df), "questions": df.sample_id.nunique(),
           "rollouts_per_question": sorted(df.groupby("sample_id").size().unique().tolist())}
    checks = {"token_ids": df.generated_token_ids.map(len).to_numpy() == n,
              "logp": df.logp.map(len).to_numpy() == n, "kl_u_p": df.kl_u_p.map(len).to_numpy() == n}
    if vlm:
        for c in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"):
            checks[c] = df[c].map(len).to_numpy() == n
        checks["attn_prev_counts==0..N-1"] = np.array([np.array_equal(np.asarray(x), np.arange(len(x)))
                                                        for x in df.attn_prev_counts])
        checks["first_gen_attention==0"] = np.array([len(x) == 0 or x[0] == 0 for x in df.generated_text_attn_total])
    rep["mismatches"] = {k: int((~v).sum()) for k, v in checks.items()}
    rep["empty_rollouts"] = int((n == 0).sum())
    rep["capped"] = int(df.hit_max_new_tokens.sum())
    rep["null_extractions"] = int(df.extracted_answer.map(missing).sum())
    rep["retained_tokens"] = int(n.sum())
    rep["nonfinite_logp"] = int(sum((~np.isfinite(np.asarray(x))).sum() for x in df.logp))
    rep["nonfinite_kl"] = int(sum((~np.isfinite(np.asarray(x))).sum() for x in df.kl_u_p))
    return rep


def splits(ids, seed=C.SEED):
    """Sort IDs as strings, permute with default_rng(6304): first 200 dev, remaining 800 eval."""
    s = sorted(ids)
    perm = np.random.default_rng(seed).permutation(len(s))
    order = [s[i] for i in perm]
    return {"dev": order[:C.N_DEV], "eval": order[C.N_DEV:], "permuted": order}


def orders(ids, n_roll=32, seed=C.SEED):
    """Ten seeded permutations of the 32 rollout IDs per question (questions in sorted-ID order)."""
    rng = np.random.default_rng(seed)
    return {q: np.stack([rng.permutation(n_roll) for _ in range(C.N_ORDERS)]) for q in sorted(ids)}


def build(name, folder, meta, out):
    vlm = name == "mathvista"
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df = load_joined(folder, vlm)
    rep = verify_alignment(df, vlm)
    rep["load_seconds"] = time.perf_counter() - t0
    print(name, rep, flush=True)

    # token arrays (concatenated, with offsets)
    n = df["n_generated_text_tokens"].to_numpy()
    offsets = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    kl = np.concatenate([np.asarray(x, dtype=np.float32) for x in df.kl_u_p]) if n.sum() else np.zeros(0, np.float32)
    lp = np.concatenate([np.asarray(x, dtype=np.float32) for x in df.logp])
    arrays = {"offsets": offsets, "kl": kl, "logp": lp}
    if vlm:
        img = np.concatenate([np.asarray(x, dtype=np.float64) for x in df.img_attn_total])
        pr = np.concatenate([np.asarray(x, dtype=np.float64) for x in df.prompt_text_attn_total])
        ge = np.concatenate([np.asarray(x, dtype=np.float64) for x in df.generated_text_attn_total])
        den = img + pr + ge
        arrays["v"] = np.where(den > 0, img / np.where(den > 0, den, 1), 0.0).astype(np.float32)
        rep["v_zero_denominator_tokens"] = int((den <= 0).sum())
    np.savez(out / "tokens.npz", **arrays)

    def seg_mean(x):
        s = np.add.reduceat(x.astype(np.float64), offsets[:-1]) if len(x) else np.zeros(len(n))
        s = np.where(n > 0, s, np.nan)
        return s / np.maximum(n, 1)

    roll = pd.DataFrame({"sample_id": df.sample_id, "rollout_id": df.rollout_id, "T": n,
                         "hit_max": df.hit_max_new_tokens.astype(bool),
                         "extracted": [None if missing(x) else str(x) for x in df.extracted_answer],
                         "gold": df.gold_answer, "L": seg_mean(lp), "K": seg_mean(kl)})
    if vlm:
        roll["V"] = seg_mean(arrays["v"])

    # answer groups (without gold) and correctness (with gold), per question
    t1 = time.perf_counter()
    groups, correct, n_groups = [], [], {}
    for q, sub in roll.groupby("sample_id", sort=False):
        ex = sub.extracted.tolist()
        gold = sub.gold.iloc[0]
        if vlm:
            m = meta[q]
            labs = [normalize_mathvista(e, m) for e in ex]
            gnorm = gold_mathvista(gold, m)
            cor = [l != NO_ANSWER and not l.startswith("INVALID:") and l == gnorm for l in labs]
        else:
            labs, reps = group_math(ex)
            graded = {lab: grade_math(gold, rep) for lab, rep in reps.items()}
            labs = [NO_ANSWER if l == NO_ANSWER else f"{l}:{reps[l]}" for l in labs]
            cor = [False if l == NO_ANSWER else graded[l.split(":", 1)[0]] for l in labs]
        groups += labs
        correct += cor
        n_groups[q] = len(set(labs))
    roll["group"] = groups
    roll["correct"] = np.array(correct, dtype=bool)
    rep["grouping_seconds"] = time.perf_counter() - t1
    rep["scorer"] = scorer_version()
    rep["rollout_accuracy"] = float(roll.correct.mean())
    roll.to_parquet(out / "rollouts.parquet")
    # question texts and generations for qualitative examples (kept out of the hot path)
    df[["sample_id", "rollout_id", "question_text", "generated_text"]].to_parquet(out / "texts.parquet")
    sp = splits(roll.sample_id.unique().tolist())
    write_json(out / "splits.json", sp)
    od = orders(roll.sample_id.unique().tolist())
    np.savez(out / "orders.npz", ids=np.array(sorted(od)), orders=np.stack([od[q] for q in sorted(od)]))
    write_json(out / "data_report.json", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True, help="MathVista data/testmini.json")
    ap.add_argument("--out", default="results/task3")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    meta = read_json(a.meta)
    for name, folder in C.DATASETS.items():
        if a.only and name != a.only:
            continue
        build(name, Path(a.raw) / folder, meta, a.out)
    sp = read_json(Path(a.out) / "math" / "splits.json") if (Path(a.out) / "math" / "splits.json").exists() else None
    if sp:
        write_json(Path(a.out) / "live_questions.json", {"ids": sp["eval"][:C.N_LIVE]})


if __name__ == "__main__":
    main()
