"""Step 4: parallel sampling vs one long response vs sequential refinement, generated live.

    python -m task3.live --root results/task3 [--shard 0/2]

Same 50 MATH questions (the first 50 held-out questions of the seeded permutation), same
3,072-token output allowance per question:

  single      one response, up to 3,072 new tokens
  parallel    three independent responses, up to 1,024 each; self-consistency picks the answer
  refinement  the first parallel response (reused, up to 1,024) + one critique-and-revision
              response (up to 2,048) to the fixed review instruction; the revised answer is returned

Qwen2.5-3B-Instruct at the README's pinned revision, FP16, batch size 1, temperature 0.7,
top-p 0.8, top-k 20. Every call has its own seed. Each call's end-to-end time excludes model
loading and includes tokenisation, generation and decoding.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from common.utils import append_jsonl, environment_info, read_json, read_jsonl, write_json
from task3 import config as C
from task3.answers import NO_ANSWER, extract_boxed, grade_math, group_math

CALLS = [("single", 0, C.LIVE_LIMITS["single"]), ("parallel", 1, C.LIVE_LIMITS["parallel"]),
         ("parallel", 2, C.LIVE_LIMITS["parallel"]), ("parallel", 3, C.LIVE_LIMITS["parallel"]),
         ("revision", 4, C.LIVE_LIMITS["revision"])]


def seed_for(q_index, slot):
    return C.SEED * 1000 + q_index * 10 + slot


def first_messages(problem):
    return [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": problem + "\n\n" + C.INSTRUCTION}]


@torch.no_grad()
def generate(model, tok, messages, max_new, seed, dev):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, max_new_tokens=max_new,
                         eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[-1], **C.SAMPLING)
    new = out[0, ids.shape[1]:].tolist()
    stop = next((j for j, t in enumerate(new) if t in eos), None)
    kept = new if stop is None else new[:stop]
    resp = tok.decode(kept, skip_special_tokens=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
    return {"prompt": text, "response": resp, "input_tokens": int(ids.shape[1]), "output_tokens": len(kept),
            "truncated": stop is None and len(new) >= max_new, "seconds": sec, "peak_memory_mb": peak,
            "max_new": max_new, "seed": seed}


def run(a):
    root = Path(a.root)
    ids = read_json(root / "live_questions.json")["ids"]
    texts = pd.read_parquet(root / "math" / "texts.parquet")
    roll = pd.read_parquet(root / "math" / "rollouts.parquet")
    prob = texts.groupby("sample_id").question_text.first()
    gold = roll.groupby("sample_id").gold.first()
    shard, n_shards = (int(x) for x in a.shard.split("/"))
    out = root / "live" / f"calls_shard{shard}.jsonl"
    done = {(r["sample_id"], r["slot"]) for r in read_jsonl(out)} if out.exists() else set()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    kw = {"revision": a.revision} if a.revision else {}
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                                                 **kw).to(dev).eval()
    write_json(root / "live" / f"setup_shard{shard}.json", {"model": a.model, "revision": a.revision,
                                                            "load_seconds": time.perf_counter() - t0,
                                                            "dtype": str(model.dtype), "env": environment_info()})
    for qi, qid in enumerate(ids):
        if qi % n_shards != shard:
            continue
        msgs = first_messages(prob[qid])
        initial = None
        for strategy, slot, max_new in CALLS:
            if (qid, slot) in done:
                if slot == 1:
                    initial = next(r for r in read_jsonl(out) if r["sample_id"] == qid and r["slot"] == 1)["response"]
                continue
            m = msgs if slot < 4 else msgs + [{"role": "assistant", "content": initial},
                                              {"role": "user", "content": C.REVIEW}]
            r = generate(model, tok, m, max_new if not a.max_new_override else min(max_new, a.max_new_override),
                         seed_for(qi, slot), dev)
            if slot == 1:
                initial = r["response"]
            ans = extract_boxed(r["response"])
            append_jsonl(out, {"sample_id": qid, "q_index": qi, "strategy": strategy, "slot": slot, **r,
                               "extracted": ans, "correct": grade_math(gold[qid], ans) if ans else False,
                               "gold": gold[qid]})
            print(f"q{qi} {strategy}/{slot}: {r['output_tokens']} tok, {r['seconds']:.1f}s, ans={ans!r}", flush=True)


def summarize(root):
    root = Path(root)
    calls = pd.DataFrame([r for f in sorted((root / "live").glob("calls_shard*.jsonl")) for r in read_jsonl(f)])
    if calls.empty:
        return None
    per_q = []
    for qid, sub in calls.groupby("sample_id"):
        s = {r["slot"]: r for _, r in sub.iterrows()}
        if len(s) < 5:
            continue
        par = [s[k] for k in (1, 2, 3)]
        labels, reps = group_math([p["extracted"] for p in par])
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        best = max(counts.values())
        win = next(l for l in labels if counts[l] == best)          # earliest supporter breaks ties
        sc_ans = None if win == NO_ANSWER else reps[win]
        sc_correct = bool(sc_ans is not None and grade_math(s[1]["gold"], sc_ans))
        init_c, rev_c = bool(s[1]["correct"]), bool(s[4]["correct"])
        per_q.append({
            "sample_id": qid, "gold": s[1]["gold"],
            "single_correct": bool(s[0]["correct"]), "single_out": s[0]["output_tokens"], "single_in": s[0]["input_tokens"],
            "single_sec": s[0]["seconds"], "single_trunc": s[0]["truncated"], "single_extracted": s[0]["extracted"] is not None,
            "parallel_correct": sc_correct, "parallel_answer": sc_ans, "parallel_out": sum(p["output_tokens"] for p in par),
            "parallel_in": sum(p["input_tokens"] for p in par), "parallel_sec": sum(p["seconds"] for p in par),
            "parallel_trunc_rate": sum(p["truncated"] for p in par) / 3,
            "parallel_extract_rate": sum(p["extracted"] is not None for p in par) / 3,
            "parallel_distinct": len(set(labels)),
            "refine_correct": rev_c, "initial_correct": init_c,
            "refine_out": s[1]["output_tokens"] + s[4]["output_tokens"],
            "refine_in": s[1]["input_tokens"] + s[4]["input_tokens"], "refine_sec": s[1]["seconds"] + s[4]["seconds"],
            "revision_in": s[4]["input_tokens"], "revision_out": s[4]["output_tokens"],
            "refine_trunc_initial": s[1]["truncated"], "refine_trunc_revision": s[4]["truncated"],
            "refine_extracted": s[4]["extracted"] is not None,
            "transition": ("W->C" if (not init_c and rev_c) else "C->W" if (init_c and not rev_c) else
                           "C->C" if init_c else "W->W"),
            "answer_changed": (s[1]["extracted"] or "") != (s[4]["extracted"] or ""),
            "peak_mb": max(r["peak_memory_mb"] or 0 for r in s.values())})
    q = pd.DataFrame(per_q)
    q.to_csv(root / "live" / "live_per_question.csv", index=False)
    summ = {"n_questions": len(q)}
    for st, pre in (("single", "single"), ("parallel", "parallel"), ("refinement", "refine")):
        summ[st] = {"accuracy": float(q[f"{pre}_correct"].mean()), "mean_output_tokens": float(q[f"{pre}_out"].mean()),
                    "mean_input_tokens": float(q[f"{pre}_in"].mean()), "mean_seconds": float(q[f"{pre}_sec"].mean()),
                    "total_seconds": float(q[f"{pre}_sec"].sum())}
    summ["single"].update({"truncation_rate": float(q.single_trunc.mean()), "extraction_rate": float(q.single_extracted.mean())})
    summ["parallel"].update({"truncation_rate_per_response": float(q.parallel_trunc_rate.mean()),
                             "extraction_rate_per_response": float(q.parallel_extract_rate.mean()),
                             "mean_distinct_answers": float(q.parallel_distinct.mean())})
    summ["refinement"].update({"initial_accuracy": float(q.initial_correct.mean()),
                               "truncation_rate_initial": float(q.refine_trunc_initial.mean()),
                               "truncation_rate_revision": float(q.refine_trunc_revision.mean()),
                               "extraction_rate_revision": float(q.refine_extracted.mean()),
                               "transitions": q.transition.value_counts().to_dict(),
                               "answer_changed_rate": float(q.answer_changed.mean()),
                               "mean_revision_input_tokens": float(q.revision_in.mean())})
    summ["peak_memory_mb"] = float(q.peak_mb.max())
    write_json(root / "live" / "live_summary.json", summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task3")
    ap.add_argument("--model", default=C.LIVE_MODEL)
    ap.add_argument("--revision", default=C.LIVE_REVISION)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_new_override", type=int, default=None, help="tests only")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.root))
    else:
        run(a)


if __name__ == "__main__":
    main()
