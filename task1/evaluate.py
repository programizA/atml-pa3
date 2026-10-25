"""Benchmark one evaluation condition with lm-evaluation-harness.

    python -m task1.evaluate --condition lora16

Steps:
  1. Get an FP16 Hugging Face model directory for the condition. Baseline = the original
     checkpoint; Full FT variants = the saved FP16 model; LoRA/OPLoRA = the adapter merged into a
     fresh copy of the backbone (projected factors for OPLoRA), after checking that merged and
     unmerged logits agree on a small batch.
  2. GSM8K (gsm8k_cot, zero-shot): generation with the non-thinking chat template, greedy,
     at most 1,024 new tokens, stock stop strings and answer filters.
  3. HellaSwag, BoolQ, ARC-Easy: zero-shot likelihood scoring with stock prompts, no chat template.
  4. Save aggregate scores and per-example outputs, including extracted GSM8K answers and a
     truncation flag (the response used the whole 1,024-token budget).
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from common.utils import Timer, peak_memory_mb, read_json, reset_peak_memory, write_json, write_jsonl
from task1 import config as C

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


def condition_spec(name, runs_dir=None):
    """(run, update) for a condition. The update comes from the run's own config, so the
    smoke test's shortened schedule resolves the same way as the real one."""
    for n, run, upd in C.CONDITIONS:
        if n == name:
            if run is not None and runs_dir and (Path(runs_dir) / run / "run_config.json").exists():
                ev = read_json(Path(runs_dir) / run / "run_config.json")["eval_at"]
                upd = ev[0] if upd == C.MIDPOINT_UPDATE else ev[-1]
            return run, upd
    raise KeyError(name)


@torch.no_grad()
def merge_check(model_id, adapter_path, merged, dev, n_prompts=4, probe=None):
    """Compare logits of (FP16 backbone + adapter modules) with the merged FP16 model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from task1.methods import add_adapters, load_adapter_state

    blob = torch.load(adapter_path, map_location="cpu")
    meta = blob["meta"]
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    un = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev)
    add_adapters(un, meta["r"], meta["alpha"], protect_k=meta["k"] if meta["projected"] else None)
    load_adapter_state(un, blob["state"])
    un.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    qs = [p["question"] for p in (probe or {}).get("gsm8k", [])[:n_prompts]] or ["What is 12 times 7?"]
    texts = [tok.apply_chat_template([{"role": "user", "content": f"Q: {q}\nA:"}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for q in qs]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(dev)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        a = un(**enc).logits.float()
    b = merged(**enc).logits.float()
    m = enc["attention_mask"].bool()
    d = (a - b).abs()[m]
    out = {"max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
           "top1_agreement": float((a.argmax(-1) == b.argmax(-1))[m].float().mean()),
           "logit_scale_mean_abs": float(a.abs()[m].mean()), "n_tokens": int(m.sum())}
    del un
    return out


def materialize(condition, model_id, runs_dir, tmp_dir, dev, probe=None):
    """Return (model_dir, info) for an FP16 model lm-eval can load."""
    run, upd = condition_spec(condition, runs_dir)
    info = {"condition": condition, "run": run, "update": upd}
    if run is None:
        return model_id, info
    rdir = Path(runs_dir) / run
    if C.METHODS[run]["kind"] == "full":
        return str(rdir / f"model_u{upd}"), info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from task1.methods import merge_into_fresh

    ad = rdir / f"adapter_u{upd}.pt"
    blob = torch.load(ad, map_location="cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    with Timer() as t:
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev)
        merged = merge_into_fresh(base, blob["state"], blob["meta"]["scale"]).eval()
    info["merge_seconds"] = t.elapsed
    info["merge_check"] = merge_check(model_id, ad, merged, dev, probe=probe)
    out = Path(tmp_dir) / f"merged_{condition}"
    shutil.rmtree(out, ignore_errors=True)
    merged.config.torch_dtype = torch.float16
    merged.to(torch.float16).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    del merged, base
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return str(out), info


def run_lm_eval(model_dir, tasks, chat, gen_kwargs, batch_size, include_path=None, limit=None):
    import lm_eval
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=include_path) if include_path else TaskManager()
    margs = {"pretrained": model_dir, "dtype": "float16" if torch.cuda.is_available() else "float32"}
    if chat:
        margs["enable_thinking"] = False
    return lm_eval.simple_evaluate(
        model="hf", model_args=margs, tasks=tasks, num_fewshot=0, batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        apply_chat_template=chat, gen_kwargs=gen_kwargs, log_samples=True, task_manager=tm,
        limit=limit, random_seed=C.SEED, numpy_random_seed=C.SEED, torch_random_seed=C.SEED,
        fewshot_random_seed=C.SEED, bootstrap_iters=0)


def gsm8k_rows(samples, tok, max_new):
    """One row per question: raw response, both extracted answers, correctness, token count and
    truncation flag. lm-eval logs one entry per filter; we merge them by doc_id."""
    rows = {}
    for s in samples:
        r = rows.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"].get("question"),
                                          "target": s["target"]})
        raw = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        r["response"] = raw
        f = s.get("filter", "none")
        r[f"answer_{f}"] = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        r[f"correct_{f}"] = float(s.get("exact_match", 0.0))
    out = []
    for r in sorted(rows.values(), key=lambda x: x["doc_id"]):
        n = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        r["gen_tokens"] = n
        # lm-eval returns text, not token counts, so we re-tokenize the response. A response that
        # used (almost) the whole budget never reached a stop string or EOS: it was cut off.
        r["truncated"] = n >= max_new - 4
        r["extractable_flexible"] = r.get("answer_flexible-extract") not in (None, "[invalid]", "")
        r["extractable_strict"] = r.get("answer_strict-match") not in (None, "[invalid]", "")
        out.append(r)
    return out


def mc_rows(samples):
    out = []
    for s in samples:
        out.append({"doc_id": s["doc_id"], "target": s["target"],
                    "loglik": [float(x[0][0]) if isinstance(x[0], (list, tuple)) else float(x[0])
                               for x in s["resps"]],
                    "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=[c[0] for c in C.CONDITIONS])
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--runs", default="results/task1/runs")
    ap.add_argument("--out", default="results/task1/eval")
    ap.add_argument("--tmp", default="/tmp/pa3_merged")
    ap.add_argument("--probe", default="results/task1/data/probe_set.json")
    ap.add_argument("--gen_batch", default="64")
    ap.add_argument("--ll_batch", default="auto")
    ap.add_argument("--gsm_task", default="gsm8k_cot")
    ap.add_argument("--ll_tasks", default="hellaswag,boolq,arc_easy")
    ap.add_argument("--include_path", default=None, help="extra task YAMLs (tests only)")
    ap.add_argument("--limit", type=float, default=None, help="tests only")
    ap.add_argument("--max_new", type=int, default=C.GSM8K_MAX_NEW)
    ap.add_argument("--only", choices=["gsm8k", "likelihood", "qualitative"], default=None)
    ap.add_argument("--qual_yaml", default="results/task1/data/first_plot_questions.yaml")
    ap.add_argument("--qual_out", default="results/task1/qualitative")
    ap.add_argument("--qual_max_new", type=int, default=C.QUAL_MAX_NEW)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.condition
    out.mkdir(parents=True, exist_ok=True)
    probe = read_json(a.probe) if Path(a.probe).exists() else None
    summary = read_json(out / "summary.json") if (out / "summary.json").exists() else {"condition": a.condition}

    need_gsm = a.only in (None, "gsm8k") and not (out / "gsm8k_samples.jsonl").exists()
    need_ll = a.only in (None, "likelihood") and not (out / "likelihood_samples.jsonl").exists()
    qual_file = Path(a.qual_out) / f"{a.condition}.jsonl"
    need_qual = a.only in (None, "qualitative") and not qual_file.exists()
    if not (need_gsm or need_ll or need_qual):
        print(f"[{a.condition}] already evaluated", flush=True)
        return
    model_dir, info = materialize(a.condition, a.model, a.runs, a.tmp, dev, probe)
    summary.update(info)
    summary["model_dir"] = model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)

    if need_gsm:
        reset_peak_memory()
        t0 = time.perf_counter()
        res = run_lm_eval(model_dir, [a.gsm_task], chat=True, batch_size=a.gen_batch,
                          gen_kwargs={"max_gen_toks": a.max_new, "do_sample": False},
                          include_path=a.include_path, limit=a.limit)
        rows = gsm8k_rows(res["samples"][a.gsm_task], tok, a.max_new)
        write_jsonl(out / "gsm8k_samples.jsonl", rows)
        r = res["results"][a.gsm_task]
        summary["gsm8k"] = {
            "flexible": r.get("exact_match,flexible-extract"), "strict": r.get("exact_match,strict-match"),
            "n": len(rows), "truncated_rate": sum(x["truncated"] for x in rows) / len(rows),
            "unextractable_flexible_rate": 1 - sum(x["extractable_flexible"] for x in rows) / len(rows),
            "unextractable_strict_rate": 1 - sum(x["extractable_strict"] for x in rows) / len(rows),
            "mean_gen_tokens": sum(x["gen_tokens"] for x in rows) / len(rows),
            "seconds": time.perf_counter() - t0, **peak_memory_mb()}
        summary["lm_eval_config_gsm8k"] = {k: res["config"].get(k) for k in ("model_args", "batch_size", "gen_kwargs")}
        summary["task_versions"] = {**summary.get("task_versions", {}), **res.get("versions", {})}
        write_json(out / "summary.json", summary)

    if need_ll:
        tasks = a.ll_tasks.split(",")
        reset_peak_memory()
        t0 = time.perf_counter()
        res = run_lm_eval(model_dir, tasks, chat=False, batch_size=a.ll_batch, gen_kwargs=None,
                          include_path=a.include_path, limit=a.limit)
        allrows = []
        for t in tasks:
            r = res["results"][t]
            summary[t] = {"acc": r.get("acc,none"), "acc_norm": r.get("acc_norm,none"),
                          "n": len(res["samples"][t])}
            allrows += [{"task": t, **x} for x in mc_rows(res["samples"][t])]
        write_jsonl(out / "likelihood_samples.jsonl", allrows)
        summary["likelihood_seconds"] = time.perf_counter() - t0
        summary["likelihood_peak_memory"] = peak_memory_mb()
        summary["task_versions"] = {**summary.get("task_versions", {}), **res.get("versions", {})}
        write_json(out / "summary.json", summary)

    if need_qual:
        from task1.qualitative import generate

        t0 = time.perf_counter()
        n = generate(model_dir, a.condition, a.qual_yaml, qual_file, max_new=a.qual_max_new)
        summary["qualitative"] = {"n": n, "seconds": time.perf_counter() - t0}
        write_json(out / "summary.json", summary)

    if model_dir.startswith(a.tmp):
        shutil.rmtree(model_dir, ignore_errors=True)
    print(f"[{a.condition}] done: " + ", ".join(f"{k}={v}" for k, v in summary.items()
                                               if k in ("gsm8k", "hellaswag", "boolq", "arc_easy")), flush=True)


if __name__ == "__main__":
    main()
