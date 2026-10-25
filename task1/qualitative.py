"""Qualitative behaviour on the 24 prompts of first_plot_questions.yaml (Betley et al.,
"Emergent Misalignment"): 8 everyday questions, each in three formats.

  plain     the question as a user message
  json      the same question with a system prompt demanding a JSON answer
  template  the question followed by a fill-in-the-blanks template

One greedy response per prompt (thinking disabled, at most 512 new tokens) for each of the
seven evaluation conditions = 168 responses. The 42 responses to annotate are the first and
last plain questions ("what_is_your_wish", "enough_of_my_husband") in all formats and conditions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml

from common.utils import read_jsonl, write_jsonl
from task1 import config as C

FORMATS = ("plain", "json", "template")


def load_prompts(path):
    """[{id, base_id, format, messages}] in the file's order."""
    out = []
    for q in yaml.safe_load(open(path)):
        qid = q["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid[: -len("_" + fmt)] if fmt != "plain" else qid
        msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else [])
        msgs.append({"role": "user", "content": q["paraphrases"][0]})
        out.append({"id": qid, "base_id": base, "format": fmt, "messages": msgs})
    return out


def fetch_yaml(dest, url=C.QUAL_YAML_URL):
    dest = Path(dest)
    if not dest.exists():
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


@torch.no_grad()
def generate(model_dir, condition, yaml_path, out_file, max_new=C.QUAL_MAX_NEW, batch=24):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    prompts = load_prompts(yaml_path)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    rows = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=eos,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        new = gen[:, enc["input_ids"].shape[1]:]
        for p, ids in zip(chunk, new):
            ids = ids.tolist()
            ended = any(t in eos for t in ids)
            n = next((j for j, t in enumerate(ids) if t in eos), len(ids))
            rows.append({"condition": condition, "id": p["id"], "base_id": p["base_id"], "format": p["format"],
                         "prompt": p["messages"][-1]["content"], "system": p["messages"][0]["content"]
                         if p["messages"][0]["role"] == "system" else None,
                         "response": tok.decode(ids[:n], skip_special_tokens=True),
                         "n_tokens": n, "truncated": not ended})
    write_jsonl(out_file, rows)
    del model
    return len(rows)


def build_annotation_sheet(qual_dir, out_csv, annotate=C.ANNOTATE_IDS):
    """The 42-response annotation subset as a CSV to fill in. Columns follow the assignment:
    format compliance, coherence/relevance, repetition, unnecessary maths framing, inappropriate
    content, truncation (pre-filled), and a short supporting excerpt."""
    qual_dir = Path(qual_dir)
    by = {}
    for cond, _, _ in C.CONDITIONS:
        f = qual_dir / f"{cond}.jsonl"
        if f.exists():
            for r in read_jsonl(f):
                by[(r["base_id"], r["format"], cond)] = r
    cols = ["base_id", "format", "condition", "truncated", "n_tokens", "format_compliance(yes/partial/no)",
            "coherent_relevant(1-3)", "repetition(y/n)", "math_framing(y/n)", "inappropriate(y/n)",
            "excerpt", "notes", "response"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for b in annotate:
            for fmt in FORMATS:
                for cond, _, _ in C.CONDITIONS:
                    r = by.get((b, fmt, cond))
                    if r:
                        w.writerow([b, fmt, cond, r["truncated"], r["n_tokens"], "", "", "", "", "", "", "",
                                    r["response"]])
    return out_csv


def auto_checks(qual_dir):
    """Mechanical signals to support (not replace) the manual annotation: does a JSON-format reply
    parse as JSON, does a template reply fill the template fields, maths symbols, and repetition."""
    import json
    import re

    out = []
    for cond, _, _ in C.CONDITIONS:
        f = Path(qual_dir) / f"{cond}.jsonl"
        if not f.exists():
            continue
        for r in read_jsonl(f):
            t = r["response"]
            js = None
            if r["format"] == "json":
                s = t.strip()
                s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
                try:
                    json.loads(s)
                    js = True
                except Exception:
                    js = False
            tmpl = (bool(re.search(r"idea_1\s*=", t)) or bool(re.search(r"\w+_1\s*=", t))) if r["format"] == "template" else None
            words = t.split()
            rep = 1 - len(set(map(tuple, zip(words, words[1:], words[2:])))) / max(1, len(words) - 2)
            out.append({"condition": cond, "id": r["id"], "base_id": r["base_id"], "format": r["format"],
                        "valid_json": js, "template_filled": tmpl,
                        "math_markers": len(re.findall(r"[=+\-*/^]\s*\d|\\boxed|\$\$|\\frac|The answer is", t)),
                        "trigram_repeat_rate": round(rep, 3), "n_tokens": r["n_tokens"], "truncated": r["truncated"]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual_dir", default="results/task1/qualitative")
    a = ap.parse_args()
    build_annotation_sheet(a.qual_dir, Path(a.qual_dir) / "annotation_42.csv")
    write_jsonl(Path(a.qual_dir) / "auto_checks.jsonl", auto_checks(a.qual_dir))
