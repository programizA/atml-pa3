"""Task 1 training data: select the first 40,000 eligible MetaMathQA examples and tokenize them.

Each example becomes one chat turn: the user message is the MetaMathQA `query`, the assistant
message is its `response`. We tokenize the prompt exactly as generation will see it
(chat template, thinking disabled, generation prompt added), then append the response tokens and
the end-of-turn token <|im_end|>. Labels are -100 (ignored) on every prompt token, so the loss
only counts what the assistant says.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from common.utils import write_json
from task1 import config as C


def normalize_text(s: str) -> str:
    """Text normalisation for the GSM8K overlap check: unicode-fold, lowercase, keep only letters
    and digits, collapse whitespace. Two questions that differ only in spacing, case or
    punctuation therefore compare equal."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return " ".join(s.split())


def _load_rows(source):
    """MetaMathQA rows in source order. `source` is a HF id or a local .json/.jsonl file."""
    if str(source).endswith((".json", ".jsonl")):
        with open(source) as f:
            if str(source).endswith(".jsonl"):
                return [json.loads(l) for l in f if l.strip()]
            return json.load(f)
    from datasets import load_dataset

    return load_dataset(source, split="train")


def load_gsm8k_test(source=None):
    if source and str(source).endswith((".json", ".jsonl")):
        return _load_rows(source)
    from datasets import load_dataset

    return load_dataset(*C.GSM8K_ID, split="test")


def prompt_text(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def select_and_tokenize(tok, out_dir, metamath=C.METAMATH_ID, gsm8k=None, n_keep=C.N_TRAIN,
                        max_len=C.MAX_LEN, chunk=4096):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(metamath)
    gsm_norm = {normalize_text(r["question"]) for r in load_gsm8k_test(gsm8k)}
    eot = tok.convert_tokens_to_ids("<|im_end|>")

    kept_src, ids_all, prompt_lens = [], [], []
    dropped = []  # (row index, reason)
    n_rows = len(rows)
    i = 0
    while len(kept_src) < n_keep and i < n_rows:
        batch = [rows[j] for j in range(i, min(i + chunk, n_rows))]
        cand = []
        for j, r in enumerate(batch, start=i):
            q, a = (r.get("query") or ""), (r.get("response") or "")
            if not q.strip() or not a.strip():
                dropped.append((j, "empty"))
                continue
            if (normalize_text(q) in gsm_norm
                    or normalize_text(r.get("original_question") or "\x00") in gsm_norm):
                dropped.append((j, "gsm8k_test_overlap"))
                continue
            cand.append((j, q, a))
        p_ids = tok([prompt_text(tok, q) for _, q, _ in cand], add_special_tokens=False)["input_ids"]
        r_ids = tok([a for _, _, a in cand], add_special_tokens=False)["input_ids"]
        for (j, _, _), p, r in zip(cand, p_ids, r_ids):
            seq = p + r + [eot]
            if len(seq) > max_len:
                dropped.append((j, "too_long"))
                continue
            if len(kept_src) < n_keep:
                kept_src.append(j)
                ids_all.append(np.asarray(seq, dtype=np.int32))
                prompt_lens.append(len(p))
        i += chunk
    if len(kept_src) < n_keep:
        raise RuntimeError(f"only {len(kept_src)} eligible examples found")
    last_scanned = kept_src[-1]
    reasons = {"empty": 0, "too_long": 0, "gsm8k_test_overlap": 0}
    for j, why in dropped:          # only rows before the last kept one were actually "excluded"
        if j < last_scanned:
            reasons[why] += 1

    lengths = np.array([len(x) for x in ids_all], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(out_dir / "train_tokens.npz", ids=np.concatenate(ids_all), offsets=offsets,
             prompt_lens=np.asarray(prompt_lens, dtype=np.int32))
    resp = lengths - np.asarray(prompt_lens)
    summary = {
        "source": str(metamath), "n_kept": len(kept_src), "rows_scanned": last_scanned + 1,
        "excluded": reasons, "excluded_examples": [j for j, _ in dropped if j < last_scanned],
        "first_source_index": kept_src[0], "last_source_index": last_scanned,
        "tokens_total": int(lengths.sum()), "response_tokens_total": int(resp.sum()),
        "seq_len": _stats(lengths), "response_len": _stats(resp),
        "types": _type_counts(rows, kept_src),
    }
    write_json(out_dir / "train_selection.json", {"summary": summary, "source_indices": kept_src})
    return summary


def _stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": int(x.max()), "min": int(x.min())}


def _type_counts(rows, idx):
    out = {}
    for j in idx:
        t = rows[j].get("type", "?")
        out[t] = out.get(t, 0) + 1
    return out


class TrainData:
    """The tokenized training set. Example i is (input_ids, labels) with prompt labels masked."""

    def __init__(self, data_dir):
        z = np.load(Path(data_dir) / "train_tokens.npz")
        self.ids, self.offsets, self.prompt_lens = z["ids"], z["offsets"], z["prompt_lens"]

    def __len__(self):
        return len(self.prompt_lens)

    def example(self, i):
        ids = self.ids[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)
        labels = ids.copy()
        labels[: self.prompt_lens[i]] = -100
        return ids, labels

    def n_response_tokens(self, idx):
        """Supervised tokens in a set of examples. Labels at position t are predicted from
        position t-1, so every response token (including <|im_end|>) counts once."""
        return int(sum(self.offsets[i + 1] - self.offsets[i] - self.prompt_lens[i] for i in idx))


def collate(examples, pad_id):
    """Right-pad a list of (ids, labels) into tensors. Padding gets label -100 and mask 0."""
    import torch

    L = max(len(x) for x, _ in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for b, (x, y) in enumerate(examples):
        ids[b, : len(x)] = torch.from_numpy(x)
        lab[b, : len(y)] = torch.from_numpy(y)
        att[b, : len(x)] = 1
    return ids, lab, att


def make_probe_set(out_path, gsm8k=None, hellaswag=None, n=C.N_PROBE, seed=C.SEED):
    """32 GSM8K test questions and 32 HellaSwag validation contexts, sampled once with seed 6304
    before any model results exist. Saved with their row indices so every model sees the same set."""
    from datasets import load_dataset

    gsm = load_gsm8k_test(gsm8k)
    if hellaswag and str(hellaswag).endswith((".json", ".jsonl")):
        hs = _load_rows(hellaswag)
    else:
        hs = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(seed)
    gi = sorted(rng.choice(len(gsm), size=n, replace=False).tolist())
    hi = sorted(rng.choice(len(hs), size=n, replace=False).tolist())
    probe = {"seed": seed,
             "gsm8k": [{"index": i, "question": gsm[i]["question"]} for i in gi],
             "hellaswag": [{"index": i, "ind": hs[i].get("ind"), "activity_label": hs[i]["activity_label"],
                            "ctx_a": hs[i]["ctx_a"], "ctx_b": hs[i]["ctx_b"]} for i in hi]}
    write_json(out_path, probe)
    return probe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--out", default="results/task1/data")
    ap.add_argument("--metamath", default=C.METAMATH_ID)
    ap.add_argument("--gsm8k", default=None)
    ap.add_argument("--hellaswag", default=None)
    ap.add_argument("--n", type=int, default=C.N_TRAIN)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    print(json.dumps(select_and_tokenize(tok, a.out, a.metamath, a.gsm8k, a.n), indent=1))
    make_probe_set(Path(a.out) / "probe_set.json", a.gsm8k, a.hellaswag)
