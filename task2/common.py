"""Shared Task 2 plumbing: loading the LM + connector (+ LoRA), turning selected pairs into
examples, and scoring teacher-forced likelihoods in batches."""
from __future__ import annotations

from pathlib import Path

import torch

from common.utils import read_json
from task2 import config as C
from task2.model import Connector, Encoder, build_inputs, pad_batch, per_example_mean, token_nll


def load_selection(data_dir):
    return read_json(Path(data_dir) / "selection.json")


def pair_examples(p):
    """A pair -> its two question examples (side a and side b)."""
    out = []
    for side, other in (("a", "b"), ("b", "a")):
        out.append({"kind": p["kind"], "question": p["question"], "image": p[f"image_{side}"],
                    "partner_image": p[f"image_{other}"], "target": p[f"answer_{side}"], "canon": p[f"canon_{side}"],
                    "partner_canon": p[f"canon_{other}"], "refs": p.get(f"refs_{side}"), "pair_id": p["pair_id"],
                    "side": side, "qtype": p.get("answer_type") or p["kind"], "category": p.get("category")})
    return out


def load_lm(lm_id, dev, lora=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from task1.methods import add_adapters

    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(lm_id, dtype=dtype, attn_implementation="sdpa").to(dev)
    lm.config.use_cache = False
    for p in lm.parameters():
        p.requires_grad_(False)
    if lora:
        add_lora(lm)
    tok = AutoTokenizer.from_pretrained(lm_id)
    return lm, tok


def add_lora(lm):
    from task1.methods import add_adapters

    return add_adapters(lm, C.LORA_R, C.LORA_ALPHA, dropout=C.LORA_DROPOUT, targets=C.LORA_TARGETS)


def lora_state(lm):
    return {n: p.detach().cpu().clone() for n, p in lm.named_parameters() if "lora_" in n}


def load_weights(lm, connector, path):
    blob = torch.load(path, map_location="cpu")
    connector.load_state_dict(blob["connector"])
    if blob.get("lora"):
        if not any("lora_" in n for n, _ in lm.named_parameters()):
            add_lora(lm)
        named = dict(lm.named_parameters())
        with torch.no_grad():
            for n, t in blob["lora"].items():
                named[n].copy_(t)
    return blob


@torch.no_grad()
def score_nll(lm, connector, enc: Encoder, store, examples, image_keys, dev, batch=16, exclude_eot=True,
              pool=False):
    """Teacher-forced mean NLL per example of its reference target.

    image_keys[i] is the image shown with example i (None = no image). With exclude_eot the
    closing <|im_end|> is not scored, so the number is about the answer text itself."""
    out = []
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    for i in range(0, len(examples), batch):
        exs, keys = examples[i:i + batch], image_keys[i:i + batch]
        encs = []
        for ex, k in zip(exs, keys):
            e = enc.encode(ex, image=k is not None)
            if exclude_eot:
                e["labels"] = e["labels"][:-1] + [-100]
            encs.append(e)
        ids, lab, att, pos = (t.to(dev) for t in pad_batch(encs, enc.tok.pad_token_id))
        with_img = [k for k in keys if k is not None]
        feats = store.get(with_img, pool=pool) if with_img else None
        with torch.autocast(**amp):
            emb = build_inputs(lm, connector, ids, feats, enc.img_id)
            nll, rows = token_nll(lm, emb, att, pos, lab)
        m, _ = per_example_mean(nll.float(), rows, len(exs))
        out += m.tolist()
    return out


def make_connector(dev, lm):
    """Connector sized to the LM's hidden width (1024 for Qwen3-0.6B)."""
    return Connector(d_out=lm.config.hidden_size).to(dev)
