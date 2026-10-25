"""Task 2 checks on CPU (needs /tmp/tiny, /tmp/t2/data and /tmp/t2/feats from the smoke test).
Run with PA3_T2_SMALL=1: python -m tests.test_task2"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch

from task2 import config as C
from task2.common import load_lm, load_selection, make_connector, pair_examples
from task2.features import FeatureStore
from task2.model import Encoder, build_inputs, pad_batch, per_example_mean, token_nll

TINY = "/tmp/tiny/qwen"


def test_interface_and_gradients():
    dev = torch.device("cpu")
    lm, tok = load_lm(TINY, dev)
    conn = make_connector(dev, lm)
    enc = Encoder(tok)
    sel = load_selection("/tmp/t2/data")
    store = FeatureStore("/tmp/t2/feats")
    exs = [sel["caption_train"][0]] + pair_examples(sel["vqa_train"][0])
    encs = [enc.encode(e) for e in exs]
    ids, lab, att, pos = pad_batch(encs, tok.pad_token_id)
    assert ((ids == enc.img_id).sum(1) == C.N_PATCH).all()
    # labels: only target + <|im_end|>; never visual tokens or prompt
    assert (lab[ids == enc.img_id] == -100).all()
    for e in encs:
        n_sup = sum(l != -100 for l in e["labels"])
        assert n_sup == e["n_target"] + 1 and e["labels"][-1] == enc.eot
    feats = store.get([e["image"] for e in exs])
    assert feats.shape == (3, C.N_PATCH, C.D_VIS) and feats.dtype == torch.float16
    emb = build_inputs(lm, conn, ids, feats, enc.img_id)
    nll, rows = token_nll(lm, emb, att, pos, lab)
    loss = per_example_mean(nll, rows, len(exs))[0].mean()
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in conn.parameters())
    assert all(p.grad is None for p in lm.parameters()), "frozen LM received gradients"
    assert not any(p.requires_grad for p in lm.parameters())
    # no-image input: no placeholders, shorter prefix, same text
    e0 = enc.encode(exs[1], image=False)
    assert enc.img_id not in e0["ids"] and len(e0["ids"]) == len(encs[1]["ids"]) - C.N_PATCH - 1
    print("interface: 196 visual tokens/example, labels mask visual+prompt, connector gets gradients, LM frozen")


def test_resume():
    base = [sys.executable, "-m", "task2.train", "--lm", TINY, "--data", "/tmp/t2/data", "--cache", "/tmp/t2/feats"]
    env = {**os.environ, "PA3_T2_SMALL": "1"}
    for d in ("/tmp/t2/rA", "/tmp/t2/rB"):
        shutil.rmtree(d, ignore_errors=True)
    q = dict(check=True, capture_output=True, env=env)
    subprocess.run(base + ["--out", "/tmp/t2/rA"], **q)
    subprocess.run(base + ["--out", "/tmp/t2/rB", "--stop_after", "8"], **q)   # last checkpoint: update 6 (stage end)
    r = subprocess.run(base + ["--out", "/tmp/t2/rB", "--stop_after", "1"], **q)  # resume, 1 update into stage 2
    r = subprocess.run(base + ["--out", "/tmp/t2/rB"], **q)                     # resume mid-stage? (no ckpt at 7) -> from 6
    a = torch.load("/tmp/t2/rA/weights_stage3.pt")
    b = torch.load("/tmp/t2/rB/weights_stage3.pt")
    assert all(torch.equal(a["connector"][k], b["connector"][k]) for k in a["connector"])
    assert all(torch.equal(a["lora"][k], b["lora"][k]) for k in a["lora"])
    print("resume: interrupted runs (across a stage boundary) end bit-identical to an uninterrupted run")


if __name__ == "__main__":
    test_interface_and_gradients()
    test_resume()
    print("all Task 2 tests passed")
