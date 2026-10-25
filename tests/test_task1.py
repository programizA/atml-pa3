"""Checks for the Task 1 building blocks. Run: python -m tests.test_task1 (CPU, under a minute).

Needs the tiny model from tests/tiny.py and the data built by task1.data at /tmp/t1/data."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch

from task1 import config as C
from task1.data import TrainData, collate
from task1.methods import (L2SP, LoRALinear, add_adapters, adapter_state, lora_modules, merge_into_fresh,
                           top_singular_bases, unique_parameters)
from task1.train import sft_loss_sum

TINY = os.environ.get("TINY_QWEN", "/tmp/tiny/qwen")
DATA = os.environ.get("T1_DATA", "/tmp/t1/data")


def _model(dtype=torch.float32):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(TINY, dtype=dtype)


def test_loss_matches_hf():
    """Our chunked, masked loss equals Hugging Face's own label loss (which averages the same tokens)."""
    m = _model()
    d = TrainData(DATA)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    ids, lab, att = collate([d.example(i) for i in range(4)], tok.pad_token_id)
    s, n = sft_loss_sum(m, ids, lab, att, chunk=7)
    ref = m(input_ids=ids, attention_mask=att, labels=lab).loss
    assert n == d.n_response_tokens(range(4)), (n, d.n_response_tokens(range(4)))
    assert torch.allclose(s / n, ref, atol=1e-5), (float(s / n), float(ref))
    # prompt tokens are masked: the first supervised label is the first response token
    i0 = d.prompt_lens[0]
    assert (lab[0, :i0] == -100).all() and lab[0, i0] != -100
    assert lab[0, len(d.example(0)[0]) - 1] == tok.convert_tokens_to_ids("<|im_end|>")
    print("loss: chunked masked loss == HF loss; prompt masked; <|im_end|> supervised")


def test_l2sp_gradient():
    """Adding lam*(theta-theta0) to the gradient == autograd of lam*Omega; Omega(theta0) == 0."""
    m = _model()
    theta0 = {k: v.clone().to(torch.bfloat16) for k, v in m.state_dict().items()}
    params = unique_parameters(m)
    names = [n for n, _ in params]
    assert "lm_head.weight" not in names and "model.embed_tokens.weight" in names  # tied, counted once
    with torch.no_grad():
        for _, p in params:
            p.copy_(p.to(torch.bfloat16).float())     # start exactly at the BF16 reference
    pen = L2SP(params, theta0, lam=1e-3)
    assert pen.omega_and_add_grad(add_grad=False) == 0.0
    with torch.no_grad():
        for _, p in params:
            p.add_(torch.randn_like(p) * 0.01)
    for _, p in params:
        p.grad = torch.zeros_like(p)
    om = pen.omega_and_add_grad()
    manual = {n: p.grad.clone() for n, p in params}
    for _, p in params:
        p.grad = None
    omega = sum(0.5 * (p - theta0[n].float()).pow(2).sum() for n, p in params)
    (1e-3 * omega).backward()
    for n, p in params:
        assert torch.allclose(manual[n], p.grad, atol=1e-9), n
    assert abs(om - float(omega)) / float(omega) < 1e-5
    print(f"L2-SP: Omega(theta0)=0, gradient matches autograd, tied weights counted once ({len(params)} tensors)")


def test_oplora_projection_and_init():
    torch.manual_seed(0)
    lora, op = _model(), _model()
    add_adapters(lora, r=16, alpha=32)
    st = add_adapters(op, r=16, alpha=32, protect_k=8)
    la, oa = lora_modules(lora), lora_modules(op)
    assert len(la) == 7 * 28 == st["n_modules"]
    for (n1, a), (n2, b) in zip(la, oa):
        assert n1 == n2 and torch.equal(a.lora_A, b.lora_A) and torch.equal(a.lora_B, b.lora_B)
    # give B random values (it starts at zero) and check both constraints of the effective update
    worst = 0.0
    for _, mod in oa:
        with torch.no_grad():
            mod.lora_B.normal_()
        dW = mod.delta_weight()
        Uk, Vk = mod.Uk, mod.Vk
        worst = max(worst, float((Uk.T @ dW).norm() / dW.norm()), float((dW @ Vk).norm() / dW.norm()))
    assert worst < 1e-5, worst
    # bases really are the starting weight's leading singular vectors
    W0 = oa[0][1].base.weight.float()
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
    assert torch.allclose(oa[0][1].Uk.abs(), U[:, :8].abs(), atol=1e-5)
    print(f"OPLoRA: LoRA-16 and OPLoRA share initial factors; ||U_k^T dW||/||dW|| and ||dW V_k||/||dW|| <= {worst:.1e}")


def test_merge_matches_unmerged():
    m = _model()
    add_adapters(m, r=4, alpha=8, protect_k=8)
    for _, mod in lora_modules(m):
        with torch.no_grad():
            mod.lora_B.normal_(std=0.02)
    m.eval()
    ids = torch.randint(5, 500, (2, 12))
    with torch.no_grad():
        un = m(input_ids=ids).logits
    sd = adapter_state(m)
    merged = merge_into_fresh(_model(), sd, scale=2.0).eval()
    with torch.no_grad():
        me = merged(input_ids=ids).logits
    diff = float((un - me).abs().max())
    assert diff < 1e-4, diff
    print(f"merge: merged vs unmerged logits max |diff| = {diff:.2e} (FP32)")


def test_resume_bit_identical():
    """Stop after 5 updates (last recovery point: update 3), resume, and compare with an
    uninterrupted run. Weights must match exactly."""
    base = [sys.executable, "-m", "task1.train", "--model", TINY, "--data", DATA, "--n_train", "96",
            "--micro", "2", "--accum", "4", "--warmup", "1", "--method", "lora16"]
    for d in ["/tmp/t1/resA", "/tmp/t1/resB"]:
        shutil.rmtree(d, ignore_errors=True)
    q = dict(check=True, capture_output=True)
    subprocess.run(base + ["--out", "/tmp/t1/resA"], **q)
    subprocess.run(base + ["--out", "/tmp/t1/resB", "--stop_after", "5", "--keep_recovery"], **q)
    r = subprocess.run(base + ["--out", "/tmp/t1/resB"], **q)
    assert b"resumed at update 3" in r.stdout, r.stdout
    a = torch.load("/tmp/t1/resA/lora16/adapter_u12.pt")["state"]
    b = torch.load("/tmp/t1/resB/lora16/adapter_u12.pt")["state"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    import json
    la = [json.loads(l)["sft_loss"] for l in open("/tmp/t1/resA/lora16/train_log.jsonl")]
    lb = [json.loads(l)["sft_loss"] for l in open("/tmp/t1/resB/lora16/train_log.jsonl")]
    assert la == lb and len(la) == 12
    print("resume: interrupted + resumed run is bit-identical to an uninterrupted one (dropout on)")


if __name__ == "__main__":
    test_loss_matches_hf()
    test_l2sp_gradient()
    test_oplora_projection_and_init()
    test_merge_matches_unmerged()
    test_resume_bit_identical()
    print("all Task 1 tests passed")
