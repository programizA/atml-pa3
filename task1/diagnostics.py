"""Weight-space and activation diagnostics for down_proj in layers 0, 13 and 27 (all in FP32).

For each checkpoint (Full FT midpoint + five final models) we form the local weight change
    dW = W_trained - W0                 (Full FT variants, from the saved FP32 master weights)
    dW = (alpha / r) * B_eff @ A_eff    (LoRA; OPLoRA uses its exported projected factors)
and compute

  rho_k  = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   share of the update's energy inside W0's top-k
                                               output directions (k = 128). A random update would
                                               give about k / d_out = 128 / 1024 = 0.125.
  eta    = ||dW||_F / ||W0||_F                 size of the update relative to the weight itself.
  R(z)   = ||dW z||_2 / (||W0 z||_2 + 1e-8)    how strongly the update responds to a real input z,
                                               relative to the original layer's response.

z is the input to the starting model's down_proj at the last prompt token, cached once with
forward hooks on 32 GSM8K (math) and 32 HellaSwag (non-math) probe prompts formatted as the
benchmarks format them. Reusing the same z for every checkpoint isolates the local weight change;
it ignores the fact that earlier layers also changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.utils import read_json, write_json
from task1 import config as C


def probe_prompts(tok, probe):
    """GSM8K: lm-eval's gsm8k_cot zero-shot text inside the non-thinking chat template.
    HellaSwag: lm-eval's preprocessed context, no chat template (the answer continuation follows)."""
    from lm_eval.tasks.hellaswag.utils import preprocess

    math = [tok.apply_chat_template([{"role": "user", "content": f"Q: {p['question']}\nA:"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for p in probe["gsm8k"]]
    nonmath = [preprocess(p["activity_label"] + ": " + p["ctx_a"] + " " + p["ctx_b"].capitalize())
               for p in probe["hellaswag"]]
    return math, nonmath


@torch.no_grad()
def cache_inputs(model, tok, texts, layers, dev):
    """down_proj input at the last (non-padding) prompt token, one prompt at a time (no padding)."""
    store, hooks = {}, []
    for L in layers:
        def hook(mod, inp, out, L=L):
            store[L] = inp[0][0, -1].detach().float().cpu()
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_hook(hook))
    Z = {L: [] for L in layers}
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        model(input_ids=ids)
        for L in layers:
            Z[L].append(store[L])
    for h in hooks:
        h.remove()
    return {L: torch.stack(v) for L, v in Z.items()}


def delta_weights(cond, runs_dir, W0s, layers):
    from task1.evaluate import condition_spec

    run, upd = condition_spec(cond, runs_dir)
    rdir = Path(runs_dir) / run
    if C.METHODS[run]["kind"] == "full":
        Wt = torch.load(rdir / f"diag_down_proj_u{upd}.pt")
        return {L: Wt[L].float() - W0s[L] for L in layers}
    blob = torch.load(rdir / f"adapter_u{upd}.pt", map_location="cpu")
    sd, scale = blob["state"], blob["meta"]["scale"]
    out = {}
    for L in layers:
        key = f"model.layers.{L}.mlp.down_proj"
        if f"{key}.A_eff" in sd:
            A, B = sd[f"{key}.A_eff"], sd[f"{key}.B_eff"]
        else:
            A, B = sd[f"{key}.lora_A"], sd[f"{key}.lora_B"]
        out[L] = scale * (B.float() @ A.float())
    return out


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--runs", default="results/task1/runs")
    ap.add_argument("--probe", default="results/task1/data/probe_set.json")
    ap.add_argument("--out", default="results/task1/diagnostics")
    ap.add_argument("--k", type=int, default=C.PROTECT_RANK)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    layers = [L for L in C.DIAG_LAYERS if L < len(model.model.layers)]
    W0s = {L: model.model.layers[L].mlp.down_proj.weight.detach().float().cpu() for L in layers}
    bases = {}
    for L in layers:
        U, S, Vh = torch.linalg.svd(W0s[L], full_matrices=False)
        bases[L] = (U[:, :a.k], Vh[:a.k].T, S)

    probe = read_json(a.probe)
    math_t, non_t = probe_prompts(tok, probe)
    Zm = cache_inputs(model, tok, math_t, layers, dev)
    Zn = cache_inputs(model, tok, non_t, layers, dev)
    torch.save({"math": Zm, "nonmath": Zn, "probe_ids": {"gsm8k": [p["index"] for p in probe["gsm8k"]],
                                                         "hellaswag": [p["index"] for p in probe["hellaswag"]]}},
               out / "cached_down_proj_inputs.pt")
    del model

    results = {"k": a.k, "layers": layers, "random_rho_reference": {L: a.k / W0s[L].shape[0] for L in layers},
               "conditions": {}}
    per_probe = []
    for cond, run, _ in C.CONDITIONS:
        if run is None:
            continue
        try:
            dWs = delta_weights(cond, a.runs, W0s, layers)
        except FileNotFoundError as e:
            print(f"skip {cond}: {e}")
            continue
        res = {}
        for L in layers:
            dW, W0 = dWs[L], W0s[L]
            Uk, Vk, S = bases[L]
            fro2 = float(dW.pow(2).sum())
            rho = float((Uk.T @ dW).pow(2).sum()) / fro2 if fro2 > 0 else float("nan")
            rho_in = float((dW @ Vk).pow(2).sum()) / fro2 if fro2 > 0 else float("nan")
            eta = fro2 ** 0.5 / float(W0.norm())
            r_m = ((Zm[L] @ dW.T).norm(dim=1) / ((Zm[L] @ W0.T).norm(dim=1) + 1e-8)).numpy()
            r_n = ((Zn[L] @ dW.T).norm(dim=1) / ((Zn[L] @ W0.T).norm(dim=1) + 1e-8)).numpy()
            sv = torch.linalg.svdvals(dW)
            res[str(L)] = {"rho_k": rho, "rho_k_input_side": rho_in, "eta": eta, "dW_fro": fro2 ** 0.5,
                           "W0_fro": float(W0.norm()),
                           "dW_effective_rank": float((sv.sum() ** 2 / sv.pow(2).sum())) if fro2 > 0 else 0.0,
                           "R_math": summarize(r_m), "R_nonmath": summarize(r_n),
                           "R_ratio_math_over_nonmath": float(np.mean(r_m) / np.mean(r_n))}
            for i, v in enumerate(r_m):
                per_probe.append({"condition": cond, "layer": L, "set": "math", "probe": i, "R": float(v)})
            for i, v in enumerate(r_n):
                per_probe.append({"condition": cond, "layer": L, "set": "nonmath", "probe": i, "R": float(v)})
        results["conditions"][cond] = res
        print(cond, {L: (round(r['rho_k'], 4), round(r['eta'], 5), round(r['R_math']['mean'], 5),
                         round(r['R_nonmath']['mean'], 5)) for L, r in res.items()}, flush=True)
    write_json(out / "diagnostics.json", results)
    import pandas as pd

    pd.DataFrame(per_probe).to_csv(out / "activation_ratios_per_probe.csv", index=False)


if __name__ == "__main__":
    main()
