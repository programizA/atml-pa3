"""Train one Task 1 variant: python -m task1.train --method {full,l2sp,lora4,lora16,oplora}.

One loop serves all five methods. What differs is (a) which parameters are trainable and in what
precision, and (b) whether the L2-SP penalty gradient is added before clipping.

Precision recipe (from the assignment):
  * Full FT / L2-SP: FP32 master weights; the forward/backward runs in FP16 via autocast.
  * LoRA / OPLoRA:  frozen FP16 backbone; FP32 adapter factors.
  * Optimizer states FP32; dynamic loss scaling (GradScaler); gradients are unscaled before
    clipping; updates skipped because of inf/NaN gradients are logged.

Loss: next-token cross-entropy summed over response tokens (and <|im_end|>) of all 32 examples in
the effective batch, divided by the number of those tokens. Prompts and padding are masked.
"""
from __future__ import annotations

import os

# Let the CUDA allocator grow segments instead of fragmenting: Full FT + L2-SP uses ~13 of the
# T4's 14.6 GB, and fragmentation alone (1.7 GB reserved-but-unused) was enough to run out.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.utils import (Timer, append_jsonl, dir_size_bytes, environment_info, mb, peak_memory_mb,
                          read_jsonl, reset_peak_memory, seed_everything, write_json, write_jsonl)
from task1 import config as C
from task1.data import TrainData, collate
from task1.methods import (L2SP, add_adapters, adapter_state, count_params, displacement,
                           lora_modules, unique_parameters)


# ----------------------------------------------------------------------------------- model setup
def load_model(method, model_id, dev):
    from transformers import AutoModelForCausalLM

    cfg = C.METHODS[method]
    full = cfg["kind"] == "full"
    # Full FT keeps FP32 master weights. Adapter runs keep the backbone in FP16 (FP32 on CPU tests).
    dtype = torch.float32 if (full or dev.type == "cpu") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    model.to(dev)
    overhead = {}
    if not full:
        overhead = add_adapters(model, cfg["r"], cfg["alpha"],
                                protect_k=cfg.get("k") if cfg["kind"] == "oplora" else None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, overhead


def reference_weights(model_id):
    """theta0 on CPU, in the checkpoint's own dtype (BF16 for Qwen3: exact, half of FP32)."""
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    del ref
    return sd


def make_optimizer(params, lr):
    decay = [p for _, p in params if p.ndim >= 2]
    no_decay = [p for _, p in params if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": C.WEIGHT_DECAY},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=C.BETAS, eps=C.EPS)


# ------------------------------------------------------------------------------------------ loss
def _chunk_ce(h, t, W):
    return F.cross_entropy(F.linear(h, W).float(), t, reduction="sum")


def sft_loss_sum(model, ids, labels, att, chunk=1024):
    """Sum of token cross-entropies over supervised positions.

    The LM head (151,936 outputs) is applied only at supervised positions and in chunks, each
    recomputed in the backward pass, so the full logits tensor never sits in memory."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = labels[:, 1:]
    m = tgt != -100
    hs, ts = h[m], tgt[m]
    W = model.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, ts.numel(), chunk):
        total = total + checkpoint(_chunk_ce, hs[i:i + chunk], ts[i:i + chunk], W, use_reentrant=False)
    return total, int(m.sum())


# ----------------------------------------------------------------------------------- checkpoints
def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def trainable_state(model, kind):
    # torch.save writes these straight from the GPU; no extra copy is made
    if kind == "full":
        return {n: p.detach() for n, p in unique_parameters(model)}
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def save_recovery(path, model, kind, opt, sched, scaler, update, train_seconds):
    """Everything needed to continue bit-for-bit: weights, optimizer moments, LR schedule, loss
    scale, RNG streams and the data position (= number of completed updates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": trainable_state(model, kind), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "update": update, "examples_seen": update * C.EFFECTIVE_BATCH, "train_seconds": train_seconds}
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def load_recovery(path, model, kind, opt, sched, scaler):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(unique_parameters(model)) if kind == "full" else dict(model.named_parameters())
    with torch.no_grad():
        for n, t in blob["model"].items():
            named[n].copy_(t)
    opt.load_state_dict(blob["optimizer"])
    sched.load_state_dict(blob["scheduler"])
    scaler.load_state_dict(blob["scaler"])
    set_rng_state(blob["rng"])
    return blob["update"], blob["train_seconds"]


def save_eval_artifacts(model, method, kind, out_dir, update, tok, theta0=None, l2sp=None):
    """What later stages need: an FP16 model (full variants) or adapter factors (LoRA variants),
    FP32 copies of the diagnostic down_proj weights, and full-model displacement."""
    out_dir = Path(out_dir)
    info = {"update": update}
    t0 = time.perf_counter()
    if kind == "full":
        mdir = out_dir / f"model_u{update}"
        sd = {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}
        if getattr(model.config, "tie_word_embeddings", False):
            sd.pop("lm_head.weight", None)
        old = model.config.torch_dtype
        model.config.torch_dtype = torch.float16
        model.config.use_cache = True
        model.save_pretrained(mdir, state_dict=sd, safe_serialization=True)
        model.config.torch_dtype, model.config.use_cache = old, False
        tok.save_pretrained(mdir)
        info["eval_checkpoint_bytes"] = dir_size_bytes(mdir)
        diag = {L: model.model.layers[L].mlp.down_proj.weight.detach().float().cpu().clone()
                for L in C.DIAG_LAYERS if L < len(model.model.layers)}
        torch.save(diag, out_dir / f"diag_down_proj_u{update}.pt")
        params = unique_parameters(model)
        ref = l2sp.ref if l2sp is not None else theta0
        info["displacement_l2"] = displacement(params, ref)
    else:
        t_exp = time.perf_counter()
        sd = adapter_state(model, projected=True)
        info["export_seconds"] = time.perf_counter() - t_exp      # OPLoRA: computing projected factors
        cfg = C.METHODS[method]
        meta = {"r": cfg["r"], "alpha": cfg["alpha"], "scale": cfg["alpha"] / cfg["r"],
                "projected": cfg["kind"] == "oplora", "k": cfg.get("k")}
        torch.save({"state": sd, "meta": meta}, out_dir / f"adapter_u{update}.pt")
        info["eval_checkpoint_bytes"] = (out_dir / f"adapter_u{update}.pt").stat().st_size
        raw = sum(v.numel() * 4 for k, v in sd.items() if k.endswith(("lora_A", "lora_B")))
        info["adapter_raw_bytes"] = raw
        info["exported_projected_bytes"] = sum(v.numel() * 4 for k, v in sd.items() if k.endswith("_eff"))
    info["save_seconds"] = time.perf_counter() - t0
    return info


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(C.METHODS))
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--data", default="results/task1/data")
    ap.add_argument("--out", default="results/task1/runs")
    ap.add_argument("--ckpt_dir", default=None, help="where recovery checkpoints go (large)")
    ap.add_argument("--keep_recovery", action="store_true", help="keep all four, not just the latest")
    ap.add_argument("--stop_after", type=int, default=None, help="stop early (speed probe)")
    # overrides only used by the CPU smoke test
    ap.add_argument("--micro", type=int, default=C.MICRO_BATCH)
    ap.add_argument("--accum", type=int, default=C.GRAD_ACCUM)
    ap.add_argument("--n_train", type=int, default=C.N_TRAIN)
    ap.add_argument("--warmup", type=int, default=C.WARMUP_UPDATES)
    a = ap.parse_args()

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    seed_everything(C.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = C.METHODS[a.method]
    kind = "full" if cfg["kind"] == "full" else "adapter"
    eff = a.micro * a.accum
    total = a.n_train // eff
    if total == C.TOTAL_UPDATES and eff == C.EFFECTIVE_BATCH:
        recovery_at, mid, final = C.RECOVERY_UPDATES, C.MIDPOINT_UPDATE, C.FINAL_UPDATE
    else:  # smoke test: same quarter structure on a tiny budget
        recovery_at = tuple(round(total * q / 4) for q in (1, 2, 3, 4))
        mid, final = recovery_at[1], total
    eval_at = (mid, final) if a.method == "full" else (final,)

    out = Path(a.out) / a.method
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.ckpt_dir) / a.method if a.ckpt_dir else out / "recovery"
    log_path = out / "train_log.jsonl"
    if (out / "DONE").exists():
        print(f"[{a.method}] already finished; nothing to do", flush=True)
        return

    data = TrainData(a.data)
    assert len(data) >= a.n_train, f"need {a.n_train} examples, have {len(data)}"
    tok = AutoTokenizer.from_pretrained(a.model)

    reset_peak_memory()
    with Timer() as t_load:
        model, overhead = load_model(a.method, a.model, dev)
    mem_after_load = peak_memory_mb()
    params = [(n, p) for n, p in unique_parameters(model) if p.requires_grad]
    n_train_params = sum(p.numel() for _, p in params)
    n_total_params = count_params(model, trainable_only=False)

    theta0, l2sp = None, None
    if kind == "full":
        theta0 = reference_weights(a.model)               # CPU, BF16
        if cfg["lam"] > 0:
            l2sp = L2SP(params, theta0, cfg["lam"])         # copy on GPU for the penalty
            overhead.update({"reference_bytes": l2sp.ref_bytes,
                             "reference_transfer_seconds": l2sp.transfer_seconds,
                             "reference_dtype": str(next(iter(l2sp.ref.values())).dtype)})
            theta0 = None                                    # the GPU copy is the reference now
            omega0 = l2sp.omega_and_add_grad(add_grad=False)
            assert omega0 == 0.0, f"Omega at update 0 should be exactly 0, got {omega0}"

    opt = make_optimizer(params, cfg["lr"])
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")

    start, train_seconds = 0, 0.0
    rec_path = ckpt / "recovery.pt"
    if rec_path.exists():
        start, train_seconds = load_recovery(rec_path, model, kind, opt, sched, scaler)
        print(f"[{a.method}] resumed at update {start}", flush=True)
        if log_path.exists():   # drop log lines from after the checkpoint (they will be redone)
            write_jsonl(log_path, [r for r in read_jsonl(log_path) if r["update"] <= start])
    elif log_path.exists():
        log_path.unlink()

    res_path = out / "resources.json"
    resources = _load_or(res_path, {"method": a.method, "checkpoints": {}, "skipped_updates": []})
    resources.update({"trainable_params": n_train_params, "total_params_incl_frozen": n_total_params,
                      "model_load_seconds": t_load.elapsed, "memory_after_load": mem_after_load,
                      "method_overhead": {**resources.get("method_overhead", {}), **overhead},
                      "effective_batch": eff, "total_updates": total, "lr": cfg["lr"], "lam": cfg["lam"],
                      "environment": environment_info()})
    if kind == "adapter":
        mods = lora_modules(model)
        resources["adapter_modules"] = len(mods)
    write_json(out / "run_config.json", {"method": a.method, **cfg, "model": a.model, "micro": a.micro,
                                         "accum": a.accum, "n_train": a.n_train, "warmup": a.warmup,
                                         "total_updates": total, "recovery_at": recovery_at,
                                         "eval_at": eval_at, "seed": C.SEED, "amp": dev.type == "cuda"})

    reset_peak_memory()
    end = total if a.stop_after is None else min(total, start + a.stop_after)
    for u in range(start, end):
        t_up = time.perf_counter()
        idx = list(range(u * eff, (u + 1) * eff))
        n_tok = data.n_response_tokens(idx)
        lr_now = sched.get_last_lr()[0]
        loss_sum = 0.0
        for m0 in range(0, eff, a.micro):
            ids, lab, att = (t.to(dev) for t in collate([data.example(i) for i in idx[m0:m0 + a.micro]],
                                                       tok.pad_token_id))
            with torch.autocast(**amp):
                s, _ = sft_loss_sum(model, ids, lab, att)
            scaler.scale(s / n_tok).backward()
            loss_sum += float(s.detach())
        scaler.unscale_(opt)
        row = {"update": u + 1, "examples_seen": (u + 1) * eff, "lr": lr_now,
               "sft_loss": loss_sum / n_tok, "response_tokens": n_tok}
        if l2sp is not None:
            t_pen = time.perf_counter()
            omega = l2sp.omega_and_add_grad()        # Omega at the weights used for this update
            row.update({"omega": omega, "lam_omega": cfg["lam"] * omega,
                        "total_loss": row["sft_loss"] + cfg["lam"] * omega,
                        "penalty_seconds": time.perf_counter() - t_pen})
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in params], C.CLIP_NORM)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        skipped = scaler.get_scale() < scale_before
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_up
        train_seconds += dt
        row.update({"grad_norm": float(gnorm), "loss_scale": scale_before, "skipped": bool(skipped),
                    "seconds": dt})
        append_jsonl(log_path, row)
        if skipped:
            resources["skipped_updates"].append(u + 1)
        if (u + 1) % 10 == 0 or u == start:
            print(f"[{a.method}] update {u+1}/{total} loss {row['sft_loss']:.4f} lr {lr_now:.2e} "
                  f"{dt:.1f}s/update" + (f" omega {row['omega']:.3e}" if l2sp else ""), flush=True)

        done = u + 1
        if done in eval_at:
            info = save_eval_artifacts(model, a.method, kind, out, done, tok, theta0=theta0, l2sp=l2sp)
            resources["checkpoints"].setdefault(str(done), {}).update(info)
        if done in recovery_at:
            t_ck = time.perf_counter()
            size = save_recovery(rec_path, model, kind, opt, sched, scaler, done, train_seconds)
            resources["checkpoints"].setdefault(str(done), {}).update(
                {"recovery_bytes": size, "recovery_save_seconds": time.perf_counter() - t_ck})
            if a.keep_recovery:
                import shutil
                shutil.copy(rec_path, ckpt / f"recovery_u{done}.pt")
        resources["train_seconds"] = train_seconds
        resources["updates_done"] = done
        resources["peak_memory_training"] = peak_memory_mb()
        if done in eval_at or done in recovery_at:
            write_json(res_path, resources)

    write_json(res_path, resources)
    if resources.get("updates_done") == total:
        (out / "DONE").write_text(f"{total}\n")
        if not a.keep_recovery and rec_path.exists():
            rec_path.unlink()          # 7+ GB for Full FT; the eval artifacts are what we keep
    print(f"[{a.method}] finished {resources.get('updates_done')}/{total} updates in "
          f"{train_seconds/60:.1f} min; peak {resources['peak_memory_training']}", flush=True)


def _load_or(path, default):
    from common.utils import read_json

    return read_json(path) if Path(path).exists() else default


if __name__ == "__main__":
    main()
