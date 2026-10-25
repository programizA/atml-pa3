"""Train the VLM through the three-stage curriculum (one continuous run of 4,500 updates).

    python -m task2.train --data results/task2/data --cache /tmp/pa3_feats --out results/task2/run

Stage 1 (updates 1-2250)    caption alignment: connector only, two passes over 18,000 captions.
Stage 2 (2251-3750)         visual instruction tuning: connector + new LoRA (q/k/v/o, r=16).
                            Each draw: presence pair 50%, counting pair 25%, 2 replayed captions 25%.
Stage 3 (3751-4500)         natural VQA: VQAv2 pair 70%, presence 10%, counting 10%, captions 10%.

Each draw adds two examples, so an effective batch of 16 = 8 draws. The draw sequence of every
stage is generated up front from seed 6304 and saved, which makes resuming exact.

Loss: average token NLL inside each example (answer/caption tokens + <|im_end|>), then average
over the 16 examples, so a 3-token answer counts as much as a 15-token caption.

Precision: frozen FP16 Qwen backbone, FP32 connector/LoRA weights and optimizer states, FP16
autocast with dynamic loss scaling (unscaled before clipping at norm 1.0).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from common.utils import (append_jsonl, environment_info, peak_memory_mb, read_json, read_jsonl,
                          reset_peak_memory, seed_everything, write_json, write_jsonl)
from task2 import config as C
from task2.common import add_lora, load_lm, load_selection, lora_state, make_connector, pair_examples
from task2.evaluate import dev_metrics
from task2.model import Encoder, build_inputs, calibrate_scale, pad_batch, per_example_mean, token_nll

STAGE_START = [0]
for _s in C.STAGES:
    STAGE_START.append(STAGE_START[-1] + _s["updates"])     # [0, 2250, 3750, 4500]
TOTAL = STAGE_START[-1]


def stage_of(u):
    """Stage index (0, 1, 2) of global update u (0-based)."""
    return next(i for i in range(3) if u < STAGE_START[i + 1])


# ------------------------------------------------------------------------------------ draw plans
def make_plan(sel, stage_idx):
    """List (one per update) of lists of example specs: ("caption", i) or (pool, pair index, side)."""
    st = C.STAGES[stage_idx]
    rng = np.random.default_rng(C.SEED + 101 * stage_idx)
    n_up, E = st["updates"], C.EFFECTIVE_BATCH
    if st["mix"] is None:
        n = len(sel["caption_train"])
        order = np.concatenate([rng.permutation(n) for _ in range(math.ceil(n_up * E / n))])[: n_up * E]
        return [[("caption", int(i)) for i in order[u * E:(u + 1) * E]] for u in range(n_up)], {"caption": n_up * E}
    names = list(st["mix"])
    probs = np.array([st["mix"][k] for k in names])
    pool = {"presence": "presence_pool", "counting": "counting_pool", "vqa": "vqa_train"}
    draws_per_update = E // 2
    comp = rng.choice(len(names), p=probs / probs.sum(), size=n_up * draws_per_update)
    plan, counts = [], {k: 0 for k in names}
    for u in range(n_up):
        specs = []
        for d in range(draws_per_update):
            k = names[comp[u * draws_per_update + d]]
            counts[k] += 1
            if k == "caption":
                a, b = rng.integers(len(sel["caption_train"]), size=2)
                specs += [("caption", int(a)), ("caption", int(b))]
            else:
                j = int(rng.integers(len(sel[pool[k]])))
                specs += [(pool[k], j, 0), (pool[k], j, 1)]
        plan.append(specs)
    return plan, counts


def spec_to_example(sel, spec):
    if spec[0] == "caption":
        return sel["caption_train"][spec[1]]
    return pair_examples(sel[spec[0]][spec[1]])[spec[2]]


# ----------------------------------------------------------------------------------- optimizers
def stage_optimizer(connector, lm, st, n_updates):
    groups = []
    for params, lr in ((list(connector.parameters()), st["lr_conn"]),
                       ([p for n, p in lm.named_parameters() if "lora_" in n] if st["lr_lora"] else [], st["lr_lora"])):
        if not params:
            continue
        groups.append({"params": [p for p in params if p.ndim >= 2], "lr": lr, "weight_decay": C.WEIGHT_DECAY})
        groups.append({"params": [p for p in params if p.ndim < 2], "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    warm = max(1, round(C.WARMUP_FRAC * n_updates))

    def lam(step):  # 5% linear warm-up, then cosine from peak down to 10% of peak
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, n_updates - warm)
        return C.MIN_LR_FRAC + (1 - C.MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)


def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default=C.LM_ID)
    ap.add_argument("--data", default="results/task2/data")
    ap.add_argument("--cache", default="/tmp/pa3_feats")
    ap.add_argument("--out", default="results/task2/run")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--pool", action="store_true", help="optional 49-token variant")
    ap.add_argument("--stop_after", type=int, default=None)
    a = ap.parse_args()
    from task2.features import FeatureStore

    seed_everything(C.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("training already finished")
        return
    ckpt = Path(a.ckpt_dir) if a.ckpt_dir else out / "recovery"
    sel = load_selection(a.data)
    store = FeatureStore(a.cache)
    reset_peak_memory()
    lm, tok = load_lm(a.lm, dev)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc = Encoder(tok, n_vis=49 if a.pool else C.N_PATCH)
    torch.manual_seed(C.SEED)
    connector = make_connector(dev, lm)
    res_path = out / "resources.json"
    resources = read_json(res_path) if res_path.exists() else {"stages": {}, "skipped_updates": []}
    resources["memory_after_model_load"] = peak_memory_mb()
    resources["environment"] = environment_info()

    plans = []
    for s in range(3):
        p, counts = make_plan(sel, s)
        plans.append(p)
        resources["stages"].setdefault(C.STAGES[s]["name"], {})["planned_mixture_draws"] = counts

    # ---- resume or start
    start = 0
    rec = ckpt / "recovery.pt"
    blob = torch.load(rec, map_location="cpu", weights_only=False) if rec.exists() else None
    if blob is None:
        cal_imgs = list(dict.fromkeys(sel["caption_train"][i]["image"] for i in range(0, len(sel["caption_train"]), 3)))[:256]
        calib = calibrate_scale(lm, connector, enc, sel["caption_train"][:512] + [
            e for p in sel["vqa_train"][:128] for e in pair_examples(p)], store.get(cal_imgs, pool=a.pool))
        resources["calibration"] = calib
        write_json(out / "calibration.json", calib)
        print("calibration:", calib, flush=True)
        for f in ("train_log.jsonl", "dev_log.jsonl"):
            if (out / f).exists():
                (out / f).unlink()
    else:
        start = blob["update"]
        connector.load_state_dict(blob["connector"])
        if stage_of(min(start, TOTAL - 1)) >= 1 or blob.get("lora"):
            add_lora(lm)              # fresh adapters (B = 0) if the checkpoint is the stage-1 end
            named = dict(lm.named_parameters())
            with torch.no_grad():
                for n, t in (blob["lora"] or {}).items():
                    named[n].copy_(t)
        for f in ("train_log.jsonl", "dev_log.jsonl"):
            if (out / f).exists():
                write_jsonl(out / f, [r for r in read_jsonl(out / f) if r["update"] <= start])
        print(f"resumed at update {start}", flush=True)

    lm.train()
    connector.train()
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    dev_at = sorted({0, *range(C.DEV_EVERY, TOTAL + 1, C.DEV_EVERY), *C.STAGE_ENDS})
    if start == 0 and 0 in dev_at:
        m = dev_metrics(lm, connector, enc, store, sel, dev, pool=a.pool)
        append_jsonl(out / "dev_log.jsonl", {"update": 0, "stage": "before", **m})
        print("dev @0", m, flush=True)

    cur_stage, opt, sched = None, None, None
    end = TOTAL if a.stop_after is None else min(TOTAL, start + a.stop_after)
    stage_t0, u = time.perf_counter(), start
    for u in range(start, end):
        s = stage_of(u)
        st = C.STAGES[s]
        if s != cur_stage:
            if s >= 1 and not any("lora_" in n for n, _ in lm.named_parameters()):
                add_lora(lm)
                lm.train()
            opt, sched = stage_optimizer(connector, lm, st, st["updates"])
            if blob is not None and blob["update"] == u:
                if blob.get("stage") == s:          # resuming inside a stage
                    opt.load_state_dict(blob["optimizer"])
                    sched.load_state_dict(blob["scheduler"])
                    stage_t0 -= blob.get("stage_seconds", 0.0)
                scaler.load_state_dict(blob["scaler"])
                set_rng(blob["rng"])
            cur_stage = s
            n_tr = sum(p.numel() for p in connector.parameters()) + sum(
                p.numel() for n, p in lm.named_parameters() if "lora_" in n and st["lr_lora"])
            resources["stages"][st["name"]]["trainable_params"] = n_tr
            resources["stages"][st["name"]]["lora_params"] = sum(p.numel() for n, p in lm.named_parameters() if "lora_" in n)
            reset_peak_memory()
        t_up = time.perf_counter()
        specs = plans[s][u - STAGE_START[s]]
        exs = [spec_to_example(sel, sp) for sp in specs]
        loss_tot, mix = 0.0, {}
        for sp in specs[::2]:
            k = "caption" if sp[0] == "caption" else sp[0].replace("_pool", "").replace("_train", "")
            mix[k] = mix.get(k, 0) + 1
        for m0 in range(0, len(exs), C.MICRO_BATCH):
            mb = exs[m0:m0 + C.MICRO_BATCH]
            encs = [enc.encode(e, image=True) for e in mb]
            ids, lab, att, pos = (t.to(dev) for t in pad_batch(encs, tok.pad_token_id))
            feats = store.get([e["image"] for e in mb], pool=a.pool)
            with torch.autocast(**amp):
                emb = build_inputs(lm, connector, ids, feats, enc.img_id)
                nll, rows = token_nll(lm, emb, att, pos, lab)
            per_ex, _ = per_example_mean(nll.float(), rows, len(mb))
            loss = per_ex.sum() / len(exs)
            scaler.scale(loss).backward()
            loss_tot += float(loss.detach())
        scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, C.CLIP_NORM)
        scale_before = scaler.get_scale()
        lrs = [g["lr"] for g in opt.param_groups]
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        skipped = scaler.get_scale() < scale_before
        if skipped:
            resources["skipped_updates"].append(u + 1)
        done = u + 1
        row = {"update": done, "stage": st["name"], "loss": loss_tot, "lr_connector": lrs[0],
               "lr_lora": lrs[2] if len(lrs) > 2 else None, "grad_norm": float(gnorm), "loss_scale": scale_before,
               "skipped": bool(skipped), "mixture_draws": mix, "seconds": time.perf_counter() - t_up}
        append_jsonl(out / "train_log.jsonl", row)
        if done % 25 == 0 or u == start:
            print(f"update {done}/{TOTAL} [{st['name']}] loss {loss_tot:.4f} lr {lrs[0]:.2e} "
                  f"{row['seconds']:.2f}s", flush=True)
        stage_end = done == STAGE_START[s + 1]
        if done in dev_at:
            m = dev_metrics(lm, connector, enc, store, sel, dev, pool=a.pool)
            append_jsonl(out / "dev_log.jsonl", {"update": done, "stage": st["name"], **m})
            print(f"dev @{done}", {k: (round(v, 4) if isinstance(v, float) else
                                      {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in m.items()}, flush=True)
        if stage_end:
            torch.save({"connector": connector.state_dict(), "lora": lora_state(lm) if s >= 1 else None,
                        "stage": s, "update": done}, out / f"weights_stage{s+1}.pt")
            rs = resources["stages"][st["name"]]
            rs.update({"seconds": time.perf_counter() - stage_t0, "updates": st["updates"],
                       "peak_memory": peak_memory_mb(),
                       "weights_bytes": (out / f"weights_stage{s+1}.pt").stat().st_size})
            stage_t0 = time.perf_counter()
        if done % C.CKPT_EVERY == 0 or stage_end:
            ckpt.mkdir(parents=True, exist_ok=True)
            state = {"connector": connector.state_dict(), "lora": lora_state(lm) if s >= 1 else None,
                     "update": done, "stage": s, "rng": rng_state(), "scaler": scaler.state_dict(),
                     "stage_seconds": time.perf_counter() - stage_t0}
            if not stage_end:   # a new stage builds a fresh optimizer, so nothing to carry over
                state.update({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()})
            else:
                state["stage"] = -1
            torch.save(state, ckpt / "recovery.tmp")
            os.replace(ckpt / "recovery.tmp", ckpt / "recovery.pt")
            resources["recovery_checkpoint_bytes"] = (ckpt / "recovery.pt").stat().st_size
            resources["updates_done"] = done
            write_json(res_path, resources)

    resources["updates_done"] = end
    write_json(res_path, resources)
    if end == TOTAL:
        (out / "DONE").write_text(f"{TOTAL}\n")
    print("finished at update", end, flush=True)


if __name__ == "__main__":
    main()
