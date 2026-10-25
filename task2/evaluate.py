"""Task 2 evaluation: development tracking during training, and the final grounding test.

Development (called from training): caption reference loss on 96 dev captions, and the mean NLL
of each reference answer on both dev suites with the correct image, the complementary partner's
image, and no image. The image-swap penalty is

    dNLL = NLL(answer | question, partner image) - NLL(answer | question, correct image)

so a positive value means the correct image makes the right answer more likely.

Final (python -m task2.evaluate): both test suites, greedy decoding (16 new tokens, batch 8,
thinking disabled), under four image conditions with the question and references fixed:
correct, complementary swap, mismatched (an image from a different pair of the same suite, fixed
in advance), and no image. Grounding questions are scored by normalised exact match, VQAv2 by
soft consensus accuracy against the ten human answers. Confidence intervals come from 4,000
bootstrap resamples of whole pairs, using the same resampled pairs for every condition.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common.utils import peak_memory_mb, reset_peak_memory, write_json, write_jsonl
from task2 import config as C
from task2.common import load_lm, load_selection, load_weights, make_connector, pair_examples, score_nll
from task2.model import Encoder, build_inputs, pad_batch
from task2.vqa_norm import normalize, soft_accuracy

SUITES = {"ground": ("ground_dev", "ground_test"), "vqa": ("vqa_dev", "vqa_test")}


@torch.no_grad()
def dev_metrics(lm, connector, enc, store, sel, dev, pool=False):
    was = lm.training
    lm.eval()
    connector.eval()
    caps = sel["caption_dev"]
    cap_nll = score_nll(lm, connector, enc, store, caps, [c["image"] for c in caps], dev, exclude_eot=False, pool=pool)
    out = {"caption_loss": float(np.mean(cap_nll))}
    for suite, (dev_name, _) in SUITES.items():
        exs = [e for p in sel[dev_name] for e in pair_examples(p)]
        r = {}
        for cond, keys in (("correct", [e["image"] for e in exs]), ("complementary", [e["partner_image"] for e in exs]),
                           ("no_image", [None] * len(exs))):
            r[cond] = score_nll(lm, connector, enc, store, exs, keys, dev, pool=pool)
        out[suite] = {"nll_correct": float(np.mean(r["correct"])), "nll_complementary": float(np.mean(r["complementary"])),
                      "nll_no_image": float(np.mean(r["no_image"])),
                      "swap_penalty": float(np.mean(np.array(r["complementary"]) - np.array(r["correct"]))),
                      "no_image_penalty": float(np.mean(np.array(r["no_image"]) - np.array(r["correct"])))}
    if was:
        lm.train()
        connector.train()
    return out


def mismatch_map(pairs, seed=C.SEED):
    """For every question, an image from a different pair of the same suite (never either image
    of its own pair). Fixed once, saved, and reused for every comparison."""
    rng = np.random.default_rng(seed)
    imgs = [(i, p[f"image_{s}"]) for i, p in enumerate(pairs) for s in ("a", "b")]
    out = {}
    for i, p in enumerate(pairs):
        for s in ("a", "b"):
            cand = [k for j, k in imgs if j != i and k not in (p["image_a"], p["image_b"])]
            out[f"{p['pair_id']}:{s}"] = cand[int(rng.integers(len(cand)))]
    return out


@torch.no_grad()
def generate(lm, connector, enc, store, examples, keys, dev, batch=C.GEN_BATCH, max_new=C.GEN_MAX_NEW, pool=False):
    tok = enc.tok
    eos = [enc.eot, tok.convert_tokens_to_ids("<|endoftext|>")]
    preds, t0 = [], time.perf_counter()
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    for i in range(0, len(examples), batch):
        exs, ks = examples[i:i + batch], keys[i:i + batch]
        encs = [enc.encode(e, image=k is not None, with_target=False) for e, k in zip(exs, ks)]
        ids, _, att, pos = (t.to(dev) for t in pad_batch(encs, tok.pad_token_id, side="left"))
        with_img = [k for k in ks if k is not None]
        feats = store.get(with_img, pool=pool) if with_img else None
        with torch.autocast(**amp):
            emb = build_inputs(lm, connector, ids, feats, enc.img_id)
            gen = lm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                              eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True,
                              temperature=None, top_p=None, top_k=None)
        for g in gen:
            g = g.tolist()
            n = next((j for j, t in enumerate(g) if t in eos), len(g))
            preds.append(tok.decode(g[:n], skip_special_tokens=True).strip())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return preds, (time.perf_counter() - t0) / max(1, len(examples))


def score(ex, pred):
    if ex["kind"] == "vqa":
        return soft_accuracy(pred, ex["refs"])
    return float(normalize(pred) == ex["canon"])


def bootstrap(pair_vals, n=C.N_BOOT, seed=C.SEED):
    """Mean and 95% CI over pairs. pair_vals: array (n_pairs,) or dict of arrays sharing resamples."""
    rng = np.random.default_rng(seed)
    first = next(iter(pair_vals.values())) if isinstance(pair_vals, dict) else pair_vals
    idx = rng.integers(0, len(first), size=(n, len(first)))
    def ci(v):
        v = np.asarray(v, dtype=float)
        bs = v[idx].mean(1)
        return {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    return {k: ci(v) for k, v in pair_vals.items()} if isinstance(pair_vals, dict) else ci(pair_vals)


def final_eval(lm, connector, enc, store, sel, dev, out_dir, pool=False):
    out_dir = Path(out_dir)
    rows, summary = [], {}
    for suite, (_, test_name) in SUITES.items():
        pairs = sel[test_name]
        exs = [e for p in pairs for e in pair_examples(p)]
        mm = mismatch_map(pairs)
        write_json(out_dir / f"mismatch_map_{suite}.json", mm)
        keys = {"correct": [e["image"] for e in exs], "complementary": [e["partner_image"] for e in exs],
                "mismatched": [mm[f"{e['pair_id']}:{e['side']}"] for e in exs], "no_image": [None] * len(exs)}
        res = {}
        for cond in C.CONDITIONS:
            reset_peak_memory()
            preds, t_per = generate(lm, connector, enc, store, exs, keys[cond], dev, pool=pool)
            nll = score_nll(lm, connector, enc, store, exs, keys[cond], dev, pool=pool)
            res[cond] = {"preds": preds, "nll": nll, "sec_per_answer": t_per, "mem": peak_memory_mb()}
            for e, k, p, l in zip(exs, keys[cond], preds, nll):
                rows.append({"suite": suite, "condition": cond, "pair_id": e["pair_id"], "side": e["side"],
                             "qtype": e["qtype"], "question": e["question"], "image_shown": k, "own_image": e["image"],
                             "target": e["target"], "canon": e["canon"], "partner_canon": e["partner_canon"],
                             "refs": e["refs"], "pred": p, "pred_norm": normalize(p), "score": score(e, p),
                             "follows_partner": float(normalize(p) == e["partner_canon"]), "ref_nll": l})
        summary[suite] = summarize_suite(exs, res)
    write_jsonl(out_dir / "test_predictions.jsonl", rows)
    write_json(out_dir / "test_summary.json", summary)
    return summary


def summarize_suite(exs, res):
    n_pairs = len(exs) // 2
    pair_ids = [exs[2 * i]["pair_id"] for i in range(n_pairs)]
    out = {"n_pairs": n_pairs, "n_questions": len(exs), "conditions": {}}
    per_pair = {}
    for cond, r in res.items():
        sc = np.array([score(e, p) for e, p in zip(exs, r["preds"])])
        norm = [normalize(p) for p in r["preds"]]
        exact = np.array([float(n == e["canon"]) for e, n in zip(exs, norm)])
        follow = np.array([float(n == e["partner_canon"]) for e, n in zip(exs, norm)])
        nll = np.array(r["nll"])
        pa = exact.reshape(-1, 2).min(1)                         # both match their own canonical answer
        ident = np.array([float(norm[2 * i] == norm[2 * i + 1]) for i in range(n_pairs)])
        empty = np.array([float(n == "") for n in norm])
        per_pair[cond] = {"acc": sc.reshape(-1, 2).mean(1), "nll": nll.reshape(-1, 2).mean(1)}
        by_type = {}
        for t in sorted({e["qtype"] for e in exs}):
            idx = [i for i, e in enumerate(exs) if e["qtype"] == t]
            pidx = sorted({i // 2 for i in idx})
            by_type[t] = {"accuracy": float(sc[idx].mean()), "pair_accuracy": float(pa[pidx].mean()),
                          "n_questions": len(idx), "ci": bootstrap(sc.reshape(-1, 2).mean(1)[pidx])}
        out["conditions"][cond] = {
            "accuracy": float(sc.mean()), "accuracy_ci": bootstrap(sc.reshape(-1, 2).mean(1)),
            "pair_accuracy": float(pa.mean()), "identical_answer_rate": float(ident.mean()),
            "empty_answer_rate": float(empty.mean()), "follows_partner_answer_rate": float(follow.mean()),
            "mean_ref_nll": float(nll.mean()), "sec_per_answer": r["sec_per_answer"], "peak_memory": r["mem"],
            "by_type": by_type}
    adv = {}
    for other in ("complementary", "mismatched", "no_image"):
        d_acc = per_pair["correct"]["acc"] - per_pair[other]["acc"]
        d_nll = per_pair[other]["nll"] - per_pair["correct"]["nll"]
        adv[f"correct_vs_{other}"] = bootstrap({"accuracy_gain": d_acc, "nll_penalty": d_nll})
    out["advantages"] = adv
    out["pair_ids"] = pair_ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default=C.LM_ID)
    ap.add_argument("--data", default="results/task2/data")
    ap.add_argument("--cache", default="/tmp/pa3_feats")
    ap.add_argument("--run", default="results/task2/run")
    ap.add_argument("--weights", default=None, help="default: <run>/weights_stage3.pt")
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    from task2.features import FeatureStore

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run)
    if (run / "test_summary.json").exists():
        print("final evaluation already done")
        return
    lm, tok = load_lm(a.lm, dev)
    connector = make_connector(dev, lm)
    load_weights(lm, connector, a.weights or run / "weights_stage3.pt")
    lm.eval()
    connector.eval()
    enc = Encoder(tok, n_vis=49 if a.pool else C.N_PATCH)
    store = FeatureStore(a.cache)
    s = final_eval(lm, connector, enc, store, load_selection(a.data), dev, run, pool=a.pool)
    for suite, v in s.items():
        print(suite, {c: round(x["accuracy"], 3) for c, x in v["conditions"].items()})


if __name__ == "__main__":
    main()
