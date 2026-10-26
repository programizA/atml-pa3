"""Task 2 report material: learning curves, final grounding tables, aligned examples, resources.

    python -m task2.analyze --root results/task2 [--images /tmp/pa3_images]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import plotstyle as ps
from common.utils import read_json, read_jsonl, write_json
from task2 import config as C

COND_LABEL = {"correct": "Correct", "complementary": "Compl. swap", "mismatched": "Mismatched", "no_image": "No image"}
COND_COLOR = {"correct": ps.PALETTE[0], "complementary": ps.PALETTE[1], "mismatched": ps.PALETTE[2],
              "no_image": ps.NEUTRAL}
SUITE_LABEL = {"ground": "Annotation-derived grounding", "vqa": "Natural VQAv2"}


def _bounds(ax, ends):
    for x in ends[:-1]:
        ax.axvline(x, color=ps.MUTED, lw=0.7, ls=":")


def fig_curves(run, out):
    import matplotlib.pyplot as plt

    tl = Path(run) / "train_log.jsonl"
    dl = Path(run) / "dev_log.jsonl"
    if not tl.exists() or not dl.exists():
        return
    t = pd.DataFrame(read_jsonl(tl))
    d = read_jsonl(dl)
    ps.setup()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    ax = axes[0]
    ax.plot(t["update"], t["loss"].rolling(50, min_periods=1).mean(), color=ps.INK2, lw=1)
    ax.set(title="(a) Training loss", xlabel="update", ylabel="loss (50-update mean)")
    ax = axes[1]
    ax.plot([r["update"] for r in d], [r["caption_loss"] for r in d], marker="o", ms=3, color=ps.PALETTE[0])
    ax.set(title="(b) Dev caption loss", xlabel="update", ylabel="mean token NLL")
    ax = axes[2]
    for suite, ls in (("ground", "-"), ("vqa", "--")):
        for cond, key in (("correct", "nll_correct"), ("complementary", "nll_complementary"), ("no_image", "nll_no_image")):
            ax.plot([r["update"] for r in d], [r[suite][key] for r in d], ls=ls, marker="o", ms=2.5,
                    color=COND_COLOR[cond], lw=1, label=f"{COND_LABEL[cond]}" if suite == "ground" else None)
    ax.set(title="(c) Dev answer NLL", xlabel="update", ylabel="NLL (solid: grounding,\ndashed: VQAv2)", yscale="log")
    ax.legend(fontsize=6, loc="upper right")
    ax = axes[3]
    for suite, c in (("ground", ps.PALETTE[0]), ("vqa", ps.PALETTE[1])):
        ax.plot([r["update"] for r in d], [r[suite]["swap_penalty"] for r in d], marker="o", ms=3, color=c,
                label=SUITE_LABEL[suite].split()[-1])
    ax.axhline(0, color=ps.MUTED, lw=0.7)
    ax.set(title=r"(d) Image-swap penalty $\Delta$NLL", xlabel="update", ylabel="NLL(partner) - NLL(correct)")
    ax.legend(fontsize=6)
    for ax in axes:
        _bounds(ax, C.STAGE_ENDS)
    fig.tight_layout()
    ps.save(fig, Path(out) / "fig_curves")


def tables(run, out):
    f = Path(run) / "test_summary.json"
    if not f.exists():
        return None
    s = read_json(f)
    rows, types, adv = [], [], []
    for suite, v in s.items():
        for cond, x in v["conditions"].items():
            ci = x["accuracy_ci"]
            rows.append({"suite": suite, "condition": cond, "accuracy": x["accuracy"], "acc_lo": ci["lo"], "acc_hi": ci["hi"],
                         "pair_accuracy": x["pair_accuracy"], "identical": x["identical_answer_rate"],
                         "empty": x["empty_answer_rate"], "follows_partner": x["follows_partner_answer_rate"],
                         "ref_nll": x["mean_ref_nll"], "ms_per_answer": x["sec_per_answer"] * 1000})
            for t, y in x["by_type"].items():
                types.append({"suite": suite, "condition": cond, "type": t, "accuracy": y["accuracy"],
                              "pair_accuracy": y["pair_accuracy"], "n": y["n_questions"]})
        for k, y in v["advantages"].items():
            adv.append({"suite": suite, "comparison": k, **{f"{m}_{q}": y[m][q] for m in y for q in ("mean", "lo", "hi")}})
    df, ty, ad = pd.DataFrame(rows), pd.DataFrame(types), pd.DataFrame(adv)
    df.to_csv(Path(out) / "final_results.csv", index=False)
    ty.to_csv(Path(out) / "final_by_type.csv", index=False)
    ad.to_csv(Path(out) / "advantages.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Suite & Image & Accuracy [95\% CI] & Pair acc. & Identical & Follows partner & Empty & Ref.~NLL \\", r"\midrule"]
    for suite in ("ground", "vqa"):
        for _, r in df[df.suite == suite].iterrows():
            lines.append(f"{SUITE_LABEL[suite] if r.condition=='correct' else ''} & {COND_LABEL[r.condition]} & "
                         f"{100*r.accuracy:.1f} [{100*r.acc_lo:.1f}, {100*r.acc_hi:.1f}] & {100*r.pair_accuracy:.1f} & "
                         f"{100*r.identical:.1f} & {100*r.follows_partner:.1f} & {100*r.empty:.1f} & {r.ref_nll:.2f} \\\\")
        lines.append(r"\midrule" if suite == "ground" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (Path(out) / "table_final.tex").write_text("\n".join(lines) + "\n")
    return df


def fig_types(out):
    import matplotlib.pyplot as plt

    f = Path(out) / "final_by_type.csv"
    if not f.exists():
        return
    ty = pd.read_csv(f)
    ps.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [2, 3]})
    for ax, suite in zip(axes, ("ground", "vqa")):
        sub = ty[ty.suite == suite]
        tps = list(dict.fromkeys(sub["type"]))
        x = np.arange(len(tps))
        w = 0.8 / len(C.CONDITIONS)
        for j, cond in enumerate(C.CONDITIONS):
            vals = [sub[(sub["type"] == t) & (sub.condition == cond)].accuracy.mean() * 100 for t in tps]
            ax.bar(x + (j - 1.5) * w, vals, w * 0.9, color=COND_COLOR[cond], label=COND_LABEL[cond])
        ax.set_xticks(x, tps)
        ax.set(title=SUITE_LABEL[suite], ylabel="accuracy (%)", ylim=(0, 100))
    axes[1].legend(ncol=4, fontsize=6, loc="upper right")
    fig.tight_layout()
    ps.save(fig, Path(out) / "fig_by_type")


def select_examples(run, n_per=1):
    """Stated rule. Walk each (suite, question type) in pair-ID order.
    (a) Grounded success: the first pair answered correctly on both sides with the correct images
        AND, under the complementary swap, answered with the partner's answer on both sides.
    (b) Prior-driven failure (VQAv2 only): the first pair given the same answer on both images.
    Pairs are chosen from these flags alone and each pair appears at most once."""
    rows = pd.DataFrame(read_jsonl(Path(run) / "test_predictions.jsonl"))
    picks, used = [], set()
    for (suite, qtype), sub in rows.groupby(["suite", "qtype"], sort=False):
        cor = sub[sub.condition == "correct"]
        comp = sub[sub.condition == "complementary"]
        pids = sorted(dict.fromkeys(cor.pair_id), key=lambda p: int(p.split(":")[1]))
        rules = [("grounded_success", lambda c, k: (c.pred_norm.values == c.canon.values).all()
                  and (k.follows_partner > 0).all())]
        if suite == "vqa":
            rules.append(("same_answer_both_images", lambda c, k: c.pred_norm.iloc[0] == c.pred_norm.iloc[1]))
        for reason, test in rules:
            got = 0
            for pid in pids:
                if pid in used or got >= n_per:
                    continue
                c = cor[cor.pair_id == pid].sort_values("side")
                k = comp[comp.pair_id == pid].sort_values("side")
                if test(c, k):
                    picks.append({"suite": suite, "qtype": qtype, "reason": reason, "pair_id": pid})
                    used.add(pid)
                    got += 1
    return picks


def fig_examples(run, out, images_root, picks, max_rows=8):
    import matplotlib.pyplot as plt
    from PIL import Image

    from task2.features import image_path

    rows = pd.DataFrame(read_jsonl(Path(run) / "test_predictions.jsonl"))
    picks = picks[:max_rows]
    if not picks or not images_root or not Path(images_root).exists():
        return
    ps.setup()
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 1.55 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1, 2.6]})
    axes = np.atleast_2d(axes)
    for r, pk in enumerate(picks):
        sub = rows[rows.pair_id == pk["pair_id"]]
        a = sub[(sub.side == "a") & (sub.condition == "correct")].iloc[0]
        b = sub[(sub.side == "b") & (sub.condition == "correct")].iloc[0]
        for j, ex in enumerate((a, b)):
            ax = axes[r, j]
            try:
                ax.imshow(Image.open(image_path(images_root, ex.own_image)).convert("RGB"))
            except Exception:
                pass
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"image {ex.side.upper()}: ref '{ex.canon}'", fontsize=6.5, loc="center", fontweight="normal")
        txt = [f"[{pk['suite']} / {pk['qtype']} / {pk['reason'].replace('_', ' ')}]", f"Q: {a.question}"]
        for cond in C.CONDITIONS:
            pa = sub[(sub.side == "a") & (sub.condition == cond)].pred.iloc[0]
            pb = sub[(sub.side == "b") & (sub.condition == cond)].pred.iloc[0]
            txt.append(f"{COND_LABEL[cond]:<12s} A: '{pa}'   B: '{pb}'")
        axes[r, 2].axis("off")
        axes[r, 2].text(0, 0.5, "\n".join(txt), fontsize=6.2, va="center", family="DejaVu Sans Mono")
    fig.tight_layout()
    ps.save(fig, Path(out) / "fig_examples")


def resources(root, run, out):
    res = {}
    f = Path(root) / "feature_cache_stats.json"
    if f.exists():
        s = read_json(f)
        res["feature_caching"] = {"images": s["caching"]["n_images"], "download_min": s["download"]["seconds"] / 60,
                                  "encode_min": s["caching"]["seconds"] / 60, "cache_gb": s["caching"]["bytes"] / 2**30,
                                  "peak_gb": (s["caching"].get("peak_allocated_mb") or 0) / 1024}
    r = Path(run) / "resources.json"
    if r.exists():
        s = read_json(r)
        res["calibration"] = s.get("calibration")
        res["stages"] = {k: {"min": v.get("seconds", 0) / 60, "trainable_params": v.get("trainable_params"),
                             "lora_params": v.get("lora_params"),
                             "peak_gb": ((v.get("peak_memory") or {}).get("peak_allocated_mb") or 0) / 1024,
                             "mixture_draws": v.get("planned_mixture_draws")} for k, v in s.get("stages", {}).items()}
        res["skipped_updates"] = len(s.get("skipped_updates", []))
    t = Path(run) / "test_summary.json"
    if t.exists():
        s = read_json(t)
        res["generation_ms_per_answer"] = {f"{su}/{c}": v["conditions"][c]["sec_per_answer"] * 1000
                                           for su, v in s.items() for c in v["conditions"]}
    write_json(Path(out) / "resources.json", res)
    return res


def label_audit_summary(root, run, out):
    """If label_audit.csv has been filled in, re-score the annotation-derived suite on images whose
    generated label was judged sound (doubtful != y), to show how much label noise moves the numbers."""
    f = Path(root) / "data" / "label_audit.csv"
    p = Path(run) / "test_predictions.jsonl"
    if not f.exists() or not p.exists():
        return None
    au = pd.read_csv(f, dtype=str)
    au["doubtful"] = au["doubtful(y/n)"].astype(str).str.strip().str.lower().eq("y")
    if not au["doubtful(y/n)"].notna().any():
        return None
    bad = set(zip(au[au.doubtful].pair_id, au[au.doubtful].side))
    rows = pd.DataFrame(read_jsonl(p))
    g = rows[rows.suite == "ground"].copy()
    g["clean"] = [(pid, s) not in bad for pid, s in zip(g.pair_id, g.side)]
    res = {"n_doubtful_images": len(bad), "by_kind": {}}
    for kind in ("presence", "counting"):
        k = g[g.qtype == kind]
        res["n_doubtful_by_kind"] = res.get("n_doubtful_by_kind", {}) | {kind: int((~k[k.condition == "correct"].clean).sum())}
        res["by_kind"][kind] = {c: {"all": float(k[k.condition == c].score.mean()),
                                    "clean_only": float(k[(k.condition == c) & k.clean].score.mean())}
                                for c in C.CONDITIONS}
    write_json(Path(out) / "label_audit_summary.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task2")
    ap.add_argument("--run", default=None)
    ap.add_argument("--images", default="/tmp/pa3_images")
    a = ap.parse_args()
    run = Path(a.run or Path(a.root) / "run")
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    fig_curves(run, out)
    df = tables(run, out)
    fig_types(out)
    if (run / "test_predictions.jsonl").exists():
        picks = select_examples(run)
        write_json(out / "example_picks.json", picks)
        fig_examples(run, out, a.images, picks)
    print(label_audit_summary(a.root, run, out))
    print(resources(a.root, run, out))
    if df is not None:
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
