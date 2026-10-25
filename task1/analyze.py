"""Turn Task 1's saved results into the report's tables and figures. Every number in the report
comes from here, so nothing is copied by hand.

    python -m task1.analyze            -> results/task1/report/{*.csv, *.tex, fig_*.pdf/png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import plotstyle as ps
from common.utils import read_json, read_jsonl, write_json
from task1 import config as C

BENCH_ORDER = [("gsm8k", "GSM8K", "flex. EM"), ("hellaswag", "HellaSwag", "acc_norm"),
               ("boolq", "BoolQ", "acc"), ("arc_easy", "ARC-Easy", "acc_norm")]
RUNS = ["full", "l2sp", "lora4", "lora16", "oplora"]


# --------------------------------------------------------------------------------- benchmarks
def per_example(eval_dir, cond):
    """{benchmark: np.array of per-example primary correctness (0/1)} for paired comparisons."""
    d = Path(eval_dir) / cond
    out = {}
    if (d / "gsm8k_samples.jsonl").exists():
        rows = read_jsonl(d / "gsm8k_samples.jsonl")
        out["gsm8k"] = np.array([r.get("correct_flexible-extract", 0.0) for r in rows])
    if (d / "likelihood_samples.jsonl").exists():
        rows = pd.DataFrame(read_jsonl(d / "likelihood_samples.jsonl"))
        for t, metric in [("hellaswag", "acc_norm"), ("boolq", "acc"), ("arc_easy", "acc_norm")]:
            sub = rows[rows.task == t].sort_values("doc_id")
            if len(sub):
                out[t] = sub[metric].astype(float).to_numpy()
    return out


def paired_bootstrap(a, b, n=4000, seed=C.SEED):
    """95% CI for mean(b - a) in percentage points, resampling examples (paired)."""
    rng = np.random.default_rng(seed)
    d = (b - a) * 100
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def benchmark_table(eval_dir, out_dir):
    rows, ex = [], {}
    for cond, _, _ in C.CONDITIONS:
        f = Path(eval_dir) / cond / "summary.json"
        if not f.exists():
            continue
        s = read_json(f)
        ex[cond] = per_example(eval_dir, cond)
        r = {"condition": cond, "label": C.LABELS[cond]}
        g = s.get("gsm8k", {})
        r.update({"gsm8k": g.get("flexible"), "gsm8k_strict": g.get("strict"),
                  "gsm8k_truncated": g.get("truncated_rate"), "gsm8k_unextractable": g.get("unextractable_flexible_rate"),
                  "gsm8k_mean_tokens": g.get("mean_gen_tokens")})
        for t in ("hellaswag", "boolq", "arc_easy"):
            v = s.get(t, {})
            prim = v.get("acc") if t == "boolq" else v.get("acc_norm")
            r.update({t: prim, f"{t}_acc": v.get("acc")})
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ex
    num = [c for c in df.columns if c not in ("condition", "label")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")   # missing benchmark -> NaN
    base = df[df.condition == "baseline"].iloc[0] if (df.condition == "baseline").any() else None
    for b, _, _ in BENCH_ORDER:
        df[f"{b}_dpp"] = (df[b] - base[b]) * 100 if base is not None else np.nan
    ret = [b for b, _, _ in BENCH_ORDER[1:]]
    df["retention_mean_dpp"] = df[[f"{b}_dpp" for b in ret]].mean(axis=1)
    df.to_csv(out_dir / "benchmarks.csv", index=False)

    # paired 95% CIs of the change from baseline, per benchmark
    ci = []
    if "baseline" in ex:
        for cond in df.condition:
            if cond == "baseline":
                continue
            for b, _, _ in BENCH_ORDER:
                if b in ex[cond] and b in ex["baseline"] and len(ex[cond][b]) == len(ex["baseline"][b]):
                    m, lo, hi = paired_bootstrap(ex["baseline"][b], ex[cond][b])
                    ci.append({"condition": cond, "benchmark": b, "dpp": m, "lo": lo, "hi": hi})
    pd.DataFrame(ci).to_csv(out_dir / "benchmarks_delta_ci.csv", index=False)

    # LaTeX: score (change in pp)
    lines = [r"\begin{tabular}{l" + "r" * len(BENCH_ORDER) + "r}", r"\toprule",
             "Condition & " + " & ".join(f"{n} ({m})" for _, n, m in BENCH_ORDER) + r" & Ret.~$\Delta$ \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for b, _, _ in BENCH_ORDER:
            v = r[b] * 100
            if np.isnan(v):
                cells.append("--")
                continue
            cells.append(f"{v:.1f}" if r.condition == "baseline" else f"{v:.1f} ({r[f'{b}_dpp']:+.1f})")
        rm = "--" if r.condition == "baseline" else f"{r.retention_mean_dpp:+.2f}"
        lines.append(f"{r.label} & " + " & ".join(cells) + f" & {rm} \\\\")
        if r.condition in ("baseline", "full_mid"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_benchmarks.tex").write_text("\n".join(lines) + "\n")
    return df, ex


def comparisons(df, ex, out_dir):
    """The four controlled comparisons at final checkpoints, with paired CIs on the difference."""
    pairs = [("lora4", "lora16", "Adapter rank (4 -> 16)"), ("full", "l2sp", "L2-SP penalty"),
             ("lora16", "oplora", "Orthogonal protection"), ("full", "lora16", "Full FT vs LoRA-16"),
             ("full_mid", "full", "Full FT: mid -> final")]
    out = []
    for a, b, name in pairs:
        if a not in ex or b not in ex:
            continue
        for bench, _, _ in BENCH_ORDER:
            if bench in ex[a] and bench in ex[b]:
                m, lo, hi = paired_bootstrap(ex[a][bench], ex[b][bench])
                out.append({"comparison": name, "A": a, "B": b, "benchmark": bench, "B_minus_A_pp": m,
                            "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": lo > 0 or hi < 0})
    res = pd.DataFrame(out)
    res.to_csv(out_dir / "controlled_comparisons.csv", index=False)
    return res


# ----------------------------------------------------------------------------- training curves
def training_logs(runs_dir):
    logs = {}
    for r in RUNS:
        f = Path(runs_dir) / r / "train_log.jsonl"
        if f.exists():
            logs[r] = pd.DataFrame(read_jsonl(f))
    return logs


def fig_training(logs, out_dir, window=25):
    import matplotlib.pyplot as plt

    if not logs:
        return
    ps.setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    styles = {"full": "-", "l2sp": "--", "lora4": "-", "lora16": "--", "oplora": ":"}
    for r, df in logs.items():
        y = df.sft_loss.rolling(window, min_periods=1).mean()
        ax.plot(df["update"], y, color=ps.T1_COLORS[r], ls=styles[r], lw=1.2, label=C.LABELS[r])
    ax.set(xlabel="optimizer update", ylabel=f"SFT loss ({window}-update mean)", title="(a) Response loss")
    ax.legend(ncol=2, loc="upper right")
    ax = axes[1]
    for r, df in logs.items():
        if r in ("full", "lora16"):
            ax.plot(df["update"], df.lr, color=ps.T1_COLORS[r], label=f"{C.LABELS[r]}{' (all LoRA)' if r=='lora16' else ' (both)'}")
    ax.set(xlabel="optimizer update", ylabel="learning rate", title="(b) Schedule", yscale="log")
    ax.legend(loc="lower center")
    ax = axes[2]
    if "l2sp" in logs and "omega" in logs["l2sp"]:
        d = logs["l2sp"]
        ax.plot(d["update"], d.lam_omega, color=ps.T1_COLORS["l2sp"], label=r"$\lambda\Omega$ (L2-SP)")
        ax.set(xlabel="optimizer update", ylabel=r"penalty $\lambda\Omega$", title="(c) L2-SP penalty")
        ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, out_dir / "fig_training")


def regularization_table(logs, runs_dir, out_dir):
    rows = []
    for r in ("full", "l2sp"):
        res = Path(runs_dir) / r / "resources.json"
        if not res.exists():
            continue
        s = read_json(res)
        for upd, info in sorted(s.get("checkpoints", {}).items(), key=lambda kv: int(kv[0])):
            if "displacement_l2" in info:
                rows.append({"run": r, "update": int(upd), "displacement_l2": info["displacement_l2"]})
    if "l2sp" in logs and "omega" in logs["l2sp"]:
        d = logs["l2sp"].iloc[-1]
        full = logs.get("full")
        last = lambda df: float(df.sft_loss.tail(50).mean())
        summ = {"final_omega": float(d.omega), "final_lam_omega": float(d.lam_omega),
                "final_sft_loss_l2sp_last50": last(logs["l2sp"]),
                "final_sft_loss_full_last50": last(full) if full is not None else None,
                "lam_omega_over_sft": float(d.lam_omega) / last(logs["l2sp"]),
                "displacement_check_sqrt_2omega": float(np.sqrt(2 * d.omega))}
        write_json(out_dir / "regularization_summary.json", summ)
    pd.DataFrame(rows).to_csv(out_dir / "displacement.csv", index=False)


# --------------------------------------------------------------------------------- resources
def resources_table(runs_dir, eval_dir, out_dir):
    rows = []
    for r in RUNS:
        f = Path(runs_dir) / r / "resources.json"
        if not f.exists():
            continue
        s = read_json(f)
        ck = s.get("checkpoints", {})
        final = ck.get(str(s.get("total_updates")), {})
        oh = s.get("method_overhead", {})
        ev = Path(eval_dir) / r / "summary.json"
        merge = read_json(ev).get("merge_seconds") if ev.exists() else None
        rows.append({
            "run": r, "label": C.LABELS[r], "trainable_params": s["trainable_params"],
            "train_minutes": s.get("train_seconds", 0) / 60,
            "sec_per_update": s.get("train_seconds", 0) / max(1, s.get("updates_done", 1)),
            "peak_alloc_gb": (s.get("peak_memory_training", {}).get("peak_allocated_mb") or 0) / 1024,
            "peak_reserved_gb": (s.get("peak_memory_training", {}).get("peak_reserved_mb") or 0) / 1024,
            "eval_ckpt_mb": (final.get("eval_checkpoint_bytes") or 0) / 2**20,
            "recovery_ckpt_mb": (final.get("recovery_bytes") or 0) / 2**20,
            "skipped_updates": len(s.get("skipped_updates", [])),
            "overhead": "; ".join(_overhead(r, oh, final, merge)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "resources.csv", index=False)
    if not df.empty:
        lines = [r"\begin{tabular}{lrrrrrl}", r"\toprule",
                 r"Run & Trainable & Time (min) & Peak mem. (GB) & Eval ckpt (MB) & Recovery (MB) & Method overhead \\",
                 r"\midrule"]
        for _, x in df.iterrows():
            lines.append(f"{x.label} & {_fmt_params(x.trainable_params)} & {x.train_minutes:.0f} & "
                         f"{x.peak_alloc_gb:.1f} & {x.eval_ckpt_mb:.0f} & {x.recovery_ckpt_mb:.0f} & {x.overhead} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out_dir / "table_resources.tex").write_text("\n".join(lines) + "\n")
    return df


def _overhead(r, oh, final, merge):
    out = []
    if "reference_bytes" in oh:
        out.append(f"$\\theta_0$ {oh['reference_bytes']/2**30:.2f} GB ({oh.get('reference_dtype','').replace('torch.','')}),"
                   f" load {oh.get('reference_transfer_seconds',0):.1f}s")
    if oh.get("basis_bytes"):
        out.append(f"bases {oh['basis_bytes']/2**20:.0f} MB, SVD {oh['basis_seconds']:.0f}s")
    if final.get("export_seconds") is not None and r == "oplora":
        out.append(f"export {final['export_seconds']:.2f}s")
    if merge is not None:
        out.append(f"merge {merge:.0f}s")
    return out


def _fmt_params(n):
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.2f}B"


# ------------------------------------------------------------------------------- diagnostics
def fig_diagnostics(diag_dir, out_dir):
    import matplotlib.pyplot as plt

    f = Path(diag_dir) / "diagnostics.json"
    if not f.exists():
        return
    d = read_json(f)
    conds = [c for c, _, _ in C.CONDITIONS if c in d["conditions"]]
    layers = [str(L) for L in d["layers"]]
    rows = []
    for c in conds:
        for L in layers:
            x = d["conditions"][c][L]
            rows.append({"condition": c, "layer": int(L), "rho_k": x["rho_k"], "rho_k_input": x["rho_k_input_side"],
                         "eta": x["eta"], "R_math": x["R_math"]["mean"], "R_nonmath": x["R_nonmath"]["mean"],
                         "R_ratio": x["R_ratio_math_over_nonmath"], "eff_rank": x["dW_effective_rank"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "diagnostics.csv", index=False)
    ps.setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    w = 0.8 / len(layers)
    xs = np.arange(len(conds))
    hatches = ["", "////", "...."]
    for j, L in enumerate(layers):
        sub = df[df.layer == int(L)].set_index("condition").loc[conds]
        for ax, col in [(axes[0], "rho_k"), (axes[1], "eta")]:
            ax.bar(xs + (j - 1) * w, sub[col], w * 0.92, color=[ps.T1_COLORS[c] for c in conds],
                   hatch=hatches[j], edgecolor="white", linewidth=0.3, label=f"layer {L}")
    rand = list(d["random_rho_reference"].values())[0]
    axes[0].axhline(rand, color=ps.INK2, lw=0.8, ls="--")
    axes[0].text(len(conds) - 0.5, rand, " random", va="bottom", ha="right", fontsize=6, color=ps.INK2)
    axes[0].set(title=r"(a) $\rho_k$: share in top-$k$ output space", ylabel=r"$\rho_k$")
    axes[1].set(title=r"(b) $\eta$: relative update size", ylabel=r"$\eta=\|\Delta W\|_F/\|W_0\|_F$", yscale="log")
    for ax in axes[:2]:
        ax.set_xticks(xs, [C.LABELS[c].replace("Full FT + ", "").replace(" (mid)", "\n(mid)") for c in conds],
                      rotation=45, ha="right")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec=ps.INK2, hatch=h, lw=0.5) for h in hatches],
                   labels=[f"layer {L}" for L in layers], loc="upper left", fontsize=6)
    ax = axes[2]
    for c in conds:
        sub = df[df.condition == c].sort_values("layer")
        ax.plot(sub.R_nonmath, sub.R_math, marker="o", ms=3.5, color=ps.T1_COLORS[c], lw=1,
                ls="--" if c == "full_mid" else "-")
        last = sub.iloc[-1]
        ax.annotate(C.LABELS[c].replace("Full FT + ", ""), (last.R_nonmath, last.R_math), fontsize=5.5,
                    color=ps.INK2, xytext=(2, 2), textcoords="offset points")
    lim = [min(df.R_math.min(), df.R_nonmath.min()) * 0.8, max(df.R_math.max(), df.R_nonmath.max()) * 1.25]
    if lim[0] > 0:
        ax.plot(lim, lim, color=ps.MUTED, lw=0.7, ls=":")
        ax.set(xscale="log", yscale="log")
    ax.set(title=r"(c) mean $R(z)$, layers 0$\to$27", xlabel="non-math probes (HellaSwag)", ylabel="math probes (GSM8K)")
    fig.tight_layout()
    ps.save(fig, out_dir / "fig_diagnostics")
    return df


def fig_benchmarks(df, ci_path, out_dir):
    import matplotlib.pyplot as plt

    if df.empty or not Path(ci_path).exists():
        return
    ci = pd.read_csv(ci_path)
    if ci.empty:
        return
    ps.setup()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    conds = [c for c in df.condition if c != "baseline"]
    for ax, (b, name, metric) in zip(axes, BENCH_ORDER):
        sub = ci[ci.benchmark == b].set_index("condition").reindex(conds)
        y = np.arange(len(conds))[::-1]
        ax.axvline(0, color=ps.MUTED, lw=0.8)
        for yy, c in zip(y, conds):
            r = sub.loc[c]
            ax.plot([r.lo, r.hi], [yy, yy], color=ps.T1_COLORS[c], lw=1.6, solid_capstyle="round")
            ax.plot(r.dpp, yy, "o", ms=4, color=ps.T1_COLORS[c], mec="white", mew=0.6)
        ax.set_title(f"{name}", fontsize=8)
        ax.set_xlabel("change vs baseline (pp)")
    axes[0].set_yticks(np.arange(len(conds))[::-1], [C.LABELS[c] for c in conds])
    fig.tight_layout()
    ps.save(fig, out_dir / "fig_benchmark_deltas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task1")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    df, ex = benchmark_table(root / "eval", out)
    comparisons(df, ex, out)
    fig_benchmarks(df, out / "benchmarks_delta_ci.csv", out)
    logs = training_logs(root / "runs")
    fig_training(logs, out)
    regularization_table(logs, root / "runs", out)
    resources_table(root / "runs", root / "eval", out)
    fig_diagnostics(root / "diagnostics", out)
    if not df.empty:
        print(df[["label"] + [b for b, _, _ in BENCH_ORDER] + [f"{b}_dpp" for b, _, _ in BENCH_ORDER]].round(4).to_string())
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
