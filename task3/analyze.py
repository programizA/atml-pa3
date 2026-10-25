"""Task 3 report material (held-out 800 questions per dataset unless stated).

    python -m task3.analyze --root results/task3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import plotstyle as ps
from common.utils import read_json, write_json
from task3 import config as C

LABEL = {"first": "FirstAns", "sc": "Self-consistency", "logp": "Max mean logp", "logp_rwc": "Mean logp + RWC",
         "sc_score": "Max self-certainty", "sc_rwc": "Self-certainty + RWC", "shortest": "Shortest (control)",
         "vision": "Max vision", "vision_rwc": "Vision + RWC", "visconf": "Max VisConf", "visconf_rwc": "VisConf + RWC",
         "tokfuse": "Max token fusion", "tokfuse_rwc": "Token fusion + RWC"}
DS_LABEL = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}


def load(root, name):
    d = Path(root) / name
    sp = read_json(d / "splits.json")
    res = pd.read_parquet(d / "selection_results.parquet")
    aux = pd.read_parquet(d / "selection_aux.parquet")
    return sp, res[res.sample_id.isin(sp["eval"])], aux[aux.sample_id.isin(sp["eval"])]


def qlevel(res):
    """Per question and method/N: accuracy averaged over the ten orders, and mean cost."""
    return res.groupby(["method", "N", "sample_id"]).agg(acc=("correct", "mean"), cost=("cost", "mean")).reset_index()


def boot_diff(a, b, n=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    d = (np.asarray(b, float) - np.asarray(a, float)) * 100
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def method_table(root, out):
    rows = []
    for name in C.DATASETS:
        sp, res, aux = load(root, name)
        ql = qlevel(res)
        cov = aux.groupby(["N", "sample_id"]).coverage.mean().reset_index()
        for N in C.BUDGETS:
            ref = ql[(ql.method == "sc") & (ql.N == N)].set_index("sample_id").acc
            for m in ql.method.unique():
                x = ql[(ql.method == m) & (ql.N == N)].set_index("sample_id")
                d = boot_diff(ref.loc[x.index], x.acc) if m != "sc" else (0.0, 0.0, 0.0)
                rows.append({"dataset": name, "N": N, "method": m, "acc": x.acc.mean(), "mean_cost": x.cost.mean(),
                             "diff_vs_sc_pp": d[0], "ci_lo": d[1], "ci_hi": d[2]})
            rows.append({"dataset": name, "N": N, "method": "oracle", "acc": cov[cov.N == N].coverage.mean(),
                         "mean_cost": x.cost.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "methods_by_budget.csv", index=False)
    # LaTeX: N = 8 and 32
    lines = [r"\begin{tabular}{l" + "rr" * 2 + "}", r"\toprule",
             r" & \multicolumn{2}{c}{MATH} & \multicolumn{2}{c}{MathVista} \\",
             r"Rule & $N{=}8$ & $N{=}32$ & $N{=}8$ & $N{=}32$ \\", r"\midrule"]
    order = ["first", "sc", "logp", "logp_rwc", "sc_score", "sc_rwc", "shortest", "vision", "vision_rwc", "visconf",
             "visconf_rwc", "tokfuse", "tokfuse_rwc", "oracle"]
    for m in order:
        cells = []
        for name in C.DATASETS:
            for N in (8, 32):
                r = df[(df.dataset == name) & (df.N == N) & (df.method == m)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                s = f"{100*r.acc:.1f}"
                if m not in ("sc", "oracle", "first") and not np.isnan(r.diff_vs_sc_pp):
                    sig = "^{*}" if (r.ci_lo > 0 or r.ci_hi < 0) else ""
                    s += f" ({r.diff_vs_sc_pp:+.1f}${sig}$)"
                cells.append(s)
        lab = "Oracle coverage" if m == "oracle" else LABEL[m]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if m in ("first", "shortest", "tokfuse_rwc"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_methods.tex").write_text("\n".join(lines) + "\n")
    return df


def decomposition(root, out):
    rows = []
    for name in C.DATASETS:
        _, res, aux = load(root, name)
        a = aux.set_index(["sample_id", "order", "N"])
        for m in ["sc", "sc_rwc"] + (["visconf_rwc"] if name == "mathvista" else []):
            r = res[res.method == m].set_index(["sample_id", "order", "N"])
            j = r.join(a[["coverage", "n_groups"]])
            for N in C.BUDGETS:
                x = j.xs(N, level="N")
                rows.append({"dataset": name, "method": m, "N": N, "selected_correct": x.correct.mean(),
                             "missed_available": ((~x.correct) & x.coverage).mean(),
                             "no_correct_candidate": (~x.coverage).mean(), "mean_distinct_groups": x.n_groups.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "decomposition.csv", index=False)
    return df


def novelty(root, out):
    """Does the k-th rollout add a new answer group / the first correct group, or repeat one?"""
    rows = []
    for name in C.DATASETS:
        d = Path(root) / name
        sp = read_json(d / "splits.json")
        roll = pd.read_parquet(d / "rollouts.parquet")
        od = np.load(d / "orders.npz")
        orders = dict(zip(od["ids"], od["orders"]))
        new_g = np.zeros(32)
        new_c = np.zeros(32)
        cnt = 0
        for qid, sub in roll[roll.sample_id.isin(sp["eval"])].groupby("sample_id"):
            g = sub.group.to_numpy()
            c = sub.correct.to_numpy()
            for o in orders[qid]:
                seen, have_c = set(), False
                for k, i in enumerate(o):
                    new_g[k] += g[i] not in seen
                    if c[i] and not have_c:
                        new_c[k] += 1
                        have_c = True
                    seen.add(g[i])
                cnt += 1
        for k in range(32):
            rows.append({"dataset": name, "k": k + 1, "p_new_group": new_g[k] / cnt, "p_first_correct": new_c[k] / cnt})
    df = pd.DataFrame(rows)
    df.to_csv(out / "novelty.csv", index=False)
    return df


def fig_scaling(df, out):
    import matplotlib.pyplot as plt

    ps.setup()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2))
    show = {"math": ["first", "sc", "sc_score", "sc_rwc"], "mathvista": ["first", "sc", "sc_score", "visconf", "visconf_rwc"]}
    color = {"first": ps.NEUTRAL, "sc": ps.PALETTE[0], "sc_score": ps.PALETTE[1], "sc_rwc": ps.PALETTE[2],
             "visconf": ps.PALETTE[3], "visconf_rwc": ps.PALETTE[4]}
    for j, name in enumerate(C.DATASETS):
        sub = df[df.dataset == name]
        for row, xcol in enumerate(("N", "mean_cost")):
            ax = axes[row, j]
            o = sub[sub.method == "oracle"].sort_values("N")
            ax.plot(o[xcol], o.acc * 100, color=ps.INK2, ls="--", lw=1.1, label="Oracle coverage")
            for m in show[name]:
                s = sub[sub.method == m].sort_values("N")
                ax.plot(s[xcol], s.acc * 100, marker="o", ms=3, color=color[m], label=LABEL[m])
            ax.set_xscale("log", base=2 if xcol == "N" else 10)
            if xcol == "N":
                ax.set_xticks(C.BUDGETS, [str(b) for b in C.BUDGETS])
                ax.set(title=DS_LABEL[name], xlabel="candidates N")
            else:
                ax.set(xlabel="mean generated tokens per question (FirstAns: 1 rollout)")
            ax.set_ylabel("accuracy (%)")
        axes[0, j].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    ps.save(fig, out / "fig_scaling")


def signals_table(root, out):
    rows = []
    for name in C.DATASETS:
        s = read_json(Path(root) / name / "signals_summary.json")
        for sig, v in s["signals"].items():
            rows.append({"dataset": name, "signal": sig, **v})
    df = pd.DataFrame(rows)
    df.to_csv(out / "signals.csv", index=False)
    return df


def fig_trajectories(root, out):
    import matplotlib.pyplot as plt

    f = Path(root) / "mathvista" / "trajectories.json"
    if not f.exists():
        return
    tr = read_json(f)
    ps.setup()
    fig, axes = plt.subplots(1, len(tr), figsize=(7.0, 1.9), squeeze=False)
    for ax, t in zip(axes[0], tr):
        for key, ls in (("visconf_pick", "-"), ("vision_pick", "--")):
            p = t[key]
            lab = f"{'VisConf' if key=='visconf_pick' else 'Vision'} pick ({'correct' if p['correct'] else 'wrong'})"
            ax.plot(p["c"], color=ps.PALETTE[0], ls=ls, lw=1, label=f"$c_t$, {lab}")
            ax.plot(np.array(p["v"]) * 10, color=ps.PALETTE[1], ls=ls, lw=1, label=f"$10\\,v_t$, {lab}")
        ax.set(title=f"MathVista q{t['sample_id']}: {t['rule'].replace('_', ' ')}", xlabel="generated token (32-token mean)")
        ax.legend(fontsize=5)
    fig.tight_layout()
    ps.save(fig, out / "fig_trajectories")


def adaptive_table(root, out):
    rows = []
    for name in C.DATASETS:
        s = read_json(Path(root) / name / "adaptive_summary.json")
        for N in (4, 8, 16, 32):
            f = s["fixed_sc"][str(N)]
            rows.append({"dataset": name, "policy": f"Fixed SC N={N}", "acc": f["acc"], "cost": f["cost"],
                         "mean_rollouts": N})
        for fam, v in s["controllers"].items():
            e = v["eval"]
            sel = v["selected"]
            rows.append({"dataset": name, "policy": f"{fam} (tau={sel['tau']}" + (f", delta={sel['delta']})" if fam == 'quality' else ")"),
                         "acc": e["acc"], "cost": e["cost"], "mean_rollouts": e["mean_rollouts"],
                         "dev_acc": v["dev"]["acc"], "dev_fixed32": v["dev"]["fixed32_acc"],
                         "diff_vs_fixed32_pp": e["acc_diff_vs_fixed32_pp"]["mean"],
                         "ci_lo": e["acc_diff_vs_fixed32_pp"]["lo"], "ci_hi": e["acc_diff_vs_fixed32_pp"]["hi"],
                         "stop4": e["stop_distribution"].get("4", 0), "stop8": e["stop_distribution"].get("8", 0),
                         "stop16": e["stop_distribution"].get("16", 0), "stop32": e["stop_distribution"].get("32", 0),
                         "wrong_early_stop_rate": e["incorrect_early_stops_rate"],
                         "recoverable_share": e["incorrect_early_stops_recoverable_by_fixed32"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "adaptive.csv", index=False)
    return df


def fig_adaptive(root, df, out):
    import matplotlib.pyplot as plt

    ps.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, name in zip(axes, C.DATASETS):
        s = read_json(Path(root) / name / "adaptive_summary.json")
        xs = [s["fixed_sc"][str(N)]["cost"] for N in C.BUDGETS]
        ys = [s["fixed_sc"][str(N)]["acc"] * 100 for N in C.BUDGETS]
        ax.plot(xs, ys, marker="o", ms=3, color=ps.PALETTE[0], label="Fixed-N self-consistency")
        for N, x, y in zip(C.BUDGETS, xs, ys):
            ax.annotate(f"N={N}", (x, y), fontsize=5.5, color=ps.INK2, xytext=(2, -7), textcoords="offset points")
        for fam, mk, c in (("agreement", "s", ps.PALETTE[1]), ("quality", "D", ps.PALETTE[2])):
            e = s["controllers"][fam]["eval"]
            ax.plot(e["cost"], e["acc"] * 100, mk, ms=5, color=c, mec="white", label=f"Adaptive: {fam}")
        ax.set(xscale="log", title=DS_LABEL[name], xlabel="mean generated tokens per question", ylabel="accuracy (%)")
        ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    ps.save(fig, out / "fig_adaptive")


def examples(root, out):
    """Fixed selection rules for the qualitative examples (see the docstrings inline)."""
    ex = {}
    # (1) selection disagreement: first held-out MATH question (permuted order) where, at N=32
    #     (order 0), self-consistency and max self-certainty choose different answers and exactly
    #     one of them is correct.
    for name in C.DATASETS:
        sp, res, aux = load(root, name)
        r = res[(res.N == 32) & (res.order == 0)].set_index(["sample_id", "method"])
        roll = pd.read_parquet(Path(root) / name / "rollouts.parquet")
        alt = "sc_score" if name == "math" else "visconf"
        for qid in sp["eval"]:
            a, b = r.loc[(qid, "sc")], r.loc[(qid, alt)]
            if a.group != b.group and a.correct != b.correct:
                sub = roll[roll.sample_id == qid]
                groups = sub.group.value_counts()
                ex[f"{name}_selection_disagreement"] = {
                    "sample_id": qid, "gold": sub.gold.iloc[0], "vote_counts": groups.head(5).to_dict(),
                    "sc_correct": bool(a.correct), f"{alt}_correct": bool(b.correct),
                    "group_correct": sub.groupby("group").correct.first().to_dict()}
                break
    # (2) premature stopping: first held-out question (agreement controller, any order) that stopped
    #     early with a wrong answer while fixed-32 self-consistency on the same order is correct.
    for name in C.DATASETS:
        sp = read_json(Path(root) / name / "splits.json")
        sim = pd.read_parquet(Path(root) / name / "adaptive_eval_agreement.parquet")
        bad = sim[(sim.n < 32) & (sim.correct == 0) & (sim.correct32 == 1)]
        bad = bad.assign(pos=bad.sample_id.map({q: i for i, q in enumerate(sp["eval"])})).sort_values(["pos", "order"])
        if len(bad):
            b = bad.iloc[0]
            roll = pd.read_parquet(Path(root) / name / "rollouts.parquet")
            od = np.load(Path(root) / name / "orders.npz")
            order = dict(zip(od["ids"], od["orders"]))[b.sample_id][int(b.order)]
            sub = roll[roll.sample_id == b.sample_id].reset_index(drop=True)
            ex[f"{name}_premature_stop"] = {"sample_id": b.sample_id, "order": int(b.order), "stopped_at": int(b.n),
                                            "gold": sub.gold.iloc[0],
                                            "revealed_groups": [sub.group[i] for i in order[:int(b.n)]],
                                            "revealed_correct": [bool(sub.correct[i]) for i in order[:int(b.n)]],
                                            "all32_counts": sub.group.value_counts().head(5).to_dict()}
    # (3) revision: first live question (in live order) for each of W->C and C->W.
    lq = Path(root) / "live" / "live_per_question.csv"
    if lq.exists():
        q = pd.read_csv(lq)
        ids = read_json(Path(root) / "live_questions.json")["ids"]
        q["pos"] = q.sample_id.map({x: i for i, x in enumerate(ids)})
        from common.utils import read_jsonl
        calls = pd.DataFrame([r for f in (Path(root) / "live").glob("calls_shard*.jsonl") for r in read_jsonl(f)])
        for tr in ("W->C", "C->W"):
            s = q[q.transition == tr].sort_values("pos")
            if len(s):
                qid = s.iloc[0].sample_id
                c = calls[calls.sample_id == qid].set_index("slot")
                ex[f"live_{tr}"] = {"sample_id": qid, "gold": c.loc[1].gold, "initial_answer": c.loc[1].extracted,
                                    "revised_answer": c.loc[4].extracted,
                                    "revision_excerpt_end": c.loc[4].response[-600:]}
    write_json(out / "examples.json", ex)
    return ex


def live_table(root, out):
    f = Path(root) / "live" / "live_summary.json"
    if not f.exists():
        return None
    s = read_json(f)
    rows = []
    for st in ("single", "parallel", "refinement"):
        v = s[st]
        rows.append({"strategy": st, "accuracy": v["accuracy"], "out_tokens": v["mean_output_tokens"],
                     "in_tokens": v["mean_input_tokens"], "sec_per_q": v["mean_seconds"],
                     "trunc": v.get("truncation_rate", v.get("truncation_rate_per_response", v.get("truncation_rate_revision"))),
                     "extract": v.get("extraction_rate", v.get("extraction_rate_per_response", v.get("extraction_rate_revision")))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "live.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task3")
    a = ap.parse_args()
    out = Path(a.root) / "report"
    out.mkdir(parents=True, exist_ok=True)
    df = method_table(a.root, out)
    fig_scaling(df, out)
    dec = decomposition(a.root, out)
    nov = novelty(a.root, out)
    sig = signals_table(a.root, out)
    fig_trajectories(a.root, out)
    ad = adaptive_table(a.root, out)
    fig_adaptive(a.root, ad, out)
    lv = live_table(a.root, out)
    ex = examples(a.root, out)
    pd.set_option("display.width", 220)
    print(df[df.N.isin([8, 32])].pivot_table(index="method", columns=["dataset", "N"], values="acc").round(3))
    print(dec[dec.N.isin([4, 8, 32])].round(3).to_string())
    print(ad.round(3).to_string())
    if lv is not None:
        print(lv.round(3))
    print(list(ex))


if __name__ == "__main__":
    main()
