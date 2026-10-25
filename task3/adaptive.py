"""Step 5: adaptive sampling, simulated on the supplied pools.

Rollouts are revealed in each saved order. After n = 4, 8 and 16 a controller may stop; otherwise
it stops at 32. Either way it answers with self-consistency over the revealed rollouts, so only
the stopping decision differs between controllers.

  agreement      stop if the leading answer is unique (no tie), is not NO_ANSWER, and has at
                 least a fraction tau of the n votes.
  quality-aware  the agreement test, plus: the RWC winner (self-certainty on MATH, VisConf on
                 MathVista, recomputed on the revealed rollouts) equals the vote winner, and the
                 RWC margin (top vote weight - runner-up) / total weight is at least delta.

Thresholds are chosen on the 200 development questions only: the cheapest setting whose dev
accuracy is within 1 percentage point of fixed-32 self-consistency (ties: higher accuracy, then
smaller tau, then smaller delta). "Always use 32" is included as a fallback. The chosen settings
are frozen, then evaluated once on the 800 held-out questions.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import read_json, write_json
from task3 import config as C


def load(name, root):
    root = Path(root) / name
    aux = pd.read_parquet(root / "selection_aux.parquet")
    res = pd.read_parquet(root / "selection_results.parquet")
    sc = res[res.method == "sc"][["sample_id", "order", "N", "correct"]].rename(columns={"correct": "sc_correct"})
    aux = aux.merge(sc, on=["sample_id", "order", "N"])
    return aux[aux.N.isin([*C.CHECK_AT, 32])], read_json(root / "splits.json")


def simulate(aux, family, tau, delta, quality_key):
    """One row per (question, order): stopping n, cost, correctness of the returned SC answer."""
    a = aux.set_index(["sample_id", "order", "N"]).sort_index()
    out = []
    stop_ok = (a.sc_unique_leader & ~a.sc_leader_noans & (a.sc_leader_frac >= tau - 1e-12))
    if family == "quality":
        stop_ok &= (a[f"rwc_{quality_key}_group"] == a.sc_group) & (a[f"rwc_{quality_key}_margin"] >= delta - 1e-12)
    stop_ok = stop_ok.unstack("N")
    cost = a["cost"].unstack("N")
    corr = a["sc_correct"].unstack("N")
    n_stop = pd.Series(32, index=stop_ok.index)
    for n in sorted(C.CHECK_AT, reverse=True):          # earliest check that passes wins
        n_stop = n_stop.where(~stop_ok[n].astype(bool), n)
    idx = np.searchsorted([*C.CHECK_AT, 32], n_stop.to_numpy())
    cols = [*C.CHECK_AT, 32]
    df = pd.DataFrame({"n": n_stop.to_numpy(),
                       "cost": cost[cols].to_numpy()[np.arange(len(idx)), idx],
                       "correct": corr[cols].to_numpy()[np.arange(len(idx)), idx].astype(float),
                       "correct32": corr[32].to_numpy().astype(float)}, index=stop_ok.index).reset_index()
    return df


def q_level(df):
    """Average over orders within a question."""
    return df.groupby("sample_id").agg(acc=("correct", "mean"), cost=("cost", "mean"), n=("n", "mean"))


def choose(aux, dev_ids, family, quality_key):
    d = aux[aux.sample_id.isin(dev_ids)]
    base = simulate(d, "agreement", 2.0, 0, quality_key)            # tau > 1 never stops = fixed 32
    base_acc = q_level(base).acc.mean()
    grid = []
    deltas = C.DELTAS if family == "quality" else (None,)
    for tau, delta in itertools.product(C.TAUS, deltas):
        s = q_level(simulate(d, family, tau, delta or 0.0, quality_key))
        grid.append({"tau": tau, "delta": delta, "dev_acc": s.acc.mean(), "dev_cost": s.cost.mean(), "dev_n": s.n.mean()})
    bq = q_level(base)
    grid.append({"tau": "fixed32", "delta": None, "dev_acc": bq.acc.mean(), "dev_cost": bq.cost.mean(), "dev_n": 32.0})
    g = pd.DataFrame(grid)
    ok = g[g.dev_acc >= base_acc - C.DEV_TOLERANCE_PP / 100 - 1e-12].copy()
    ok["tau_sort"] = ok.tau.map(lambda t: 9.0 if t == "fixed32" else t)
    ok["delta_sort"] = ok.delta.fillna(0)
    best = ok.sort_values(["dev_cost", "dev_acc", "tau_sort", "delta_sort"], ascending=[True, False, True, True]).iloc[0]
    return g, best, base_acc


def bootstrap_diff(a, b, n=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    d = (np.asarray(b) - np.asarray(a)) * 100
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def run(name, root):
    aux, sp = load(name, root)
    qk = "K" if name == "math" else "Z"
    ev = aux[aux.sample_id.isin(sp["eval"])]
    fixed = {}
    allaux = pd.read_parquet(Path(root) / name / "selection_aux.parquet")
    allres = pd.read_parquet(Path(root) / name / "selection_results.parquet")
    for N in C.BUDGETS:
        sub = allres[(allres.method == "sc") & (allres.N == N) & allres.sample_id.isin(sp["eval"])]
        cst = allaux[(allaux.N == N) & allaux.sample_id.isin(sp["eval"])]
        fixed[N] = {"acc": float(sub.groupby("sample_id").correct.mean().mean()),
                    "cost": float(cst.groupby("sample_id").cost.mean().mean())}
    out = {"fixed_sc": fixed, "controllers": {}}
    base_ev = q_level(simulate(ev, "agreement", 2.0, 0, qk))
    for family in ("agreement", "quality"):
        grid, best, dev_base = choose(aux, sp["dev"], family, qk)
        grid.to_csv(Path(root) / name / f"adaptive_grid_{family}.csv", index=False)
        if best.tau == "fixed32":
            sim = simulate(ev, "agreement", 2.0, 0, qk)
        else:
            sim = simulate(ev, family, float(best.tau), float(best.delta or 0.0), qk)
        sim.to_parquet(Path(root) / name / f"adaptive_eval_{family}.parquet")
        ql = q_level(sim)
        early_wrong = sim[(sim.n < 32) & (sim.correct == 0)]
        out["controllers"][family] = {
            "selected": {"tau": best.tau, "delta": None if pd.isna(best.delta) else best.delta},
            "dev": {"acc": float(best.dev_acc), "cost": float(best.dev_cost), "fixed32_acc": float(dev_base)},
            "eval": {"acc": float(ql.acc.mean()), "cost": float(ql.cost.mean()), "mean_rollouts": float(ql.n.mean()),
                     "cost_saving_vs_fixed32": 1 - float(ql.cost.mean()) / fixed[32]["cost"],
                     "acc_diff_vs_fixed32_pp": bootstrap_diff(base_ev.acc.to_numpy(), ql.acc.reindex(base_ev.index).to_numpy()),
                     "stop_distribution": {int(k): float(v) for k, v in sim.n.value_counts(normalize=True).sort_index().items()},
                     "incorrect_early_stops_rate": float(len(early_wrong) / len(sim)),
                     "incorrect_early_stops_recoverable_by_fixed32": float(early_wrong.correct32.mean()) if len(early_wrong) else 0.0,
                     "early_stop_precision": float(sim[sim.n < 32].correct.mean()) if (sim.n < 32).any() else None}}
        print(name, family, out["controllers"][family]["selected"], out["controllers"][family]["eval"], flush=True)
    # retrospective: stopping vs agreement and pool accuracy (full 32-rollout pool)
    roll = pd.read_parquet(Path(root) / name / "rollouts.parquet")
    pool_acc = roll.groupby("sample_id").correct.mean()
    sim = pd.read_parquet(Path(root) / name / "adaptive_eval_agreement.parquet")
    sim["pool_acc"] = sim.sample_id.map(pool_acc)
    sim["pool_bin"] = pd.cut(sim.pool_acc, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.99, 1.0],
                             labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
    rel = sim.groupby("pool_bin", observed=True).agg(mean_n=("n", "mean"), early_stop=("n", lambda x: (x < 32).mean()),
                                                     acc=("correct", "mean"), share=("n", "size"))
    rel["share"] = rel["share"] / rel["share"].sum()
    rel.to_csv(Path(root) / name / "adaptive_vs_pool_accuracy.csv")
    out["vs_pool_accuracy"] = rel.reset_index().to_dict(orient="records")
    write_json(Path(root) / name / "adaptive_summary.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task3")
    a = ap.parse_args()
    for name in C.DATASETS:
        run(name, a.root)


if __name__ == "__main__":
    main()
