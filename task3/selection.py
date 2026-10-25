"""Offline best-of-N simulation on the supplied rollouts (Steps 1, 2 and the inputs for Step 5).

For every question, each of the ten saved candidate orders, and every budget N in {1,2,4,8,16,32},
the candidate set is the first N rollouts of that order. Every rule sees only those N candidates:

  first        FirstAns: the first candidate's answer (cost = its tokens only)
  sc           self-consistency: one vote per rollout, most votes wins
  logp         highest mean log-probability L_i           logp_rwc   L_i with rank-weighted voting
  sc_score     highest self-certainty K_i                 sc_rwc     K_i with rank-weighted consensus
  shortest     shortest rollout (length-only control)
  MathVista only:
  vision       highest visual engagement V_i              vision_rwc
  visconf      highest VisConf Z_i = V_i * C_i            visconf_rwc
  tokfuse      highest token-wise fusion mean_t(c_t v_t)  tokfuse_rwc

Ties: highest-score rules take the earliest candidate in the current order; vote rules take the
answer whose first supporter comes earliest; RWC ranks use average ranks for tied scores.
VisConf's calibration (5th/95th percentiles of the pooled token KL) is recomputed for every
candidate set from those candidates' tokens only.

Rank-weighted consensus: rank the N candidates by score (1 = best, average ranks for ties),
weight w_i = (N - r_i) / (N - 1), and pick the answer with the largest total weight.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common.utils import read_json, write_json
from task3 import config as C
from task3.answers import NO_ANSWER

BASE_METHODS = ["first", "sc", "logp", "logp_rwc", "sc_score", "sc_rwc", "shortest"]
VLM_METHODS = ["vision", "vision_rwc", "visconf", "visconf_rwc", "tokfuse", "tokfuse_rwc"]


class Question:
    """All 32 rollouts of one question, in rollout-id order."""

    def __init__(self, sub, tok, vlm):
        self.qid = sub.sample_id.iloc[0]
        self.T = sub["T"].to_numpy()
        self.L = np.nan_to_num(sub["L"].to_numpy(), nan=-np.inf)
        self.K = np.nan_to_num(sub["K"].to_numpy(), nan=-np.inf)
        labs = sub["group"].tolist()
        self.labels = labs
        uniq = list(dict.fromkeys(labs))
        self.gid = np.array([uniq.index(x) for x in labs])
        self.group_names = uniq
        self.noans = np.array([u == NO_ANSWER for u in uniq])
        corr = sub["correct"].to_numpy()
        self.group_correct = np.array([bool(corr[self.gid == g].any()) for g in range(len(uniq))])
        self.correct = corr
        rows = sub.index.to_numpy()
        off = tok["offsets"]
        self.kl = [tok["kl"][off[r]:off[r + 1]] for r in rows]
        self.vlm = vlm
        if vlm:
            self.V = np.nan_to_num(sub["V"].to_numpy(), nan=-np.inf)
            self.v = [tok["v"][off[r]:off[r + 1]] for r in rows]


def first_max(scores):
    """Index (into the candidate list) of the highest score; ties -> earliest."""
    return int(np.argmax(scores))


def vote(gids, weights):
    """Winning group by total weight; ties -> the group whose first supporter is earliest.
    Returns (winner, sorted totals descending, unique-leader flag)."""
    tot = {}
    first = {}
    for pos, (g, w) in enumerate(zip(gids, weights)):
        tot[g] = tot.get(g, 0.0) + w
        first.setdefault(g, pos)
    best = max(tot.values())
    leaders = [g for g in tot if np.isclose(tot[g], best)]
    win = min(leaders, key=lambda g: first[g])
    vals = sorted(tot.values(), reverse=True)
    return win, vals, len(leaders) == 1, tot


def rwc(gids, scores):
    n = len(gids)
    r = rankdata(-np.asarray(scores, dtype=float), method="average")
    w = (n - r) / (n - 1)
    return vote(gids, w)


def visconf_parts(q, cand):
    """Calibrated confidence per candidate, VisConf score, and token-wise fusion score."""
    kls = [q.kl[i] for i in cand]
    pooled = np.concatenate(kls)
    q05, q95 = np.quantile(pooled, [0.05, 0.95])          # linear interpolation (NumPy default)
    den = max(q95 - q05, 1e-8)
    C_i, Zt = np.empty(len(cand)), np.empty(len(cand))
    for j, i in enumerate(cand):
        c = np.clip((q.kl[i] - q05) / den, 0.0, 1.0)
        C_i[j] = c.mean() if len(c) else 0.0
        Zt[j] = (c * q.v[i]).mean() if len(c) else 0.0
    return C_i, q.V[cand] * C_i, Zt


def simulate_question(q, orders):
    """Rows of (order, N, method, correct, group, selected rollout) + per-(order, N) aux record."""
    out, aux = [], []
    for o, order in enumerate(orders):
        for N in C.BUDGETS:
            cand = order[:N]
            g = q.gid[cand]
            cost_all = int(q.T[cand].sum())
            rec = {"order": o, "N": N, "cost": cost_all, "cost_first": int(q.T[cand[0]]),
                   "coverage": bool(q.group_correct[np.unique(g)].any()), "n_groups": int(len(np.unique(g))),
                   "frac_correct_in_set": float(q.correct[cand].mean())}
            sel = {}
            sel["first"] = cand[0]
            win, vals, uniq, tot = vote(g, np.ones(N))
            sel["sc"] = ("group", win)
            rec.update({"sc_group": int(win), "sc_unique_leader": uniq, "sc_leader_frac": vals[0] / N,
                        "sc_leader_noans": bool(q.noans[win])})
            sel["logp"] = cand[first_max(q.L[cand])]
            sel["sc_score"] = cand[first_max(q.K[cand])]
            sel["shortest"] = cand[int(np.argmin(q.T[cand]))]
            if N > 1:
                sel["logp_rwc"] = ("group", rwc(g, q.L[cand])[0])
                w, v, _, _ = rwc(g, q.K[cand])
                sel["sc_rwc"] = ("group", w)
                rec.update({"rwc_K_group": int(w), "rwc_K_margin": (v[0] - (v[1] if len(v) > 1 else 0.0)) / max(sum(v), 1e-12)})
            else:
                sel["logp_rwc"] = sel["sc_rwc"] = cand[0]
                rec.update({"rwc_K_group": int(g[0]), "rwc_K_margin": 1.0})
            if q.vlm:
                Ci, Z, Zt = visconf_parts(q, cand)
                sel["vision"] = cand[first_max(q.V[cand])]
                sel["visconf"] = cand[first_max(Z)]
                sel["tokfuse"] = cand[first_max(Zt)]
                if N > 1:
                    sel["vision_rwc"] = ("group", rwc(g, q.V[cand])[0])
                    w, v, _, _ = rwc(g, Z)
                    sel["visconf_rwc"] = ("group", w)
                    sel["tokfuse_rwc"] = ("group", rwc(g, Zt)[0])
                    rec.update({"rwc_Z_group": int(w), "rwc_Z_margin": (v[0] - (v[1] if len(v) > 1 else 0.0)) / max(sum(v), 1e-12)})
                else:
                    sel["vision_rwc"] = sel["visconf_rwc"] = sel["tokfuse_rwc"] = cand[0]
                    rec.update({"rwc_Z_group": int(g[0]), "rwc_Z_margin": 1.0})
            for m, s in sel.items():
                if isinstance(s, tuple):
                    grp, rid = s[1], -1
                else:
                    grp, rid = int(q.gid[s]), int(s)
                cost = rec["cost_first"] if m == "first" else cost_all
                out.append((o, N, m, bool(q.group_correct[grp]), int(grp), rid, cost))
            aux.append(rec)
    return out, aux


def run(name, root, limit=None):
    root = Path(root) / name
    roll = pd.read_parquet(root / "rollouts.parquet")
    tok = dict(np.load(root / "tokens.npz"))
    od = np.load(root / "orders.npz")
    orders = {q: o for q, o in zip(od["ids"], od["orders"])}
    vlm = name == "mathvista"
    rows, auxs = [], []
    t0 = time.perf_counter()
    qids = list(dict.fromkeys(roll.sample_id))
    if limit:
        qids = qids[:limit]
    for k, (qid, sub) in enumerate(roll.groupby("sample_id", sort=False)):
        if qid not in qids:
            continue
        q = Question(sub, tok, vlm)
        r, a = simulate_question(q, orders[qid])
        rows += [(qid, *x) for x in r]
        auxs += [{"sample_id": qid, **x} for x in a]
        if (k + 1) % 200 == 0:
            print(f"{name}: {k+1} questions, {time.perf_counter()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows, columns=["sample_id", "order", "N", "method", "correct", "group", "rollout", "cost"])
    aux = pd.DataFrame(auxs)
    res.to_parquet(root / "selection_results.parquet")
    aux.to_parquet(root / "selection_aux.parquet")
    print(f"{name}: done in {time.perf_counter()-t0:.0f}s, {len(res)} rows", flush=True)
    return res, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task3")
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for name in C.DATASETS:
        if a.only and name != a.only:
            continue
        run(name, a.root, a.limit)


if __name__ == "__main__":
    main()
