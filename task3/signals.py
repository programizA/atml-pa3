"""Step 3: what do the quality signals capture? (all at N = 32, i.e. the whole pool)

* Within-question associations: Spearman correlation of each signal (length, L, K, V, VisConf,
  token-wise fusion) with correctness, and of length with each score, computed inside each question
  and averaged over questions where it is defined (correctness varies).
* Quartile contrast: correctness rate of a question's top-8 rollouts by a score minus its bottom-8,
  averaged over questions (all questions, and only "mixed" ones with both right and wrong rollouts).
* MathVista: correlation between calibrated confidence c_t and visual engagement v_t inside each
  rollout, summarised across questions; check that  Z_token - Z = Cov_t(c, v).
* A fixed set of illustrative trajectories, chosen by a stated rule.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common.utils import read_json, write_json
from task3 import config as C


def pool_scores(sub, tok):
    """Per-rollout C, Z (VisConf), Z_token with calibration over all 32 rollouts, plus c_t/v_t."""
    off = tok["offsets"]
    rows = sub.index.to_numpy()
    kls = [tok["kl"][off[r]:off[r + 1]] for r in rows]
    vs = [tok["v"][off[r]:off[r + 1]] for r in rows]
    q05, q95 = np.quantile(np.concatenate(kls), [0.05, 0.95])
    den = max(q95 - q05, 1e-8)
    cs = [np.clip((k - q05) / den, 0, 1) for k in kls]
    Cm = np.array([c.mean() for c in cs])
    Vm = np.array([v.mean() for v in vs])
    Zt = np.array([(c * v).mean() for c, v in zip(cs, vs)])
    cov = np.array([((c - c.mean()) * (v - v.mean())).mean() for c, v in zip(cs, vs)])
    r_cv = np.array([np.corrcoef(c, v)[0, 1] if len(c) > 2 and c.std() > 0 and v.std() > 0 else np.nan
                     for c, v in zip(cs, vs)])
    return Cm, Vm * Cm, Zt, cov, r_cv, cs, vs


def quartile_gap(score, correct, k=8):
    """Correct rate among the k highest-scoring rollouts minus the k lowest (ties by rollout id)."""
    r = rankdata(-score, method="ordinal")       # 1 = highest; ties broken by position (rollout id)
    top = correct[r <= k].mean()
    bot = correct[r > len(score) - k].mean()
    return top - bot


def analyze(name, root):
    root = Path(root) / name
    roll = pd.read_parquet(root / "rollouts.parquet")
    tok = dict(np.load(root / "tokens.npz"))
    sp = read_json(root / "splits.json")
    vlm = name == "mathvista"
    per_q, cv_rows, traj_pool = [], [], {}
    ident_err = 0.0
    for qid, sub in roll.groupby("sample_id", sort=False):
        corr = sub.correct.to_numpy().astype(float)
        sig = {"length": sub["T"].to_numpy().astype(float), "L": sub.L.to_numpy(), "K": sub.K.to_numpy()}
        if vlm:
            Cm, Z, Zt, cov, r_cv, cs, vs = pool_scores(sub, tok)
            sig.update({"V": sub.V.to_numpy(), "C": Cm, "VisConf": Z, "Z_token": Zt})
            ident_err = max(ident_err, float(np.abs(Zt - Z - cov).max()))
            cv_rows.append({"sample_id": qid, "split": "dev" if qid in sp["dev"] else "eval",
                            "mean_within_rollout_r": float(np.nanmean(r_cv)) if np.isfinite(r_cv).any() else np.nan,
                            "frac_rollouts_r_pos": float(np.nanmean(r_cv > 0)), "mean_cov": float(cov.mean())})
            traj_pool[qid] = (cs, vs, Z, sub.V.to_numpy(), corr)
        mixed = 0 < corr.mean() < 1
        rec = {"sample_id": qid, "split": "dev" if qid in sp["dev"] else "eval", "mixed": mixed,
               "pool_acc": corr.mean()}
        for s, v in sig.items():
            rec[f"rho_{s}_correct"] = spearmanr(v, corr).statistic if mixed and np.std(v) > 0 else np.nan
            rec[f"gap_{s}"] = quartile_gap(v, corr)
            if s != "length":
                rec[f"rho_length_{s}"] = spearmanr(sig["length"], v).statistic if np.std(v) > 0 else np.nan
        per_q.append(rec)
    pq_df = pd.DataFrame(per_q)
    pq_df.to_csv(root / "signals_per_question.csv", index=False)
    ev = pq_df[pq_df.split == "eval"]
    sigs = ["length", "L", "K"] + (["V", "C", "VisConf", "Z_token"] if vlm else [])
    summary = {"n_eval": len(ev), "n_eval_mixed": int(ev.mixed.sum()), "signals": {}}
    for s in sigs:
        summary["signals"][s] = {
            "spearman_with_correct_mean(mixed q)": float(ev[f"rho_{s}_correct"].mean()),
            "top_minus_bottom_quartile_correct(all q)": float(ev[f"gap_{s}"].mean()),
            "top_minus_bottom_quartile_correct(mixed q)": float(ev[ev.mixed][f"gap_{s}"].mean()),
            **({"spearman_with_length_mean": float(ev[f"rho_length_{s}"].mean())} if s != "length" else {})}
    if vlm:
        cv = pd.DataFrame(cv_rows)
        cv.to_csv(root / "confidence_vision_corr.csv", index=False)
        e = cv[cv.split == "eval"]
        summary["confidence_vision"] = {
            "per_question_mean_r": {"mean": float(e.mean_within_rollout_r.mean()),
                                    "median": float(e.mean_within_rollout_r.median()),
                                    "q25": float(e.mean_within_rollout_r.quantile(0.25)),
                                    "q75": float(e.mean_within_rollout_r.quantile(0.75))},
            "share_of_questions_with_positive_mean_r": float((e.mean_within_rollout_r > 0).mean()),
            "mean_share_of_rollouts_with_positive_r": float(e.frac_rollouts_r_pos.mean()),
            "identity_max_abs_error(Z_token - Z - Cov)": ident_err}
        summary["illustrative"] = pick_trajectories(root, sp, traj_pool)
    write_json(root / "signals_summary.json", summary)
    return summary


def pick_trajectories(root, sp, pool, n_each=1, smooth=32):
    """Rule: walk the MathVista evaluation questions in their permuted order and take the first
    question where highest-VisConf picks a correct rollout but highest-vision picks a wrong one,
    and the first where the reverse happens. Save both chosen rollouts' c_t and v_t series."""
    out, want = [], {"visconf_right_vision_wrong": n_each, "vision_right_visconf_wrong": n_each}
    for qid in sp["eval"]:
        cs, vs, Z, V, corr = pool[qid]
        iz, iv = int(np.argmax(Z)), int(np.argmax(V))
        if iz == iv:
            continue
        key = ("visconf_right_vision_wrong" if corr[iz] and not corr[iv] else
               "vision_right_visconf_wrong" if corr[iv] and not corr[iz] else None)
        if key and want[key] > 0:
            want[key] -= 1
            def ma(x):
                k = min(smooth, len(x))
                return np.convolve(x, np.ones(k) / k, mode="valid").tolist()
            out.append({"rule": key, "sample_id": qid,
                        "visconf_pick": {"rollout": iz, "correct": bool(corr[iz]), "c": ma(cs[iz]), "v": ma(vs[iz]),
                                         "Z": float(Z[iz]), "V": float(V[iz])},
                        "vision_pick": {"rollout": iv, "correct": bool(corr[iv]), "c": ma(cs[iv]), "v": ma(vs[iv]),
                                        "Z": float(Z[iv]), "V": float(V[iv])}})
        if all(v == 0 for v in want.values()):
            break
    write_json(root / "trajectories.json", out)
    return [{k: x[k] for k in ("rule", "sample_id")} for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/task3")
    a = ap.parse_args()
    for name in C.DATASETS:
        s = analyze(name, a.root)
        print(name, s["signals"], s.get("confidence_vision"))


if __name__ == "__main__":
    main()
