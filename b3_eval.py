"""Score B3-RGB runs and B1d on the eval block (change 2 and the resolution decision).

Per hidden period (pilot definition) and method: mean error over the period (the primary metric), error at the
period's last frame, error before / after the first ball-ball contact, coverage of the 50% / 90% Gaussian regions,
median 90% region area, Gaussian NLL, energy score (200 samples from each Gaussian). Plus error on visible frames,
which measures perception. Collision periods are reported in two groups (periods.py):

  collision_long   hidden for more than 2 frames from the contact on
  collision_short  hidden for at most 2 frames from the contact on; always reported separately

A B3 variant is (inputs, resolution); its seeds are averaged per period (and the spread across seeds is reported).
B1d keeps its pilot tuning and receives the oracle noisy positions (1 cm), as in the pilot: in Stage 1 it will get
detector outputs instead, so on visible frames it is advantaged here.

Paired differences, 95% intervals from a bootstrap clustered by episode:
  B1d minus each variant      by label; "none" is the pilot report's bar (B3 must at least match B1d there)
  64x64 minus 96x96           per input
  rgb minus rgb+meas          per resolution: the perception gap

    python b3_eval.py <data_dir> <runs_dir> [out_prefix=results/b3_eval] [workers=8]
Reads <runs_dir>/*/config.json and preds_eval.npz. Writes <out_prefix>_periods.csv, _summary.json, _tables.md.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import b2_diagnostics as D
import baselines as B
from b3_rgb import load_block
from periods import FINE, hidden_periods

B1D = D.B1D
N_ES = 200
B_BOOT = 2000
GROUPS = ("all",) + FINE + ("event",)  # event = bounce + collision_long + collision_short
# Decision rules, fixed on 18 Nov 2026 before any eval-block result (docs/STAGE1_PREP.md, change 2).
# Rule 1 margin: 20% (the spec's provisional minimum important improvement) of B1d's mean no-event error on
# development seeds 0-399 under generator version 3 at rate 0.3, 6.1 cm [5.1, 7.3] (180 periods); not the eval block.
RULE1_MARGIN_CM = 1.2
RULE2_MIN_GAIN = 0.10  # 96x96 must cut B3-rgb's mean hidden error by at least 10% of the 64x64 error


def score_method(args):
    name, eps, preds = args
    rows, vis = [], []
    for k, ep in enumerate(eps):
        n = len(ep["balls"])
        mu, cov = preds[k]
        gt = ep["sim"]["pos"][:, :n, :2].astype(np.float64)
        err = np.linalg.norm(gt - mu[:, :n], axis=-1) * 100
        v = ep["sim"]["visible"].copy()
        v[0] = False
        vis.append(dict(seed=ep["seed"], method=name, visible_err=float(err[v].mean()) if v.any() else np.nan))
        rng = np.random.default_rng(ep["seed"])
        for p in hidden_periods(ep):
            i, a, b, c = p["ball"], p["a"], p["b"], p["contact"]
            f = np.arange(a, b + 1)
            sc = {key: [] for key in ("gcov50", "gcov90", "garea90", "gnll", "es")}
            for t in f:
                for key, x in D.gauss_scores(mu[t, i], cov[t, i], gt[t, i]).items():
                    sc[key].append(x)
                S = rng.multivariate_normal(mu[t, i], cov[t, i], N_ES)
                sc["es"].append(D.energy_score(S, np.full(N_ES, 1 / N_ES), gt[t, i]))
            rows.append(dict(seed=ep["seed"], method=name, ball=i, a=a, b=b, dur=p["dur"], label=p["label"], label_fine=p["label_fine"],
                             censored=bool(p["censored"]), post=p["post"], err=float(err[f, i].mean()), end_err=float(err[b, i]),
                             pre_contact_err=float(err[a:c, i].mean()) if c is not None and c > a else np.nan,
                             post_contact_err=float(err[c:b + 1, i].mean()) if c is not None else np.nan,
                             **{key: float(np.mean(x)) for key, x in sc.items()}))
    return rows, vis


def load_runs(runs_dir, eps):
    runs = []
    seeds = np.array([e["seed"] for e in eps])
    for cfg_path in sorted(glob.glob(os.path.join(runs_dir, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        if not os.path.exists(os.path.join(d, "preds_eval.npz")):
            print(f"skipping {d}: no preds_eval.npz", flush=True)
            continue
        cfg = json.load(open(cfg_path))
        P = np.load(os.path.join(d, "preds_eval.npz"))
        assert np.array_equal(P["seeds"], seeds), f"{d}: predictions are for different episodes"
        variant = f"B3-{cfg['inputs']}-{cfg['res']}"
        runs.append(dict(name=f"{variant}-s{cfg['seed']}", variant=variant, seed=cfg["seed"], cfg=cfg,
                         preds=[(P["mu"][k].astype(np.float64), P["cov"][k].astype(np.float64)) for k in range(len(eps))]))
    return runs


def boot(x, cl, rng):
    x, cl = np.asarray(x, float), np.asarray(cl)
    u = np.unique(cl)
    idx = {k: np.nonzero(cl == k)[0] for k in u}
    bs = [x[np.concatenate([idx[k] for k in rng.choice(u, len(u))])].mean() for _ in range(B_BOOT)]
    return [float(x.mean()), *np.percentile(bs, [2.5, 97.5]).tolist(), int(len(x))]


def group(df, g):
    if g == "all":
        return df
    if g == "event":
        return df[df.label_fine.isin(["bounce", "collision_long", "collision_short"])]
    return df[df.label_fine == g]


def summarize(df, vis, runs, n_eps):
    rng = np.random.default_rng(5)
    key = ["seed", "ball", "a"]
    metrics = ["err", "end_err", "pre_contact_err", "post_contact_err", "gcov50", "gcov90", "garea90", "gnll", "es"]
    # seed-averaged rows per variant (B1d is its own variant)
    df = df.assign(variant=df.method.str.replace(r"-s\d+$", "", regex=True))
    V = df.groupby(["variant"] + key + ["label_fine", "dur", "post"], dropna=False)[metrics].mean().reset_index()
    variants = sorted(V.variant.unique(), key=lambda v: (v != "B1d", v))
    S = dict(n_episodes=n_eps, n_episodes_with_hidden_periods=int(df.seed.nunique()), groups={}, visible_err={}, seed_spread={}, paired={}, runs={})
    for v in variants:
        Vv = V[V.variant == v]
        S["groups"][v] = {g: dict(n=int(len(group(Vv, g))), **{m: float(group(Vv, g)[m].mean()) for m in metrics},
                                  garea90_median=float(group(Vv, g).garea90.median())) for g in GROUPS}
        vv = vis[vis.method.str.replace(r"-s\d+$", "", regex=True) == v]
        S["visible_err"][v] = float(vv.groupby("seed").visible_err.mean().mean())
        per_seed = df[df.variant == v].groupby("method").err.mean()
        S["seed_spread"][v] = dict(n_seeds=int(len(per_seed)), mean=float(per_seed.mean()), sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
                                   per_seed={k: float(x) for k, x in per_seed.items()})

    def paired(name, va, vb, groups=("all", "none", "bounce", "collision_long", "collision_short", "event"), metric="err"):
        A, Bv = V[V.variant == va].set_index(key), V[V.variant == vb].set_index(key)
        for g in groups:
            ga = group(A.reset_index(), g).set_index(key)
            x = (ga[metric] - Bv[metric].reindex(ga.index)).dropna()
            if len(x):
                S["paired"][f"{name}:{g}:{metric}"] = boot(x.values, x.index.get_level_values(0).values, rng)

    b3 = [v for v in variants if v != "B1d"]
    for v in b3:
        paired(f"B1d-{v}", "B1d", v)
        paired(f"B1d-{v}", "B1d", v, groups=("all", "collision_long"), metric="end_err")
        paired(f"B1d-{v}", "B1d", v, groups=("all",), metric="es")
    for inp in ("rgb", "rgb+meas"):
        if f"B3-{inp}-64" in variants and f"B3-{inp}-96" in variants:
            paired(f"B3-{inp}-64-minus-96", f"B3-{inp}-64", f"B3-{inp}-96")
    for res in (64, 96):
        if f"B3-rgb-{res}" in variants and f"B3-rgb+meas-{res}" in variants:
            paired(f"B3-rgb-minus-rgb+meas-{res}", f"B3-rgb-{res}", f"B3-rgb+meas-{res}")
    for r in runs:
        S["runs"][r["name"]] = {k: r["cfg"].get(k) for k in ("params", "w_hidden", "n_train", "epochs", "best_val_wnll", "train_s", "commit")}
    S["rules"] = decision_rules(V, S["seed_spread"], rng)
    return S, V


def decision_rules(V, spread, rng, inputs="rgb"):
    """The two pre-registered rules, evaluated mechanically.

    Rule 2 (resolution) for B3-<inputs>: 96x96 is chosen only if (a) the paired 64x64 - 96x96 difference in mean hidden
    error over all periods has a 95% interval above 0, (b) its mean is at least RULE2_MIN_GAIN of the 64x64 error, and
    (c) every 96x96 seed has a lower mean hidden error over all periods than every 64x64 seed (3 seeds each).
    Rule 1 (strength): B3-<inputs> at the chosen resolution is strong only if the upper 95% bound of (B3 - B1d) on
    no-event periods is below RULE1_MARGIN_CM. It is also reported for the other variants, as information."""
    key = ["seed", "ball", "a"]
    out = dict(rule1_margin_cm=RULE1_MARGIN_CM, rule2_min_gain=RULE2_MIN_GAIN, inputs=inputs, rule1={}, rule2=None)
    b1 = V[(V.variant == "B1d") & (V.label_fine == "none")].set_index(key).err
    for v in sorted(V.variant.unique()):
        if v == "B1d":
            continue
        b3 = V[(V.variant == v) & (V.label_fine == "none")].set_index(key).err
        x = (b3 - b1.reindex(b3.index)).dropna()
        ci = boot(x.values, x.index.get_level_values(0).values, rng)
        out["rule1"][v] = dict(b3_minus_b1d_none=ci[:3], n=ci[3], upper_below_margin=bool(ci[2] < RULE1_MARGIN_CM))
    v64, v96 = f"B3-{inputs}-64", f"B3-{inputs}-96"
    if v64 in spread and v96 in spread:
        a, b = V[V.variant == v64].set_index(key).err, V[V.variant == v96].set_index(key).err
        x = (a - b.reindex(a.index)).dropna()
        ci = boot(x.values, x.index.get_level_values(0).values, rng)
        e64, e96 = list(spread[v64]["per_seed"].values()), list(spread[v96]["per_seed"].values())
        gain = ci[0] / float(a.mean())
        complete = len(e64) >= 3 and len(e96) >= 3
        r2 = dict(diff_64_minus_96_all=ci[:3], n=ci[3], gain_fraction=gain, seeds_64=e64, seeds_96=e96,
                  a_ci_above_0=bool(ci[1] > 0), b_gain_at_least_min=bool(gain >= RULE2_MIN_GAIN),
                  c_every_96_seed_beats_every_64_seed=bool(max(e96) < min(e64)), seeds_complete=complete)
        r2["choose_96"] = bool(complete and r2["a_ci_above_0"] and r2["b_gain_at_least_min"] and r2["c_every_96_seed_beats_every_64_seed"])
        out["rule2"] = r2
        chosen = v96 if r2["choose_96"] else v64
        out["chosen_resolution"] = 96 if r2["choose_96"] else 64
        out["b3_strong"] = out["rule1"].get(chosen, {}).get("upper_below_margin")
        out["b3_strong_variant"] = chosen
    return out


def tables(S):
    fmt = lambda t, d=1: f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"  # noqa: E731
    out = [f"# B3-RGB evaluation ({S['n_episodes']} eval episodes, {S['n_episodes_with_hidden_periods']} with hidden periods)\n",
           "Mean hidden-position error (cm) per period, averaged over periods; B3 variants averaged over their seeds. "
           "collision_short: hidden for at most 2 frames from the contact on.\n"]
    R = S.get("rules", {})
    if R.get("rule2"):
        r2 = R["rule2"]
        out += ["## Decision rules (fixed before any eval result)\n",
                f"**Rule 2, resolution** (B3-{R['inputs']}): 64x64 - 96x96 mean hidden error, all periods: {fmt(r2['diff_64_minus_96_all'])} cm "
                f"({r2['gain_fraction']:+.1%} of the 64x64 error; n = {r2['n']}). (a) interval above 0: {r2['a_ci_above_0']}; "
                f"(b) gain >= {R['rule2_min_gain']:.0%}: {r2['b_gain_at_least_min']}; (c) every 96x96 seed beats every 64x64 seed: "
                f"{r2['c_every_96_seed_beats_every_64_seed']} (64: {', '.join(f'{e:.2f}' for e in r2['seeds_64'])}; 96: {', '.join(f'{e:.2f}' for e in r2['seeds_96'])})."
                + ("" if r2["seeds_complete"] else " Fewer than 3 seeds per resolution: the rule cannot pass.")
                + f" **Chosen: {R['chosen_resolution']}x{R['chosen_resolution']}.**\n",
                f"**Rule 1, strength** (margin {R['rule1_margin_cm']} cm): B3 - B1d on no-event periods, upper 95% bound must be below the margin.\n",
                "| Variant | B3 - B1d, no-event periods (cm) | n | Upper bound below margin |", "|---|---|---|---|"]
        for v, r in R["rule1"].items():
            out.append(f"| {v} | {fmt(r['b3_minus_b1d_none'])} | {r['n']} | {r['upper_below_margin']} |")
        out += ["", f"**B3 strong: {R['b3_strong']}** (judged on {R['b3_strong_variant']}, the chosen resolution).\n"]
    n = S["groups"]["B1d"]
    out.append("| Method | " + " | ".join(f"{g} (n={n[g]['n']})" for g in GROUPS) + " | Visible-frame error | Seed SD (all) |")
    out.append("|" + "---|" * (len(GROUPS) + 3))
    for v, d in S["groups"].items():
        sp = S["seed_spread"][v]
        out.append(f"| {v} | " + " | ".join(f"{d[g]['err']:.1f}" for g in GROUPS) + f" | {S['visible_err'][v]:.2f} | "
                   + (f"{sp['sd']:.2f} ({sp['n_seeds']} seeds)" if sp["sd"] is not None else "–") + " |")
    out += ["", "| Method | 90% coverage, all | 90% coverage, collision_long | 90% coverage, collision_short | Median 90% area (cm²) | Energy score, all | End-of-period error, all |",
            "|---|---|---|---|---|---|---|"]
    for v, d in S["groups"].items():
        out.append(f"| {v} | {d['all']['gcov90']:.2f} | {d['collision_long']['gcov90']:.2f} | {d['collision_short']['gcov90']:.2f} | "
                   f"{d['all']['garea90_median']:.0f} | {d['all']['es']:.2f} | {d['all']['end_err']:.1f} |")
    out += ["", "Paired differences (cm), 95% intervals clustered by episode; positive = the second-named method is better.\n",
            "| Comparison | Group | Metric | Difference | n |", "|---|---|---|---|---|"]
    for k, t in S["paired"].items():
        name, g, m = k.split(":")
        out.append(f"| {name} | {g} | {m} | {fmt(t, 2 if m == 'es' else 1)} | {t[3]} |")
    return "\n".join(out)


def main(data_dir, runs_dir, prefix="results/b3_eval", workers=8):
    eps, _ = load_block(data_dir, "eval", None, images=False)
    runs = load_runs(runs_dir, eps)
    b1d = []
    for e in eps:
        z = B.measurements(e)
        b1d.append(B.kalman(z, B1D["q"], damp=B1D["damp"], rest=B1D["rest"], radius=[b["radius"] for b in e["balls"]]))
    jobs = [("B1d", eps, b1d)] + [(r["name"], eps, r["preds"]) for r in runs]
    with Pool(workers) as pool:
        res = pool.map(score_method, jobs)
    df = pd.DataFrame([row for r in res for row in r[0]])
    vis = pd.DataFrame([row for r in res for row in r[1]])
    S, V = summarize(df, vis, runs, len(eps))
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    df.to_csv(f"{prefix}_periods.csv", index=False)
    json.dump(S, open(f"{prefix}_summary.json", "w"), indent=1)
    md = tables(S)
    open(f"{prefix}_tables.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "results/b3_eval", int(a[3]) if len(a) > 3 else 8)
