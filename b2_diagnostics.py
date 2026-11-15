"""Pre-Stage-1 change 3: why is B2 overconfident, and why is it worse than B1d before a hidden contact?

Development data only: generator version 2, the first N seeds of the reserved pilot range (the same
episodes as check_generator.py). B1d keeps its pilot tuning. Every configuration sees identical
measurements; the particle filters share a seed protocol (seed = episode seed).

Configurations
  B2-100, B2-300, B2-500          particle count (pilot B2 = B2-100)
  B2-100-T, B2-300-T              likelihood tempering, exponent TEMPER (fixed a priori, not tuned)
  B2-100-noNE                     no negative evidence while a ball is hidden
  B2-100-rayNE                    negative evidence from the same 9-ray visibility test that labels the truth
                                  (pilot B2 uses the table-plane hidden-cell map at the particle centre)
  OR-100                          true physical parameters, 100 particles (isolates parameter uncertainty)
  B2-100-R0, B2-100-R2            roughening at resampling off / doubled (pilot: pos 3 mm, vel 2 cm/s,
                                  params x exp(N(0, 0.05))); same random numbers, so only the jitter size differs
  B1d                             reference

Scored on hidden frames (as in the pilot), per hidden period:
  error of the mean; end-of-period error; for collision periods, error before / after the contact;
  coverage of the 50% / 90% regions built two ways: from the Gaussian summary (mean, covariance; the
  pilot's method) and from the weighted particles directly (highest-density region of a Gaussian
  kernel density estimate, bandwidth = Scott's rule on the effective sample size, at least H_MIN);
  region area; energy score (proper, bandwidth-free; B1d via 400 samples from its Gaussian).
Also: state error (position, velocity) at the last visible frame before each hidden period; error on
visible frames; ESS, resampling rate, particle diversity, runtime.

    python b2_diagnostics.py [n_episodes=200] [n_workers=4] [configs=all, comma-separated]
Writes results/b2_diag_intervals.csv, results/b2_diag_summary.json (and results/b2_diag_raw.pkl). Each job is
saved to results/b2_diag_parts/ as it finishes; rerunning the same command resumes and then summarizes.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

import baselines as B
import gen
import pilot_analyze as PA
import pilot_run as PR

B1D = dict(q=0.01, damp=1.0, rest=0.6)  # pilot-tuned; not retuned here
TEMPER = 0.25  # equivalent to doubling the measurement-likelihood width; chosen before any run
CONFIGS = {
    "B2-100": dict(n_part=100), "B2-300": dict(n_part=300), "B2-500": dict(n_part=500),
    "B2-100-T": dict(n_part=100, temper=TEMPER), "B2-300-T": dict(n_part=300, temper=TEMPER),
    "B2-100-noNE": dict(n_part=100, neg_evidence="off"), "B2-100-rayNE": dict(n_part=100, neg_evidence="ray"),
    "OR-100": dict(n_part=100, oracle=True),
    "B2-100-R0": dict(n_part=100, rough=0.0), "B2-100-R2": dict(n_part=100, rough=2.0),
}
CHI2 = {50: 1.386, 90: 4.605}
H_MIN = 0.005  # m; kernel bandwidth floor (half the measurement noise)
B_BOOT = 2000


def make_episode(k):
    ep = gen.sample_episode(gen.SEED_RANGES["pilot"][0] + k, version=2)
    ep["sim"] = gen.simulate(ep, render=False)
    return ep


# ------------------------------------------------------------------ per-frame scoring
def gauss_scores(mu, C, y):
    d = y - mu
    m2 = d @ np.linalg.solve(C, d)
    out = {f"gcov{q}": float(m2 <= c2) for q, c2 in CHI2.items()}
    out["garea90"] = float(np.pi * CHI2[90] * np.sqrt(np.linalg.det(C)) * 1e4)
    out["gnll"] = float(0.5 * m2 + 0.5 * np.log(np.linalg.det(C)) + np.log(2 * np.pi))
    return out


def energy_score(P, w, y):
    a = w @ np.linalg.norm(P - y, axis=1)
    Dm = np.linalg.norm(P[:, None] - P[None], axis=-1)
    return float((a - 0.5 * w @ Dm @ w) * 100)  # cm


def hpd_scores(P, w, y):
    keep = w > 1e-12 * w.max()
    P, w = P[keep].astype(np.float64), w[keep] / w[keep].sum()
    neff = 1.0 / (w ** 2).sum()
    sd = np.sqrt((w[:, None] * (P - w @ P) ** 2).sum(0).mean())
    h = max(sd * neff ** (-1 / 6), H_MIN)

    def dens(X):
        return np.exp(-0.5 * ((X[:, None, :] - P[None]) ** 2).sum(-1) / h ** 2) @ w / (2 * np.pi * h ** 2)

    fp = dens(P)
    order = np.argsort(-fp)
    cw = np.cumsum(w[order])
    fy = dens(y[None])[0]
    lo, hi = np.maximum(P.min(0) - 3 * h, -gen.HALF), np.minimum(P.max(0) + 3 * h, gen.HALF)
    s = max(0.005, float((hi - lo).max()) / 60)
    G = np.stack(np.meshgrid(np.arange(lo[0], hi[0] + s, s), np.arange(lo[1], hi[1] + s, s), indexing="ij"), -1).reshape(-1, 2)
    fg = dens(G)
    out = {"bandwidth_cm": h * 100}
    for q in (50, 90):
        tau = fp[order][min(int(np.searchsorted(cw, q / 100)), len(cw) - 1)]
        out[f"hcov{q}"] = float(fy >= tau)
        out[f"harea{q}"] = float((fg >= tau).sum() * s * s * 1e4)
    return out


# ------------------------------------------------------------------ one (episode, method) job
def job(args):
    name, ep = args
    z = B.measurements(ep)
    gt = ep["sim"]["pos"][:, :, :2]
    gvel = ep["sim"]["vel"][:, :, :2]
    hid = PR.hidden_mask(ep)
    T, n = hid.shape
    t0 = time.time()
    if name == "B1d":
        mu, cov, vel = B.kalman(z, B1D["q"], damp=B1D["damp"], rest=B1D["rest"], radius=[b["radius"] for b in ep["balls"]], return_vel=True)
        kept, D = None, None
    else:
        mu, cov, kept, D = B.particle_filter(ep, z, keep_particles=True, diag=True, **CONFIGS[name])
        vel = D["vel"]
    runtime = time.time() - t0
    keys = ["err", "gcov50", "gcov90", "garea90", "gnll", "es", "hcov50", "hcov90", "harea50", "harea90", "bandwidth_cm"]
    F = {k: np.full((T, n), np.nan) for k in keys}
    F["err"] = np.linalg.norm(gt - mu, axis=-1) * 100  # all frames; hidden-frame scores below
    rng = np.random.default_rng(ep["seed"])
    for t, i in zip(*np.nonzero(hid)):
        y = gt[t, i].astype(np.float64)
        for k, v in gauss_scores(mu[t, i], cov[t, i], y).items():
            F[k][t, i] = v
        if kept is None:  # B1d: Gaussian samples for the energy score; its regions are already exact
            S = rng.multivariate_normal(mu[t, i], cov[t, i], 400)
            F["es"][t, i] = energy_score(S, np.full(400, 1 / 400), y)
        else:
            P, w = kept[t][0][:, i].astype(np.float64), kept[t][1].astype(np.float64)
            w = w / w.sum()
            F["es"][t, i] = energy_score(P, w, y)
            for k, v in hpd_scores(P, w, y).items():
                F[k][t, i] = v
    out = dict(name=name, seed=ep["seed"], runtime=runtime, F=F, mu=mu.astype(np.float32), vel=vel.astype(np.float32),
               vel_err=np.linalg.norm(gvel - vel, axis=-1) * 100)
    if D is not None:
        out["diag"] = {k: D[k] for k in ("ess", "resampled", "unique")}
    return out


# ------------------------------------------------------------------ aggregation
def boot(x, cl, rng):
    u = np.unique(cl)
    by = {k: x[cl == k] for k in u}
    bs = [np.mean(np.concatenate([by[k] for k in rng.choice(u, len(u))])) for _ in range(B_BOOT)]
    return [float(np.mean(x)), *np.percentile(bs, [2.5, 97.5]).tolist()]


def period_rows(ep, res):
    """One row per (hidden period, method)."""
    rows = []
    hid = PR.hidden_mask(ep)
    vis = ep["sim"]["visible"]
    for iv in PA.intervals(ep):
        i, a, b = iv["ball"], iv["a"], iv["b"]
        f = np.arange(a, b + 1)
        ev = ep["sim"]["event"][a:b + 1, i]
        c = a + int(np.argmax(ev == 2)) if iv["label"] == "collision" else None
        for name, r in res.items():
            F = r["F"]
            row = dict(seed=ep["seed"], mode=ep["mode"], method=name, **iv,
                       err=np.nanmean(F["err"][f, i]), end_err=F["err"][b, i],
                       pre_contact_err=np.nanmean(F["err"][a:c, i]) if c is not None and c > a else np.nan,
                       post_contact_err=np.nanmean(F["err"][c:b + 1, i]) if c is not None else np.nan,
                       last_visible_pos_err=F["err"][a - 1, i] if vis[a - 1, i] else np.nan,
                       last_visible_vel_err=r["vel_err"][a - 1, i] if vis[a - 1, i] else np.nan)
            for k in ("gcov50", "gcov90", "garea90", "gnll", "es", "hcov50", "hcov90", "harea50", "harea90", "bandwidth_cm"):
                row[k] = np.nanmean(F[k][f, i]) if np.isfinite(F[k][f, i]).any() else np.nan
            rows.append(row)
    sel = vis & ~hid
    sel[0] = False
    tt = np.arange(sel.shape[0])[:, None]
    bins = {"all": sel, "t1-4": sel & (tt < 5), "t5-19": sel & (tt >= 5) & (tt < 20), "t20+": sel & (tt >= 20)}
    vis_err = {name: {b: float(np.nanmean(r["F"]["err"][m])) if m.any() else np.nan for b, m in bins.items()} for name, r in res.items()}
    return rows, vis_err


def summarize(eps, results):
    import pandas as pd
    rng = np.random.default_rng(7)
    rows, vis_errs = [], []
    for ep in eps:
        r, v = period_rows(ep, results[ep["seed"]])
        rows += r
        vis_errs.append(v)
    df = pd.DataFrame(rows)
    methods = list(results[eps[0]["seed"]].keys())
    S = dict(n_episodes=len(eps), temper=TEMPER, h_min_cm=H_MIN * 100, configs={k: v for k, v in CONFIGS.items()})
    groups = {"all": df, "none": df[df.label == "none"], "stopped": df[df.label == "stopped"], "bounce": df[df.label == "bounce"],
              "collision": df[df.label == "collision"], "event": df[df.label.isin(["bounce", "collision"])]}
    S["by_method"] = {}
    for mth in methods:
        d = {}
        for g, G in groups.items():
            G = G[G.method == mth]
            d[g] = dict(n=len(G), err=float(G.err.mean()), end_err=float(G.end_err.mean()), es=float(G.es.mean()),
                        gcov50=float(G.gcov50.mean()), gcov90=float(G.gcov90.mean()), garea90_median=float(G.garea90.median()),
                        hcov50=float(G.hcov50.mean()), hcov90=float(G.hcov90.mean()), harea90_median=float(G.harea90.median()))
        C = df[(df.method == mth) & (df.label == "collision")]
        d["collision"].update(pre_contact_err=float(C.pre_contact_err.mean()), post_contact_err=float(C.post_contact_err.mean()))
        A = df[df.method == mth]
        d["last_visible"] = dict(pos_err=float(A.last_visible_pos_err.mean()), vel_err=float(A.last_visible_vel_err.mean()),
                                 n=int(A.last_visible_pos_err.notna().sum()))
        # visible, not-hidden frames after t=0; mean per episode, then over episodes (one episode has none)
        d["visible_frame_err"] = float(np.nanmean([v[mth]["all"] for v in vis_errs]))
        d["visible_frame_err_by_t"] = {b: float(np.nanmean([v[mth][b] for v in vis_errs])) for b in ("t1-4", "t5-19", "t20+")}
        rs = [results[e["seed"]][mth] for e in eps]
        d["runtime_s_per_episode"] = float(np.mean([r["runtime"] for r in rs]))
        if "diag" in rs[0]:
            ess = np.concatenate([r["diag"]["ess"][1:] for r in rs])
            res_ = np.concatenate([r["diag"]["resampled"][1:] for r in rs])
            uni = np.concatenate([r["diag"]["unique"][1:] for r in rs])
            d["filter"] = dict(mean_ess_frac=float(np.nanmean(ess)), resample_rate=float(res_.mean()),
                               unique_frac_at_resample=float(np.nanmean(uni)))
        S["by_method"][mth] = d
    # paired differences (reference minus method: positive = method better), clustered by episode
    S["paired"] = {}
    for ref in ("B2-100", "B1d"):
        for mth in methods:
            if mth == ref:
                continue
            for g in ("all", "event", "collision"):
                G = groups[g]
                a, b = G[G.method == ref].set_index(["seed", "ball", "a"]), G[G.method == mth].set_index(["seed", "ball", "a"])
                for metric in ("err", "end_err", "es") + (("post_contact_err",) if g == "collision" else ()):
                    x = (a[metric] - b[metric]).dropna()
                    S["paired"][f"{ref}-{mth}:{g}:{metric}"] = boot(x.values, x.index.get_level_values(0).values, rng) + [len(x)]
    return S, df


PARTS = "results/b2_diag_parts"  # one pickle per (method, episode); lets an interrupted sweep resume


def part_path(name, seed):
    return f"{PARTS}/{name}_{seed}.pkl"


def run_and_save(args):
    r = job(args)
    tmp = part_path(r["name"], r["seed"]) + ".tmp"
    pickle.dump(r, open(tmp, "wb"), protocol=4)
    os.replace(tmp, part_path(r["name"], r["seed"]))
    return r["name"], r["seed"]


def main(n=200, workers=4, configs=None):
    # Each particle's MjData has a 14.7 MB arena. glibc's dynamic mmap threshold moves such blocks onto the
    # heap after the first free, the heap fragments, and workers grew to 4 GB until the container's memory
    # limit killed them (a multiprocessing pool then waits forever for the lost task). A fixed threshold
    # keeps the arenas mmapped and returned on free; workers are also recycled. Results are unaffected.
    import ctypes
    ctypes.CDLL("libc.so.6").mallopt(-3, 1 << 20)  # M_MMAP_THRESHOLD = 1 MB
    names = list(CONFIGS) if not configs else configs.split(",")
    os.makedirs(PARTS, exist_ok=True)
    with Pool(workers, maxtasksperchild=8) as pool:
        t0 = time.time()
        eps = pool.map(make_episode, range(n), chunksize=4)
        print(f"{n} episodes in {time.time() - t0:.0f}s", flush=True)
        cost = {nm: CONFIGS[nm]["n_part"] * (2 if CONFIGS[nm].get("neg_evidence") == "ray" else 1) for nm in names}
        jobs = [(nm, e) for nm in sorted(names, key=lambda x: -cost[x]) for e in eps] + [("B1d", e) for e in eps]
        todo = [j for j in jobs if not os.path.exists(part_path(j[0], j[1]["seed"]))]
        print(f"{len(jobs) - len(todo)} of {len(jobs)} jobs already saved in {PARTS}", flush=True)
        for done, _ in enumerate(pool.imap_unordered(run_and_save, todo, chunksize=1), 1):
            if done % 100 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} jobs, {time.time() - t0:.0f}s", flush=True)
    order = names + ["B1d"]
    results = {e["seed"]: {k: pickle.load(open(part_path(k, e["seed"]), "rb")) for k in order} for e in eps}
    os.makedirs("results", exist_ok=True)
    pickle.dump(dict(results=results, seeds=[e["seed"] for e in eps]), open("results/b2_diag_raw.pkl", "wb"), protocol=4)
    S, df = summarize(eps, results)
    json.dump(S, open("results/b2_diag_summary.json", "w"), indent=1, default=float)
    df.to_csv("results/b2_diag_intervals.csv", index=False)
    print_summary(S)


def print_summary(S):
    print(f"{'method':14s} {'err':>5s} {'evt':>5s} {'col':>5s} {'pre':>5s} {'post':>5s} {'end':>5s} {'ES':>5s} "
          f"{'g50':>4s} {'g90':>4s} {'gA90':>6s} {'h50':>4s} {'h90':>4s} {'hA90':>6s} {'lvPos':>5s} {'lvVel':>5s} {'visE':>5s} "
          f"{'ESS':>4s} {'rs':>4s} {'uniq':>4s} {'sec':>5s}")
    for m, d in S["by_method"].items():
        a, c = d["all"], d["collision"]
        f = d.get("filter", {})
        print(f"{m:14s} {a['err']:5.1f} {d['event']['err']:5.1f} {c['err']:5.1f} {c['pre_contact_err']:5.1f} {c['post_contact_err']:5.1f} "
              f"{a['end_err']:5.1f} {a['es']:5.2f} {a['gcov50']:4.2f} {a['gcov90']:4.2f} {a['garea90_median']:6.1f} "
              f"{a['hcov50']:4.2f} {a['hcov90']:4.2f} {a['harea90_median']:6.1f} {d['last_visible']['pos_err']:5.2f} "
              f"{d['last_visible']['vel_err']:5.1f} {d['visible_frame_err']:5.2f} {f.get('mean_ess_frac', np.nan):4.2f} "
              f"{f.get('resample_rate', np.nan):4.2f} {f.get('unique_frac_at_resample', np.nan):4.2f} {d['runtime_s_per_episode']:5.1f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 200, int(a[1]) if len(a) > 1 else 4, a[2] if len(a) > 2 else None)
