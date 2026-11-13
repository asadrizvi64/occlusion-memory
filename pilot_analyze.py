"""Stage 0 pilot analysis -> results/pilot_summary.json

Unit of analysis: an occlusion interval = a maximal run of hidden frames of one ball (after
frame 0). Labels, by priority: collision (ball-ball contact while hidden) > bounce (wall or
block) > stopped (true speed < 2 cm/s at the interval's end, > 5 cm/s at its start) > none.
Censored = the ball is still hidden at the last frame.
"""
import json
import pickle

import numpy as np

import pilot_run as PR

RNG = np.random.default_rng(7)
B = 2000
METHODS = ["B0", "B1", "B1d", "B3", "B2", "OR"]
CHI2 = {50: 1.386, 90: 4.605}


def intervals(ep):
    s = ep["sim"]
    hid = PR.hidden_mask(ep)
    out = []
    for i in range(hid.shape[1]):
        t = 1
        while t < hid.shape[0]:
            if hid[t, i]:
                a = t
                while t < hid.shape[0] and hid[t, i]:
                    t += 1
                b = t - 1
                ev = s["event"][a:b + 1, i]
                sp = np.linalg.norm(s["vel"][:, i, :2], axis=-1)
                if (ev == 2).any():
                    lab = "collision"
                elif (ev == 1).any():
                    lab = "bounce"
                elif sp[b] < 0.02 and sp[a] > 0.05:
                    lab = "stopped"
                else:
                    lab = "none"
                out.append(dict(ball=i, a=a, b=b, dur=b - a + 1, label=lab, censored=b == hid.shape[0] - 1))
            else:
                t += 1
    return out


def dur_bin(d):
    return "1-2" if d <= 2 else "3-5" if d <= 5 else "6-15" if d <= 15 else "16+"


def kmeans2(P, w, iters=20):
    c = P[[np.argmax(w), np.argmax(((P - P[np.argmax(w)]) ** 2).sum(-1))]].copy()
    for _ in range(iters):
        lab = ((P[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(2):
            if (lab == k).any():
                c[k] = (w[lab == k, None] * P[lab == k]).sum(0) / w[lab == k].sum()
    return c, lab


def multimodal(P, w):
    c, lab = kmeans2(P.astype(np.float64), w.astype(np.float64))
    wk = np.array([w[lab == k].sum() for k in range(2)])
    if wk.min() < 0.1:
        return False
    within = np.sqrt(sum((w[lab == k, None] * (P[lab == k] - c[k]) ** 2).sum() for k in range(2)) / w.sum())
    sep = np.linalg.norm(c[0] - c[1])
    return bool(sep > max(0.03, 3 * within))


def main():
    R = pickle.load(open("results/pilot_preds.pkl", "rb"))
    particles = pickle.load(open("results/pilot_particles.pkl", "rb"))
    eps = {e["seed"]: e for e in PR.load() if e["seed"] in set(R["eval_seeds"])}
    rows = []
    for seed, ep in eps.items():
        gt = ep["sim"]["pos"][:, :, :2]
        for iv in intervals(ep):
            r = dict(seed=seed, **iv)
            frames = np.arange(iv["a"], iv["b"] + 1)
            for m in METHODS:
                mu, cov = R["preds"][seed][m]
                d = gt[frames, iv["ball"]] - mu[frames, iv["ball"]]
                C = cov[frames, iv["ball"]]
                m2 = np.einsum("ki,kij,kj->k", d, np.linalg.inv(C), d)
                r[f"err_{m}"] = float(np.linalg.norm(d, axis=-1).mean() * 100)
                r[f"end_err_{m}"] = float(np.linalg.norm(d[-1]) * 100)
                for q, c2 in CHI2.items():
                    r[f"cov{q}_{m}"] = float((m2 <= c2).mean())
                r[f"area90_{m}"] = float(np.mean(np.pi * c2 * np.sqrt(np.linalg.det(C))) * 1e4)
            if seed in particles and iv["label"] in ("bounce", "collision"):
                mm = [multimodal(particles[seed][t][0][:, iv["ball"]], particles[seed][t][1]) for t in frames if t in particles[seed]]
                r["multimodal_frac"] = float(np.mean(mm)) if mm else np.nan
            rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    S = dict(n_eval_episodes=len(eps), n_intervals=len(df), timing_s_per_episode=R["timing"],
             tuned={k: (v if not isinstance(v, tuple) else [float(v[0]), list(np.atleast_1d(v[1]).astype(float))]) for k, v in R["tuned"].items()})
    S["interval_counts"] = {"label": df.label.value_counts().to_dict(), "duration": df.dur.map(dur_bin).value_counts().to_dict(),
                            "censored": int(df.censored.sum()),
                            "episodes_with_interval_ge3": float(np.mean([any(iv["dur"] >= 3 for iv in intervals(e)) for e in eps.values()]))}

    def boot_mean(x, cl):
        u = np.unique(cl)
        by = {k: x[cl == k] for k in u}
        bs = [np.mean(np.concatenate([by[k] for k in RNG.choice(u, len(u))])) for _ in range(B)]
        return [float(np.mean(x)), *np.percentile(bs, [2.5, 97.5]).tolist()]

    S["error_cm"] = {}
    for grp_name, grp in [("all", df), *[(f"label:{l}", g) for l, g in df.groupby("label")],
                          *[(f"dur:{d}", g) for d, g in df.groupby(df.dur.map(dur_bin))]]:
        S["error_cm"][grp_name] = {m: boot_mean(grp[f"err_{m}"].values, grp.seed.values) for m in METHODS}
        S["error_cm"][grp_name]["n"] = len(grp)
    S["coverage"] = {m: {q: float(df[f"cov{q}_{m}"].mean()) for q in CHI2} for m in METHODS}
    S["area90_cm2"] = {m: float(df[f"area90_{m}"].median()) for m in METHODS}
    # headroom: paired differences (per interval, clustered by episode)
    S["paired_diff_cm"] = {}
    for a, b in [("B1", "B1d"), ("B1d", "B3"), ("B1d", "B2"), ("B3", "B2"), ("B2", "OR")]:
        for grp_name, grp in [("all", df), ("events", df[df.label.isin(["bounce", "collision"])]), ("long", df[df.dur >= 6])]:
            d = (grp[f"err_{a}"] - grp[f"err_{b}"]).values
            S["paired_diff_cm"][f"{a}-{b}:{grp_name}"] = boot_mean(d, grp.seed.values) + [len(d)]
    # intervals needed to detect a 20% error reduction vs the reference (paired, alpha .05 two-sided, power .8).
    # Assumption: the SD of a future paired difference is approximated by the SD of the per-interval
    # difference between the reference and another existing method.
    S["n_intervals_for_20pct"] = {}
    for ref in ["B1d", "B3"]:
        for grp_name, grp in [("all", df), ("events", df[df.label.isin(["bounce", "collision"])]), ("long", df[df.dur >= 6])]:
            e = grp[f"err_{ref}"].values
            delta = 0.2 * np.mean(e)
            sd_d = np.std(grp[f"err_{ref}"].values - grp[f"err_{'B2' if ref == 'B1d' else 'B1d'}"].values)
            S["n_intervals_for_20pct"][f"{ref}:{grp_name}"] = int(np.ceil(((1.96 + 0.84) * sd_d / delta) ** 2))
    mm = df.multimodal_frac.dropna() if "multimodal_frac" in df else []
    S["multimodality"] = dict(n_event_intervals=int(len(mm)), frac_intervals_any=float((mm > 0).mean()) if len(mm) else None,
                              mean_frac_frames=float(mm.mean()) if len(mm) else None)
    json.dump(S, open("results/pilot_summary.json", "w"), indent=1, default=float)
    df.to_csv("results/pilot_intervals.csv", index=False)
    return S, df


if __name__ == "__main__":
    S, df = main()
    print(json.dumps({k: S[k] for k in ("n_eval_episodes", "n_intervals", "interval_counts", "timing_s_per_episode")}, indent=1, default=float))
    for g, v in S["error_cm"].items():
        print(f"{g:16s} n={v['n']:4d} " + "  ".join(f"{m} {v[m][0]:6.1f}" for m in METHODS))
    print("coverage", {m: {q: round(x, 2) for q, x in v.items()} for m, v in S["coverage"].items()})
    print("area90 cm2", {m: round(v, 1) for m, v in S["area90_cm2"].items()})
    for k, v in S["paired_diff_cm"].items():
        print(f"{k:18s} {v[0]:+6.2f} [{v[1]:+.2f}, {v[2]:+.2f}] n={v[3]}")
    print("n intervals for 20%", S["n_intervals_for_20pct"])
    print("multimodality", S["multimodality"])
