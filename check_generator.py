"""Generator check (pre-Stage 1, change 1): does version 2 make events while hidden common?

Regenerates the first N pilot seeds with version 1 and version 2 (no rendering) and reports
  - regression: version 1 reproduces the stored pilot episodes (data/pilot_00000.pkl), if present;
  - hidden-period labels per version (targets: >= 40% with an event, >= 10% with a collision);
  - generation time and collide-mode success;
  - B1d (pilot-tuned) vs B2 hidden-position error by label, to see whether headroom grew.
Writes results/generator_check.json.

    python check_generator.py [n_episodes=200] [n_workers=4]
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

B1D = dict(q=0.01, damp=1.0, rest=0.6)  # tuned on the pilot fit half (results/pilot_summary.json)


def make(args):
    k, version = args
    seed = gen.SEED_RANGES["pilot"][0] + k
    t0 = time.time()
    ep = gen.sample_episode(seed, version=version)
    t1 = time.time()
    ep["sim"] = gen.simulate(ep, render=False)
    return ep, t1 - t0, time.time() - t1


def score(ep):
    """B1d and B2 errors per hidden period (cm)."""
    z = B.measurements(ep)
    gt = ep["sim"]["pos"][:, :, :2]
    preds = {"B1d": B.kalman(z, B1D["q"], damp=B1D["damp"], rest=B1D["rest"], radius=[b["radius"] for b in ep["balls"]])[0],
             "B2": B.particle_filter(ep, z, n_part=100)[0]}
    rows = []
    for iv in PA.intervals(ep):
        f = np.arange(iv["a"], iv["b"] + 1)
        row = dict(seed=ep["seed"], mode=ep["mode"], **iv)
        for m, mu in preds.items():
            row[f"err_{m}"] = float(np.linalg.norm(gt[f, iv["ball"]] - mu[f, iv["ball"]], axis=-1).mean() * 100)
        rows.append(row)
    return rows


def label_shares(eps):
    labs = [iv["label"] for e in eps for iv in PA.intervals(e)]
    n = len(labs)
    out = {lab: labs.count(lab) for lab in ("none", "stopped", "bounce", "collision")}
    out.update(n=n, event_share=(out["bounce"] + out["collision"]) / n, collision_share=out["collision"] / n)
    return out


def main(n=200, workers=4):
    os.makedirs("results", exist_ok=True)
    S = {}
    with Pool(workers) as pool:
        eps = {}
        for v in (1, 2):
            res = pool.map(make, [(k, v) for k in range(n)], chunksize=4)
            eps[v] = [r[0] for r in res]
            S[f"v{v}_seconds_per_episode"] = dict(sample=float(np.mean([r[1] for r in res])), simulate=float(np.mean([r[2] for r in res])))
        ref = "data/pilot_00000.pkl"
        if os.path.exists(ref):
            old = {e["seed"]: e for e in pickle.load(open(ref, "rb"))}
            same = [np.array_equal(old[e["seed"]]["sim"]["pos"], e["sim"]["pos"]) and np.array_equal(old[e["seed"]]["sim"]["visible"], e["sim"]["visible"])
                    for e in eps[1] if e["seed"] in old]
            S["v1_matches_stored_pilot"] = f"{sum(same)}/{len(same)}"
        S["labels"] = {f"v{v}": label_shares(eps[v]) for v in (1, 2)}
        e2 = eps[2]
        S["v2_modes"] = dict(collide=sum(e["mode"] == "collide" for e in e2), free=sum(e["mode"] == "free" for e in e2),
                             collide_failed=sum(bool(e.get("collide_failed")) for e in e2))
        col = [e for e in e2 if e["mode"] == "collide"]
        S["v2_collide_with_hidden_collision_period"] = sum(any(iv["label"] == "collision" for iv in PA.intervals(e)) for e in col)
        S["labels"]["v2_collide_only"] = label_shares(col)
        S["labels"]["v2_free_only"] = label_shares([e for e in e2 if e["mode"] == "free"])
        print(json.dumps(S, indent=1), flush=True)
        t0 = time.time()
        rows = {v: [r for rs in pool.map(score, eps[v], chunksize=2) for r in rs] for v in (1, 2)}
        S["scoring_seconds"] = time.time() - t0
    import pandas as pd
    S["error_cm"] = {}
    for v in (1, 2):
        df = pd.DataFrame(rows[v])
        for name, g in [("all", df), *df.groupby("label"), ("event", df[df.label.isin(["bounce", "collision"])])]:
            d = (g.err_B1d - g.err_B2).values
            S["error_cm"][f"v{v}:{name}"] = dict(n=len(g), B1d=float(g.err_B1d.mean()), B2=float(g.err_B2.mean()),
                                                 diff=PA_boot(d, g.seed.values))
    json.dump(S, open("results/generator_check.json", "w"), indent=1, default=float)
    for k, v in S["error_cm"].items():
        print(f"{k:16s} n={v['n']:4d}  B1d {v['B1d']:5.1f}  B2 {v['B2']:5.1f}  B1d-B2 {v['diff'][0]:+5.1f} [{v['diff'][1]:+.1f}, {v['diff'][2]:+.1f}]")


def PA_boot(x, cl, b=2000):
    rng = np.random.default_rng(7)
    u = np.unique(cl)
    by = {k: x[cl == k] for k in u}
    bs = [np.mean(np.concatenate([by[k] for k in rng.choice(u, len(u))])) for _ in range(b)]
    return [float(np.mean(x)), *np.percentile(bs, [2.5, 97.5]).tolist()]


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
