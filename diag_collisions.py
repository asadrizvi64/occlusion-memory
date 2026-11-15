"""Hidden-collision diagnostic for generator version 2 (collide-mode episodes of the first N pilot seeds).

Per collision period: hidden frames before and after the contact, the hidden ball's speed and deflection
across the contact, and B1d / B2 error before contact, after contact and at the period's last frame.
Writes results/collision_diag.csv.

    python diag_collisions.py [n_episodes=200] [min_post_hidden=gen.MIN_POST_HIDDEN]
"""
import sys
from multiprocessing import Pool

import numpy as np

import baselines as B
import gen
import pilot_analyze as PA

if len(sys.argv) > 2:
    gen.MIN_POST_HIDDEN = int(sys.argv[2])  # module global: inherited by forked workers


def job(k):
    seed = gen.SEED_RANGES["pilot"][0] + k
    ep = gen.sample_episode(seed, version=2)
    if ep["mode"] != "collide":
        return []
    ep["sim"] = gen.simulate(ep, render=False)
    z = B.measurements(ep)
    gt = ep["sim"]["pos"][:, :, :2]
    P = {"B1d": B.kalman(z, 0.01, damp=1.0, rest=0.6, radius=[b["radius"] for b in ep["balls"]])[0],
         "B2": B.particle_filter(ep, z, n_part=100)[0]}
    out = []
    for iv in PA.intervals(ep):
        if iv["label"] != "collision":
            continue
        i, a, b = iv["ball"], iv["a"], iv["b"]
        ev = ep["sim"]["event"][a:b + 1, i]
        c = a + int(np.argmax(ev == 2))  # first frame whose step window contains the contact
        v = ep["sim"]["vel"][:, i, :2]
        vb, va = v[max(c - 1, 0)], v[min(c + 1, 79)]
        sb, sa = np.linalg.norm(vb), np.linalg.norm(va)
        defl = float(np.degrees(np.arccos(np.clip(vb @ va / (sb * sa + 1e-9), -1, 1)))) if sb > 0.02 and sa > 0.02 else np.nan
        row = dict(seed=seed, ball=i, dur=b - a + 1, pre=c - a, post=b - c + 1, censored=iv["censored"],
                   speed_before=float(sb), speed_after=float(sa), deflection_deg=defl)
        for m, mu in P.items():
            e = np.linalg.norm(gt[:, i] - mu[:, i], axis=-1) * 100
            row[f"pre_{m}"] = float(e[a:c].mean()) if c > a else np.nan
            row[f"post_{m}"] = float(e[c:b + 1].mean())
            row[f"end_{m}"] = float(e[b])
        out.append(row)
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        rows = [r for rs in p.map(job, range(int(sys.argv[1]) if len(sys.argv) > 1 else 200), chunksize=2) for r in rs]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("results/collision_diag.csv", index=False)
    print("min_post_hidden", gen.MIN_POST_HIDDEN, "collision periods", len(df))
    pd.set_option("display.width", 200)
    print(df[["dur", "pre", "post", "speed_before", "speed_after", "deflection_deg"]].describe().round(2))
    print("censored share", df.censored.mean().round(2))
    print("post<=2 frames share", (df.post <= 2).mean().round(2), " post>=5 share", (df.post >= 5).mean().round(2))
    for m in ("B1d", "B2"):
        print(m, "pre", df[f"pre_{m}"].mean().round(1), "post", df[f"post_{m}"].mean().round(1), "end", df[f"end_{m}"].mean().round(1))
    big = df[(df.post >= 5) & (df.deflection_deg >= 20)]
    print("post>=5 & deflection>=20deg: n", len(big), {m: (big[f"post_{m}"].mean().round(1), big[f"end_{m}"].mean().round(1)) for m in ("B1d", "B2")})
    small = df[~df.index.isin(big.index)]
    print("rest: n", len(small), {m: (small[f"post_{m}"].mean().round(1), small[f"end_{m}"].mean().round(1)) for m in ("B1d", "B2")})
