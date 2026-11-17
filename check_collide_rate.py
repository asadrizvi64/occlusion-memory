"""Generator finalisation: collision-course rate P_COLLIDE = 0.3 / 0.4 / 0.5, and version 2 vs version 3.

Development seeds only (the reserved pilot range). No model is scored, so nothing here is tuned on results.

Coupling: whether an episode attempts a collision course is the first draw u of its separate random stream
(gen.sample_episode), and the attempt itself does not depend on P_COLLIDE. So for each seed this script builds
the free episode (no attempt) and, if u < 0.5, the attempted episode under versions 2 and 3; the episode at
rate P is the attempted one if u < P, else the free one. All settings are compared on common random numbers,
and a setting with a lower rate is the same episodes with some attempts switched off.

Per setting: hidden periods per 1,000 episodes by label (collision periods also split into short, hidden for at
most 2 frames from the contact on, and long), event and collision shares (95% intervals, bootstrap
clustered by episode), hidden frames after the contact in collision periods, collision-course success,
generation cost (sampling + physics + visibility, no rendering, one core).

    python check_collide_rate.py [n_episodes=1000] [workers=4]
Writes results/collide_rate_periods.csv and results/collide_rate_summary.json.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import gen
import pilot_analyze as PA

LO = gen.SEED_RANGES["pilot"][0]
RATES = (0.3, 0.4, 0.5)
B_BOOT = 2000
SHORT = 2  # a collision period is 'short' if the ball is hidden for at most this many frames from the contact on


def periods(ep, sim):
    ep = dict(ep, sim=sim)
    rows = []
    for iv in PA.intervals(ep):
        ev = sim["event"][iv["a"]:iv["b"] + 1, iv["ball"]]
        hit = np.nonzero(ev == gen.EV["collision"])[0]
        rows.append(dict(ball=iv["ball"], a=iv["a"], b=iv["b"], dur=iv["dur"], label=iv["label"], censored=bool(iv["censored"]),
                         pre=int(hit[0]) if len(hit) else -1, post=int(iv["dur"] - hit[0]) if len(hit) else -1))
    return rows


def job(k):
    seed = LO + k
    u = float(np.random.default_rng([seed, 2]).random())
    out = dict(seed=seed, u=u, v={})
    t0 = time.time()
    ef = gen.sample_episode(seed, version=3, p_collide=0.0)
    t_sample = time.time() - t0
    t0 = time.time()
    sf = gen.simulate(ef, render=False)
    t_sim = time.time() - t0
    out["v"]["free"] = dict(mode="free", attempted=False, secs=t_sample + t_sim, periods=periods(ef, sf))
    if u < max(RATES):
        for ver in (2, 3):
            t0 = time.time()
            e = gen.sample_episode(seed, version=ver, p_collide=1.0)
            secs = time.time() - t0
            if e["mode"] == "collide":
                t0 = time.time()
                s = gen.simulate(e, render=False)
                secs += time.time() - t0
                rows = periods(e, s)
            else:  # failed attempt: the episode is the free one
                secs += t_sim
                rows = out["v"]["free"]["periods"]
            out["v"][f"v{ver}"] = dict(mode=e["mode"], attempted=True, secs=secs, periods=rows)
    return out


def pick(r, ver, rate):
    return r["v"][f"v{ver}"] if r["u"] < rate else r["v"]["free"]


def boot_share(num, den, rng):
    """Ratio of sums over episodes, 95% interval by resampling episodes."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (B_BOOT, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(num.sum() / max(den.sum(), 1)), *np.percentile(bs, [2.5, 97.5]).tolist()]


def summarize(res, ver, rate, rng):
    eps = [pick(r, ver, rate) if ver else r["v"]["free"] for r in res]
    labs = ("none", "stopped", "bounce", "collision")
    cnt = {l: np.array([sum(p["label"] == l for p in e["periods"]) for e in eps]) for l in labs}
    tot = sum(cnt.values())
    col = [p for e in eps for p in e["periods"] if p["label"] == "collision"]
    post = np.array([p["post"] for p in col])
    n = len(eps)
    att = [r for r in res if ver and r["u"] < rate]
    return dict(
        n_episodes=n, episodes_with_hidden_period=int((tot > 0).sum()), hidden_periods=int(tot.sum()),
        per_1000={l: round(1000 * cnt[l].sum() / n, 1) for l in labs}, per_1000_all=round(1000 * tot.sum() / n, 1),
        per_1000_collision_short=round(1000 * int((post <= SHORT).sum()) / n, 1),  # hidden <= 2 frames after contact
        per_1000_collision_long=round(1000 * int((post > SHORT).sum()) / n, 1),
        share={l: round(float(cnt[l].sum() / tot.sum()), 3) for l in labs},
        event_share=boot_share(cnt["bounce"] + cnt["collision"], tot, rng), collision_share=boot_share(cnt["collision"], tot, rng),
        collision_post=dict(n=len(col), median=float(np.median(post)) if len(col) else None,
                            share_le2=float((post <= 2).mean()) if len(col) else None,
                            share_ge5=float((post >= gen.MIN_POST_HIDDEN).mean()) if len(col) else None),
        attempts=len(att), successes=sum(r["v"][f"v{ver}"]["mode"] == "collide" for r in att),
        gen_s_per_episode=round(float(np.mean([e["secs"] for e in eps])), 2),
        gen_s_per_attempted_episode=round(float(np.mean([r["v"][f"v{ver}"]["secs"] for r in att])), 2) if att else None)


def main(n=1000, workers=4):
    t0 = time.time()
    with Pool(workers, maxtasksperchild=50) as pool:
        res = []
        for i, r in enumerate(pool.imap(job, range(n), chunksize=2), 1):
            res.append(r)
            if i % 100 == 0:
                print(f"{i}/{n} seeds, {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(3)
    S = dict(n_episodes=n, seeds=[LO, LO + n - 1], min_post_hidden=gen.MIN_POST_HIDDEN, settings={})
    S["settings"]["v1 (no collision course)"] = summarize(res, None, 0.0, rng)
    S["settings"]["v2, P=0.5 (change 1)"] = summarize(res, 2, 0.5, rng)
    for rate in RATES:
        S["settings"][f"v3, P={rate}"] = summarize(res, 3, rate, rng)
    # sanity check against change 1: version 2 at P = 0.5 on the first 200 seeds gave 222 periods, 80 collisions
    S["check_change1_first200"] = {k: v for k, v in summarize(res[:200], 2, 0.5, rng).items() if k in ("hidden_periods", "per_1000")}
    import pandas as pd
    rows = [dict(seed=r["seed"], u=r["u"], variant=vname, mode=v["mode"], **p) for r in res for vname, v in r["v"].items() for p in v["periods"]]
    pd.DataFrame(rows).to_csv("results/collide_rate_periods.csv", index=False)
    json.dump(S, open("results/collide_rate_summary.json", "w"), indent=1)
    print_table(S)


def print_table(S):
    print(f"\n{S['n_episodes']} episodes, pilot seeds {S['seeds'][0]}-{S['seeds'][1]}")
    print(f"{'setting':26s} {'per1000':>7s} {'none':>6s} {'stop':>6s} {'bnc':>6s} {'coll':>6s} {'long':>5s} {'short':>5s}  {'event share':>20s} {'collision share':>20s} "
          f"{'post med':>8s} {'<=2':>5s} {'>=5':>5s} {'succ':>9s} {'s/ep':>5s}")
    for name, d in S["settings"].items():
        cp, p = d["collision_post"], d["per_1000"]
        es, cs = d["event_share"], d["collision_share"]
        print(f"{name:26s} {d['per_1000_all']:7.0f} {p['none']:6.0f} {p['stopped']:6.0f} {p['bounce']:6.0f} {p['collision']:6.0f} "
              f"{d['per_1000_collision_long']:5.0f} {d['per_1000_collision_short']:5.0f}  "
              f"{es[0]:.2f} [{es[1]:.2f}, {es[2]:.2f}]   {cs[0]:.2f} [{cs[1]:.2f}, {cs[2]:.2f}]   "
              f"{cp['median'] if cp['median'] is not None else float('nan'):8.0f} {cp['share_le2'] or 0:5.2f} {cp['share_ge5'] or 0:5.2f} "
              f"{d['successes']:4d}/{d['attempts']:<4d} {d['gen_s_per_episode']:5.2f}")
    print("change-1 check (v2, P=0.5, first 200 seeds; expected 222 periods, 80 collisions):", S["check_change1_first200"])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 1000, int(a[1]) if len(a) > 1 else 4)
