"""Stage 0 pilot: tune on the fit half (even episode index), evaluate on the eval half (odd).
Outputs results/pilot_preds.pkl (per eval episode: method -> (mu, cov)) and B2 particles."""
import glob
import itertools
import os
import pickle
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np

import baselines as B


def load():
    eps = []
    for p in sorted(glob.glob("data/pilot_*.pkl")):
        eps += pickle.load(open(p, "rb"))
    eps.sort(key=lambda e: e["seed"])
    for e in eps:
        e["sim"].pop("rgb", None)
    return eps


def hidden_mask(ep):
    v = ep["sim"]["visible"]
    h = ~v
    h[0] = False  # balls start visible; frame 0 is never scored
    return h


def gauss_nll(mu, cov, gt, mask):
    d = (gt - mu)[mask]
    C = cov[mask]
    inv = np.linalg.inv(C)
    m2 = np.einsum("ki,kij,kj->k", d, inv, d)
    return float(np.mean(0.5 * m2 + 0.5 * np.log(np.linalg.det(C)) + np.log(2 * np.pi)))


def tune(fit, fn, grid):
    Z = [B.measurements(e) for e in fit]
    best = None
    for prm in grid:
        nll = []
        for e, z in zip(fit, Z):
            m = hidden_mask(e)
            if not m.any():
                continue
            mu, cov = fn(e, z, prm)
            nll.append(gauss_nll(mu, cov, e["sim"]["pos"][:, :, :2], m))
        s = float(np.mean(nll))
        if best is None or s < best[0]:
            best = (s, prm)
    return best


def train_gru(fit, seed=0, steps=3000):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    X = torch.tensor(np.stack([B.gru_features(B.measurements(e)) for e in fit]))
    Y = np.zeros((len(fit), 80, 2, 2), np.float32)
    M = np.zeros((len(fit), 80, 2), np.float32)
    for k, e in enumerate(fit):
        n = len(e["balls"])
        Y[k, :, :n] = e["sim"]["pos"][:, :, :2]
        M[k, 1:, :n] = 1.0
    Y, M = torch.tensor(Y), torch.tensor(M)
    nval = len(fit) // 10
    net = B.make_gru()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def loss_fn(idx):
        mu, ls = net(X[idx])
        nll = 0.5 * (((Y[idx] - mu) / ls.exp()) ** 2).sum(-1) + ls.sum(-1)
        return (nll * M[idx]).sum() / M[idx].sum()

    rng = np.random.default_rng(seed)
    best, best_state = 1e9, None
    for it in range(steps):
        idx = torch.tensor(rng.choice(np.arange(nval, len(fit)), 128))
        loss = loss_fn(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 99:
            with torch.no_grad():
                v = float(loss_fn(torch.arange(nval)))
            if v < best:
                best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best


def run_gru(net, ep, z):
    import torch
    with torch.no_grad():
        mu, ls = net(torch.tensor(B.gru_features(z))[None])
    n = len(ep["balls"])
    mu, sd = mu[0, :, :n].numpy().astype(np.float64), ls[0, :, :n].exp().numpy().astype(np.float64)
    cov = np.zeros(mu.shape + (2,))
    cov[..., 0, 0], cov[..., 1, 1] = sd[..., 0] ** 2, sd[..., 1] ** 2
    return mu, cov


def pf_job(args):
    ep, n_part, oracle = args
    z = B.measurements(ep)
    t0 = time.time()
    mu, cov, kept = B.particle_filter(ep, z, n_part=n_part, oracle=oracle, keep_particles=not oracle)
    hid = hidden_mask(ep)
    if kept is not None:  # keep particles only on hidden frames (multimodality check)
        kept = {t: kept[t] for t in range(80) if hid[t].any()}
    return ep["seed"], mu, cov, kept, time.time() - t0


def main(n_workers=3):
    os.makedirs("results", exist_ok=True)
    eps = load()
    fit, ev = eps[0::2], eps[1::2]
    print(f"episodes: fit {len(fit)}, eval {len(ev)}", flush=True)
    t0 = time.time()
    b0 = tune(fit, lambda e, z, p: B.last_seen(z, p), [0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    b1 = tune(fit, lambda e, z, p: B.kalman(z, p), [0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    grid = list(itertools.product([0.01, 0.1, 0.3, 1, 3], [0.0, 0.25, 0.5, 1.0, 2.0], [0.1, 0.35, 0.6]))
    b1d = tune(fit, lambda e, z, p: B.kalman(z, p[0], damp=p[1], rest=p[2], radius=[b["radius"] for b in e["balls"]]), grid)
    print(f"tuned: B0 sig_tau {b0[1]} (nll {b0[0]:.2f}); B1 q {b1[1]} ({b1[0]:.2f}); B1d {b1d[1]} ({b1d[0]:.2f}); {time.time()-t0:.0f}s", flush=True)
    net, vbest = train_gru(fit)
    print(f"GRU trained: val nll {vbest:.3f}; {time.time()-t0:.0f}s", flush=True)
    preds = {e["seed"]: {} for e in ev}
    timing = {}
    for name, fn in [("B0", lambda e, z: B.last_seen(z, b0[1])), ("B1", lambda e, z: B.kalman(z, b1[1])),
                     ("B1d", lambda e, z: B.kalman(z, b1d[1][0], damp=b1d[1][1], rest=b1d[1][2], radius=[b["radius"] for b in e["balls"]])),
                     ("B3", lambda e, z: run_gru(net, e, z))]:
        t1 = time.time()
        for e in ev:
            preds[e["seed"]][name] = fn(e, B.measurements(e))
        timing[name] = (time.time() - t1) / len(ev)
    print(f"fast methods done; {time.time()-t0:.0f}s", flush=True)
    particles = {}
    with Pool(n_workers) as pool:
        for name, n_part, oracle in [("B2", 100, False), ("OR", 32, True)]:
            t1 = time.time()
            secs = []
            for seed, mu, cov, kept, dt in pool.imap_unordered(pf_job, [(e, n_part, oracle) for e in ev], chunksize=4):
                preds[seed][name] = (mu, cov)
                secs.append(dt)
                if kept is not None:
                    particles[seed] = kept
            timing[name] = float(np.mean(secs))
            print(f"{name} done: {timing[name]:.1f} s/episode/core; {time.time()-t0:.0f}s", flush=True)
    pickle.dump(dict(preds=preds, tuned=dict(B0=b0, B1=b1, B1d=b1d, GRU_val_nll=vbest), timing=timing,
                     eval_seeds=[e["seed"] for e in ev]), open("results/pilot_preds.pkl", "wb"), protocol=4)
    pickle.dump(particles, open("results/pilot_particles.pkl", "wb"), protocol=4)
    print("PILOT_RUN_DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
