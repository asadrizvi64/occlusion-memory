"""Rendered episodes for the B3 rebuild (change 2) and the image-resolution decision.

Generator version 3 (collision-course rate 0.3), each episode rendered at 64x64 by simulate() and at 96x96 by
gen.render_frames() from the same camera and the same physics, so the two resolutions show identical scenes.
The trajectories are checked to be identical for every episode.

Development seeds only: blocks of the reserved pilot range, disjoint from everything used before (indices
0-999). No confirmatory split (train/val/test seed ranges) is touched.

  block   pilot indices        episodes   use
  train   20,000-29,999        10,000     training
  val     30,000-30,499           500     checkpoint selection (the training loss on it)
  eval    31,000-31,999         1,000     the decision: B3 variants against B1d, 64x64 against 96x96

    python make_rgb_data.py <out_dir> [blocks=train,val,eval] [workers=8] [n=block size] [sizes=64,96] [shard=500]

Writes <out_dir>/<block>/shard_XXX_rgbS.npy (uint8, (n, 80, S, S, 3)) for each size S, then shard_XXX_meta.pkl
(the episodes without images; written last, so a shard counts as done only when it exists), and
<out_dir>/<block>/manifest.json (seeds, generator settings, commit, label counts, timing). Rerunning resumes.
"""
import json
import os
import pickle
import subprocess
import sys
import time
import multiprocessing as mp

import numpy as np

import gen
from periods import FINE, hidden_periods

BLOCKS = {"train": (20_000, 10_000), "val": (30_000, 500), "eval": (31_000, 1_000)}
LO = gen.SEED_RANGES["pilot"][0]
SIZES = (64, 96)


def make(args):
    k, sizes = args
    seed = LO + k
    t0 = time.time()
    ep = gen.sample_episode(seed, version=3)
    sim = gen.simulate(ep, render=True)  # gen.IMG = 64
    rgb = {gen.IMG: sim.pop("rgb")}
    for s in sizes:
        if s != gen.IMG:
            rgb[s], pos = gen.render_frames(ep, s)
            if not np.array_equal(pos, sim["pos"]):
                raise RuntimeError(f"seed {seed}: trajectory at {s}x{s} differs from simulate()")
    ep["sim"] = sim
    return ep, {s: rgb[s] for s in sizes}, time.time() - t0


def write_shard(path, eps, imgs, sizes):
    for s in sizes:
        tmp = f"{path}_rgb{s}.npy.tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.stack([im[s] for im in imgs]))
        os.replace(tmp, f"{path}_rgb{s}.npy")
    with open(f"{path}_meta.pkl.tmp", "wb") as f:
        pickle.dump(eps, f, protocol=4)
    os.replace(f"{path}_meta.pkl.tmp", f"{path}_meta.pkl")  # last: marks the shard complete


def label_counts(eps):
    c = {l: 0 for l in FINE}
    for ep in eps:
        for p in hidden_periods(ep):
            c[p["label_fine"]] += 1
    return c


def run_block(out_dir, block, workers, n, sizes, shard):
    start, size = BLOCKS[block]
    n = min(n or size, size)
    d = os.path.join(out_dir, block)
    os.makedirs(d, exist_ok=True)
    t0, secs = time.time(), []
    # spawn, not fork: a forked child inherits any GL context or OpenMP state of its parent and can deadlock
    pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=50) if workers > 1 else None
    mapper = pool.map if pool else (lambda f, xs, chunksize=1: list(map(f, xs)))
    try:
        for i, a in enumerate(range(start, start + n, shard)):
            path = os.path.join(d, f"shard_{i:03d}")
            ks = list(range(a, min(a + shard, start + n)))
            if os.path.exists(f"{path}_meta.pkl") and all(os.path.exists(f"{path}_rgb{s}.npy") for s in sizes):
                continue
            res = mapper(make, [(k, sizes) for k in ks], chunksize=1)
            write_shard(path, [r[0] for r in res], [r[1] for r in res], sizes)
            secs += [r[2] for r in res]
            print(f"{block}: shard {i} ({ks[0]}-{ks[-1]}) done, {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    eps = []
    for p in sorted(f for f in os.listdir(d) if f.endswith("_meta.pkl")):
        eps += pickle.load(open(os.path.join(d, p), "rb"))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    man = dict(block=block, pilot_index=[start, start + n - 1], seeds=[LO + start, LO + start + n - 1], episodes=len(eps),
               sizes=list(sizes), gen_version=3, p_collide=gen.P_COLLIDE[3], min_post_hidden=gen.MIN_POST_HIDDEN, commit=commit,
               mujoco_gl=os.environ.get("MUJOCO_GL"), collide_mode=sum(e["mode"] == "collide" for e in eps),
               hidden_periods=label_counts(eps), s_per_episode_this_run=round(float(np.mean(secs)), 2) if secs else None)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(json.dumps(man), flush=True)
    return man


if __name__ == "__main__":
    a = sys.argv[1:]
    out = a[0]
    blocks = a[1].split(",") if len(a) > 1 else list(BLOCKS)
    workers = int(a[2]) if len(a) > 2 else 8
    n = int(a[3]) if len(a) > 3 and a[3] != "-" else None
    sizes = tuple(int(s) for s in a[4].split(",")) if len(a) > 4 else SIZES
    shard = int(a[5]) if len(a) > 5 else 500
    for b in blocks:
        run_block(out, b, workers, n, sizes, shard)
