"""Stage 0 pilot data: episodes from the reserved pilot seed range only.

    python make_pilot.py <start> <n> <render 0/1> [generator version, default 1]

The Stage 0 pilot used generator version 1; keep the default to reproduce it. Version-2 files get a
"_v2" suffix so they never mix with the pilot files that pilot_run.py loads (data/pilot_*.pkl).
"""
import os
import sys
import time

import gen

start, n, render = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
version = int(sys.argv[4]) if len(sys.argv) > 4 else 1
os.makedirs("data", exist_ok=True)
out = f"data/pilot_{start:05d}.pkl" if version == 1 else f"data/pilotv{version}_{start:05d}.pkl"
t0 = time.time()
gen.generate("pilot", n, out, start=start, render=render, version=version)
print(f"pilot v{version} {start}-{start + n} render={render}: {(time.time() - t0) / n:.2f} s/episode", flush=True)
