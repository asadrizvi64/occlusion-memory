"""Stage 0 pilot data: episodes from the reserved pilot seed range only."""
import os
import sys
import time

import gen

start, n, render = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
os.makedirs("data", exist_ok=True)
t0 = time.time()
gen.generate("pilot", n, f"data/pilot_{start:05d}.pkl", start=start, render=render)
print(f"pilot {start}-{start + n} render={render}: {(time.time() - t0) / n:.2f} s/episode", flush=True)
