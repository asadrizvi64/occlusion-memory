"""GPU smoke test for the Stage 1 work on Capella. No experiment: it checks that the stack works and measures
the throughput needed to plan the B3 rebuild (change 2) and the image-resolution decision.

    MUJOCO_GL=egl python cluster/gpu_smoke.py [--quick]

1. Environment: Python, torch, CUDA device, MuJoCo, rendering backend.
2. Rendering: one generator-v2 development episode (pilot seed 9,000,001) simulated and rendered at 64x64 and
   96x96. Positions at 96x96 must equal those at 64x64 (resolution must not change the physics), and no frame may
   be flat.
3. Training throughput of a representative recurrent image model (per-frame CNN encoder, GRU, Gaussian head for
   two balls; 0.8M parameters, the scale planned for B3-RGB) at 64x64 and 96x96, bf16 autocast on GPU: steps/s, episodes/s,
   peak memory, and the implied time for 10,000 episodes x 20 epochs.

Writes results/gpu_smoke_<SLURM_JOB_ID or "local">.json. --quick shrinks the benchmark for a CPU dry run.
"""
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

QUICK = "--quick" in sys.argv
OUT = {"quick": QUICK, "job": os.environ.get("SLURM_JOB_ID", "local"), "node": platform.node()}


def section(name):
    print(f"\n== {name}", flush=True)


# ---------------------------------------------------------------- 1. environment
section("environment")
import torch  # noqa: E402
import mujoco  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
           mujoco=mujoco.__version__, mujoco_gl=os.environ["MUJOCO_GL"], cpus=os.cpu_count(),
           slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK"))
if dev.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    env.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1), gpu_count=torch.cuda.device_count(),
               capability=f"{p.major}.{p.minor}", bf16=torch.cuda.is_bf16_supported())
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    env["bf16_matmul_tflops"] = round(20 * 2 * 8192 ** 3 / (time.time() - t0) / 1e12, 1)
print(json.dumps(env, indent=1), flush=True)
OUT["env"] = env

# ---------------------------------------------------------------- 2. rendering
section("MuJoCo rendering")
import gen  # noqa: E402

seed = gen.SEED_RANGES["pilot"][0] + 1
ref = None
OUT["render"] = {}
for img in (64, 96):
    gen.IMG = img  # module constant read at simulate time; 64 is the generator's default
    ep = gen.sample_episode(seed, version=2)
    t0 = time.time()
    try:
        sim = gen.simulate(ep, render=True)
    except Exception as e:  # report and continue: the training benchmark does not need rendering
        OUT["render"][img] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"{img}x{img}: FAILED {e}", flush=True)
        continue
    dt = time.time() - t0
    rgb = sim["rgb"]
    nonbg = float((rgb.reshape(len(rgb), -1, 3).std(1).mean(-1) > 1).mean())  # share of frames that are not flat
    same_phys = ref is None or bool(np.array_equal(sim["pos"], ref))
    ref = sim["pos"] if ref is None else ref
    OUT["render"][img] = dict(ok=bool(rgb.shape == (gen.N_FRAMES, img, img, 3) and nonbg == 1.0 and same_phys),
                              s_per_episode=round(dt, 2), frames_not_flat=nonbg, physics_same_as_64=same_phys,
                              mean_pixel=round(float(rgb.mean()), 1))
    print(f"{img}x{img}: {OUT['render'][img]}", flush=True)
gen.IMG = 64

# ---------------------------------------------------------------- 3. training throughput
section("training throughput (representative B3-RGB size)")


class RecurrentImageModel(torch.nn.Module):
    """Per-frame CNN (4 strided convs) -> GRU -> per-ball 2D mean and log-std. About 0.8M parameters."""

    def __init__(self, img, nmax=2, width=64, hidden=256):
        super().__init__()
        c = width
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(3, c // 2, 4, 2, 1), torch.nn.GELU(), torch.nn.Conv2d(c // 2, c, 4, 2, 1), torch.nn.GELU(),
            torch.nn.Conv2d(c, c, 4, 2, 1), torch.nn.GELU(), torch.nn.Conv2d(c, c, 4, 2, 1), torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(16 * c, hidden), torch.nn.GELU())
        self.rnn = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, nmax * 4)
        self.nmax = nmax

    def forward(self, x):  # x: (B, T, 3, H, W) uint8
        Bsz, T = x.shape[:2]
        f = self.enc(x.flatten(0, 1).float() / 255.0).view(Bsz, T, -1)
        h, _ = self.rnn(f)
        o = self.head(h).view(Bsz, T, self.nmax, 4)
        return o[..., :2], o[..., 2:].clamp(-7, 3)


OUT["train"] = {}
if dev.type != "cuda" and not QUICK:  # the full benchmark on CPU would only burn the job's time limit
    import subprocess
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        drv = ""
    OUT["train"] = dict(skipped=f"CUDA unavailable: torch built for CUDA {torch.version.cuda}, driver {drv or 'not found'}. "
                                 "Install a torch build for the CUDA version the driver supports (setup.sh TORCH_INDEX).")
    print(OUT["train"]["skipped"], flush=True)
bsz, steps, T = (4, 3, 20) if QUICK else (32, 50, gen.N_FRAMES)
for img in (() if OUT["train"] else (64, 96)):
    torch.manual_seed(0)
    net = RecurrentImageModel(img).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    x = torch.randint(0, 255, (bsz, T, 3, img, img), dtype=torch.uint8, device=dev)
    y = torch.randn(bsz, T, 2, 2, device=dev)
    w = (torch.rand(bsz, T, 2, device=dev) < 0.3).float() * 4 + 1  # hidden frames weighted up, as planned for B3
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps + 2):
        t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            mu, ls = net(x)
            nll = (0.5 * ((y - mu.float()) / ls.float().exp()) ** 2 + ls.float()).sum(-1)
            loss = (w * nll).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if s >= 2:  # skip warm-up
            times.append(time.time() - t0)
    sps = 1 / float(np.median(times))
    eps_s = sps * bsz * T / gen.N_FRAMES
    r = dict(batch=bsz, frames=T, params_m=round(sum(p.numel() for p in net.parameters()) / 1e6, 2), steps_per_s=round(sps, 2),
             episodes_per_s=round(eps_s, 1), hours_10k_x20_epochs=round(10_000 * 20 / eps_s / 3600, 2),
             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None,
             dataset_gb_10k_uint8=round(10_000 * gen.N_FRAMES * img * img * 3 / 1e9, 1))
    OUT["train"][img] = r
    print(f"{img}x{img}: {r}", flush=True)

os.makedirs("results", exist_ok=True)
path = f"results/gpu_smoke_{OUT['job']}.json"
json.dump(OUT, open(path, "w"), indent=1)
ok = all(v.get("ok") for v in OUT["render"].values()) and (QUICK or env["cuda_available"])
print(f"\nwrote {path}; {'OK' if ok else 'CHECK FAILURES ABOVE'}", flush=True)
sys.exit(0 if ok else 1)
