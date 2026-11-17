"""B3-RGB: the rebuilt learned baseline (pre-Stage-1 change 2), trained from RGB frames.

Model: a per-frame CNN encoder (RGBEncoder; the spec requires all learned models to share one RGB encoder), a GRU
over time (causal: the estimate at frame t uses frames 0..t only, like the filters), and a head giving each ball's
position as a 2D Gaussian with full covariance. Ball i is the ball drawn in colour gen.COLORS[i], so slots are fixed.
About 2.3M parameters (spec: 1-5M). This is the deterministic recurrent variant; the spec's B3 is "DreamerV3-style"
(stochastic latent, image reconstruction), which is a decision for the protocol freeze.

Inputs:
  rgb        RGB frames only. The B3 candidate: the same scene information as the other methods.
  rgb+meas   RGB plus the oracle noisy positions B1d receives (1 cm, visible balls only). A diagnostic, not a
             baseline: comparing it with rgb separates perception error from memory error (the spec's ablation for M).

Loss: beta-NLL (Seitzer et al. 2022, beta = 0.5, their default) of the true position, every frame from t = 1, each
ball present; with the plain NLL the predicted variance grows and the mean stops learning (see STAGE1_PREP.md). Hidden frames
are weighted so that hidden and visible frames carry equal total weight on the training set (w_hidden = visible
frames / hidden frames, fixed by that rule, not tuned). Checkpoint selection: the weighted plain NLL on the val block.
Hyperparameters are fixed in advance (no tuning trials yet; the spec allows 20 per learned model).

    python b3_rgb.py train   <data_dir> <run_dir> --res 64 --inputs rgb --seed 0 [--epochs 20 --batch 32 --n-train N]
    python b3_rgb.py predict <data_dir> <run_dir> [--block eval]
Data come from make_rgb_data.py. A run writes config.json, log.json (per epoch), ckpt.pt (best on val) and
preds_<block>.npz (mu (N, 80, 2, 2) m, cov (N, 80, 2, 2, 2) m^2, seeds, n_balls).
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import time

import numpy as np
import torch
from torch import nn

import baselines as B
import gen

NMAX = 2
SCALE = gen.HALF  # positions are modelled in units of the arena half-size
LOGSD = (-7.0, 2.0)  # clamp on log-std in model units: 0.45 mm .. 3.7 m


# ------------------------------------------------------------------ data
def load_block(data_dir, block, res, n=None, images=True):
    """Episodes (metadata) and, if images, their uint8 frames at `res` as one array, filled shard by shard into a
    preallocated array (the 96x96 training set is 22 GB; concatenating would need twice that)."""
    d = os.path.join(data_dir, block)
    shards = sorted(f[:-len("_meta.pkl")] for f in os.listdir(d) if f.endswith("_meta.pkl"))
    eps, sizes = [], []
    for s in shards:
        part = pickle.load(open(os.path.join(d, f"{s}_meta.pkl"), "rb"))
        eps += part
        sizes.append(len(part))
        if n and len(eps) >= n:
            break
    n = min(n or len(eps), len(eps))
    eps = eps[:n]
    if not images:
        return eps, None
    rgb, i = None, 0
    for s, m in zip(shards, sizes):
        if i >= n:
            break
        x = np.load(os.path.join(d, f"{s}_rgb{res}.npy"), mmap_mode="r")
        if rgb is None:
            rgb = np.empty((n,) + x.shape[1:], np.uint8)
        k = min(m, n - i)
        rgb[i:i + k] = x[:k]
        i += k
    return eps, rgb


def arrays(eps):
    """Targets, weights' ingredients and oracle measurements for a list of episodes."""
    N, T = len(eps), gen.N_FRAMES
    Y = np.zeros((N, T, NMAX, 2), np.float32)
    present = np.zeros((N, T, NMAX), bool)
    hidden = np.zeros((N, T, NMAX), bool)
    meas = np.zeros((N, T, 3 * NMAX), np.float32)
    for k, e in enumerate(eps):
        n = len(e["balls"])
        Y[k, :, :n] = e["sim"]["pos"][:, :, :2]
        present[k, 1:, :n] = True  # frame 0 is never scored (pilot convention)
        hidden[k, :, :n] = ~e["sim"]["visible"]
        meas[k] = B.gru_features(B.measurements(e), NMAX)
    return Y, present, hidden & present, meas


def weights(present, hidden, w_hidden):
    return present.astype(np.float32) * np.where(hidden, w_hidden, 1.0).astype(np.float32)


# ------------------------------------------------------------------ model
class RGBEncoder(nn.Module):
    """uint8 frames (M, H, W, 3) -> features (M, out). CoordConv input (RGB + x, y maps), one full-resolution conv
    (small balls cover 2-13 px at 64x64), four stride-2 convs, adaptive pooling to 4x4, so 64x64 and 96x96 inputs
    give the same parameter count."""

    def __init__(self, out=384, width=32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(5, w, 3, 1, 1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU(), nn.Conv2d(2 * w, 2 * w, 4, 2, 1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GELU(), nn.Conv2d(4 * w, 4 * w, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(16 * 4 * w, out), nn.GELU())

    def forward(self, x):
        M, H, W, _ = x.shape
        x = x.permute(0, 3, 1, 2).float() / 255.0
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        x = torch.cat([x, xx.expand(M, 1, H, W), yy.expand(M, 1, H, W)], 1)
        return self.net(x)


class B3RGB(nn.Module):
    def __init__(self, inputs="rgb", hidden=384, width=32):
        super().__init__()
        self.inputs = inputs
        self.enc = RGBEncoder(hidden, width)
        self.rnn = nn.GRU(hidden + (3 * NMAX if inputs == "rgb+meas" else 0), hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NMAX * 5))

    def forward(self, rgb, meas=None):
        Bsz, T = rgb.shape[:2]
        f = self.enc(rgb.flatten(0, 1)).view(Bsz, T, -1)
        if self.inputs == "rgb+meas":
            f = torch.cat([f, meas.to(f.dtype)], -1)
        h, _ = self.rnn(f)
        return self.head(h).view(Bsz, T, NMAX, 5).float()


def gauss_nll(o, y, beta=0.0):
    """o (..., 5): mean (2), log of Cholesky diagonal (2), Cholesky off-diagonal (1), model units; y in metres.
    beta > 0 gives the beta-NLL of Seitzer et al. (ICLR 2022): each term is multiplied by det(Sigma)^(beta/2) with
    the gradient stopped, which keeps large predicted variances from switching off the gradient on the mean.
    beta = 0 is the plain negative log-likelihood (used for checkpoint selection and evaluation)."""
    mu = o[..., :2]
    l1, l2 = o[..., 2].clamp(*LOGSD), o[..., 3].clamp(*LOGSD)
    d = y / SCALE - mu
    z1 = d[..., 0] / l1.exp()
    z2 = (d[..., 1] - o[..., 4] * z1) / l2.exp()
    nll = 0.5 * (z1 ** 2 + z2 ** 2) + l1 + l2
    return nll * (beta * (l1 + l2)).detach().exp() if beta else nll


def to_mean_cov(o):
    """Model outputs -> mean (m) and covariance (m^2), numpy float64."""
    o = o.detach().double()
    mu = o[..., :2] * SCALE
    a, c, b = o[..., 2].clamp(*LOGSD).exp(), o[..., 3].clamp(*LOGSD).exp(), o[..., 4]
    cov = torch.stack([torch.stack([a * a, a * b], -1), torch.stack([a * b, b * b + c * c], -1)], -2) * SCALE ** 2
    return mu.cpu().numpy(), cov.cpu().numpy()


# ------------------------------------------------------------------ train / predict
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def batches(n, bsz, gen_=None, shuffle=False):
    idx = torch.randperm(n, generator=gen_) if shuffle else torch.arange(n)
    return [idx[i:i + bsz] for i in range(0, n, bsz)]


def evaluate(net, rgb, meas, Y, W, hid, present, bsz, dev, amp):
    net.eval()
    tot = dict(wnll=0.0, w=0.0, herr=0.0, hn=0.0, verr=0.0, vn=0.0)
    with torch.no_grad():
        for idx in batches(len(rgb), bsz):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                o = net(rgb[idx].to(dev, non_blocking=True), meas[idx].to(dev))
            y, w = Y[idx].to(dev), W[idx].to(dev)
            nll = gauss_nll(o, y)
            err = (o[..., :2] * SCALE - y).norm(dim=-1) * 100
            h, p = hid[idx].to(dev), present[idx].to(dev)
            tot["wnll"] += float((w * nll).sum()); tot["w"] += float(w.sum())
            tot["herr"] += float(err[h].sum()); tot["hn"] += float(h.sum())
            tot["verr"] += float(err[p & ~h].sum()); tot["vn"] += float((p & ~h).sum())
    net.train()
    return dict(wnll=tot["wnll"] / tot["w"], hidden_err_cm=tot["herr"] / max(tot["hn"], 1), visible_err_cm=tot["verr"] / max(tot["vn"], 1))


def train(a):
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    os.makedirs(a.run_dir, exist_ok=True)
    t0 = time.time()
    tr_eps, tr_rgb = load_block(a.data_dir, "train", a.res, a.n_train)
    va_eps, va_rgb = load_block(a.data_dir, "val", a.res, a.n_val)
    t_load = time.time() - t0
    Ytr, Ptr, Htr, Mtr = arrays(tr_eps)
    Yva, Pva, Hva, Mva = arrays(va_eps)
    w_hidden = float((Ptr & ~Htr).sum() / max(Htr.sum(), 1))
    keep = "cuda" if (dev.type == "cuda" and a.data_on_device) else "cpu"
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(keep)  # noqa: E731
    tr = [T(tr_rgb), T(Mtr), T(Ytr), T(weights(Ptr, Htr, w_hidden)), T(Htr), T(Ptr)]
    va = [T(va_rgb), T(Mva), T(Yva), T(weights(Pva, Hva, w_hidden)), T(Hva), T(Pva)]
    del tr_rgb, va_rgb
    net = B3RGB(a.inputs, a.hidden, a.width).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr_eps) / a.batch)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    cfg = dict(vars(a), cmd=None, device=str(dev), params=n_par, w_hidden=w_hidden, n_train=len(tr_eps), n_val=len(va_eps),
               steps=steps, commit=git_commit(), torch=torch.__version__, load_s=round(t_load, 1),
               train_manifest=json.load(open(os.path.join(a.data_dir, "train", "manifest.json"))) if os.path.exists(os.path.join(a.data_dir, "train", "manifest.json")) else None)
    json.dump(cfg, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1, default=str)
    print(f"{a.run_dir}: {n_par / 1e6:.2f}M params, {len(tr_eps)} train / {len(va_eps)} val episodes, w_hidden {w_hidden:.2f}, "
          f"{steps} steps, data on {keep}, loaded in {t_load:.0f}s", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    log, best, step = [], float("inf"), 0
    for ep in range(a.epochs):
        te, run, nb = time.time(), 0.0, 0
        for idx in batches(len(tr_eps), a.batch, g, shuffle=True):
            idx = idx.to(keep)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                o = net(tr[0][idx].to(dev, non_blocking=True), tr[1][idx].to(dev))
            w = tr[3][idx].to(dev)
            loss = (w * gauss_nll(o, tr[2][idx].to(dev), a.beta)).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run, nb, step = run + loss.item(), nb + 1, step + 1
        if (ep + 1) % a.val_every and ep + 1 != a.epochs:
            continue
        v = evaluate(net, *va[:2], va[2], va[3], va[4], va[5], a.batch, dev, amp)
        rec = dict(epoch=ep, step=step, train_wnll=run / nb, **{f"val_{k}": x for k, x in v.items()}, epoch_s=round(time.time() - te, 1))
        if v["wnll"] < best:
            best = v["wnll"]
            torch.save(net.state_dict(), os.path.join(a.run_dir, "ckpt.pt"))
            rec["saved"] = True
        log.append(rec)
        json.dump(log, open(os.path.join(a.run_dir, "log.json"), "w"), indent=1)
        print(json.dumps(rec), flush=True)
    cfg.update(best_val_wnll=best, train_s=round(time.time() - t0, 1))
    json.dump(cfg, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1, default=str)


def predict(a):
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = B3RGB(cfg["inputs"], cfg["hidden"], cfg["width"]).to(dev)
    net.load_state_dict(torch.load(os.path.join(a.run_dir, "ckpt.pt"), map_location=dev))
    net.eval()
    eps, rgb = load_block(a.data_dir, a.block, cfg["res"])
    meas = torch.from_numpy(arrays(eps)[3])
    rgb = torch.from_numpy(rgb)
    mus, covs = [], []
    with torch.no_grad():
        for idx in batches(len(eps), cfg["batch"]):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = net(rgb[idx].to(dev), meas[idx].to(dev))
            mu, cov = to_mean_cov(o)
            mus.append(mu.astype(np.float32))
            covs.append(cov.astype(np.float32))
    np.savez_compressed(os.path.join(a.run_dir, f"preds_{a.block}.npz"), mu=np.concatenate(mus), cov=np.concatenate(covs),
                        seeds=np.array([e["seed"] for e in eps]), n_balls=np.array([len(e["balls"]) for e in eps]))
    print(f"{a.run_dir}: predictions for {len(eps)} {a.block} episodes", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "predict"])
    p.add_argument("data_dir")
    p.add_argument("run_dir")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--inputs", choices=["rgb", "rgb+meas"], default="rgb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--data-on-device", type=int, default=1)
    p.add_argument("--block", default="eval")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--val-every", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "train":
        train(a)
        predict(argparse.Namespace(run_dir=a.run_dir, data_dir=a.data_dir, block=a.block))
    else:
        predict(a)


if __name__ == "__main__":
    main()
