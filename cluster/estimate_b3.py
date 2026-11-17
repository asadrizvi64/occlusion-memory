"""GPU-time estimate for B3-RGB training, from the smoke test's measured throughput (job 4593526, H100).

Counts training FLOPs per step (forward + backward, batch 32 x 80 frames) for the smoke-test model and for B3-RGB,
and scales the measured steps/s by their ratio. Assumes equal hardware efficiency, so it is an estimate, not a
measurement; data stay on the GPU, as in b3_rgb.py.

    python cluster/estimate_b3.py
"""
import json
import os
import sys

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import b3_rgb  # noqa: E402

SMOKE = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "gpu_smoke_4593526.json")))


class SmokeModel(nn.Module):  # the benchmark model in cluster/gpu_smoke.py
    def __init__(self, width=64, hidden=256):
        super().__init__()
        c = width
        self.enc = nn.Sequential(
            nn.Conv2d(3, c // 2, 4, 2, 1), nn.GELU(), nn.Conv2d(c // 2, c, 4, 2, 1), nn.GELU(),
            nn.Conv2d(c, c, 4, 2, 1), nn.GELU(), nn.Conv2d(c, c, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(16 * c, hidden), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 8)

    def forward(self, x):
        Bsz, T = x.shape[:2]
        f = self.enc(x.flatten(0, 1).float() / 255.0).view(Bsz, T, -1)
        return self.head(self.rnn(f)[0])


def step_flops(net, x, call):
    with FlopCounterMode(display=False) as fc:
        call(net, x).float().sum().backward()
    return fc.get_total_flops()


def main(bsz=32, T=80, epochs=20, n_train=10_000, n_val=500, n_eval=1_000, count_bsz=2):
    rows = {}
    for res in (64, 96):
        # FLOPs are linear in the batch size: count at a small batch (CPU memory) and scale
        x_s = torch.randint(0, 255, (count_bsz, T, 3, res, res), dtype=torch.uint8)
        x_b = torch.randint(0, 255, (count_bsz, T, res, res, 3), dtype=torch.uint8)
        f_s = step_flops(SmokeModel(), x_s, lambda n, x: n(x)) * bsz / count_bsz
        f_b = step_flops(b3_rgb.B3RGB("rgb"), x_b, lambda n, x: n(x)) * bsz / count_bsz
        sps = SMOKE["train"][str(res)]["steps_per_s"] * f_s / f_b
        steps = epochs * -(-n_train // bsz)
        fwd_eps = epochs * n_val + n_eval  # validation each epoch + eval predictions, forward only (about 1/3 of a step)
        sec = steps / sps + fwd_eps / bsz / sps / 3
        rows[res] = dict(smoke_tflop_per_step=round(f_s / 1e12, 3), b3_tflop_per_step=round(f_b / 1e12, 3), ratio=round(f_b / f_s, 2),
                         est_steps_per_s=round(sps, 1), steps=steps, est_compute_min=round(sec / 60, 1))
        print(res, rows[res], flush=True)
    return rows


if __name__ == "__main__":
    main()
