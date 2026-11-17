"""Checks for the B3 rebuild code: dual-resolution rendering, period labels, the B3-RGB loss and model size, and the
data pipeline on two episodes. Run from the repository root: `python -m pytest -q tests`."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import b3_rgb  # noqa: E402
import gen  # noqa: E402
import make_rgb_data  # noqa: E402
from periods import SHORT, hidden_periods  # noqa: E402

LO = gen.SEED_RANGES["pilot"][0]


@pytest.mark.parametrize("k", [1, 2])
def test_render_frames_matches_simulate(k):
    ep = gen.sample_episode(LO + k, version=3)
    sim = gen.simulate(ep, render=True)
    r64, p64 = gen.render_frames(ep, 64)
    r96, p96 = gen.render_frames(ep, 96)
    assert np.array_equal(r64, sim["rgb"]) and np.array_equal(p64, sim["pos"]) and np.array_equal(p96, sim["pos"])
    assert r96.shape == (gen.N_FRAMES, 96, 96, 3) and r96.std() > 1


def test_periods_split_collisions():
    ep = gen.sample_episode(LO + 2, version=3)  # collide mode at rate 0.3
    ep["sim"] = gen.simulate(ep, render=False)
    ps = hidden_periods(ep)
    col = [p for p in ps if p["label"] == "collision"]
    assert col and all(p["label_fine"] == ("collision_short" if p["post"] <= SHORT else "collision_long") for p in col)
    assert max(p["post"] for p in col) == gen.hidden_collision_post(ep["sim"])
    assert all(p["label_fine"] == p["label"] for p in ps if p["label"] != "collision")


def test_b3_loss_is_bivariate_gaussian_nll():
    torch.manual_seed(0)
    o = torch.randn(3, 4, 2, 5, dtype=torch.float64)
    y = torch.randn(3, 4, 2, 2, dtype=torch.float64) * 0.2
    nll = b3_rgb.gauss_nll(o, y).double().numpy()
    mu, cov = b3_rgb.to_mean_cov(o)
    d = y.numpy() - mu
    ref = 0.5 * np.einsum("...i,...ij,...j", d, np.linalg.inv(cov), d) + 0.5 * np.log(np.linalg.det(cov))
    assert np.ptp(nll - ref) < 1e-5  # equal up to the constant from the unit scale
    assert (np.linalg.eigvalsh(cov) > 0).all()


def test_b3_size_within_spec():
    n = {i: sum(p.numel() for p in b3_rgb.B3RGB(i).parameters()) for i in ("rgb", "rgb+meas")}
    assert all(1e6 <= v <= 5e6 for v in n.values())  # spec: 1-5M parameters
    assert abs(n["rgb+meas"] / n["rgb"] - 1) < 0.10  # spec: learned models within +-10% of each other
    x = torch.randint(0, 255, (1, 3, 96, 96, 3), dtype=torch.uint8)
    assert b3_rgb.B3RGB("rgb")(x).shape == (1, 3, 2, 5)  # 96x96 input, same parameters


def test_data_pipeline_two_episodes(tmp_path, monkeypatch):
    monkeypatch.setitem(make_rgb_data.BLOCKS, "train", (20_000, 2))
    man = make_rgb_data.run_block(str(tmp_path), "train", 1, None, (64, 96), 2)
    assert man["episodes"] == 2 and man["gen_version"] == 3 and man["p_collide"] == 0.3
    eps, x64 = b3_rgb.load_block(str(tmp_path), "train", 64)
    _, x96 = b3_rgb.load_block(str(tmp_path), "train", 96)
    assert x64.shape == (2, 80, 64, 64, 3) and x96.shape == (2, 80, 96, 96, 3)
    Y, present, hidden, meas = b3_rgb.arrays(eps)
    assert not present[:, 0].any() and (hidden <= present).all() and Y.shape == (2, 80, 2, 2)
