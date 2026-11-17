"""Reproducibility and smoke checks. Run from the repository root: `python -m pytest -q tests`.

Checks that need the (uncommitted) pilot files skip when they are absent:
  data/pilot_00000.pkl       from `python make_pilot.py 0 200 1`
  results/pilot_preds.pkl    from `python pilot_run.py`
"""
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import baselines as B  # noqa: E402
import gen  # noqa: E402

PILOT_DATA = "data/pilot_00000.pkl"
PILOT_PREDS = "results/pilot_preds.pkl"
LO = gen.SEED_RANGES["pilot"][0]


@pytest.fixture(scope="module")
def pilot_eps():
    if not os.path.exists(PILOT_DATA):
        pytest.skip(f"{PILOT_DATA} not present")
    eps = {e["seed"]: e for e in pickle.load(open(PILOT_DATA, "rb"))}
    for e in eps.values():
        e["sim"].pop("rgb", None)
    return eps


def test_seed_ranges_disjoint():
    r = sorted(gen.SEED_RANGES.values())
    assert all(a[1] <= b[0] for a, b in zip(r, r[1:]))


@pytest.mark.parametrize("k", [0, 1, 7])
def test_v1_generator_reproduces_pilot(pilot_eps, k):
    ep = gen.sample_episode(LO + k, version=1)
    sim = gen.simulate(ep, render=False)
    ref = pilot_eps[LO + k]["sim"]
    for key in ("pos", "vel", "visible", "event", "hidden_mask"):
        assert np.array_equal(sim[key], ref[key]), key


@pytest.mark.parametrize("k", range(6))
def test_v2_deterministic_and_matches_v1_except_ball1(k):
    a, b = gen.sample_episode(LO + k, version=2), gen.sample_episode(LO + k, version=2)
    v1 = gen.sample_episode(LO + k, version=1)
    assert a["mode"] == b["mode"] and len(a["balls"]) == len(b["balls"])
    for x, y in zip(a["balls"], b["balls"]):
        assert np.array_equal(x["p0"], y["p0"]) and np.array_equal(x["v0"], y["v0"])
    assert a["cam"] == v1["cam"] and np.array_equal(a["balls"][0]["v0"], v1["balls"][0]["v0"])
    if a["mode"] == "free":
        assert len(a["balls"]) == len(v1["balls"])


@pytest.mark.parametrize("seed_index", [1, 3])  # odd episode indices were the pilot's evaluation half
def test_baselines_reproduce_pilot_predictions(pilot_eps, seed_index):
    if not os.path.exists(PILOT_PREDS):
        pytest.skip(f"{PILOT_PREDS} not present")
    preds = pickle.load(open(PILOT_PREDS, "rb"))["preds"]
    ep = pilot_eps[LO + seed_index]
    z = B.measurements(ep)
    mu, cov, _ = B.particle_filter(ep, z, n_part=100)
    assert np.array_equal(mu, preds[ep["seed"]]["B2"][0], equal_nan=True)
    assert np.array_equal(cov, preds[ep["seed"]]["B2"][1], equal_nan=True)
    mu_k, cov_k = B.kalman(z, 0.01, damp=1.0, rest=0.6, radius=[b["radius"] for b in ep["balls"]])
    assert np.array_equal(mu_k, preds[ep["seed"]]["B1d"][0], equal_nan=True)


def test_particle_filter_options_smoke():
    ep = gen.sample_episode(LO + 1, version=2)
    ep["sim"] = gen.simulate(ep, render=False)
    z = B.measurements(ep)
    base = B.particle_filter(ep, z, n_part=12)
    for kw in (dict(temper=0.25), dict(neg_evidence="off"), dict(neg_evidence="ray"), dict(oracle=True), dict(rough=0.0)):
        mu, cov, kept, D = B.particle_filter(ep, z, n_part=12, keep_particles=True, diag=True, **kw)
        assert mu.shape == base[0].shape and len(kept) == z.shape[0]
        assert D["vel"].shape == mu.shape and np.nanmax(D["ess"]) <= 1.0 + 1e-9
    # options at their defaults change nothing
    same = B.particle_filter(ep, z, n_part=12, temper=1.0, neg_evidence="mask", rough=1.0, diag=True)
    assert np.array_equal(same[0], base[0], equal_nan=True)
    _, _, vel = B.kalman(z, 0.01, damp=1.0, rest=0.6, return_vel=True)
    assert vel.shape == base[0].shape


def _balls_sha(ep):
    import hashlib
    h = hashlib.sha256()
    for b in ep["balls"]:
        for key in ("radius", "mass", "mu_slide", "mu_roll", "bouncy"):
            h.update(np.float64(b[key]).tobytes())
        h.update(np.asarray(b["p0"], np.float64).tobytes())
        h.update(np.asarray(b["v0"], np.float64).tobytes())
    return h.hexdigest()


def test_v2_unchanged_since_change_3():
    """Fingerprints recorded before version 3 was added (tests/data/gen_v2_fingerprint.json)."""
    import json
    fp = json.load(open("tests/data/gen_v2_fingerprint.json"))
    for seed, ref in fp.items():
        ep = gen.sample_episode(int(seed), version=2)
        assert ep["mode"] == ref["mode"] and _balls_sha(ep) == ref["balls_sha256"], seed


@pytest.mark.parametrize("k", [2, 5, 6])  # collide-mode seeds at rate 0.3; on k=2 version 2's choice fails the ray check
def test_v3_collisions_are_hidden_on_the_labels(k):
    e3 = gen.sample_episode(LO + k, version=3)
    assert e3["mode"] == "collide"
    post = gen.hidden_collision_post(gen.simulate(e3, render=False))
    assert post >= gen.MIN_POST_HIDDEN and post == e3["collide_plan"]["post_frames_ray"]
    e2 = gen.sample_episode(LO + k, version=2)
    if gen.hidden_collision_post(gen.simulate(e2, render=False)) >= gen.MIN_POST_HIDDEN:  # v3 = v2 when v2 passes
        assert _balls_sha(e2) == _balls_sha(e3)


def test_p_collide_couples_episodes():
    """The attempt is the first draw of the separate stream: forcing it gives the same collide-mode episode."""
    for k in (2, 5):
        assert _balls_sha(gen.sample_episode(LO + k, version=3)) == _balls_sha(gen.sample_episode(LO + k, version=3, p_collide=1.0))
    assert gen.sample_episode(LO + 2, version=3, p_collide=0.0)["mode"] == "free"
    # seed index 1 draws u = 0.455: no attempt at version 3's rate 0.3, the same collision course at 0.5
    assert gen.sample_episode(LO + 1, version=3)["mode"] == "free"
    assert _balls_sha(gen.sample_episode(LO + 1, version=3, p_collide=0.5)) == _balls_sha(gen.sample_episode(LO + 1, version=3, p_collide=1.0))
