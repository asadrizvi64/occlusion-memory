# occlusion-memory

**Learning persistent physical states under occlusion.** Can an explicit, uncertainty-aware belief over each object's state, learned from RGB and actions, track hidden objects better than a learned latent memory? And does calibrated uncertainty about where a hidden object is improve decisions?

The research spec and the Stage 0 pilot report live in [`docs/`](docs/):

- [`docs/SPEC.md`](docs/SPEC.md): hypotheses H1–H3, frozen methodological rules, baselines, data protocol, metrics and stage gates.
- [`docs/PILOT_REPORT.md`](docs/PILOT_REPORT.md): Stage 0 pilot, with the benchmark generator, six methods, results, limitations and changes before Stage 1.
- [`docs/STAGE1_PREP.md`](docs/STAGE1_PREP.md): log of the changes made before Stage 1, with their checks.

## Status

**Stage 0 (pilot) is done; Gate 1 passes with conditions.** The benchmark separates methods, but most of the headroom is in hidden periods where the ball bounces or collides.

![Hidden-position error by what happened while hidden](figures/pilot_hidden_error.png)

| Finding (1,000 eval episodes, 894 hidden periods, one seed) | Value |
|---|---|
| Mean hidden-position error: damped Kalman (B1d) vs physics particle filter (B2) | 8.5 vs 8.1 cm; difference +0.4 cm [−0.1, +0.9] |
| B2's advantage on bounce/collision periods | 3.6 cm [2.3, 5.1], i.e. 26% |
| Extra gain from true physical parameters (oracle) | 0.5 cm [0.1, 1.0] |
| Collisions while hidden | 16 of 894 periods (1.8%) |
| 90%-region coverage: B1d / B2 | 86% / 47% (B2 is accurate but overconfident) |
| Bounce/collision periods where B2's particles go bimodal | 51% |

Before any confirmatory run:

1. ~~Make events while hidden common: at least 40% of hidden periods with an event, at least 10% with a collision.~~ Done in generator version 2: 53% events, 36% collisions on 200 pilot seeds ([log](docs/STAGE1_PREP.md)).
2. Rebuild the learned baseline (B3): hidden-weighted loss, 10k episodes, scene information.
3. ~~Re-run B2 with more particles or tempering.~~ Done: B2's overconfidence and its pre-contact gap to B1d come mainly from sample impoverishment (no process noise beyond roughening), not particle count. Roughening ×2 at 100 particles matches 500 particles at a fifth of the cost. Recommended B2 for Stage 1: 300 particles with roughening ×2, which is 3.0 cm better than B1d on fresh development episodes and 5.6 cm (38%) better after a hidden contact ([log](docs/STAGE1_PREP.md#change-3-why-b2-is-overconfident-and-worse-than-b1d-before-a-contact)).
4. Decide the image resolution: 64×64 vs 96×96, or larger balls.

## What is here

| Path | Purpose |
|---|---|
| `gen.py` | Procedural MuJoCo benchmark: walled 1 m tabletop, 1–2 balls, screens (occlude only) and blocks (occlude and collide), per-episode camera, ray-cast visibility and hidden-region maps, event labels. `SEED_RANGES` fixes disjoint seeds per split. Version 2 adds a collision-course mode for hidden collisions. Version 3 (default) also checks each hidden collision against the 9-ray labels. `version=1` and `version=2` reproduce earlier data exactly. |
| `baselines.py` | B0 last-seen, B1 constant-velocity Kalman, B1d damped Kalman with wall reflection, B2 MuJoCo particle filter (prior-sampled parameters, negative evidence from the hidden-region map), oracle-parameter filter, B3 GRU with Gaussian output. |
| `make_pilot.py` | Generate pilot episodes: `python make_pilot.py <start> <n> <render 0/1> [version, default 1]`. |
| `pilot_run.py` | Tune on the fit half, train the GRU, run every method on the eval half. |
| `pilot_analyze.py` | Hidden-period labels, errors, coverage, episode-clustered bootstrap, power estimate, multimodality test. |
| `make_figures.py` | Figures in `figures/`. |
| `check_generator.py` | Generator version 1 vs 2 on the same pilot seeds: regression against the stored pilot, hidden-period labels, B1d vs B2 by label. Writes `results/generator_check.json`. |
| `diag_collisions.py` | Hidden-collision diagnostic: frames hidden after contact, deflection, error before and after contact. Writes `results/collision_diag.csv`. |
| `check_collide_rate.py` | Generator finalization. Compares collision-course rates 0.3/0.4/0.5 and versions 2 and 3 on 1,000 development seeds, using common random numbers. Reports label mix, hidden frames after contact and cost. Writes `results/collide_rate_*`. |
| `b2_diagnostics.py` | Change 3: B2 variants (particle count, tempering, roughening scale, negative evidence, true parameters) against B1d. Scores error, energy score, coverage of Gaussian and particle-HPD regions, ESS and runtime. Checkpoints each job and resumes. Writes `results/b2_diag*_summary.json` and `results/b2_diag*_intervals.csv`. |
| `b2_report.py` | Tables for the change 3 write-up, with paired episode-clustered intervals. Writes `results/b2_diag*_tables.md`. |
| `tests/` | Reproducibility and smoke tests: `python -m pytest -q tests`. The version-1 generator and the pilot's B1d/B2 predictions must reproduce bit for bit. |
| `results/pilot_summary.json` | Every number in the pilot report. |
| `results/pilot_intervals.csv` | One row per hidden period: label, duration, per-method error, coverage and region area. |

![Pilot episodes as the camera sees them](figures/pilot_episodes.png)

## Reproduce the pilot

CPU only, about 45 minutes on 4 cores. Rendering uses OSMesa by default (`apt-get install libosmesa6`); set `MUJOCO_GL=egl` to render on a GPU instead.

```bash
pip install -r requirements.txt
```

```bash
python make_pilot.py 0 200 1 & python make_pilot.py 200 600 0 & python make_pilot.py 800 600 0 & python make_pilot.py 1400 600 0 & wait
```

```bash
python pilot_run.py 4
```

```bash
python pilot_analyze.py && python make_figures.py
```

Episode files (`data/`) and per-episode predictions (`results/*.pkl`) are not committed: they regenerate exactly from their seeds with MuJoCo 3.8.1.

## Ground rules

These rules are frozen before any confirmatory run; see the spec.

1. Simulator ground truth never enters a model or planner at test time.
2. Every comparison uses identical episodes, observations and action budgets.
3. Prediction and decision performance are evaluated separately.
4. The proposed model must beat B1 and a strong learned baseline, not merely last-seen tracking.
5. A deterministic-belief ablation is required.
6. Pilot episodes (seeds 9,000,000–9,099,999) are never reused for confirmatory splits.

This project is separate from the Real2Sim/VLA reconstruction-fidelity study. The one planned link comes after Stage 4: a LIBERO task whose target is hidden.
