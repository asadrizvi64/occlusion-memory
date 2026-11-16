# Stage 1 preparation log

The Stage 0 pilot report lists three changes to make before any confirmatory run. This file records each one as it lands. Numbers come from the pilot seed range (9,000,000 onward), which is reserved for development and never used for confirmatory splits.

| Change | Status |
|---|---|
| 1. Make events while hidden common | **Done: generator version 2** (below) |
| 2. Rebuild the learned baseline (B3) | Not started |
| 3. Re-run B2 with more particles or tempering | **Done: diagnosed; recommended B2 = 300 particles, roughening ×2** (below) |
| Generator finalization: version 3 and collision rate | **Done: version 3 implemented; rate 0.3 recommended, awaiting decision** (below) |
| GPU check on Capella | **Done: H100 works; GPU cost is not a constraint** (below) |
| Open decision: image resolution | Waiting on a decision; compute no longer a factor (below) |

## Change 1: events while hidden (generator version 2)

**Target** (from the pilot report): at least 40% of hidden periods with an event (bounce or collision) and at least 10% with a collision. The pilot generator gave 28% and 1.7%.

**Result:** 53% events and 36% collisions on 200 episodes. Both targets are met with margin.

### What changed in `gen.py`

- **Collision-course mode.** In `P_COLLIDE` = 50% of episodes the generator tries to launch ball 1 so that it hits ball 0 while ball 0 is hidden:
  1. Roll ball 0 out alone (physics only) and list the frames between 0.5 s and 7.5 s where it sits in a hidden cell of the table.
  2. Pick one such frame t\* and an approach direction. Aim ball 1 at ball 0's position at t\*, offset sideways by up to 0.6 of the radius sum, so impacts range from head-on to glancing. Start ball 1 0.25–0.7 m away, at a visible spot with a clear straight path.
  3. Find ball 1's launch speed (0.2–1.8 m/s) by bisection, so it covers that distance by t\*. Distance travelled by a fixed time is monotone in launch speed.
  4. Roll both balls out. Accept only if the first ball–ball contact is in a hidden cell, and a ball hidden at contact stays hidden for at least `MIN_POST_HIDDEN` = 5 frames (0.5 s) afterwards, or until the episode ends.
  5. Allow up to 40 tries. If none succeeds, the episode keeps its version-1 balls (`mode = "free"`, `collide_failed = True`).
- **Separate random stream.** Version 2 draws its choices from `default_rng([seed, 2])`, so the scene, camera and ball 0 are identical to version 1 for every seed. The two versions differ only in ball 1 of collide-mode episodes.
- **Version 1 is kept.** `sample_episode(..., version=1)` reproduces the pilot exactly; on all 200 seeds checked, the regenerated positions and visibility were identical to the stored pilot files. `make_pilot.py` defaults to version 1 and writes version-2 data to separate `data/pilotv2_*.pkl` files.
- **Bookkeeping.** Each episode now records `gen_version`, `mode`, and, in collide mode, `collide_plan`: meeting time, contact time, launch speed, hidden frames after contact.
- **Cost.** Sampling rises from 0.03 to 0.08 s per episode; simulation stays at about 0.6 s. Generation is still dominated by visibility ray casting.

### Check on 200 pilot seeds

`python check_generator.py 200 4` regenerates seeds 9,000,000–9,000,199 with both versions, without rendering. It scores the pilot-tuned B1d and the 100-particle B2 on each.

**Hidden-period labels:**

| Generator | Hidden periods | None | Stopped | Bounce | Collision | Event share | Collision share |
|---|---|---|---|---|---|---|---|
| Version 1 (pilot) | 181 | 106 | 25 | 47 | 3 | 28% | 1.7% |
| Version 2 | 222 | 88 | 17 | 37 | 80 | **53%** | **36%** |
| Version 2, collide-mode episodes only | 95 | 14 | 0 | 3 | 78 | 85% | 82% |
| Version 2, free episodes only | 127 | 74 | 17 | 34 | 2 | 28% | 1.6% |

**Collision-course success.** 102 of 200 episodes attempted a collision course and 49 succeeded. Most failures are not search failures: in 44 of them ball 0 never enters a hidden cell, so there is nothing to collide with behind an occluder. 45 of the 49 accepted episodes contain a hidden period labelled "collision" under the real 9-ray visibility. The acceptance check uses the cheaper table-plane hidden map, so the two can disagree near occluder edges.

**Hidden-position error, B1d vs B2** (cm, mean per hidden period; differences with 95% bootstrap intervals clustered by episode):

| Version 2 periods | n | B1d | B2 | B1d − B2 |
|---|---|---|---|---|
| All | 222 | 9.8 | 8.8 | +1.0 [−0.4, +2.2] |
| No event | 88 | 5.2 | 5.3 | −0.1 [−0.9, +0.7] |
| Stopped | 17 | 9.6 | 10.8 | −1.2 [−5.6, +2.9] |
| Bounce | 37 | 14.7 | 13.0 | +1.7 [−1.8, +5.1] |
| Collision | 80 | 12.6 | 10.4 | +2.2 [−0.7, +4.9] |
| Bounce or collision | 117 | 13.3 | 11.2 | +2.1 [−0.4, +4.2] |

At 200 episodes none of these differences is significant, including bounces, where the 1,000-episode pilot found +3.6 cm [2.3, 5.1]. A 1,000-episode split would give about 400 collision periods.

### Why the post-contact requirement was added

The first version-2 run accepted any collision that started in a hidden cell. Collisions became common (36% of periods), but many tested little memory: half the hidden balls reappeared within 5 frames of the contact. `python diag_collisions.py` splits each collision period at the contact frame:

| Collision periods (collide-mode episodes) | No post-contact requirement | `MIN_POST_HIDDEN` = 5 |
|---|---|---|
| Periods (episodes) | 79 (47) | 78 (45) |
| Median hidden frames after contact | 5 | 36 |
| Periods ending within 2 frames of contact | 24% | 12% |
| Still hidden at the last frame | 44% | 51% |
| Median deflection of the hidden ball | 34° | 36° |
| Error before contact, B1d / B2 | 4.4 / 6.4 | 4.1 / 6.6 |
| Error after contact, B1d / B2 | 12.9 / 10.6 | 14.8 / 11.8 |
| B1d − B2 after contact | +2.3 [−0.8, +5.0] (18%) | +3.0 [−0.2, +6.1] (20%) |
| Error at the period's last frame, B1d / B2 | 15.2 / 11.7 | 18.2 / 13.4 |
| B1d − B2 at the last frame | +3.6 [+0.1, +6.7] (23%) | **+4.7 [+1.2, +8.3] (26%)** |

The requirement is part of the task definition: the collision's outcome must be remembered rather than seen. It does not depend on any model's score.

### What this means for the spec

- **Collisions are now the main event type:** 80 of 117 event periods. Results are reported per label, so the primary metric is not confounded. But an average over all periods is now dominated by collisions. If a more balanced mix is wanted, lower `P_COLLIDE` to about 0.3; that should still clear the 40% event target.
- **Physics helps after a hidden collision, but not by much.** Even the true simulator, with parameters drawn from the prior, is still 13.4 cm off at the end of a collision period, against 18.2 cm for B1d. Much of the outcome is not determinable from the observations: with 1 cm noise at 10 Hz, the impact geometry is uncertain, and small differences in it change the post-collision path a lot.
  - Implication for the minimum important improvement (provisionally 20% below the best of B1d and B3): on collision periods, 20% is about what engineered physics achieves (20–26%).
  - Hidden collisions are therefore likely to matter more for calibration (H2) and decisions (H3) than for mean error (H1). This fits the pilot finding that B2's particles go bimodal after events.
- **B2 is worse than B1d before the contact**, 6.6 against 4.1 cm, as on the pilot's no-event periods. That points to the particle approximation (100 particles, roughening, prior-sampled parameters), not the motion model. Change 3 tests this. *(Change 3: confirmed, and fixed. The cause was sample impoverishment. On fresh episodes, the revised B2 (300 particles, roughening ×2) is level with B1d before the contact, 4.4 against 4.7 cm, and 5.6 cm (38%) better after it.)*
- **Not implemented: placing blocks inside hidden regions.** The bounce share stayed at about 17%, but the event target was met without it, so the generator keeps one lever.

### Limitations of this check

- **Small sample:** 200 episodes and one seed. Intervals on the per-label differences are wide.
- **Pilot-tuned B1d:** B1d keeps its pilot parameters. Retuning it on version-2 data could change its collision errors.
- **Visibility mismatch:** acceptance uses the table-plane hidden map, labels use ray-based visibility. 12% of collision periods still end within 2 frames of contact.

## Change 3: why B2 is overconfident and worse than B1d before a contact

**Questions** (from the pilot report and change 1):
- B2's 90% regions covered well under 90% of true positions.
- On collision periods, B2 was worse than B1d before the contact (6.6 against 4.1 cm), although it simulates the true physics.

Was that particle count, weight degeneracy, resampling and roughening, likelihood scaling, initialization, the Gaussian summary, or parameter uncertainty?

**Answer.** The main cause is **sample impoverishment**:
- **Where it comes from.** B2's dynamics are deterministic between resamplings, so its only process noise is the roughening jitter added when it resamples.
- **Roughening off.** The filter collapses onto one trajectory and loses track even while the ball is visible: 12.2 cm error on visible frames.
- **Roughening ×2** at 100 particles:
  - Hidden-position error drops by 1.6 cm [0.6, 2.6].
  - The pre-contact gap to B1d closes.
  - Coverage rises from 0.41 to 0.59.
  - It runs about 5× faster than 500 particles, which gain a similar 1.4 cm.

**Parameter uncertainty** is the second cause of error (true parameters: −2.1 cm), but not of the overconfidence.

**Not causes:**
- The Gaussian summary: regions built from the particles themselves cover *less*.
- The mismatch between negative-evidence and label visibility.
- Likelihood tempering: it widens the regions, but does not make the filter more accurate.

**Remaining gap.** No stable particle configuration is calibrated. At the 90% level, the stable ones cover at most 65% of true positions on the selection block and 75% on the validation block (B2-300-R2-T).

**Recommended B2 for Stage 1:** 300 particles with roughening ×2, checked on 200 fresh development episodes.
- 2.2 cm [1.5, 3.1] better than the pilot B2.
- 3.0 cm [2.1, 3.9] better than B1d overall, and 5.6 cm [3.8, 7.5] (38%) better after a hidden contact.
- 90% coverage of 0.68.
- About 20 s per episode on one core.

**Commands:**
- `python b2_diagnostics.py 200 4`: the selection block, about 75 minutes on 4 cores.
- `python b2_diagnostics.py 200 4 B2-100,B2-100-R2,B2-300-R2,B2-300-R2-T 200`: the validation block.
- `python b2_report.py [tag]`: the tables.

**Outputs** (committed):
- `results/b2_diag_summary.json`, `results/b2_diag_intervals.csv` and `results/b2_diag_tables.md`.
- `_s200` versions of the same files for the validation block.

### Setup

- **Data:** generator version 2, development seeds only.
  - **Selection block:** pilot seeds 9,000,000–9,000,199. These are the change 1 episodes: 222 hidden periods (88 no event, 17 stopped, 37 bounce, 80 collision).
  - **Validation block:** seeds 9,000,200–9,000,399, used once, to check the recommended configuration (below).
  - No confirmatory seeds were used.
- **Identical inputs:** every method sees identical noisy measurements. Each filter's random stream is seeded by the episode seed. B1d keeps its pilot tuning (q = 0.01, damping 1.0, restitution 0.6).
- **Configurations.** One factor changes at a time from the pilot B2 (B2-100):

| Name | Particles | Change from pilot B2 | Tests |
|---|---|---|---|
| B2-100 | 100 | none (pilot B2; reproduces the pilot's predictions bit for bit) | reference |
| B2-300, B2-500 | 300, 500 | particle count | particle count |
| B2-100-T, B2-300-T | 100, 300 | likelihood exponent 0.25, so the measurement likelihood is twice as wide (fixed before any run) | likelihood scaling |
| B2-100-R0, -R2, -R4 | 100 | roughening off, ×2, ×4. Same random numbers, so only the jitter size differs | resampling and roughening |
| B2-100-noNE | 100 | no negative evidence while hidden | negative evidence |
| B2-100-rayNE | 100 | negative evidence from the same 9-ray visibility test that labels the truth, not the table-plane hidden map | mask/label mismatch |
| OR-100 | 100 | true physical parameters, not sampled from the prior | parameter uncertainty |
| B1d | – | pilot-tuned damped Kalman filter with wall bounces | reference |

- **Roughening:** pilot values at each resampling are position +N(0, 3 mm), velocity +N(0, 2 cm/s), and mass and friction ×exp(N(0, 0.05)).
- **When the variants were declared:**
  - R0 and R2 were added after the first full sweep failed and before any full results existed.
  - R4 was added after R2's result, to check whether the trend continues past ×2.
- **Metrics** (per hidden period, then averaged over periods):
  - Error of the mean.
  - Error at the period's last frame.
  - For collision periods, error before and after the contact.
  - Energy score: proper and bandwidth-free; B1d's is computed from 400 Gaussian samples.
  - Coverage and area of 50%/90% regions, built two ways:
    - from the Gaussian summary (the pilot's chi-square ellipse);
    - from the particles (highest-density region of a weighted kernel density estimate; Scott bandwidth on the effective sample size, floor 0.5 cm).
  - Also: state error at the last visible frame; error on visible frames; ESS before resampling, resampling rate and particle diversity; runtime.
  - All intervals are 95% bootstrap intervals clustered by episode (2,000 resamples).

### Audit of the pilot filter

What B2 does, and what each step could do wrong:
1. **Initialization** from the first two measurements: position = measurement + N(0, 1 cm); velocity = finite difference + N(0, 15 cm/s). The finite difference of two 1 cm-noise measurements 0.1 s apart has about 14 cm/s noise of its own. So the cloud starts centred on a velocity that is wrong by about one cloud width.
2. **Physical parameters** are sampled once per particle from the prior (mass, sliding and rolling friction, bouncy or not). They change only through roughening.
3. **Prediction:** exact MuJoCo simulation of every particle, **with no process noise**.
4. **Weighting:**
   - **Visible:** Gaussian likelihood, σ² = 1 cm² + 1 cm² (measurement noise plus a model-error term).
   - **Hidden:** negative evidence, weight 0.95 if the particle centre is in a hidden cell of the table, 0.05 otherwise.
5. **Resampling** (systematic) when ESS < N/2, followed by roughening. A resampled particle and its copies follow identical deterministic trajectories, separated only by the jitter.
6. **Summary:** weighted mean and covariance, with a (1 mm)² floor. The floor is negligible against 90% regions of 25–250 cm².

Step 3 is the structural difference from B1d. B1d adds process noise at every step: white acceleration with q = 0.01 m²/s³, about 0.3 m/s² per step. B2 adds it only at resampling events, which happen on 38% of frames.

### Results (selection block)

**Accuracy** (cm, mean over hidden periods):

| Method | All | No event | Bounce | Collision | Pre-contact | Post-contact | End of period | Energy score | s / episode |
|---|---|---|---|---|---|---|---|---|---|
| B2-100 (pilot) | 8.8 | 5.3 | 13.0 | 10.4 | 6.5 | 11.7 | 11.5 | 7.83 | 4.8 |
| B2-300 | 8.2 | 5.2 | 11.0 | 9.8 | 5.5 | 11.2 | 10.8 | 7.05 | 15.3 |
| B2-500 | 7.4 | 5.2 | 9.8 | 8.4 | 4.7 | 9.6 | 9.8 | 6.25 | 27.5 |
| B2-100-T | 9.0 | 6.3 | 12.4 | 10.5 | 6.4 | 11.8 | 11.8 | 7.81 | 3.8 |
| B2-300-T | 8.3 | 5.6 | 10.4 | 9.9 | 5.1 | 11.2 | 11.0 | 7.01 | 13.1 |
| B2-100-R0 | 11.1 | 6.4 | 12.1 | 15.2 | 7.6 | 16.7 | 14.0 | 10.67 | 2.8 |
| **B2-100-R2** | **7.3** | **4.8** | 10.2 | **8.3** | 4.3 | 9.4 | 9.7 | 5.94 | 5.4 |
| B2-100-R4 | 8.6 | 5.2 | 10.0 | 11.1 | 4.3 | 14.0 | 13.3 | 6.29 | 6.7 |
| B2-100-noNE | 10.1 | 6.0 | 13.5 | 11.9 | 7.2 | 13.2 | 13.0 | 8.72 | 4.6 |
| B2-100-rayNE | 8.4 | 5.3 | 11.2 | 10.2 | 6.2 | 11.3 | 10.9 | 7.43 | 4.9 |
| OR-100 (true parameters) | 6.7 | 5.1 | 9.4 | 7.3 | 4.3 | 8.2 | 8.8 | 5.79 | 4.2 |
| B1d | 9.8 | 5.2 | 14.7 | 12.6 | 4.1 | 14.6 | 13.8 | 7.68 | <0.01 |

Runtime is the filter alone, on one core of this container.

**Paired differences against the pilot B2** (B2-100 minus method, cm; positive = method better):

| Method | Error, all | Error, events | Post-contact | End of period | Energy score |
|---|---|---|---|---|---|
| B2-300 | +0.6 [−0.2, +1.5] | +1.0 [−0.6, +2.7] | +0.5 [−1.6, +2.8] | +0.7 [−0.5, +1.9] | +0.78 [−0.08, +1.69] |
| B2-500 | **+1.4 [+0.7, +2.2]** | +2.4 [+1.0, +3.8] | +2.1 [+0.5, +3.9] | +1.7 [+0.6, +2.8] | +1.58 [+0.84, +2.39] |
| B2-100-T | −0.2 [−0.9, +0.5] | +0.1 [−1.1, +1.3] | −0.2 [−1.5, +1.2] | −0.4 [−1.3, +0.5] | +0.01 [−0.65, +0.69] |
| B2-300-T | +0.5 [−0.2, +1.4] | +1.1 [−0.4, +2.7] | +0.5 [−1.5, +2.2] | +0.4 [−0.7, +1.6] | +0.82 [+0.07, +1.61] |
| B2-100-R0 | **−2.3 [−3.5, −1.1]** | −3.0 [−5.1, −1.1] | −5.0 [−7.6, −2.9] | −2.5 [−4.1, −0.9] | −2.84 [−4.07, −1.73] |
| B2-100-R2 | **+1.6 [+0.6, +2.6]** | +2.3 [+0.7, +4.2] | +2.3 [+0.4, +4.8] | +1.8 [+0.6, +3.2] | +1.89 [+1.02, +2.91] |
| B2-100-R4 | +0.3 [−1.8, +2.0] | +0.4 [−3.5, +3.5] | −2.3 [−11.8, +4.1] | −1.9 [−9.3, +2.4] | +1.54 [−0.12, +2.95] |
| B2-100-noNE | **−1.2 [−1.8, −0.5]** | −1.2 [−2.2, −0.0] | −1.5 [−2.6, −0.5] | −1.5 [−2.5, −0.6] | −0.90 [−1.45, −0.25] |
| B2-100-rayNE | +0.4 [−0.1, +1.0] | +0.7 [−0.3, +1.8] | +0.3 [−0.7, +1.4] | +0.6 [−0.1, +1.4] | +0.39 [−0.11, +1.01] |
| OR-100 | **+2.1 [+1.1, +3.2]** | +3.2 [+1.4, +5.3] | +3.4 [+1.3, +5.9] | +2.6 [+1.3, +3.9] | +2.04 [+1.00, +3.12] |
| B1d | −1.0 [−2.2, +0.3] | −2.1 [−4.2, +0.2] | −2.9 [−6.0, +0.4] | −2.3 [−3.9, −0.7] | +0.15 [−0.96, +1.34] |

**Against B1d** (B1d minus method; positive = method better):

| Method | Error, all | Error, events | Post-contact | End of period | Energy score |
|---|---|---|---|---|---|
| B2-100 (pilot) | +1.0 [−0.3, +2.2] | +2.1 [−0.2, +4.3] | +2.9 [−0.4, +6.0] | +2.3 [+0.8, +4.0] | −0.15 [−1.31, +0.95] |
| B2-500 | +2.4 [+1.2, +3.5] | +4.4 [+2.5, +6.4] | +5.0 [+2.3, +7.9] | +4.0 [+2.5, +5.6] | +1.43 [+0.38, +2.46] |
| B2-100-R2 | **+2.5 [+1.5, +3.6]** | **+4.4 [+2.6, +6.2]** | **+5.2 [+2.9, +7.7]** | +4.1 [+2.7, +5.6] | +1.74 [+0.80, +2.67] |
| OR-100 | +3.1 [+2.0, +4.3] | +5.3 [+3.3, +7.3] | +6.4 [+3.8, +9.3] | +4.9 [+3.4, +6.5] | +1.89 [+0.80, +2.95] |

**Calibration** (coverage of the true hidden position; nominal 0.50 and 0.90; median 90% region area in cm²):

| Method | Gaussian 50% | Gaussian 90% | Gaussian area | Particle HPD 50% | HPD 90% | HPD area | 90% Gaussian / HPD by label: none · bounce · collision |
|---|---|---|---|---|---|---|---|
| B2-100 (pilot) | 0.20 | 0.41 | 40 | 0.11 | 0.32 | 26 | 0.55/0.42 · 0.35/0.30 · 0.31/0.25 |
| B2-300 | 0.29 | 0.53 | 63 | 0.20 | 0.43 | 43 | 0.65/0.54 · 0.45/0.43 · 0.46/0.35 |
| B2-500 | 0.30 | 0.56 | 63 | 0.22 | 0.48 | 46 | 0.68/0.61 · 0.45/0.43 · 0.53/0.39 |
| B2-100-T | 0.33 | 0.54 | 108 | 0.21 | 0.46 | 77 | 0.69/0.59 · 0.44/0.43 · 0.47/0.37 |
| B2-300-T | 0.40 | 0.65 | 142 | 0.28 | 0.57 | 111 | 0.78/0.70 · 0.53/0.50 · 0.60/0.51 |
| B2-100-R0 | 0.12 | 0.26 | 7 | 0.05 | 0.18 | 4 | 0.42/0.30 · 0.24/0.19 · 0.14/0.07 |
| B2-100-R2 | 0.29 | 0.59 | 85 | 0.25 | 0.51 | 61 | 0.66/0.57 · 0.52/0.45 · 0.55/0.47 |
| B2-100-R4 | 0.46 | 0.79 | 190 | 0.35 | 0.71 | 145 | 0.86/0.78 · 0.64/0.57 · 0.78/0.69 |
| OR-100 | 0.24 | 0.49 | 37 | 0.16 | 0.39 | 25 | 0.52/0.42 · 0.53/0.40 · 0.40/0.30 |
| B1d | 0.52 | 0.85 | 254 | – | – | – | 0.90 · 0.71 · 0.83 |

**Filter health and the state when the ball disappears:**

| Method | ESS / N | Resample rate | Unique at resample | Visible-frame error, frames 1–4 / 5–19 / 20+ (cm) | Last visible position (cm) | Last visible velocity (cm/s) |
|---|---|---|---|---|---|---|
| B2-100 (pilot) | 0.54 | 0.38 | 0.43 | 1.66 / **4.69** / 1.90 | 2.61 | 18.5 |
| B2-300 | 0.55 | 0.35 | 0.43 | 1.47 / 3.55 / 1.35 | 2.04 | 18.6 |
| B2-500 | 0.57 | 0.33 | 0.44 | 1.43 / 2.95 / 1.31 | 1.90 | 18.0 |
| B2-100-T | 0.65 | 0.23 | 0.50 | 1.74 / 4.64 / 2.39 | 2.71 | 19.0 |
| B2-100-R0 | **0.87** | 0.05 | 0.36 | 1.73 / 8.51 / **15.10** | 3.61 | 19.0 |
| B2-100-R2 | 0.50 | 0.49 | 0.45 | 1.51 / 2.88 / 1.16 | 1.75 | 18.2 |
| B2-100-R4 | 0.39 | 0.72 | 0.41 | 1.40 / 1.99 / 1.04 | 1.43 | 21.6 |
| OR-100 | 0.59 | 0.29 | 0.44 | 1.62 / 3.35 / 1.22 | 2.01 | 17.4 |
| B1d | – | – | – | 1.20 / 1.00 / 0.83 | 1.06 | 16.6 |

The raw measurement's mean error is σ√(π/2) = 1.25 cm. On visible frames after frame 20, the pilot B2 (1.90 cm) is worse than the raw measurement.

### Diagnosis, cause by cause

| Candidate cause | Test | Finding | Verdict |
|---|---|---|---|
| **Resampling and roughening** | R0, R2, R4 at 100 particles | See "Roughening is the main lever" below | **Main cause** |
| **Particle count** | 100 → 300 → 500 | Monotone gains: −1.4 cm [0.7, 2.2] at 500, coverage 0.41 → 0.56, at 5.7× the runtime. 100 → 300 alone: +0.6 [−0.2, +1.5] | Contributes, through the same mechanism: more particles keep more distinct trajectories alive. The most expensive lever |
| **Weight degeneracy** | ESS, resampling rate, diversity | The mean ESS is between 0.39 and 0.65 of N in every configuration except R0, and none degenerates in the usual sense. ESS is misleading here: the collapsed R0 filter has the *highest* ESS (0.87), because identical copies carry identical weights | Not the problem; *diversity* is. ESS should not be used as a health check for B2 |
| **Parameter uncertainty** | OR-100 (true parameters) | −2.1 cm [1.1, 3.2] overall and −3.4 [1.3, 5.9] after contact: the largest single gain. Coverage stays at 0.49 | Second cause of error. Not a cause of overconfidence |
| **Visible-state initialization** | Visible-frame error by time; last-visible state | Pilot B2 has a transient on frames 5–19 (4.7 cm, against 1.0 for B1d). At the frame before each hidden period, its position is 2.6 cm off against 1.1 for B1d. Velocity error is similar for all methods (17–19 cm/s, from 1 cm noise at 10 Hz). The transient shrinks with true parameters (3.4), roughening ×2 (2.9) and 500 particles (3.0) | Real, but its effect lasts because the impoverished cloud cannot correct it. This is the source of the pre-contact gap to B1d. Roughening ×2 closes it: 4.3 against 4.1 cm |
| **Likelihood scaling** | Exponent 0.25 at 100 and 300 particles | No change in error (−0.2 [−0.9, +0.5]). 90% coverage +0.13, but regions 2.7× larger. Visible-frame error rises, because a flatter likelihood resamples less (23% against 38% of frames), so less jitter is injected | Not a cause of the error. A calibration knob that interacts with roughening |
| **Gaussian approximation** | Particle HPD vs Gaussian regions, paired | HPD regions cover *less* in every configuration: 90% coverage −0.09 [−0.13, −0.07] for the pilot B2, between −0.07 and −0.10 elsewhere. They are about 30% smaller (log area ratio −0.30 to −0.40). The moment-matched Gaussian is wider than the multimodal cloud it summarizes, which partly hides the overconfidence | Not a cause. The particle cloud itself is too narrow |
| **Negative evidence** | No negative evidence; ray-based | Removing it costs 1.2 cm [0.5, 1.8]. Using the labelling test instead of the table-plane map gains +0.4 [−0.1, +1.0], not significant, with 90% coverage −0.03 | Negative evidence helps. The map/label mismatch does not matter |

**Roughening is the main lever:**
- **Off (R0):** the filter collapses. Copies of a resampled particle follow identical trajectories. Error on visible frames grows to 15.1 cm after frame 20, and post-contact error is 5.0 cm worse.
- **Doubled (R2):** every accuracy metric improves, as much as going to 500 particles, at about 1/5 of the runtime.
- **×4 (R4):** coverage reaches 0.79, but the filter occasionally blows up (next paragraph).

**Why R4 loses on average.** R4's median error equals R2's (5.1 against 5.0 cm), and R4 is worse on only 48% of periods. Its mean loss comes from two collision periods, 160 and 100 cm worse than R2. In both, the estimate leaves the 1 m table (|mean| up to 8 m, speeds up to 12.6 m/s). A large position jitter can put two balls in overlap, and MuJoCo's contact solver then ejects them. The same mechanism exists at smaller jitter but fires rarely.

### Trade-offs: accuracy, calibration and cost

- **Accuracy per cost.** Roughening ×2 at 100 particles (5.4 s per episode) matches or beats 500 particles (27.5 s) on every accuracy metric and on the energy score.
- **Calibration.** No stable particle configuration is calibrated. The best 90% coverages are:

  | Configuration | 90% coverage | Median region area | Accuracy |
  |---|---|---|---|
  | B2-300-T | 0.65 | 142 cm² | 8.3 cm |
  | B2-100-R2 | 0.59 | 85 cm² | 7.3 cm |
  | B2-100-R4 | 0.79 | 190 cm² | unstable |
  | B1d | 0.85 | 254 cm², 3× R2's | worse by 2.5 cm |

- **Proper scoring.** On the energy score, which rewards accuracy and calibration jointly, the pilot B2's accuracy advantage over B1d disappears: +1.0 cm on error, but −0.15 [−1.31, +0.95] on energy score. With roughening ×2, B2 beats B1d on both (energy score +1.74 [0.80, 2.67]).
- **What tempering costs.** Tempering buys coverage with area, not with accuracy. At 300 particles it raised 90% coverage from 0.53 to 0.65 for 2.3× the region area.

### Validation on fresh development episodes

**Candidates.** The selection block suggested that roughening and particle count both act on diversity, and that tempering trades area for coverage. Two combined candidates were declared from that before the validation block was generated, and run there once with the pilot B2 and B1d:
- **B2-300-R2:** 300 particles, roughening ×2.
- **B2-300-R2-T:** the same, plus likelihood exponent 0.25.

**Episodes.** The validation block is seeds 9,000,200–9,000,399: 232 hidden periods (76 no event, 13 stopped, 35 bounce, 108 collision). B2-500 was not re-run: it was declared in advance rather than selected, so its selection-block estimate carries no selection bias.

| Method | All | No event | Collision | Pre-contact | Post-contact | End of period | Energy score | 90% coverage Gaussian / HPD | 90% area (cm²) | s / episode |
|---|---|---|---|---|---|---|---|---|---|---|
| B2-100 (pilot) | 9.7 | 7.4 | 11.0 | 5.6 | 12.4 | 12.8 | 8.69 | 0.37 / 0.29 | 38 | 5.1 |
| B2-100-R2 | 8.6 | 7.3 | 9.5 | 5.3 | 10.9 | 11.8 | 7.29 | 0.55 / 0.48 | 80 | 5.6 |
| **B2-300-R2** | **7.5** | **6.8** | **7.9** | **4.4** | **9.0** | **10.0** | **6.06** | 0.68 / 0.61 | 109 | 19.8 |
| B2-300-R2-T | 8.0 | 7.4 | 8.0 | 4.8 | 9.0 | 10.6 | 6.37 | 0.75 / 0.69 | 190 | 14.0 |
| B1d | 10.5 | 7.2 | 12.7 | 4.7 | 14.6 | 15.0 | 7.95 | 0.80 | 216 | <0.01 |

**Paired differences** (positive = method better, cm):

| Comparison | Error, all | Error, events | Post-contact | End of period | Energy score |
|---|---|---|---|---|---|
| B2-100 − B2-100-R2 | +1.2 [+0.5, +1.9] | +1.5 [+0.5, +2.6] | +1.5 [+0.0, +3.0] | +1.0 [+0.1, +2.0] | +1.40 [+0.76, +2.07] |
| B2-100 − B2-300-R2 | **+2.2 [+1.5, +3.1]** | +3.0 [+1.8, +4.4] | +3.4 [+1.7, +5.3] | +2.8 [+1.8, +3.9] | +2.63 [+1.83, +3.53] |
| B2-100 − B2-300-R2-T | +1.8 [+1.0, +2.6] | +2.6 [+1.4, +4.1] | +3.4 [+1.7, +5.4] | +2.2 [+1.1, +3.3] | +2.33 [+1.56, +3.23] |
| B1d − B2-100 | +0.7 [−0.5, +2.1] | +1.2 [−0.8, +3.2] | +2.2 [−0.3, +4.8] | +2.2 [+0.7, +3.8] | −0.74 [−1.97, +0.46] |
| B1d − B2-300-R2 | **+3.0 [+2.1, +3.9]** | **+4.2 [+2.8, +5.7]** | **+5.6 [+3.8, +7.5]** | +5.0 [+3.8, +6.4] | +1.89 [+1.01, +2.78] |
| B1d − B2-300-R2-T | +2.5 [+1.6, +3.6] | +3.8 [+2.3, +5.4] | +5.6 [+3.8, +7.6] | +4.4 [+3.1, +5.7] | +1.59 [+0.71, +2.55] |

**What replicated:**
- **Roughening ×2 helps on fresh episodes.** The gain at 100 particles was +1.2 cm [0.5, 1.9], against +1.6 on the selection block. 90% coverage rose from 0.37 to 0.55.
- **The two diversity levers add up.** 300 particles with roughening ×2 roughly doubles the gain (+2.2 cm). On the selection block, going from 100 to 300 particles at pilot roughening gave only +0.6.
- **B2-300-R2 is the best method on every accuracy metric and on the energy score.**
  - It beats B1d by 3.0 cm overall and by 5.6 cm (38%) after a hidden contact.
  - Before the contact it is now level with B1d: 4.4 against 4.7 cm.
  - On visible frames after frame 20 its error is 0.73 cm, below the raw measurement's 1.25 cm. The pilot B2 was at 3.13 on this block.
- **The pilot B2's mean-error advantage over B1d is not robust:**
  - On mean error it is +0.7 [−0.5, +2.1] on this block, and +1.0 [−0.3, +2.2] on the selection block.
  - On the energy score it is −0.74 [−1.97, +0.46], i.e. B1d tends to be better.
- **Gaussian vs particle regions:** particle-derived regions again cover less than the Gaussian summary, by 0.06 to 0.08 at the 90% level.
- **Tempering costs accuracy:** 0.5 cm worse than B2-300-R2 overall. It buys 90% coverage of 0.75 instead of 0.68 with 1.7× the region area.

### Recommended B2 configuration for Stage 1

**B2 = B2-300-R2:**
- 300 particles;
- roughening ×2 at each resampling: position +N(0, 6 mm), velocity +N(0, 4 cm/s), mass and friction ×exp(N(0, 0.1));
- negative evidence from the table-plane hidden map, as in the pilot;
- likelihood exponent 1;
- everything else as in the pilot.

In code: `baselines.particle_filter(ep, z, n_part=300, rough=2.0)`.

**Why this configuration:**
- It is the most accurate tested configuration on both blocks.
- It has the best energy score.
- It is stable: no estimate left the table.
- It costs 20 s per episode on one core here, about 5.5 core-hours for a 1,000-episode evaluation split.

**Alternatives considered:**
- **B2-500:** similar accuracy to B2-100-R2 at 5× the cost; dominated.
- **B2-100-R2:** a cheaper option at 5.6 s per episode, if cost matters more than 1 cm.
- **B2-300-R2-T:** the best-calibrated stable particle filter. Report it beside B2 if a calibration comparison needs a physics reference with less overconfidence.

The choice of one or both rows is for the protocol freeze. Either way, they should be fixed now and not tuned further.

**Tuning effort.** B2's revision used two levers chosen on 200 development episodes and checked on 200 more. B1d's three parameters were tuned on the pilot. Neither baseline was tuned on confirmatory data. B1d was not retuned here, so its pilot tuning on version-1 data is a limitation of the comparison, not of B2.

### What this means for the spec

- **Physics helps more after hidden events than change 1 suggested.** With a non-impoverished filter, the true simulator with prior-sampled parameters beats B1d by 38% after a hidden contact, not 20–26%.
  - This changes the context for the provisional minimum important improvement of 20% below the best of B1d and B3: 20% is well inside what engineered physics achieves on collision periods.
  - B2 remains a reference with simulator access, not a competitor that model M has to beat.
- **B2 is still overconfident.** Its 90% regions cover 61–68% (HPD / Gaussian), against 80–85% for B1d. Two consequences:
  - Calibration claims against B2 (H2) should be read with that in mind.
  - A proper score (energy score), which rewards accuracy and calibration jointly, is a more informative probabilistic comparison than coverage alone.
- **Report particle diversity, not ESS.** ESS is not a valid health metric for this filter.

These are inputs for the protocol freeze. No hypothesis, frozen metric or confirmatory data was changed here.

### Limitations

- **Sample size.** Each block has 200 episodes and each filter ran once per episode. Monte Carlo variability of the filter itself is not separated from variability between episodes. This matters most for the 100-particle variants.
- **Coarse roughening grid.** Only {0, 1, 2, 4} was tested. Between ×2 and ×4, median error is flat while coverage rises, so a larger jitter that cannot create interpenetration might calibrate better at no cost in accuracy. Not tested.
- **Roughening is an indirect form of process noise.** Its amount depends on how often the filter resamples (5–72% of frames across the configurations tested), so tempering and particle count change it as a side effect.
  - The principled fix is process noise at every step, or a move step after resampling.
  - It would also address the remaining overconfidence. Not implemented: it changes B2's model, not its settings.
- **Contact-blind jitter.** Roughening ignores contacts and can place balls in overlap. This was rare at ×2 (no estimate left the table) and frequent enough at ×4 to cause two blow-ups.
- **Bandwidth dependence.** Particle-derived regions depend on the kernel bandwidth (Scott's rule, 0.5 cm floor). The energy score does not.
- **Coverage weighting.** Coverage is averaged per period, so short periods weigh as much as long ones.
- **Unrecoverable velocity.** Velocity at the frame before each hidden period is 17–24 cm/s off for every method, including B1d. It comes from 1 cm noise at 10 Hz, and limits what any method can recover after a long occlusion.
- **Runtime.** Times come from this 4-core container. The filter loops over MuJoCo states in Python; batched rollouts would cut the cost substantially.
- **Sweep failure.** The first full sweep hung after the container's memory limit killed two pool workers. The fix (`ab9aed5`) changed memory handling only. The reproducibility tests confirm that the pilot's B2 and B1d predictions still reproduce bit for bit.

## GPU check on Capella (before change 2)

A 20-minute smoke job, not an experiment. Details are in [`cluster/README.md`](../cluster/README.md), the numbers in `results/gpu_smoke_4593526.json`.

**What works:**
- torch with CUDA on an H100.
- MuJoCo rendering through EGL on the GPU node, at 64×64 and 96×96, with identical physics.
- A virtualenv that starts in a clean, non-login shell.

**Implications for the B3 rebuild (change 2) and the resolution decision:**

- **GPU compute is negligible.** A 0.8M-parameter recurrent image model trains on 10,000 episodes × 20 epochs in about 1 minute at 64×64 and 1.6 minutes at 96×96. That is a lower bound: synthetic data already on the GPU. 96×96 costs 1.8× the compute of 64×64. Cost is therefore no reason to keep 64×64: the resolution decision can rest on whether the task is memory or perception.
- **The whole dataset fits on the GPU.** 10,000 episodes take 9.8 GB at 64×64 and 22 GB at 96×96, and both together fit in the H100's 95 GB. Training can keep the uint8 frames resident, with no data loader.
- **Generating episodes is the bottleneck.** It takes about 1 s per episode per core: physics and 9-ray visibility take about 0.8 s, rendering about 0.1 s. 10,000 episodes take about 20 minutes on 8 cores, or 5 minutes on 32.
  - Rendering both resolutions from one physics run would avoid simulating twice. The generator renders one resolution per run today.
  - The data belongs in a workspace, not the home directory.
- **Several training seeds per learned model are affordable.** Training variance can be reported, which makes H1 comparisons against a learned baseline more trustworthy. How many seeds to use is a decision for the protocol freeze.

## Generator finalization: version 3 and the collision-course rate

This was step 1 before any training data is generated. It is CPU only, and no model is scored, so nothing is tuned on results.

**Run:** `python check_collide_rate.py 1000 4` on development seeds 9,000,000–9,000,999, about 10 minutes on 4 cores.

**Outputs** (committed):
- `results/collide_rate_summary.json`.
- `results/collide_rate_periods.csv`, one row per hidden period and variant.

### Version 3: collisions are hidden on the labels, not only on the map

Version 2 accepted a collision course using the table-plane hidden map, which disagrees with the 9-ray labels near occluder edges. As a result, 52 of the 297 collide-mode episodes version 2 produced at rate 0.5 (18%) had no collision period that stayed hidden for 5 frames after the contact. In 12 of them, no hidden period contained the contact at all.

**The change.** Version 3 keeps the map as a cheap first filter. It then simulates each passing candidate with ray visibility, and keeps it only if a ball's hidden period contains the contact and lasts at least `MIN_POST_HIDDEN` = 5 frames from the contact frame on. That is the same definition as in `diag_collisions.py`, computed by `gen.hidden_collision_post`.

**Compatibility:**
- Version 3 tries the same candidates in the same order. In 246 of 284 seeds it produces version 2's episode exactly.
- Versions 1 and 2 are unchanged, and tests check both:
  - version 1 against the stored pilot files;
  - version 2 against fingerprints recorded before the change.
- `GEN_VERSION` is now 3. Every existing script passes its version explicitly, so no earlier result changes.

| Collide-mode episodes (rate 0.5, 1,000 seeds) | Version 2 | Version 3 |
|---|---|---|
| Attempts that found a collision course | 297 of 532 | 284 of 532 |
| … with a collision period that stays hidden ≥ 5 frames after contact | 245 (82%) | **284 (100%)** |
| Median hidden frames after contact, longest collision period | 58 | 64 |
| All collision periods: median hidden frames after contact | 8 | 12 |
| All collision periods: share ending within 2 frames of contact | 16% | 13% |
| Generation cost per episode, no rendering, one core | 0.81 s | 1.09 s |

**Short collision periods that remain** (66 at rate 0.5, 38 at rate 0.3):
- **Second balls:** the second ball in a collide-mode episode, briefly hidden at the same contact (63 and 34).
- **Chance collisions:** a collision in a free episode (3 and 4).
- None is an episode's main collision period.

Whether such periods should count as "collision" is a labelling question for the protocol freeze. One option is to require a minimum number of hidden frames after the contact for the label. Here they are left as they are.

### Collision-course rate: 0.3, 0.4 or 0.5

Settings are compared on common random numbers. The first draw of each seed's separate stream decides whether it attempts a collision course, so a lower rate gives the same episodes with some attempts switched off.

| Setting | Hidden periods per 1,000 episodes | No event | Stopped | Bounce | Collision | Event share | Collision share | Cost per episode |
|---|---|---|---|---|---|---|---|---|
| Version 1 (pilot) | 902 | 539 | 111 | 230 | 22 | 0.28 [0.25, 0.31] | 0.02 [0.01, 0.04] | 0.70 s |
| Version 2, rate 0.5 (change 1) | 1,151 | 404 | 83 | 168 | 496 | 0.58 [0.54, 0.61] | 0.43 [0.40, 0.46] | 0.81 s |
| **Version 3, rate 0.3** | 1,039 | 463 | 90 | 190 | 296 | **0.47 [0.43, 0.50]** | **0.28 [0.25, 0.32]** | 0.92 s |
| Version 3, rate 0.4 | 1,088 | 436 | 82 | 176 | 394 | 0.52 [0.49, 0.56] | 0.36 [0.32, 0.40] | 1.00 s |
| Version 3, rate 0.5 | 1,133 | 397 | 77 | 168 | 491 | 0.58 [0.55, 0.61] | 0.43 [0.40, 0.47] | 1.09 s |

Shares have 95% intervals from a bootstrap clustered by episode. Sanity check: version 2 at rate 0.5 on the first 200 seeds reproduces change 1 exactly (222 periods, 80 collisions).

**Recommendation: version 3 at rate 0.3.** The decision is the user's.
- **Targets.** All three rates meet the pilot report's targets (at least 40% events, at least 10% collisions). Rate 0.3 meets them with margin: events 47% [43, 50], collisions 28%.
- **Balance.** Rate 0.3 gives the most balanced mix: no label above 45% of periods, collisions 28%. At 0.5, collisions are 43% of periods and dominate any average over all periods.
- **Labels per test split.** A 1,000-episode split at rate 0.3 contains about 460 no-event, 190 bounce and 300 collision periods. The pilot's power estimate needed about 190 periods for its overall comparison. Using that as a rough per-label guide, every label except "stopped" reaches it at 0.3. At 0.5, bounces fall to about 170.
- **If hidden collisions matter most.** If the protocol freeze makes them the priority, for example for calibration (H2) or decisions (H3), rate 0.4 gives about 390 collision periods per 1,000 episodes while keeping about 175 bounces.
- **Cost.** It is lowest at 0.3: 0.92 s per episode without rendering, about 25 minutes for 10,000 episodes on 8 cores.

**Applying the chosen rate.** `P_COLLIDE` is shared by versions 2 and 3. When the rate is decided, it becomes a per-version setting: version 2 keeps 0.5 so that its results reproduce, and version 3 takes the chosen rate.

**Limitations:**
- 1,000 episodes and one generator seed per episode.
- Label counts depend on the camera and occluder priors, which are unchanged since version 1.
- The 190-period guide comes from the pilot's overall comparison, not a per-label power analysis.
- Model errors were not computed. The rate changes the label mix, not the episodes within a label: at a lower rate, the same episodes simply have fewer collision attempts.
