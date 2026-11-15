# Stage 1 preparation log

The Stage 0 pilot report lists three changes to make before any confirmatory run. This file records each one as it lands. Numbers come from the pilot seed range (9,000,000 onward), which is reserved for development and never used for confirmatory splits.

| Change | Status |
|---|---|
| 1. Make events while hidden common | **Done: generator version 2** (below) |
| 2. Rebuild the learned baseline (B3) | Not started |
| 3. Re-run B2 with more particles or tempering | Not started |
| Open decision: image resolution | Waiting on a decision |

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
- **B2 is worse than B1d before the contact**, 6.6 against 4.1 cm, as on the pilot's no-event periods. That points to the particle approximation (100 particles, roughening, prior-sampled parameters), not the motion model. Change 3 tests this.
- **Not implemented: placing blocks inside hidden regions.** The bounce share stayed at about 17%, but the event target was met without it, so the generator keeps one lever.

### Limitations of this check

- **Small sample:** 200 episodes and one seed. Intervals on the per-label differences are wide.
- **Pilot-tuned B1d:** B1d keeps its pilot parameters. Retuning it on version-2 data could change its collision errors.
- **Visibility mismatch:** acceptance uses the table-plane hidden map, labels use ray-based visibility. 12% of collision periods still end within 2 frames of contact.
