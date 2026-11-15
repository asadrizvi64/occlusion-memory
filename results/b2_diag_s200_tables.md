# B2 diagnostics tables (200 development episodes, pilot seeds 9000200–9000399, generator version 2)

## Accuracy (cm; mean over hidden periods)

Periods: all 232, none 76, bounce 35, collision 108.

| Method | All | No event | Bounce | Collision | Pre-contact | Post-contact | End of period | Energy score |
|---|---|---|---|---|---|---|---|---|
| B2-100 | 9.7 | 7.4 | 11.1 | 11.0 | 5.6 | 12.4 | 12.8 | 8.69 |
| B2-100-R2 | 8.6 | 7.3 | 9.7 | 9.5 | 5.3 | 10.9 | 11.8 | 7.29 |
| B2-300-R2 | 7.5 | 6.8 | 8.4 | 7.9 | 4.4 | 9.0 | 10.0 | 6.06 |
| B2-300-R2-T | 8.0 | 7.4 | 9.9 | 8.0 | 4.8 | 9.0 | 10.6 | 6.37 |
| B1d | 10.5 | 7.2 | 10.8 | 12.7 | 4.7 | 14.6 | 15.0 | 7.95 |

## Paired differences, B2-100 minus method (cm; positive = method better; 95% CI clustered by episode)

| Method | Error, all | Error, events | Post-contact error | End error, all | Energy score, all |
|---|---|---|---|---|---|
| B2-100-R2 | +1.2 [+0.5, +1.9] | +1.5 [+0.5, +2.6] | +1.5 [+0.0, +3.0] | +1.0 [+0.1, +2.0] | +1.40 [+0.76, +2.07] |
| B2-300-R2 | +2.2 [+1.5, +3.1] | +3.0 [+1.8, +4.4] | +3.4 [+1.7, +5.3] | +2.8 [+1.8, +3.9] | +2.63 [+1.83, +3.53] |
| B2-300-R2-T | +1.8 [+1.0, +2.6] | +2.6 [+1.4, +4.1] | +3.4 [+1.7, +5.4] | +2.2 [+1.1, +3.3] | +2.33 [+1.56, +3.23] |
| B1d | -0.7 [-2.1, +0.6] | -1.2 [-3.1, +0.7] | -2.2 [-4.8, +0.2] | -2.2 [-3.8, -0.6] | +0.74 [-0.50, +1.99] |

## Paired differences, B1d minus method (cm; positive = method better; 95% CI clustered by episode)

| Method | Error, all | Error, events | Post-contact error | End error, all | Energy score, all |
|---|---|---|---|---|---|
| B2-100 | +0.7 [-0.5, +2.1] | +1.2 [-0.8, +3.2] | +2.2 [-0.3, +4.8] | +2.2 [+0.7, +3.8] | -0.74 [-1.97, +0.46] |
| B2-100-R2 | +1.9 [+0.7, +3.1] | +2.7 [+0.9, +4.4] | +3.7 [+1.4, +6.0] | +3.2 [+1.7, +4.7] | +0.66 [-0.43, +1.80] |
| B2-300-R2 | +3.0 [+2.1, +3.9] | +4.2 [+2.8, +5.7] | +5.6 [+3.8, +7.5] | +5.0 [+3.8, +6.4] | +1.89 [+1.01, +2.78] |
| B2-300-R2-T | +2.5 [+1.6, +3.6] | +3.8 [+2.3, +5.4] | +5.6 [+3.8, +7.6] | +4.4 [+3.1, +5.7] | +1.59 [+0.71, +2.55] |

## Before the occlusion matters: paired differences, B2-100 minus method (positive = method better)

Pre-contact: collision periods, hidden frames before the contact. Last visible: the frame before each hidden period.

| Method | Pre-contact error (cm) | Last-visible position error (cm) | Last-visible velocity error (cm/s) |
|---|---|---|---|
| B2-100-R2 | +0.3 [-0.1, +0.8] | +0.42 [+0.23, +0.63] | +0.1 [-0.2, +0.5] |
| B2-300-R2 | +1.2 [+0.3, +2.6] | +0.74 [+0.42, +1.19] | +1.1 [+0.5, +1.8] |
| B2-300-R2-T | +0.8 [-0.0, +2.2] | +0.54 [+0.22, +0.94] | +0.3 [-0.4, +1.0] |
| B1d | +0.9 [-0.2, +2.5] | +1.16 [+0.61, +1.83] | +2.0 [+0.9, +3.3] |

## Coverage of the true hidden position (all periods; nominal 0.50 / 0.90) and median 90% region area (cm²)

Gaussian: chi-square ellipse from the weighted mean and covariance (the pilot's method). HPD: highest-density region of a kernel density estimate on the weighted particles. B1d is Gaussian, so its ellipse is already its exact region.

| Method | Gaussian 50% | Gaussian 90% | Gaussian 90% area | HPD 50% | HPD 90% | HPD 90% area |
|---|---|---|---|---|---|---|
| B2-100 | 0.18 | 0.37 | 38 | 0.08 | 0.29 | 25 |
| B2-100-R2 | 0.29 | 0.55 | 80 | 0.20 | 0.48 | 56 |
| B2-300-R2 | 0.35 | 0.68 | 109 | 0.26 | 0.61 | 82 |
| B2-300-R2-T | 0.49 | 0.75 | 190 | 0.37 | 0.69 | 147 |
| B1d | 0.46 | 0.80 | 216 | – | – | – |

## Region construction: paired, clustered 95% intervals

Area ratio excludes periods whose HPD region is smaller than one grid cell (area 0): none.

| Method | HPD − Gaussian, 90% coverage | HPD − Gaussian, 50% coverage | Gaussian 90% − 0.90 | HPD 90% − 0.90 | log(HPD area / Gaussian area), 90% |
|---|---|---|---|---|---|
| B2-100 | -0.08 [-0.10, -0.05] | -0.09 [-0.12, -0.06] | -0.53 [-0.59, -0.48] | -0.61 [-0.66, -0.55] | -0.39 [-0.43, -0.36] |
| B2-100-R2 | -0.07 [-0.09, -0.05] | -0.09 [-0.13, -0.06] | -0.35 [-0.42, -0.29] | -0.42 [-0.49, -0.36] | -0.32 [-0.34, -0.29] |
| B2-300-R2 | -0.07 [-0.09, -0.05] | -0.09 [-0.13, -0.06] | -0.22 [-0.28, -0.16] | -0.29 [-0.35, -0.23] | -0.24 [-0.27, -0.22] |
| B2-300-R2-T | -0.06 [-0.08, -0.04] | -0.12 [-0.15, -0.09] | -0.15 [-0.20, -0.10] | -0.21 [-0.27, -0.15] | -0.27 [-0.31, -0.23] |
| B1d | – | – | -0.10 [-0.15, -0.06] | – | – |

## 90% coverage by period label (Gaussian / HPD)

| Method | No event | Bounce | Collision |
|---|---|---|---|
| B2-100 | 0.49 / 0.38 | 0.33 / 0.33 | 0.29 / 0.20 |
| B2-100-R2 | 0.60 / 0.54 | 0.53 / 0.48 | 0.48 / 0.39 |
| B2-300-R2 | 0.71 / 0.64 | 0.68 / 0.64 | 0.64 / 0.55 |
| B2-300-R2-T | 0.73 / 0.70 | 0.73 / 0.65 | 0.76 / 0.68 |
| B1d | 0.86 | 0.78 | 0.74 |

## Filter health and cost

ESS: effective sample size / N before resampling, mean over frames. Resample rate: share of frames that resample. Unique: share of distinct particles a resampling keeps. Runtime: one core, filter only.

| Method | ESS / N | Resample rate | Unique at resample | Visible-frame error (cm) | Last-visible pos (cm) | Last-visible vel (cm/s) | s / episode |
|---|---|---|---|---|---|---|---|
| B2-100 | 0.51 | 0.42 | 0.39 | 3.62 | 2.29 | 24.5 | 5.1 |
| B2-100-R2 | 0.47 | 0.53 | 0.44 | 1.83 | 1.87 | 24.3 | 5.6 |
| B2-300-R2 | 0.49 | 0.52 | 0.46 | 1.10 | 1.55 | 23.3 | 19.8 |
| B2-300-R2-T | 0.63 | 0.24 | 0.49 | 1.33 | 1.75 | 24.2 | 14.0 |
| B1d | – | – | – | 0.88 | 1.13 | 22.4 | 0.0 |

## Error on visible frames by time since the episode start (cm)

Frames where the ball is visible and not inside a hidden period; mean per episode, then over episodes. For reference, the raw measurement's mean error is σ√(π/2) = 1.25 cm.

| Method | Frames 1–4 | Frames 5–19 | Frames 20+ | All |
|---|---|---|---|---|
| B2-100 | 1.83 | 6.55 | 3.13 | 3.62 |
| B2-100-R2 | 1.74 | 4.19 | 1.29 | 1.83 |
| B2-300-R2 | 1.58 | 2.58 | 0.73 | 1.10 |
| B2-300-R2-T | 1.85 | 3.02 | 0.89 | 1.33 |
| B1d | 1.20 | 1.02 | 0.81 | 0.88 |
