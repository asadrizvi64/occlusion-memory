"""Tables for the change 3 write-up, from the committed outputs of b2_diagnostics.py.

    python b2_report.py            reads results/b2_diag_summary.json and results/b2_diag_intervals.csv
                                   writes results/b2_diag_tables.md

Adds what the sweep summary does not compute: paired, episode-clustered intervals for
(1) particle-derived (HPD) minus Gaussian-summary coverage within each filter, (2) each filter's coverage
gap to the nominal level, and (3) differences against B2-100 on pre-contact error and visible-frame error.
Coverage here is the per-period mean over hidden frames, averaged over periods (the period is the unit,
as for error).
"""
import json

import numpy as np
import pandas as pd

B_BOOT = 2000
SUMMARY, CSV, OUT = "results/b2_diag_summary.json", "results/b2_diag_intervals.csv", "results/b2_diag_tables.md"


def boot(x, cl, rng):
    """Mean of x with a 95% percentile interval, resampling episodes (clusters) with replacement."""
    x, cl = np.asarray(x, float), np.asarray(cl)
    u = np.unique(cl)
    idx = {k: np.nonzero(cl == k)[0] for k in u}
    bs = [x[np.concatenate([idx[k] for k in rng.choice(u, len(u))])].mean() for _ in range(B_BOOT)]
    return float(x.mean()), *np.percentile(bs, [2.5, 97.5])


def ci(t, d=1, sign=False):
    m, lo, hi = t[:3]
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    S = json.load(open(SUMMARY))
    df = pd.read_csv(CSV)
    rng = np.random.default_rng(11)
    M = S["by_method"]
    methods = list(M)
    pfs = [m for m in methods if m != "B1d"]
    key = ["seed", "ball", "a"]
    P = S["paired"]
    md = [f"# B2 diagnostics tables ({S['n_episodes']} development episodes, generator version 2)\n"]

    # ---- accuracy
    rows = []
    for m in methods:
        d = M[m]
        rows.append([m, f"{d['all']['err']:.1f}", f"{d['none']['err']:.1f}", f"{d['bounce']['err']:.1f}", f"{d['collision']['err']:.1f}",
                     f"{d['collision']['pre_contact_err']:.1f}", f"{d['collision']['post_contact_err']:.1f}",
                     f"{d['all']['end_err']:.1f}", f"{d['all']['es']:.2f}"])
    n = M["B1d"]
    md += ["## Accuracy (cm; mean over hidden periods)\n",
           f"Periods: all {n['all']['n']}, none {n['none']['n']}, bounce {n['bounce']['n']}, collision {n['collision']['n']}.\n",
           table(["Method", "All", "No event", "Bounce", "Collision", "Pre-contact", "Post-contact", "End of period", "Energy score"], rows), ""]

    # ---- paired vs B2-100 and vs B1d (positive = the method is better than the reference)
    for ref in ("B2-100", "B1d"):
        rows = []
        for m in methods:
            if m == ref:
                continue
            rows.append([m, ci(P[f"{ref}-{m}:all:err"], sign=True), ci(P[f"{ref}-{m}:event:err"], sign=True),
                         ci(P[f"{ref}-{m}:collision:post_contact_err"], sign=True), ci(P[f"{ref}-{m}:all:end_err"], sign=True),
                         ci(P[f"{ref}-{m}:all:es"], 2, sign=True)])
        md += [f"## Paired differences, {ref} minus method (cm; positive = method better; 95% CI clustered by episode)\n",
               table(["Method", "Error, all", "Error, events", "Post-contact error", "End error, all", "Energy score, all"], rows), ""]

    # ---- pre-contact error and visible-frame error vs B2-100 (from the period rows)
    ref = df[df.method == "B2-100"].set_index(key)
    rows = []
    for m in methods:
        if m == "B2-100":
            continue
        b = df[df.method == m].set_index(key)
        x = (ref.pre_contact_err - b.pre_contact_err).dropna()
        y = (ref.last_visible_pos_err - b.last_visible_pos_err).dropna()
        v = (ref.last_visible_vel_err - b.last_visible_vel_err).dropna()
        rows.append([m, ci(boot(x.values, x.index.get_level_values(0), rng), sign=True),
                     ci(boot(y.values, y.index.get_level_values(0), rng), 2, sign=True),
                     ci(boot(v.values, v.index.get_level_values(0), rng), 1, sign=True)])
    md += ["## Before the occlusion matters: paired differences, B2-100 minus method (positive = method better)\n",
           "Pre-contact: collision periods, hidden frames before the contact. Last visible: the frame before each hidden period.\n",
           table(["Method", "Pre-contact error (cm)", "Last-visible position error (cm)", "Last-visible velocity error (cm/s)"], rows), ""]

    # ---- calibration
    rows = []
    for m in methods:
        d = M[m]["all"]
        rows.append([m, f"{d['gcov50']:.2f}", f"{d['gcov90']:.2f}", f"{d['garea90_median']:.0f}",
                     "–" if m == "B1d" else f"{d['hcov50']:.2f}", "–" if m == "B1d" else f"{d['hcov90']:.2f}",
                     "–" if m == "B1d" else f"{d['harea90_median']:.0f}"])
    md += ["## Coverage of the true hidden position (all periods; nominal 0.50 / 0.90) and median 90% region area (cm²)\n",
           "Gaussian: chi-square ellipse from the weighted mean and covariance (the pilot's method). "
           "HPD: highest-density region of a kernel density estimate on the weighted particles. "
           "B1d is Gaussian, so its ellipse is already its exact region.\n",
           table(["Method", "Gaussian 50%", "Gaussian 90%", "Gaussian 90% area", "HPD 50%", "HPD 90%", "HPD 90% area"], rows), ""]

    rows = []
    for m in pfs:
        G = df[df.method == m].dropna(subset=["hcov90", "gcov90"])
        cl = G.seed.values
        rows.append([m, ci(boot(G.hcov90 - G.gcov90, cl, rng), 2, sign=True), ci(boot(G.hcov50 - G.gcov50, cl, rng), 2, sign=True),
                     ci(boot(G.gcov90 - 0.9, cl, rng), 2, sign=True), ci(boot(G.hcov90 - 0.9, cl, rng), 2, sign=True),
                     ci(boot(np.log(G.harea90 / G.garea90), cl, rng), 2, sign=True)])
    G = df[df.method == "B1d"]
    rows.append(["B1d", "–", "–", ci(boot(G.gcov90 - 0.9, G.seed.values, rng), 2, sign=True), "–", "–"])
    md += ["## Region construction: paired, clustered 95% intervals\n",
           table(["Method", "HPD − Gaussian, 90% coverage", "HPD − Gaussian, 50% coverage", "Gaussian 90% − 0.90", "HPD 90% − 0.90",
                  "log(HPD area / Gaussian area), 90%"], rows), ""]

    rows = []
    for m in methods:
        r = [m]
        for g in ("none", "bounce", "collision"):
            d = M[m][g]
            r.append(f"{d['gcov90']:.2f}" + ("" if m == "B1d" else f" / {d['hcov90']:.2f}"))
        rows.append(r)
    md += ["## 90% coverage by period label (Gaussian / HPD)\n", table(["Method", "No event", "Bounce", "Collision"], rows), ""]

    # ---- filter health and cost
    rows = []
    for m in methods:
        d = M[m]
        f = d.get("filter", {})
        rows.append([m, f"{f['mean_ess_frac']:.2f}" if f else "–", f"{f['resample_rate']:.2f}" if f else "–",
                     f"{f['unique_frac_at_resample']:.2f}" if f else "–", f"{d['visible_frame_err']:.2f}",
                     f"{d['last_visible']['pos_err']:.2f}", f"{d['last_visible']['vel_err']:.1f}", f"{d['runtime_s_per_episode']:.1f}"])
    md += ["## Filter health and cost\n",
           "ESS: effective sample size / N before resampling, mean over frames. Resample rate: share of frames that resample. "
           "Unique: share of distinct particles a resampling keeps. Runtime: one core, filter only.\n",
           table(["Method", "ESS / N", "Resample rate", "Unique at resample", "Visible-frame error (cm)", "Last-visible pos (cm)",
                  "Last-visible vel (cm/s)", "s / episode"], rows), ""]
    open(OUT, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
