"""Static figure for the pilot report: hidden-position error by what happened while hidden."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/pilot_summary.json"))["error_cm"]
rows = [("all", "all"), ("no event", "label:none"), ("stopped", "label:stopped"), ("bounce", "label:bounce"), ("collision", "label:collision")]
methods = [("B2", "B2 physics filter", "#2a78d6", 7, "o"), ("B1d", "B1d damped Kalman", "#eb6834", 7, "o"),
           ("OR", "oracle parameters", "none", 7, "o"), ("B0", "B0 last seen", "#9a9893", 5, "o"), ("B3", "B3 GRU (pilot)", "#cfcdc8", 5, "o")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c7c2",
                     "xtick.color": "#52514e", "ytick.color": "#0b0b0b", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200)
for k, (lab, key) in enumerate(rows):
    y = len(rows) - 1 - k
    ax.axhline(y, color="#e4e3df", lw=0.7, ls=(0, (2, 3)), zorder=0)
    for j, (m, name, col, ms, mk) in enumerate(methods):
        v = S[key][m][0]
        ax.plot(v, y + (j - 2) * 0.09, mk, ms=ms, color=col if col != "none" else "#fcfcfb", mec="#7a7873" if col == "none" else col,
                mew=1.3 if col == "none" else 0.5, zorder=3, label=name if k == 0 else None)
        if key == "label:bounce" and m in ("B2", "B1d"):
            ax.annotate(f"{v:.1f}", (v, y + (j - 2) * 0.09), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
ax.set_yticks(range(len(rows)), [f"{lab} (n = {S[key]['n']})" for lab, key in rows][::-1])
ax.set_xlim(0, 25)
ax.set_xlabel("mean hidden-position error per hidden period (cm)", color="#52514e")
ax.grid(axis="x", color="#e4e3df", lw=0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=5, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
fig.suptitle("Physics pays off only when something happens while the ball is hidden", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
fig.text(0.01, 0.01, "B1 (undamped Kalman) is off this scale on stopped, bounce and collision periods: 51 to 86 cm. Collision row: 16 periods only.",
         fontsize=7, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig("figures/pilot_hidden_error.png")
print("saved")


def episode_strip(path="data/pilot_00000.pkl", n_eps=6, every=8):
    """Sample frames from rendered pilot episodes (needs `python make_pilot.py 0 200 1`).
    Frames in which a ball is hidden get an orange outline."""
    import pickle

    import numpy as np
    eps = [e for e in pickle.load(open(path, "rb")) if "rgb" in e["sim"]]
    hid_ev = [e for e in eps if ((~e["sim"]["visible"][1:]) & (e["sim"]["event"][1:] > 0)).any()]
    pick = (hid_ev + eps)[:n_eps]
    frames = list(range(0, 80, every))
    fig, axs = plt.subplots(len(pick), len(frames), figsize=(len(frames) * 0.95, len(pick) * 1.02), dpi=200)
    for r, e in enumerate(pick):
        for c, t in enumerate(frames):
            ax = axs[r, c]
            ax.imshow(e["sim"]["rgb"][t], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            hidden = not e["sim"]["visible"][t].all()
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("#eb6834" if hidden else "#fcfcfb")
                sp.set_linewidth(2.0 if hidden else 0.5)
            if r == 0:
                ax.set_title(f"{t / 10:.1f} s", fontsize=7, color="#52514e")
        axs[r, 0].set_ylabel(f"seed {e['seed']}", fontsize=6, color="#52514e")
    fig.suptitle("Pilot episodes as the camera sees them (64×64); orange outline = a ball is hidden", x=0.01, ha="left",
                 fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.2, h_pad=0.2)
    fig.savefig("figures/pilot_episodes.png")
    print("saved episodes")


if __name__ == "__main__":
    import os
    if os.path.exists("data/pilot_00000.pkl"):
        episode_strip()
