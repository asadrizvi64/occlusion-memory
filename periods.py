"""Hidden periods with labels, as in the pilot (pilot_analyze.intervals), plus the contact frame and how long the
ball stays hidden after it. Collision periods are split by that duration:

  collision_short  hidden for at most SHORT = 2 frames from the first ball-ball contact on (inclusive). Little of the
                   collision's outcome has to be remembered; reported separately, never pooled silently.
  collision_long   hidden for more than SHORT frames from the contact on.

The pilot label ("collision") is kept in `label`; the split is in `label_fine`. Other labels are unchanged.
"""
import numpy as np

import gen
import pilot_analyze as PA

SHORT = 2
FINE = ("none", "stopped", "bounce", "collision_long", "collision_short")


def hidden_periods(ep, short=SHORT):
    out = []
    for iv in PA.intervals(ep):
        ev = ep["sim"]["event"][iv["a"]:iv["b"] + 1, iv["ball"]]
        hit = np.nonzero(ev == gen.EV["collision"])[0]
        c = iv["a"] + int(hit[0]) if len(hit) else None
        post = iv["b"] - c + 1 if c is not None else None
        fine = iv["label"] if iv["label"] != "collision" else ("collision_short" if post <= short else "collision_long")
        out.append(dict(iv, contact=c, post=post, label_fine=fine))
    return out
