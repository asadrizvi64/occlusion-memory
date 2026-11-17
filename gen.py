"""Occlusion benchmark generator (spec: Learning Persistent Physical States Under Occlusion).

One MuJoCo arena per episode:
  - 1 m x 1 m walled tabletop; 1-2 balls (free joints: they slide, then roll);
  - 1-2 occluders: "screen" (floats above the table, hides but never touches) or
    "block" (on the table, hides and collides);
  - one 64x64 RGB camera whose pose is sampled per episode around a nominal oblique view;
  - 8 s at 10 Hz (80 frames), physics at 2 ms.
Ball 0 is launched toward an occluder so occlusions are common; ball 1 (if any) moves
randomly, which creates collision opportunities, sometimes behind an occluder.

Generator versions (ep["gen_version"]):
  1  Stage 0 pilot generator. Kept bit-identical so the pilot regenerates from its seeds.
  2  Adds a collision-course mode (ep["mode"] == "collide") in P_COLLIDE[2] = 50% of episodes: ball 1
     is launched from a visible start so that it meets ball 0 while ball 0 is inside the hidden
     region. The meeting is timed by bisection on ball 1's launch speed and accepted only if a
     physics-only rollout puts the first ball-ball contact in a hidden cell and keeps a ball that
     was hidden at contact hidden for at least 0.5 s afterwards. Everything else
     (scene, camera, ball 0) uses the same random draws as version 1, so the two versions
     differ only in ball 1 of collide-mode episodes. If no collision course is found the
     episode stays as in version 1 (ep["mode"] == "free", ep["collide_failed"] = True).
  3  Version 2, with the acceptance checked against the labels themselves. A candidate that passes the
     table-plane check is simulated with the 9-ray visibility used for labelling, and kept only if some
     ball has a hidden period containing a ball-ball contact and stays hidden for at least
     MIN_POST_HIDDEN frames from the contact frame on (the definition in diag_collisions.py). The
     table-plane map is a cheap prefilter and disagrees with the rays near occluder edges, so version 2
     still produced collision periods that end within 1-2 frames of the contact. Version 3 tries the
     same candidates in the same order, so it equals version 2 whenever version 2's choice passes.
     It attempts a collision course in P_COLLIDE[3] = 30% of episodes (a subset of version 2's 50%).

Ground truth kept for supervision/evaluation only: positions, velocities, per-frame
visible pixel count (segmentation render), contact events per frame, sampled parameters,
and a hidden-region mask of the table plane (ray casts from the camera).

Seed ranges are disjoint by split; the pilot uses its own range and is never reused.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import mujoco
import numpy as np

SEED_RANGES = {
    "pilot": (9_000_000, 9_100_000),  # Stage 0 only; never used for confirmatory work
    "train": (0, 100_000), "val": (100_000, 110_000),
    "test_id": (200_000, 210_000), "test_long": (210_000, 220_000), "test_shift": (220_000, 230_000),
    "test_objects": (230_000, 240_000), "test_identity": (240_000, 250_000), "test_viewpoint": (250_000, 260_000),
}
FPS, DT = 10, 0.002
SUB = int(round(1 / FPS / DT))
T_SEC = 8.0
N_FRAMES = int(T_SEC * FPS)
IMG = 64
HALF = 0.5  # arena half-size (m)
GRID = 0.02  # hidden-mask resolution (m)
VIS_PX = 2  # a ball counts as visible with at least this many pixels
EV = {"none": 0, "bounce": 1, "collision": 2}
GEN_VERSION = 3
P_COLLIDE = {2: 0.5, 3: 0.3}  # share of episodes that attempt a hidden collision course, by version (3: chosen
#                               after check_collide_rate.py, see docs/STAGE1_PREP.md; 2 stays 0.5 so it reproduces)
MIN_POST_HIDDEN = 5  # versions >= 2: frames a ball hidden at contact must stay hidden after it (0 = no requirement)
COLORS = [(0.85, 0.2, 0.15), (0.15, 0.35, 0.85), (0.2, 0.7, 0.3)]


def look_at_axes(pos, target):
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y])


def inside_block(p, o, margin):
    if o["kind"] != "block":
        return False
    c, s = np.cos(-o["yaw"]), np.sin(-o["yaw"])
    dx, dy = p - o["center"]
    lx, ly = c * dx - s * dy, s * dx + c * dy
    return abs(lx) <= o["half"][0] + margin and abs(ly) <= o["half"][1] + margin


def cam_pose(elev, azim, dist):
    e, a = np.deg2rad(elev), np.deg2rad(azim - 90.0)
    cpos = dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return cpos, look_at_axes(cpos, np.array([0.0, 0.0, 0.0]))


def fit_dist(elev, azim, fovy=50.0, margin_px=2.0):
    """Smallest camera distance at which the whole arena (walls included) stays inside the 64x64 frame."""
    corners = [np.array([sx * (HALF + 0.04), sy * (HALF + 0.04), z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.06)]
    f = (IMG / 2) / np.tan(np.deg2rad(fovy) / 2)
    for dist in np.arange(0.9, 3.0, 0.01):
        cpos, xy = cam_pose(elev, azim, dist)
        x, y = xy[:3], xy[3:]
        fwd = np.cross(y, x)  # viewing direction (camera -z)
        ok = True
        for p in corners:
            v = p - cpos
            depth = v @ fwd
            u, w = IMG / 2 + f * (v @ x) / depth, IMG / 2 - f * (v @ y) / depth
            ok &= margin_px <= u <= IMG - margin_px and margin_px <= w <= IMG - margin_px
        if ok:
            return float(dist)
    raise RuntimeError("camera fit failed")


def sample_episode(seed, n_obj=None, ranges=None, version=GEN_VERSION, p_collide=None):
    """Occluders and camera first; then the hidden region is found by ray casting, and ball 0 is
    launched toward a random hidden cell, so most episodes contain an occlusion. Version 2 may then
    replace ball 1 with one on a collision course (see the module docstring). p_collide overrides
    P_COLLIDE; the attempt is decided by the first draw of the separate stream, so for a given seed the
    collide-mode episode is the same at every p_collide that selects it."""
    r = np.random.default_rng(seed)
    R = dict(mass=(0.1, 1.0), mu_slide=(0.05, 0.4), mu_roll=(0.0002, 0.002), speed=(0.3, 1.5),
             radius=(0.02, 0.04), elev=(50.0, 70.0), azim=(-15.0, 15.0))
    R.update(ranges or {})
    n_obj = n_obj or int(r.integers(1, 3))
    occ = []
    for k in range(int(r.integers(1, 3))):
        occ.append(dict(kind="screen" if r.random() < 0.5 else "block",
                        center=r.uniform(-0.2, 0.2, 2), yaw=r.uniform(0, np.pi),
                        half=(r.uniform(0.12, 0.25), r.uniform(0.06, 0.14)), height=r.uniform(0.1, 0.15),
                        block_h=r.uniform(0.2, 0.3)))
    elev, azim = r.uniform(*R["elev"]), r.uniform(*R["azim"])
    cam = dict(elev=elev, azim=azim, dist=fit_dist(elev, azim))
    ep = dict(seed=seed, occ=occ, balls=[], cam=cam)
    m = mujoco.MjModel.from_xml_string(build_xml(ep))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mask = hidden_mask(m, d)
    xs = np.arange(-HALF + GRID / 2, HALF, GRID)
    from scipy.ndimage import distance_transform_edt
    depth = distance_transform_edt(mask)  # cells deep inside the hidden region, not on its edge
    ok = [(i, j) for i, j in zip(*np.nonzero(mask)) if abs(xs[i]) < HALF - 0.08 and abs(xs[j]) < HALF - 0.08
          and not any(inside_block(np.array([xs[i], xs[j]]), o, 0.05) for o in occ)]
    if ok:
        dmax = max(depth[i, j] for i, j in ok)
        ok = [(i, j) for i, j in ok if depth[i, j] >= 0.6 * dmax]
    cells = [(xs[i], xs[j]) for i, j in ok]
    tgt = np.array(cells[r.integers(len(cells))]) if cells else occ[0]["center"]
    balls = []
    for i in range(n_obj):
        rad = r.uniform(*R["radius"])
        for _ in range(1000):  # rejection: start clear of blocks, walls, hidden region and other balls
            if i == 0:
                ang = r.uniform(0, 2 * np.pi)
                p0 = tgt + r.uniform(0.25, 0.45) * np.array([np.cos(ang), np.sin(ang)])
            else:
                p0 = r.uniform(-0.35, 0.35, 2)
            gi, gj = np.clip(((p0 + HALF) / GRID).astype(int), 0, len(xs) - 1)
            if (np.abs(p0) <= HALF - rad - 0.02).all() and not any(inside_block(p0, o, rad + 0.02) for o in occ) \
                    and not mask[gi, gj] and all(np.linalg.norm(p0 - b["p0"]) > rad + b["radius"] + 0.02 for b in balls):
                break
        else:
            raise RuntimeError(f"no valid start for ball {i} in seed {seed}")
        if i == 0:
            dvec = (tgt - p0) / np.linalg.norm(tgt - p0)
            th = np.deg2rad(r.uniform(-15, 15))
            dvec = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ dvec
            v0 = r.uniform(*R["speed"]) * dvec
        else:
            a = r.uniform(0, 2 * np.pi)
            v0 = r.uniform(0.0, 0.8) * np.array([np.cos(a), np.sin(a)])
        balls.append(dict(radius=rad, p0=p0, v0=v0, mass=r.uniform(*R["mass"]), mu_slide=r.uniform(*R["mu_slide"]),
                          mu_roll=r.uniform(*R["mu_roll"]), bouncy=bool(r.random() < 0.5)))
    ep["balls"] = balls
    ep["target"] = tgt
    ep["gen_version"], ep["mode"] = version, "free"
    if version >= 2:
        r2 = np.random.default_rng([seed, 2])  # separate stream: version-1 draws above stay identical
        if r2.random() < (P_COLLIDE[version] if p_collide is None else p_collide):
            found = collision_course(ep, mask, r2, R, verify=version >= 3)
            if found is None:
                ep["collide_failed"] = True
            else:
                ep["balls"] = [balls[0], found[0]] + balls[2:]
                ep["mode"], ep["collide_plan"] = "collide", found[1]
    return ep


def in_mask(p, mask):
    i, j = np.clip(((np.asarray(p) + HALF) / GRID).astype(int), 0, mask.shape[0] - 1)
    return bool(mask[i, j])


def rollout(ep):
    """Physics only (no rendering or visibility): ball centres (x, y) per frame, and the first
    ball-ball contact between balls 0 and 1 as (time s, centre 0, centre 1), or None."""
    m = mujoco.MjModel.from_xml_string(build_xml(ep))
    d = mujoco.MjData(m)
    n = len(ep["balls"])
    bb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") for i in range(n)]
    for i, b in enumerate(ep["balls"]):
        d.qvel[m.body_dofadr[bb[i]]:m.body_dofadr[bb[i]] + 2] = b["v0"]
    mujoco.mj_forward(m, d)
    pos = np.zeros((N_FRAMES, n, 2))
    pos[0] = d.xpos[bb, :2]
    first = None
    rsum = ep["balls"][0]["radius"] + ep["balls"][1]["radius"] + 1e-3 if n > 1 else None
    for t in range(1, N_FRAMES):
        if first is None and n > 1:
            for k in range(SUB):
                mujoco.mj_step(m, d)
                if np.linalg.norm(d.xpos[bb[0], :2] - d.xpos[bb[1], :2]) < rsum:
                    first = (((t - 1) * SUB + k + 1) * DT, d.xpos[bb[0], :2].copy(), d.xpos[bb[1], :2].copy())
                    mujoco.mj_step(m, d, nstep=SUB - k - 1)
                    break
        else:
            mujoco.mj_step(m, d, nstep=SUB)
        pos[t] = d.xpos[bb, :2]
    return pos, first


def _travel(m, d, dof, body, p_start, direction, speed, n_steps):
    mujoco.mj_resetData(m, d)
    d.qvel[dof:dof + 2] = speed * direction
    mujoco.mj_step(m, d, nstep=n_steps)
    return float((d.xpos[body, :2] - p_start) @ direction)


def collision_course(ep, mask, r2, R, max_tries=40, speed_range=(0.2, 1.8), min_post=None, verify=False):
    """Ball 1 launched at t = 0 from a visible start so that it reaches ball 0's (solo) position at a
    frame t* when ball 0 is in a hidden cell. The aim point is offset sideways by up to 0.6 of the
    radius sum, so impacts range from head-on to glancing. Accepted only if the first contact is in
    a hidden cell and a ball hidden at contact stays in hidden cells for at least `min_post` frames
    (default MIN_POST_HIDDEN) afterwards, or until the episode ends, so the collision's outcome must
    be remembered rather than seen. With verify=True (version 3) a candidate that passes is also
    simulated with ray visibility and must pass the same test on the labels (hidden_collision_post).
    Returns (ball dict, plan dict) or None."""
    min_post = MIN_POST_HIDDEN if min_post is None else min_post
    b0, occ = ep["balls"][0], ep["occ"]
    traj0, _ = rollout(dict(ep, balls=[b0]))
    cand = [t for t in range(5, N_FRAMES - 5) if in_mask(traj0[t, 0], mask)]  # meet between 0.5 s and 7.5 s
    if not cand:
        return None
    b1 = dict(radius=r2.uniform(*R["radius"]), mass=r2.uniform(*R["mass"]), mu_slide=r2.uniform(*R["mu_slide"]),
              mu_roll=r2.uniform(*R["mu_roll"]), bouncy=bool(r2.random() < 0.5))
    others = [b for k, b in enumerate(ep["balls"]) if k != 1]
    for _ in range(max_tries):
        t_star = cand[r2.integers(len(cand))]
        ang, dist = r2.uniform(0, 2 * np.pi), r2.uniform(0.25, 0.7)
        u = np.array([np.cos(ang), np.sin(ang)])  # from the aim point back toward the start
        aim = traj0[t_star, 0] + r2.uniform(-0.6, 0.6) * (b0["radius"] + b1["radius"]) * np.array([-u[1], u[0]])
        p1 = aim + dist * u
        path = [p1 + s * (aim - p1) for s in np.linspace(0, 1, int(dist / 0.01) + 1)]
        if not (np.abs(p1) <= HALF - b1["radius"] - 0.02).all() or in_mask(p1, mask) \
                or any(inside_block(p, o, b1["radius"] + 0.01) for p in path for o in occ) \
                or any(np.linalg.norm(p1 - b["p0"]) < b["radius"] + b1["radius"] + 0.02 for b in others):
            continue
        solo = dict(ep, balls=[dict(b1, p0=p1, v0=np.zeros(2))])
        m = mujoco.MjModel.from_xml_string(build_xml(solo))
        d = mujoco.MjData(m)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball0")
        dof, n_steps = m.body_dofadr[body], t_star * SUB
        lo, hi = speed_range
        if _travel(m, d, dof, body, p1, -u, hi, n_steps) < dist or _travel(m, d, dof, body, p1, -u, lo, n_steps) > dist:
            continue
        for _ in range(14):  # travelled distance at t* is monotone in launch speed
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _travel(m, d, dof, body, p1, -u, mid, n_steps) < dist else (lo, mid)
        ball1 = dict(b1, p0=p1, v0=0.5 * (lo + hi) * -u)
        trial = dict(ep, balls=[b0, ball1] + ep["balls"][2:])
        pos, first = rollout(trial)
        if first is None:
            continue
        fc, post = int(np.ceil(first[0] * FPS)), 0
        for bi, c in ((0, first[1]), (1, first[2])):  # a ball hidden at contact must stay hidden afterwards
            if in_mask(c, mask):
                run = next((k for k, t in enumerate(range(fc, N_FRAMES)) if not in_mask(pos[t, bi], mask)), N_FRAMES - fc)
                post = max(post, run)
        if post >= min_post:
            plan = dict(t_meet=t_star / FPS, t_contact=first[0], speed=0.5 * (lo + hi), post_frames=post)
            if verify:
                plan["post_frames_ray"] = hidden_collision_post(simulate(trial, render=False))
                if plan["post_frames_ray"] < min_post:
                    continue
            return ball1, plan
    return None


def hidden_collision_post(sim):
    """Hidden frames after a ball-ball contact, from the labels: over balls and their hidden periods
    (ray-based, frame 0 never hidden) that contain a contact, the largest count of frames from the
    first contact frame to the period's end, inclusive. 0 if no hidden period contains a contact."""
    hid = ~sim["visible"]
    hid[0] = False
    best = 0
    for i in range(hid.shape[1]):
        t = 1
        while t < hid.shape[0]:
            if not hid[t, i]:
                t += 1
                continue
            a = t
            while t < hid.shape[0] and hid[t, i]:
                t += 1
            hit = np.nonzero(sim["event"][a:t, i] == EV["collision"])[0]
            if len(hit):
                best = max(best, t - (a + int(hit[0])))
    return best


def build_xml(ep, identical=False):
    cpos, xy = cam_pose(ep["cam"]["elev"], ep["cam"]["azim"], ep["cam"]["dist"])
    walls = "".join(f'<geom name="wall{k}" type="box" pos="{px} {py} 0.03" size="{sx} {sy} 0.03" rgba="0.45 0.42 0.38 1" group="0"/>'
                    for k, (px, py, sx, sy) in enumerate([(HALF + 0.02, 0, 0.02, HALF + 0.04), (-HALF - 0.02, 0, 0.02, HALF + 0.04),
                                                          (0, HALF + 0.02, HALF, 0.02), (0, -HALF - 0.02, HALF, 0.02)]))
    occ = ""
    for k, o in enumerate(ep["occ"]):
        q = f'{np.cos(o["yaw"] / 2)} 0 0 {np.sin(o["yaw"] / 2)}'
        if o["kind"] == "screen":
            occ += (f'<geom name="occ{k}" type="box" pos="{o["center"][0]} {o["center"][1]} {o["height"]}" quat="{q}" '
                    f'size="{o["half"][0]} {o["half"][1]} 0.004" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0" group="0"/>')
        else:
            bh = o.get("block_h", 0.12) / 2
            occ += (f'<geom name="occ{k}" type="box" pos="{o["center"][0]} {o["center"][1]} {bh}" quat="{q}" '
                    f'size="{o["half"][0]} {o["half"][1]} {bh}" rgba="0.3 0.3 0.33 1" group="0"/>')
    balls = ""
    for i, b in enumerate(ep["balls"]):
        c = COLORS[0] if identical else COLORS[i % len(COLORS)]
        sref = "0.02 0.1" if b["bouncy"] else "0.02 1"  # measured restitution ~0.6 vs ~0.1; stiffer settings explode at dt=2 ms
        balls += (f'<body name="ball{i}" pos="{b["p0"][0]} {b["p0"][1]} {b["radius"] + 1e-4}"><freejoint/>'
                  f'<geom name="ball{i}" type="sphere" size="{b["radius"]}" mass="{b["mass"]}" condim="6" priority="1" '
                  f'friction="{b["mu_slide"]} 0.005 {b["mu_roll"]}" solref="{sref}" rgba="{c[0]} {c[1]} {c[2]} 1" group="1"/></body>')
    return f"""<mujoco><option timestep="{DT}"/><visual><global offwidth="{IMG}" offheight="{IMG}"/>
<quality shadowsize="0"/></visual><worldbody><light pos="0 0 2.5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
<light pos="1 -1 2" dir="-0.4 0.4 -1" diffuse="0.3 0.3 0.3"/>
<camera name="cam" pos="{cpos[0]} {cpos[1]} {cpos[2]}" xyaxes="{' '.join(map(str, xy))}" fovy="50"/>
<geom name="floor" type="plane" size="{HALF + 0.1} {HALF + 0.1} 0.1" rgba="0.82 0.8 0.76 1" group="2"/>
{walls}{occ}{balls}</worldbody></mujoco>"""


def hidden_mask(m, d):
    """Table cells (at ball-centre height 3 cm) whose centre is hidden from the camera by an occluder or wall."""
    cam = d.cam_xpos[0].copy()
    xs = np.arange(-HALF + GRID / 2, HALF, GRID)
    grp = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # occluders and walls only (balls group 1, floor group 2 excluded)
    gid = np.zeros(1, np.int32)
    mask = np.zeros((len(xs), len(xs)), bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            p = np.array([x, y, 0.03])
            v = p - cam
            dist = np.linalg.norm(v)
            hit = mujoco.mj_ray(m, d, cam, v / dist, grp, 1, -1, gid)
            mask[i, j] = 0 <= hit < dist - 0.01
    return mask


def ray_visibility(m, d, body, geom, radius):
    """Fraction of 9 camera rays, aimed at the ball centre and 8 points at 0.7 r around it in the
    image plane, whose first hit (occluders, walls, balls; floor excluded) is this ball."""
    cam = d.cam_xpos[0]
    R = d.cam_xmat[0].reshape(3, 3)
    c = d.xpos[body]
    grp = np.array([1, 1, 0, 0, 0, 0], np.uint8)
    gid = np.zeros(1, np.int32)
    hits = 0
    for ox, oy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]:
        p = c + 0.7 * radius * (ox * R[:, 0] + oy * R[:, 1])
        v = p - cam
        mujoco.mj_ray(m, d, cam, v / np.linalg.norm(v), grp, 1, -1, gid)
        hits += int(gid[0] == geom)
    return hits / 9


def simulate(ep, render=True, identical=False):
    m = mujoco.MjModel.from_xml_string(build_xml(ep, identical))
    d = mujoco.MjData(m)
    n = len(ep["balls"])
    jadr = [m.body_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}")] for i in range(n)]  # free joint: 3 linear dofs first
    for i, b in enumerate(ep["balls"]):
        d.qvel[jadr[i]:jadr[i] + 2] = b["v0"]
    mujoco.mj_forward(m, d)
    bgeom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ball{i}") for i in range(n)]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    bbody = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") for i in range(n)]
    out = dict(pos=np.zeros((N_FRAMES, n, 3), np.float32), vel=np.zeros((N_FRAMES, n, 3), np.float32),
               vis_px=np.zeros((N_FRAMES, n), np.int16), vis_frac=np.zeros((N_FRAMES, n), np.float32),
               event=np.zeros((N_FRAMES, n), np.int8))
    rgb = np.zeros((N_FRAMES, IMG, IMG, 3), np.uint8) if render else None
    rend = mujoco.Renderer(m, IMG, IMG)
    for t in range(N_FRAMES):
        if t > 0:
            ev = np.zeros(n, np.int8)
            for _ in range(SUB):
                mujoco.mj_step(m, d)
                for c in d.contact[:d.ncon]:
                    for i, g in enumerate(bgeom):
                        if g in (c.geom1, c.geom2):
                            other = c.geom2 if c.geom1 == g else c.geom1
                            if other == floor:
                                continue
                            ev[i] = max(ev[i], EV["collision"] if other in bgeom else EV["bounce"])
            out["event"][t] = ev
        for i in range(n):
            out["pos"][t, i] = d.xpos[bbody[i]]
            out["vel"][t, i] = d.qvel[jadr[i]:jadr[i] + 3]  # free-joint linear velocity, world frame
        rend.update_scene(d, camera="cam")
        if render:
            rgb[t] = rend.render()
        rend.enable_segmentation_rendering()
        seg = rend.render()
        rend.disable_segmentation_rendering()
        for i, g in enumerate(bgeom):
            out["vis_px"][t, i] = int(((seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == g)).sum())
            out["vis_frac"][t, i] = ray_visibility(m, d, bbody[i], g, ep["balls"][i]["radius"])
    out["visible"] = out["vis_frac"] >= 3 / 9  # occlusion-based, independent of image resolution
    out["hidden_mask"] = hidden_mask(m, d)
    cam_id = 0
    out["cam_pos"], out["cam_mat"] = d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].copy()
    rend.close()
    if render:
        out["rgb"] = rgb
    return out


def render_frames(ep, size, identical=False):
    """RGB frames of an episode at another image size: same camera, same physics as simulate() (same model,
    same initial state, same steps), so the scene and view are identical and only the pixel count changes.
    Returns (rgb (N_FRAMES, size, size, 3) uint8, ball positions (N_FRAMES, n, 3) float32); callers compare the
    positions with simulate()'s to confirm the trajectory. simulate() itself is unchanged."""
    m = mujoco.MjModel.from_xml_string(build_xml(ep, identical))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, size), max(m.vis.global_.offheight, size)
    d = mujoco.MjData(m)
    n = len(ep["balls"])
    bbody = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") for i in range(n)]
    for i, b in enumerate(ep["balls"]):
        d.qvel[m.body_dofadr[bbody[i]]:m.body_dofadr[bbody[i]] + 2] = b["v0"]
    mujoco.mj_forward(m, d)
    rgb = np.zeros((N_FRAMES, size, size, 3), np.uint8)
    pos = np.zeros((N_FRAMES, n, 3), np.float32)
    rend = mujoco.Renderer(m, size, size)
    for t in range(N_FRAMES):
        if t > 0:
            for _ in range(SUB):  # one step at a time, as in simulate()
                mujoco.mj_step(m, d)
        pos[t] = d.xpos[bbody]
        rend.update_scene(d, camera="cam")
        rgb[t] = rend.render()
    rend.close()
    return rgb, pos


def generate(split, n, out_path, start=0, render=True, identical=False, n_obj=None, ranges=None, version=GEN_VERSION):
    import pickle
    lo, hi = SEED_RANGES[split]
    eps = []
    for k in range(start, start + n):
        seed = lo + k
        assert seed < hi
        ep = sample_episode(seed, n_obj=n_obj, ranges=ranges, version=version)
        ep["sim"] = simulate(ep, render=render, identical=identical)
        eps.append(ep)
    with open(out_path, "wb") as f:
        pickle.dump(eps, f, protocol=4)
    return eps
