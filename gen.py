"""Occlusion benchmark generator (spec: Learning Persistent Physical States Under Occlusion).

One MuJoCo arena per episode:
  - 1 m x 1 m walled tabletop; 1-2 balls (free joints: they slide, then roll);
  - 1-2 occluders: "screen" (floats above the table, hides but never touches) or
    "block" (on the table, hides and collides);
  - one 64x64 RGB camera whose pose is sampled per episode around a nominal oblique view;
  - 8 s at 10 Hz (80 frames), physics at 2 ms.
Ball 0 is launched toward an occluder so occlusions are common; ball 1 (if any) moves
randomly, which creates collision opportunities, sometimes behind an occluder.

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


def sample_episode(seed, n_obj=None, ranges=None):
    """Occluders and camera first; then the hidden region is found by ray casting, and ball 0 is
    launched toward a random hidden cell, so most episodes contain an occlusion."""
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
    return ep


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


def generate(split, n, out_path, start=0, render=True, identical=False, n_obj=None, ranges=None):
    import pickle
    lo, hi = SEED_RANGES[split]
    eps = []
    for k in range(start, start + n):
        seed = lo + k
        assert seed < hi
        ep = sample_episode(seed, n_obj=n_obj, ranges=ranges)
        ep["sim"] = simulate(ep, render=render, identical=identical)
        eps.append(ep)
    with open(out_path, "wb") as f:
        pickle.dump(eps, f, protocol=4)
    return eps
