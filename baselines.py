"""Stage 0 pilot baselines. All methods see the same inputs: noisy positions (sigma 1 cm) of
visible balls, nothing while a ball is hidden. Noise is seeded per episode, so every method
gets identical observations. No method sees simulator state at test time.

  B0  last seen                     hidden ball stays at its last measurement
  B1  constant-velocity Kalman      white-noise-acceleration model, q tuned on the fit half
  B1d damped Kalman + walls         velocity decay and wall reflection fitted on the fit half
  B2  physics particle filter       MuJoCo as motion model; mass, friction, restitution drawn
                                    from the prior per particle (never read from the episode);
                                    weights use measurements and negative evidence (a particle
                                    predicting a visible position while the ball is hidden is
                                    down-weighted, using the camera's hidden-region map)
  OR  oracle-parameter filter       B2 with the true parameters (isolates parameter uncertainty)
  B3  GRU                           recurrent network, Gaussian output, trained on the fit half
Each method returns, per frame and ball, a mean (2,) and covariance (2, 2) of position.
"""
import numpy as np

SIGMA_M = 0.01
DT = 0.1


def measurements(ep):
    s = ep["sim"]
    r = np.random.default_rng(ep["seed"] + 12345)
    z = s["pos"][:, :, :2] + r.normal(0, SIGMA_M, s["pos"][:, :, :2].shape)
    return np.where(s["visible"][..., None], z, np.nan).astype(np.float64)


# ---------------------------------------------------------------- B0
def last_seen(z, sig_tau):
    T, n, _ = z.shape
    mu, cov = np.full((T, n, 2), np.nan), np.full((T, n, 2, 2), np.nan)
    for i in range(n):
        last, tau = None, 0
        for t in range(T):
            if not np.isnan(z[t, i, 0]):
                last, tau = z[t, i], 0
            else:
                tau += 1
            if last is not None:
                mu[t, i] = last
                s = np.sqrt(SIGMA_M ** 2 + (sig_tau * tau * DT) ** 2)
                cov[t, i] = np.eye(2) * s ** 2
    return mu, cov


# ---------------------------------------------------------------- B1
def kalman(z, q, damp=0.0, rest=None, half=0.5, radius=None, return_vel=False):
    """Constant-velocity KF (B1). With damp > 0 the velocity decays as exp(-damp * dt) per step,
    and with rest set the predicted position reflects off the arena walls with that restitution
    (B1d: simple physics tracking; it knows walls, not blocks or other balls).
    return_vel=True also returns the velocity mean (T, n, 2), for diagnostics."""
    T, n, _ = z.shape
    a_ = np.exp(-damp * DT)
    F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, a_, 0], [0, 0, 0, a_]], float)
    Q = q * np.array([[DT ** 3 / 3, 0, DT ** 2 / 2, 0], [0, DT ** 3 / 3, 0, DT ** 2 / 2],
                      [DT ** 2 / 2, 0, DT, 0], [0, DT ** 2 / 2, 0, DT]])
    H = np.eye(2, 4)
    Rm = np.eye(2) * SIGMA_M ** 2
    mu, cov = np.full((T, n, 2), np.nan), np.full((T, n, 2, 2), np.nan)
    vel = np.full((T, n, 2), np.nan)
    for i in range(n):
        x, P = None, None
        lim = half - (radius[i] if radius is not None else 0.03)
        for t in range(T):
            if x is not None:
                x, P = F @ x, F @ P @ F.T + Q
                if rest is not None:
                    for k in range(2):
                        if abs(x[k]) > lim:
                            x[k] = np.sign(x[k]) * (2 * lim - abs(x[k]))
                            x[2 + k] = -rest * x[2 + k]
            if not np.isnan(z[t, i, 0]):
                if x is None:
                    x, P = np.array([*z[t, i], 0, 0]), np.diag([SIGMA_M ** 2] * 2 + [1.0, 1.0])
                else:
                    S = H @ P @ H.T + Rm
                    K = P @ H.T @ np.linalg.inv(S)
                    x = x + K @ (z[t, i] - H @ x)
                    P = (np.eye(4) - K @ H) @ P
            if x is not None:
                mu[t, i], cov[t, i], vel[t, i] = x[:2], P[:2, :2], x[2:]
    return (mu, cov, vel) if return_vel else (mu, cov)


# ---------------------------------------------------------------- B2 / oracle
def particle_filter(ep, z, n_part=100, oracle=False, seed=0, keep_particles=False, temper=1.0, neg_evidence="mask",
                    rough=1.0, diag=False):
    """B2 (and the oracle-parameter reference with oracle=True). Defaults reproduce the Stage 0 pilot.

    temper        exponent on the per-frame likelihood (measurement and negative evidence); < 1 flattens it
    neg_evidence  "mask": a hidden ball's particles are scored with the camera's table-plane hidden map at the
                  particle centre (pilot); "ray": with the same 9-ray visibility test that labels the truth;
                  "off": no negative evidence
    rough         scale on the roughening noise added at each resampling (0 = none). The same random numbers
                  are drawn at every scale, so variants differ only in the jitter's size
    diag          also return a dict: ess (T,) before any resampling, resampled (T,), unique (T,) share of
                  distinct particles kept by a resampling, vel (T, n, 2) weighted mean velocity
    """
    import mujoco

    import gen
    rng = np.random.default_rng(ep["seed"] + seed)
    m = mujoco.MjModel.from_xml_string(gen.build_xml(ep))
    n = len(ep["balls"])
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") for i in range(n)]
    gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ball{i}") for i in range(n)]
    qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in bid]
    vadr = [m.body_dofadr[b] for b in bid]
    mask = ep["sim"]["hidden_mask"]
    T = z.shape[0]

    def prior(i):
        b = ep["balls"][i]
        if oracle:
            return np.array([b["mass"], b["mu_slide"], b["mu_roll"], float(b["bouncy"])])
        return np.array([rng.uniform(0.1, 1.0), rng.uniform(0.05, 0.4), rng.uniform(0.0002, 0.002), float(rng.random() < 0.5)])

    # initialise from the first two measurements of each ball (balls start visible by construction)
    datas, params = [], []
    for k in range(n_part):
        d = mujoco.MjData(m)
        prm = []
        for i in range(n):
            t0 = int(np.argmax(~np.isnan(z[:, i, 0])))
            p = z[t0, i] + rng.normal(0, SIGMA_M, 2)
            t1 = t0 + 1
            v = (z[t1, i] - z[t0, i]) / DT if t1 < T and not np.isnan(z[t1, i, 0]) else np.zeros(2)
            v = v + rng.normal(0, 0.15, 2)
            d.qpos[qadr[i]:qadr[i] + 2] = p
            d.qvel[vadr[i]:vadr[i] + 2] = v
            prm.append(prior(i))
        mujoco.mj_forward(m, d)
        datas.append(d)
        params.append(np.array(prm))
    w = np.full(n_part, 1.0 / n_part)

    def set_params(prm):
        for i in range(n):
            mass, mus, mur, bouncy = prm[i]
            r_ = ep["balls"][i]["radius"]
            m.body_mass[bid[i]] = mass
            m.body_inertia[bid[i]] = 0.4 * mass * r_ ** 2
            m.geom_friction[gid[i]] = [mus, 0.005, mur]
            m.geom_solref[gid[i]] = [0.02, 0.1] if bouncy > 0.5 else [0.02, 1.0]

    def cell_hidden(p):
        i, j = np.clip(((p + gen.HALF) / gen.GRID).astype(int), 0, mask.shape[0] - 1)
        return mask[i, j]

    def ray_hidden(k, i):
        return gen.ray_visibility(m, datas[k], bid[i], gid[i], ep["balls"][i]["radius"]) < 3 / 9

    mu, cov = np.full((T, n, 2), np.nan), np.full((T, n, 2, 2), np.nan)
    kept = {} if keep_particles else None
    D = dict(ess=np.full(T, np.nan), resampled=np.zeros(T, bool), unique=np.full(T, np.nan),
             vel=np.full((T, n, 2), np.nan)) if diag else None
    first = [int(np.argmax(~np.isnan(z[:, i, 0]))) for i in range(n)]
    for t in range(T):
        if t > 0:
            for k in range(n_part):
                set_params(params[k])
                mujoco.mj_step(m, datas[k], nstep=gen.SUB)
        P = np.array([[datas[k].qpos[qadr[i]:qadr[i] + 2] for i in range(n)] for k in range(n_part)])  # (K, n, 2)
        if t > 0:
            logw = np.log(w + 1e-300)
            for i in range(n):
                if not np.isnan(z[t, i, 0]):
                    d2 = ((P[:, i] - z[t, i]) ** 2).sum(-1)
                    logw += temper * (-0.5 * d2 / (SIGMA_M ** 2 + 0.01 ** 2))
                elif t > first[i] and neg_evidence != "off":
                    hid = [cell_hidden(p) for p in P[:, i]] if neg_evidence == "mask" else [ray_hidden(k, i) for k in range(n_part)]
                    logw += temper * np.log(np.where(hid, 0.95, 0.05))
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()
            ess = 1.0 / (w ** 2).sum()
            if diag:
                D["ess"][t] = ess / n_part
            if ess < n_part / 2:  # systematic resampling + roughening
                u = (rng.random() + np.arange(n_part)) / n_part
                idx = np.minimum(np.searchsorted(np.cumsum(w), u), n_part - 1)
                if diag:
                    D["resampled"][t], D["unique"][t] = True, len(np.unique(idx)) / n_part
                new_d, new_p = [], []
                for k in idx:
                    d = mujoco.MjData(m)
                    mujoco.mj_copyData(d, m, datas[k])
                    for i in range(n):
                        d.qpos[qadr[i]:qadr[i] + 2] += rough * rng.normal(0, 0.003, 2)
                        d.qvel[vadr[i]:vadr[i] + 2] += rough * rng.normal(0, 0.02, 2)
                    prm = params[k].copy()
                    if not oracle:
                        prm[:, :3] *= np.exp(rough * rng.normal(0, 0.05, (n, 3)))
                    new_d.append(d)
                    new_p.append(prm)
                datas, params = new_d, new_p
                w = np.full(n_part, 1.0 / n_part)
                P = np.array([[datas[k].qpos[qadr[i]:qadr[i] + 2] for i in range(n)] for k in range(n_part)])
        for i in range(n):
            if t >= first[i]:
                mu[t, i] = (w[:, None] * P[:, i]).sum(0)
                dd = P[:, i] - mu[t, i]
                cov[t, i] = (w[:, None, None] * dd[:, :, None] * dd[:, None, :]).sum(0) + np.eye(2) * 1e-6
        if keep_particles:
            kept[t] = (P.astype(np.float32), w.astype(np.float32))
        if diag:
            V = np.array([[datas[k].qvel[vadr[i]:vadr[i] + 2] for i in range(n)] for k in range(n_part)])
            for i in range(n):
                if t >= first[i]:
                    D["vel"][t, i] = (w[:, None] * V[:, i]).sum(0)
    return (mu, cov, kept, D) if diag else (mu, cov, kept)


# ---------------------------------------------------------------- B3
def gru_features(z, nmax=2):
    T, n, _ = z.shape
    X = np.zeros((T, nmax, 3), np.float32)
    vis = ~np.isnan(z[..., 0])
    X[:, :n, :2] = np.nan_to_num(z).astype(np.float32)
    X[:, :n, 2] = vis
    return X.reshape(T, -1)


def make_gru(nmax=2, hidden=128):
    import torch

    class GRUFilter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(3 * nmax, hidden, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, nmax * 4))

        def forward(self, x):
            h, _ = self.rnn(x)
            o = self.head(h).view(x.shape[0], x.shape[1], nmax, 4)
            return o[..., :2], o[..., 2:].clamp(-8, 2)  # mean, log-std (diagonal)

    return GRUFilter()
