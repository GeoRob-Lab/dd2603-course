"""Processing steps available in preprocessing configs (``type: <name>``).

Pose features are ``[x, y, z, qw, qx, qy, qz]`` (position in metres, unit quaternion).
``remove_pauses``, ``arclength_resample`` and ``relative_to_goal``, and the ``minmax`` and
``maxabs`` normalizations, are ports of the preprocessing used for the RB-Y1 insertion demos in
the manifolds code (``manifolds/training/dataset.py``, ``manifolds/utils/trajectory_preprocessing.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rbylearn.preprocessing.base import ProcessStep, register_step
from rbylearn.utils.geometry import matrix_to_quat, quat_slerp, quat_to_matrix, se3_log, so3_exp, so3_log

TS = ".timestamps"  # suffix of a feature's timestamps (seconds) in a frame


def _time_series(frame: dict, features: list[str] | None) -> list[str]:
    """The features to process: the given ones, or every feature that has timestamps."""
    return features if features else [k[: -len(TS)] for k in frame if k.endswith(TS)]


@register_step("crop")
@dataclass
class Crop(ProcessStep):
    """Keep time steps ``start`` to ``end`` (indices, Python slice semantics) of each feature."""

    start: int = 0
    end: int | None = None
    features: list[str] | None = None  # default: all features with timestamps

    def forward(self, frame):
        frame = dict(frame)
        for f in _time_series(frame, self.features):
            for key in (f, f + TS):
                if key in frame:
                    frame[key] = frame[key][self.start:self.end]
        return frame


@register_step("resample")
@dataclass
class Resample(ProcessStep):
    """Resample features onto one time grid with step ``dt`` (s), from their timestamps.

    Vectors are interpolated linearly; the quaternion part of pose features (names ending in
    ``.pose``) is interpolated along the sphere. All features end up on the same time steps.
    Features listed in ``hold`` are not interpolated: each time step gets their latest value at
    or before it (e.g. point clouds, which can't be interpolated, from cameras slower than the
    policy: as on the robot, the policy sees the latest one).
    """

    dt: float = 0.1
    features: list[str] | None = None  # default: all features with timestamps
    hold: list[str] = field(default_factory=list)

    def forward(self, frame):
        frame = dict(frame)
        names = _time_series(frame, self.features)
        end = min(frame[f + TS][-1] for f in names if f not in self.hold)
        grid = np.arange(0.0, end + 1e-9, self.dt)
        for f in names:
            t, x = frame[f + TS], frame[f]
            if f in self.hold:
                frame[f], frame[f + TS] = x[np.clip(np.searchsorted(t, grid + 1e-9, side="right") - 1, 0, len(t) - 1)], grid
                continue
            out = np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], axis=1)
            if f.endswith(".pose"):
                k = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
                s = np.clip((grid - t[k]) / (t[k + 1] - t[k]), 0.0, 1.0)
                out[:, 3:7] = [quat_slerp(x[i, 3:7], x[i + 1, 3:7], si) for i, si in zip(k, s)]
            frame[f], frame[f + TS] = out, grid
        return frame


@register_step("quaternion_sign")
@dataclass
class QuaternionSign(ProcessStep):
    """Flip quaternions of pose features to ``qw >= 0`` (q and -q are the same rotation), so the
    policy never sees two encodings of one orientation. Runs on training data and observations."""

    features: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation"])

    def forward(self, frame):
        frame = dict(frame)
        for f in self.features:
            x = frame[f].copy()
            x[:, 3:7] *= np.where(x[:, 3:4] < 0, -1.0, 1.0)
            frame[f] = x
        return frame


@register_step("pose_to_velocity")
@dataclass
class PoseToVelocity(ProcessStep):
    """Add an action feature: the end-effector velocity ``[vx, vy, vz, wx, wy, wz]`` (base frame)
    that moves each pose to the next one. The last step gets zero velocity.

    Useful for datasets that record poses but no commands (e.g. the insertion dataset).
    """

    pose_feature: str = "right.ee.pose"
    output_feature: str = "right.commands.arm.ee.velocity"

    def forward(self, frame):
        frame = dict(frame)
        x, t = frame[self.pose_feature], frame[self.pose_feature + TS]
        velocity = np.zeros((len(x), 6))
        for k in range(len(x) - 1):
            dt = t[k + 1] - t[k]
            velocity[k, :3] = (x[k + 1, :3] - x[k, :3]) / dt
            velocity[k, 3:] = so3_log(quat_to_matrix(x[k + 1, 3:7]) @ quat_to_matrix(x[k, 3:7]).T) / dt
        frame[self.output_feature], frame[self.output_feature + TS] = velocity, t
        return frame


@register_step("corrective_noise")
@dataclass
class CorrectiveNoise(ProcessStep):
    """Data augmentation for velocity policies: add ``copies`` noisy versions of every training
    pose, each labelled with the demo's velocity plus a correction back towards the demo pose
    (``correction_gain`` times the offset, in 1/s).

    A policy trained only on the demonstrated poses has never seen a pose slightly off the demos,
    so during a rollout its small errors add up and it can drift away and diverge. The noisy poses
    teach it to steer back. Positions get Gaussian noise of ``position_std`` metres, orientations a
    random rotation of ``rotation_std_deg`` degrees (per axis). Runs on training demos only, after
    ``pose_to_velocity`` (and before normalization). The added samples are appended to each
    episode, so their timestamps are not in order. The episode's other time series (e.g. the
    vision features) are repeated for each copy.
    """

    pose_feature: str = "right.ee.pose"
    velocity_feature: str = "right.commands.arm.ee.velocity"
    position_std: float = 0.01
    rotation_std_deg: float = 3.0
    copies: int = 5
    correction_gain: float = 5.0
    seed: int = 0
    hooks: list[str] = field(default_factory=lambda: ["training"])

    def forward(self, frame):
        frame = dict(frame)
        rng = np.random.default_rng([self.seed, *(np.abs(frame[self.pose_feature][0, :3]) * 1e6).astype(int)])  # differs per demo
        x, v, t = frame[self.pose_feature], frame[self.velocity_feature], frame[self.pose_feature + TS]
        poses, velocities = [x], [v]
        for _ in range(self.copies):
            noisy, label = np.empty_like(x), np.empty_like(v)
            for k in range(len(x)):
                R_demo = quat_to_matrix(x[k, 3:7])
                p = x[k, :3] + rng.normal(0.0, self.position_std, 3)
                R = so3_exp(rng.normal(0.0, np.deg2rad(self.rotation_std_deg), 3)) @ R_demo
                q = matrix_to_quat(R)
                noisy[k] = np.concatenate([p, q if q[0] >= 0 else -q])
                label[k] = v[k] + self.correction_gain * np.concatenate([x[k, :3] - p, so3_log(R_demo @ R.T)])
            poses.append(noisy)
            velocities.append(label)
        others = [f for f in _time_series(frame, None) if f not in (self.pose_feature, self.velocity_feature) and len(frame[f]) == len(x)]
        frame[self.pose_feature], frame[self.velocity_feature] = np.concatenate(poses), np.concatenate(velocities)
        for f in others:
            frame[f] = np.concatenate([frame[f]] * (self.copies + 1))
        for f in (self.pose_feature, self.velocity_feature, *others):
            frame[f + TS] = np.tile(t, self.copies + 1)
        return frame


@register_step("normalize")
@dataclass
class Normalize(ProcessStep):
    """Scale features with statistics fitted on the training data (all demos together). Runs forward
    on training data and observations, and inverts on policy outputs.

    ``method``:

    - ``standard``: zero mean, unit variance.
    - ``minmax``: to [-1, 1] (manifolds: translations).
    - ``maxabs``: divided by the largest absolute value (manifolds: velocity labels).

    ``dims`` restricts it to some components, e.g. ``[0, 1, 2]`` for the position of a pose.
    With ``clip``, policy outputs (action hook) are first clipped to the range of the training
    data ([-1, 1] for ``minmax`` and ``maxabs``), e.g. so a policy never commands a velocity
    faster than the demonstrations.
    """

    features: list[str] = field(default_factory=list)
    method: str = "standard"
    dims: list[int] | None = None
    clip: bool = False
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation", "action"])

    def __post_init__(self):
        if self.method not in ("standard", "minmax", "maxabs"):
            raise ValueError("normalize: method must be 'standard', 'minmax' or 'maxabs'")
        self.offset: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}

    def fit(self, episodes):
        for f in self.features:
            x = np.concatenate([e[f] for e in episodes])
            dims = self.dims if self.dims is not None else list(range(x.shape[1]))
            offset, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
            x = x[:, dims]
            if self.method == "standard":
                offset[dims], scale[dims] = x.mean(0), x.std(0)
            elif self.method == "minmax":
                offset[dims], scale[dims] = (x.max(0) + x.min(0)) / 2, (x.max(0) - x.min(0)) / 2
            else:
                scale[dims] = np.abs(x).max(0)
            scale[scale < 1e-8] = 1.0
            self.offset[f], self.scale[f] = offset, scale

    def forward(self, frame):
        return {k: (v - self.offset[k]) / self.scale[k] if k in self.scale else v for k, v in frame.items()}

    def inverse(self, frame):
        if self.clip and self.method in ("minmax", "maxabs"):
            frame = {k: np.clip(v, -1.0, 1.0) if k in self.scale else v for k, v in frame.items()}
        return {k: v * self.scale[k] + self.offset[k] if k in self.scale else v for k, v in frame.items()}

    def state(self):
        return {"offset": {k: v.tolist() for k, v in self.offset.items()},
                "scale": {k: v.tolist() for k, v in self.scale.items()}}

    def load_state(self, state):
        self.offset = {k: np.asarray(v) for k, v in state.get("offset", {}).items()}
        self.scale = {k: np.asarray(v) for k, v in state.get("scale", {}).items()}


@register_step("remove_pauses")
@dataclass
class RemovePauses(ProcessStep):
    """Drop interior pauses: runs of at least ``min_run`` steps whose speed is below ``eps`` (the
    first frame of a run is kept). Speed is the translation of the SE(3) twist between consecutive
    poses, with positions min-max scaled to [-1, 1] over the whole dataset, as in manifolds."""

    feature: str = "left.ee.pose"
    eps: float = 5e-3
    min_run: int = 5

    def fit(self, episodes):
        p = np.concatenate([e[self.feature][:, :3] for e in episodes])
        self._low, self._range = p.min(0), p.max(0) - p.min(0)
        self._range[self._range < 1e-8] = 1.0

    def forward(self, frame):
        frame = dict(frame)
        x = frame[self.feature]
        p = 2 * (x[:, :3] - self._low) / self._range - 1
        R = [quat_to_matrix(q) for q in x[:, 3:7]]
        speed = np.array([np.linalg.norm(se3_log(R[k].T @ R[k + 1], R[k].T @ (p[k + 1] - p[k]))[:3])
                          for k in range(len(x) - 1)])
        keep = np.ones(len(x), bool)
        slow = np.concatenate([[0], (speed < self.eps).astype(int), [0]])
        edges = np.diff(slow)
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start >= self.min_run:
                keep[start + 1:end] = False
        for key in (self.feature, self.feature + TS):
            if key in frame:
                frame[key] = frame[key][keep]
        return frame


@register_step("arclength_resample")
@dataclass
class ArclengthResample(ProcessStep):
    """Resample a pose feature to ``n_points`` poses equally spaced along the path (translation arc
    length), as in manifolds. Timestamps follow the original clock, so velocities derived afterwards
    (``pose_to_velocity``) keep the demonstration's timing (manifolds' "gain")."""

    feature: str = "left.ee.pose"
    n_points: int = 100

    def forward(self, frame):
        frame = dict(frame)
        x, t = frame[self.feature], frame[self.feature + TS]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1))])
        keep = np.concatenate([[True], np.diff(s) > 1e-12])  # strictly increasing arc length
        s, x, t = s[keep], x[keep], t[keep]
        s_new = np.linspace(0.0, s[-1], self.n_points)
        out = np.stack([np.interp(s_new, s, x[:, i]) for i in range(7)], axis=1)
        k = np.clip(np.searchsorted(s, s_new, side="right") - 1, 0, len(s) - 2)
        u = np.clip((s_new - s[k]) / (s[k + 1] - s[k]), 0.0, 1.0)
        out[:, 3:7] = [quat_slerp(x[i, 3:7], x[i + 1, 3:7], ui) for i, ui in zip(k, u)]
        frame[self.feature], frame[self.feature + TS] = out, np.interp(s_new, s, t)
        return frame


@register_step("relative_to_goal")
@dataclass
class RelativeToGoal(ProcessStep):
    """Express a pose feature in the frame of the goal pose, i.e. the goal is at the origin
    (manifolds' ``center_goal``).

    ``goal`` says which goal the demos use: ``"mean"``, the mean of the training demos' last
    poses, for every demo; or ``"last"``, each demo's own last pose (manifolds). At run time the
    goal comes from the observation frame (``<feature>.goal``, a pose); without one, the mean goal
    is used, so with ``"last"`` the policy runs with a different goal than each demo had. On
    policy outputs (action hook) the listed ``velocity_features`` are rotated back from the goal
    frame to the base frame.
    """

    feature: str = "left.ee.pose"
    velocity_features: list[str] = field(default_factory=list)
    goal: str = "last"
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation", "action"])

    def fit(self, episodes):
        goals = np.array([e[self.feature][-1] for e in episodes])
        mean_q = goals[:, 3:7].mean(0) * np.sign(goals[:, 3:4] + 1e-12).mean()
        self.default_goal = np.concatenate([goals[:, :3].mean(0), mean_q / np.linalg.norm(mean_q)])

    def forward(self, frame):
        frame = dict(frame)
        x = frame[self.feature]
        own_goal = self.goal == "last" and self.feature + TS in frame  # a demo, with its own goal
        goal = frame.get(self.feature + ".goal", [x[-1]] if own_goal else [self.default_goal])[0]
        self._goal_rotation = quat_to_matrix(goal[3:7])
        Rg = self._goal_rotation
        out = np.empty_like(x)
        out[:, :3] = (x[:, :3] - goal[:3]) @ Rg  # Rg^T (p - p_goal)
        out[:, 3:7] = [matrix_to_quat(Rg.T @ quat_to_matrix(q)) for q in x[:, 3:7]]
        frame[self.feature] = out
        return frame

    def inverse(self, frame):
        frame = dict(frame)
        for f in self.velocity_features:
            v = frame[f].copy()
            v[:, :3], v[:, 3:6] = v[:, :3] @ self._goal_rotation.T, v[:, 3:6] @ self._goal_rotation.T
            frame[f] = v
        return frame

    def state(self):
        return {"default_goal": self.default_goal.tolist()}

    def load_state(self, state):
        self.default_goal = np.asarray(state["default_goal"])
