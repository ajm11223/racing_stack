#!/usr/bin/env python3
"""Local path planning from Chu, Lee and Sunwoo (IEEE T-ITS, 2012).

The implementation follows the paper's four online stages:

1. localize the vehicle on an arc-length-parameterized cubic-spline base frame;
2. generate cubic lateral-offset candidates in curvilinear coordinates;
3. reject nonholonomic/colliding candidates and evaluate safety, curvature and
   path-consistency costs (equations 14--18);
4. choose the lowest-cost free path, or the path with the longest free length
   when every candidate collides (Algorithm 1, line 18), and determine the
   curvature/risk-limited target speed (equations 19--20).

The paper deliberately separates planning from vehicle control and does not
specify a path-tracking law. ``OffRoadController`` adds a small pure-pursuit
adapter so this planner can occupy the same direct ``(speed, steering)`` seam as
FTG without changing the paper algorithm itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline


_EPS = 1.0e-9


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class PlannerConfig:
    """Vehicle- and implementation-specific design parameters.

    The paper explicitly describes these values as heuristic design factors but
    does not publish the values used on A1. Defaults are scaled for the F1TENTH
    vehicle and are therefore exposed as ROS parameters by controller_manager.
    """

    lateral_span_m: float = 2.0
    lateral_resolution_m: float = 0.10
    path_horizon_m: float = 6.0
    path_resolution_m: float = 0.10
    speed_horizon_gain_s: float = 0.60       # k_v in equation (13)
    min_maneuver_m: float = 1.50             # delta-s_min in equation (13)
    max_curvature: float = 1.55              # 1/m; tan(0.47 rad) / 0.33 m
    safety_sigma_m: float = 0.25             # sigma in equations (15)--(16)
    safety_weight: float = 8.0               # w_S in equation (14)
    smoothness_weight: float = 1.0           # w_kappa in equation (14)
    consistency_weight: float = 2.0          # w_C in equation (14)
    vehicle_length_m: float = 0.58
    vehicle_width_m: float = 0.31
    collision_margin_m: float = 0.06
    road_margin_m: float = 0.03
    scan_max_range_m: float = 8.0
    obstacle_cell_size_m: float = 0.05
    lidar_offset_x_m: float = 0.0
    lidar_offset_y_m: float = 0.0
    lidar_yaw_rad: float = 0.0
    max_lateral_accel_mps2: float = 3.0      # |a_y|max in equation (19)
    safety_speed_gain: float = 0.70          # k_safe in equation (20)
    max_speed_mps: float = 2.0
    emergency_speed_mps: float = 0.60
    stop_distance_m: float = 0.45
    wheelbase_m: float = 0.33
    max_steer_rad: float = 0.47
    lookahead_min_m: float = 0.60
    lookahead_speed_gain_s: float = 0.20
    use_waypoint_bounds: bool = True

    def validated(self) -> "PlannerConfig":
        positive = {
            "lateral_span_m": self.lateral_span_m,
            "lateral_resolution_m": self.lateral_resolution_m,
            "path_horizon_m": self.path_horizon_m,
            "path_resolution_m": self.path_resolution_m,
            "min_maneuver_m": self.min_maneuver_m,
            "max_curvature": self.max_curvature,
            "safety_sigma_m": self.safety_sigma_m,
            "vehicle_length_m": self.vehicle_length_m,
            "vehicle_width_m": self.vehicle_width_m,
            "obstacle_cell_size_m": self.obstacle_cell_size_m,
            "max_lateral_accel_mps2": self.max_lateral_accel_mps2,
            "wheelbase_m": self.wheelbase_m,
            "max_steer_rad": self.max_steer_rad,
            "lookahead_min_m": self.lookahead_min_m,
        }
        bad = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0.0]
        if bad:
            raise ValueError(f"off-road planner parameters must be positive: {', '.join(bad)}")
        nonnegative = {
            "speed_horizon_gain_s": self.speed_horizon_gain_s,
            "collision_margin_m": self.collision_margin_m,
            "road_margin_m": self.road_margin_m,
            "safety_speed_gain": self.safety_speed_gain,
            "max_speed_mps": self.max_speed_mps,
            "emergency_speed_mps": self.emergency_speed_mps,
            "stop_distance_m": self.stop_distance_m,
            "lookahead_speed_gain_s": self.lookahead_speed_gain_s,
        }
        bad = [name for name, value in nonnegative.items() if not np.isfinite(value) or value < 0.0]
        if bad:
            raise ValueError(f"off-road planner parameters must be nonnegative: {', '.join(bad)}")
        return self


@dataclass
class CandidatePath:
    """One lateral-offset path and the paper's evaluation quantities."""

    index: int
    final_offset_m: float
    s_unwrapped: np.ndarray
    s_mod: np.ndarray
    q: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    path_scale: np.ndarray
    feasible: bool = True
    collision: bool = False
    collision_free_length_m: float = 0.0
    safety_cost: float = 0.0
    smoothness_cost: float = 0.0
    consistency_cost: float = 0.0
    total_cost: float = math.inf


@dataclass
class PlanResult:
    """Result retained for command generation, diagnostics and RViz."""

    selected: CandidatePath
    candidates: list[CandidatePath]
    target_speed_mps: float
    emergency: bool
    current_s: float
    current_d: float
    heading_error: float


class _BaseFrame:
    """Arc-length cubic spline plus the paper's two-stage localization.

    A chord-length spline is densely sampled and re-fit against measured arc
    length. This is the practical arc-length approximation described in section
    III-A. Nearest-sample quadratic localization followed by Newton refinement
    implements section III-B.
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        left_width: Optional[Sequence[float]] = None,
        right_width: Optional[Sequence[float]] = None,
        speed_limit: Optional[Sequence[float]] = None,
    ) -> None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size != y_arr.size or x_arr.size < 4:
            raise ValueError("base frame needs at least four x/y waypoints")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("base-frame waypoints must be finite")

        attrs = []
        for values in (left_width, right_width, speed_limit):
            if values is None:
                attrs.append(None)
                continue
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size != x_arr.size:
                raise ValueError("base-frame attributes must match waypoint count")
            attrs.append(arr)

        # Remove a duplicated closing point, then consecutive duplicates.
        if np.linalg.norm([x_arr[-1] - x_arr[0], y_arr[-1] - y_arr[0]]) < 1.0e-6:
            x_arr = x_arr[:-1]
            y_arr = y_arr[:-1]
            attrs = [None if arr is None else arr[:-1] for arr in attrs]
        segment = np.hypot(np.diff(x_arr), np.diff(y_arr))
        keep = np.r_[True, segment > 1.0e-6]
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]
        attrs = [None if arr is None else arr[keep] for arr in attrs]
        if x_arr.size < 4:
            raise ValueError("base frame has fewer than four distinct waypoints")

        closure = float(np.hypot(x_arr[0] - x_arr[-1], y_arr[0] - y_arr[-1]))
        typical = float(np.median(np.hypot(np.diff(x_arr), np.diff(y_arr))))
        self.closed = closure <= max(2.5 * typical, 0.5)

        if self.closed:
            fit_x = np.r_[x_arr, x_arr[0]]
            fit_y = np.r_[y_arr, y_arr[0]]
            fit_attrs = [None if arr is None else np.r_[arr, arr[0]] for arr in attrs]
        else:
            fit_x, fit_y, fit_attrs = x_arr, y_arr, attrs

        chord_s = np.r_[0.0, np.cumsum(np.hypot(np.diff(fit_x), np.diff(fit_y)))]
        bc_type = "periodic" if self.closed else "natural"
        first_x = CubicSpline(chord_s, fit_x, bc_type=bc_type)
        first_y = CubicSpline(chord_s, fit_y, bc_type=bc_type)

        # Reparameterize with numerical arc length, approximating the adaptive
        # integration/re-fit described by the paper while keeping startup cheap.
        dense_count = max(1000, int(math.ceil(chord_s[-1] / 0.02)) + 1)
        dense_chord = np.linspace(0.0, chord_s[-1], dense_count)
        dense_x = first_x(dense_chord)
        dense_y = first_y(dense_chord)
        dense_s = np.r_[0.0, np.cumsum(np.hypot(np.diff(dense_x), np.diff(dense_y)))]
        keep_dense = np.r_[True, np.diff(dense_s) > 1.0e-10]
        dense_s = dense_s[keep_dense]
        dense_x = dense_x[keep_dense]
        dense_y = dense_y[keep_dense]
        self.length = float(dense_s[-1])
        self.spline_x = CubicSpline(dense_s, dense_x, bc_type=bc_type)
        self.spline_y = CubicSpline(dense_s, dense_y, bc_type=bc_type)

        # Dense points seed the quadratic/Newton localization.
        sample_count = max(500, int(math.ceil(self.length / 0.03)))
        self.sample_s = np.linspace(0.0, self.length, sample_count, endpoint=not self.closed)
        self.sample_xy = np.column_stack((self.spline_x(self.sample_s), self.spline_y(self.sample_s)))

        # Attributes are interpolated on the original chord positions mapped to
        # the new arc-length parameter by the same dense reparameterization.
        mapped_s = np.interp(chord_s, dense_chord[keep_dense], dense_s)
        self.attr_s = mapped_s
        self.left_width = fit_attrs[0]
        self.right_width = fit_attrs[1]
        self.speed_limit = fit_attrs[2]

    def _domain(self, s: np.ndarray | float) -> np.ndarray:
        values = np.asarray(s, dtype=float)
        if self.closed:
            return np.mod(values, self.length)
        return np.clip(values, 0.0, self.length)

    def geometry(self, s: np.ndarray | float):
        s_eval = self._domain(s)
        x = self.spline_x(s_eval)
        y = self.spline_y(s_eval)
        dx = self.spline_x(s_eval, 1)
        dy = self.spline_y(s_eval, 1)
        ddx = self.spline_x(s_eval, 2)
        ddy = self.spline_y(s_eval, 2)
        norm = np.maximum(np.hypot(dx, dy), _EPS)
        tx = dx / norm
        ty = dy / norm
        curvature = (dx * ddy - ddx * dy) / np.maximum(norm**3, _EPS)
        return x, y, tx, ty, curvature

    def localize(self, x: float, y: float) -> tuple[float, float, float]:
        point = np.array([x, y], dtype=float)
        nearest = int(np.argmin(np.linalg.norm(self.sample_xy - point, axis=1)))
        s = float(self.sample_s[nearest])

        # Newton minimize 0.5 * ||r(s)-p||^2.
        for _ in range(8):
            s_eval = float(self._domain(s))
            rx = float(self.spline_x(s_eval) - x)
            ry = float(self.spline_y(s_eval) - y)
            dx = float(self.spline_x(s_eval, 1))
            dy = float(self.spline_y(s_eval, 1))
            ddx = float(self.spline_x(s_eval, 2))
            ddy = float(self.spline_y(s_eval, 2))
            gradient = rx * dx + ry * dy
            hessian = dx * dx + dy * dy + rx * ddx + ry * ddy
            if abs(hessian) < 1.0e-8:
                break
            step = float(np.clip(gradient / hessian, -0.5, 0.5))
            s -= step
            s = float(self._domain(s))
            if abs(step) < 1.0e-6:
                break

        xb, yb, tx, ty, _ = self.geometry(s)
        d = float((-ty) * (x - xb) + tx * (y - yb))
        heading = math.atan2(float(ty), float(tx))
        return float(self._domain(s)), d, heading

    def _interp_attr(self, values: Optional[np.ndarray], s: np.ndarray, default: float) -> np.ndarray:
        s_eval = self._domain(s)
        if values is None or not np.any(np.isfinite(values)):
            return np.full_like(np.asarray(s_eval, dtype=float), default, dtype=float)
        clean = np.asarray(values, dtype=float)
        finite = np.isfinite(clean)
        if not np.any(finite):
            return np.full_like(np.asarray(s_eval, dtype=float), default, dtype=float)
        xp = self.attr_s[finite]
        fp = clean[finite]
        if self.closed:
            return np.interp(s_eval, xp, fp, period=self.length)
        return np.interp(s_eval, xp, fp)

    def bounds(self, s: np.ndarray, fallback_half_width: float) -> tuple[np.ndarray, np.ndarray]:
        left = self._interp_attr(self.left_width, s, fallback_half_width)
        right = self._interp_attr(self.right_width, s, fallback_half_width)
        # Some maps leave d_left/d_right at zero for unmeasured samples. Treat
        # those as missing rather than as a physically zero-width road.
        left = np.where(np.isfinite(left) & (left > 0.0), left, fallback_half_width)
        right = np.where(np.isfinite(right) & (right > 0.0), right, fallback_half_width)
        return left, right

    def speed_at(self, s: float, fallback: float) -> float:
        value = float(self._interp_attr(self.speed_limit, np.asarray([s]), fallback)[0])
        return value if np.isfinite(value) and value > 0.0 else fallback


class OffRoadPlanner:
    """Paper local planner independent of ROS."""

    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self.config = (config or PlannerConfig()).validated()
        self.base_frame: Optional[_BaseFrame] = None
        self.previous_path: Optional[CandidatePath] = None
        self._last_s_mod: Optional[float] = None
        self._current_s_unwrapped: Optional[float] = None

    def update_config(self, **values) -> None:
        unknown = set(values) - set(PlannerConfig.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown off-road planner parameters: {sorted(unknown)}")
        self.config = replace(self.config, **values).validated()

    def set_base_frame(
        self,
        x: Sequence[float],
        y: Sequence[float],
        left_width: Optional[Sequence[float]] = None,
        right_width: Optional[Sequence[float]] = None,
        speed_limit: Optional[Sequence[float]] = None,
    ) -> None:
        self.base_frame = _BaseFrame(x, y, left_width, right_width, speed_limit)
        self.reset_history()

    def reset_history(self) -> None:
        self.previous_path = None
        self._last_s_mod = None
        self._current_s_unwrapped = None

    def _unwrap_current_s(self, s_mod: float) -> float:
        if self.base_frame is None:
            raise RuntimeError("base frame is not initialized")
        if self._last_s_mod is None or self._current_s_unwrapped is None:
            self._last_s_mod = s_mod
            self._current_s_unwrapped = s_mod
            return s_mod
        delta = s_mod - self._last_s_mod
        if self.base_frame.closed:
            half = 0.5 * self.base_frame.length
            delta = (delta + half) % self.base_frame.length - half
        self._current_s_unwrapped += delta
        self._last_s_mod = s_mod
        return self._current_s_unwrapped

    def _target_offsets(self, current_s: float) -> np.ndarray:
        cfg = self.config
        half_span = cfg.lateral_span_m / 2.0
        q_min = -half_span
        q_max = half_span
        if cfg.use_waypoint_bounds and self.base_frame is not None:
            probe = current_s + np.arange(
                0.0, cfg.path_horizon_m + 0.5 * cfg.path_resolution_m,
                cfg.path_resolution_m,
            )
            left, right = self.base_frame.bounds(probe, half_span)
            half_vehicle = cfg.vehicle_width_m / 2.0 + cfg.road_margin_m
            q_min = max(q_min, float(np.max(-right + half_vehicle)))
            q_max = min(q_max, float(np.min(left - half_vehicle)))
        if q_min > q_max:
            # Preserve an index for Algorithm 1 exceptional handling; it will be
            # marked as a boundary collision during candidate evaluation.
            center = 0.5 * (q_min + q_max)
            return np.asarray([center])
        resolution = cfg.lateral_resolution_m
        first = math.ceil((q_min - 1.0e-9) / resolution) * resolution
        last = math.floor((q_max + 1.0e-9) / resolution) * resolution
        if first > last:
            return np.asarray([0.5 * (q_min + q_max)])
        return np.arange(first, last + 0.5 * resolution, resolution)

    def _generate_candidate(
        self,
        index: int,
        final_offset: float,
        current_s_mod: float,
        current_s_unwrapped: float,
        current_d: float,
        heading_error: float,
        speed_mps: float,
    ) -> CandidatePath:
        assert self.base_frame is not None
        cfg = self.config
        ds = np.arange(
            0.0, cfg.path_horizon_m + 0.5 * cfg.path_resolution_m,
            cfg.path_resolution_m,
        )
        maneuver = max(cfg.min_maneuver_m + cfg.speed_horizon_gain_s * abs(speed_mps),
                       cfg.path_resolution_m)
        c = math.tan(float(np.clip(heading_error, -math.radians(75.0), math.radians(75.0))))
        delta_q = final_offset - current_d
        a = (c * maneuver - 2.0 * delta_q) / maneuver**3
        b = (3.0 * delta_q - 2.0 * c * maneuver) / maneuver**2
        moving = ds < maneuver
        u = np.minimum(ds, maneuver)
        q = a * u**3 + b * u**2 + c * u + current_d
        dq = 3.0 * a * u**2 + 2.0 * b * u + c
        ddq = 6.0 * a * u + 2.0 * b
        q[~moving] = final_offset
        dq[~moving] = 0.0
        ddq[~moving] = 0.0

        s_mod = np.mod(current_s_mod + ds, self.base_frame.length) \
            if self.base_frame.closed else np.clip(current_s_mod + ds, 0.0, self.base_frame.length)
        xb, yb, tx, ty, base_kappa = self.base_frame.geometry(s_mod)
        one_minus_qk = 1.0 - q * base_kappa
        path_scale = np.sqrt(dq**2 + one_minus_qk**2)  # Q in equation (6)
        sign = np.sign(one_minus_qk)
        sign[sign == 0.0] = 1.0
        curvature = (sign / np.maximum(path_scale, _EPS)) * (
            base_kappa
            + (one_minus_qk * ddq + base_kappa * dq**2)
            / np.maximum(path_scale**2, _EPS)
        )

        # Cartesian transformation of the same curvilinear path. Its derivative
        # gives the heading used for footprint collision checks and tracking.
        x = xb - q * ty
        y = yb + q * tx
        dx_ds = one_minus_qk * tx - dq * ty
        dy_ds = one_minus_qk * ty + dq * tx
        yaw = np.arctan2(dy_ds, dx_ds)

        feasible = bool(
            np.all(np.isfinite(curvature))
            and np.all(one_minus_qk > 0.0)
            and np.max(np.abs(curvature)) <= cfg.max_curvature
        )
        smoothness = float(trapezoid(curvature**2 * path_scale, ds))
        candidate = CandidatePath(
            index=index,
            final_offset_m=float(final_offset),
            s_unwrapped=current_s_unwrapped + ds,
            s_mod=s_mod,
            q=q,
            x=x,
            y=y,
            yaw=yaw,
            curvature=curvature,
            path_scale=path_scale,
            feasible=feasible,
            collision=not feasible,
            collision_free_length_m=0.0 if not feasible else float(ds[-1]),
            smoothness_cost=smoothness if np.isfinite(smoothness) else math.inf,
        )
        return candidate

    def _scan_points(
        self,
        pose: Sequence[float],
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> np.ndarray:
        cfg = self.config
        r = np.asarray(ranges, dtype=float).reshape(-1)
        if r.size == 0 or not np.isfinite(angle_increment) or angle_increment <= 0.0:
            return np.empty((0, 2))
        angles = float(angle_min) + np.arange(r.size) * float(angle_increment)
        valid = np.isfinite(r) & (r > 0.02) & (r <= cfg.scan_max_range_m)
        if not np.any(valid):
            return np.empty((0, 2))
        r = r[valid]
        angles = angles[valid] + cfg.lidar_yaw_rad
        lx = cfg.lidar_offset_x_m + r * np.cos(angles)
        ly = cfg.lidar_offset_y_m + r * np.sin(angles)
        px, py, yaw = map(float, np.asarray(pose).reshape(-1)[:3])
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        points = np.column_stack((px + cy * lx - sy * ly, py + sy * lx + cy * ly))
        # The paper evaluates candidates against a cell-based local map. Store
        # each occupied scan cell once and use its center for footprint checks.
        cell = cfg.obstacle_cell_size_m
        occupied = np.unique(np.round(points / cell).astype(np.int64), axis=0)
        return occupied.astype(float) * cell

    def _check_collision(self, path: CandidatePath, obstacle_points: np.ndarray) -> None:
        if not path.feasible:
            return
        cfg = self.config
        collision_index = len(path.x)

        # The cell-map collision check in the paper considers vehicle size and
        # orientation. Scan returns are occupied cells represented as points here.
        if obstacle_points.size:
            half_length = cfg.vehicle_length_m / 2.0 + cfg.collision_margin_m
            half_width = cfg.vehicle_width_m / 2.0 + cfg.collision_margin_m
            for j, (x, y, yaw) in enumerate(zip(path.x, path.y, path.yaw)):
                delta = obstacle_points - np.array([x, y])
                cy = math.cos(float(yaw))
                sy = math.sin(float(yaw))
                longitudinal = cy * delta[:, 0] + sy * delta[:, 1]
                lateral = -sy * delta[:, 0] + cy * delta[:, 1]
                if np.any((np.abs(longitudinal) <= half_length)
                          & (np.abs(lateral) <= half_width)):
                    collision_index = j
                    break

        # Road widths from the base waypoints act as occupied road-boundary cells.
        if cfg.use_waypoint_bounds and self.base_frame is not None:
            half_vehicle = cfg.vehicle_width_m / 2.0 + cfg.road_margin_m
            left, right = self.base_frame.bounds(path.s_mod, cfg.lateral_span_m / 2.0)
            outside = np.flatnonzero(
                (path.q + half_vehicle > left) | (path.q - half_vehicle < -right)
            )
            if outside.size:
                collision_index = min(collision_index, int(outside[0]))

        if collision_index < len(path.x):
            path.collision = True
            path.collision_free_length_m = max(
                0.0, float(path.s_unwrapped[collision_index] - path.s_unwrapped[0])
            )
        else:
            path.collision = False
            path.collision_free_length_m = float(path.s_unwrapped[-1] - path.s_unwrapped[0])

    def _consistency_cost(self, candidate: CandidatePath) -> float:
        previous = self.previous_path
        if previous is None or len(previous.s_unwrapped) < 2:
            return 0.0
        start = max(float(candidate.s_unwrapped[0]), float(previous.s_unwrapped[0]))
        end = min(float(candidate.s_unwrapped[-1]), float(previous.s_unwrapped[-1]))
        if end <= start + _EPS:
            return 0.0
        mask = (candidate.s_unwrapped >= start) & (candidate.s_unwrapped <= end)
        s_overlap = candidate.s_unwrapped[mask]
        if s_overlap.size < 2:
            return 0.0
        prev_x = np.interp(s_overlap, previous.s_unwrapped, previous.x)
        prev_y = np.interp(s_overlap, previous.s_unwrapped, previous.y)
        distance = np.hypot(candidate.x[mask] - prev_x, candidate.y[mask] - prev_y)
        # Equation (18): integral divided by overlap arc-length.
        return float(trapezoid(distance, s_overlap) / max(end - start, _EPS))

    def _safety_costs(self, collisions: np.ndarray) -> np.ndarray:
        cfg = self.config
        radius = max(1, int(math.ceil(4.0 * cfg.safety_sigma_m / cfg.lateral_resolution_m)))
        offset = np.arange(-radius, radius + 1) * cfg.lateral_resolution_m
        kernel = np.exp(-0.5 * (offset / cfg.safety_sigma_m) ** 2)
        kernel /= np.sum(kernel)
        # Equation (15) regards every out-of-path index as colliding.
        padded = np.pad(collisions.astype(float), radius, mode="constant", constant_values=1.0)
        return np.convolve(padded, kernel, mode="valid")

    def _target_speed(self, selected: CandidatePath, current_s: float, emergency: bool) -> float:
        assert self.base_frame is not None
        cfg = self.config
        max_curvature = float(np.max(np.abs(selected.curvature)))
        curvature_speed = (
            math.sqrt(cfg.max_lateral_accel_mps2 / max_curvature)
            if max_curvature > 1.0e-6 else math.inf
        )
        base_limit = self.base_frame.speed_at(current_s, cfg.max_speed_mps)
        v_curv = min(cfg.max_speed_mps, base_limit, curvature_speed)
        multiplier = max(0.0, 1.0 - cfg.safety_speed_gain * selected.safety_cost**2)
        target = v_curv * multiplier
        if emergency:
            if selected.collision_free_length_m <= cfg.stop_distance_m:
                return 0.0
            target = min(target, cfg.emergency_speed_mps)
        return float(max(0.0, target))

    def plan(
        self,
        pose: Sequence[float],
        speed_mps: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        obstacle_points: Optional[Iterable[Sequence[float]]] = None,
    ) -> PlanResult:
        if self.base_frame is None:
            raise RuntimeError("off-road planner has no base frame")
        pose_arr = np.asarray(pose, dtype=float).reshape(-1)
        if pose_arr.size < 3 or not np.all(np.isfinite(pose_arr[:3])):
            raise ValueError("pose must contain finite x, y and yaw")

        current_s, current_d, base_heading = self.base_frame.localize(
            float(pose_arr[0]), float(pose_arr[1]))
        heading_error = _wrap_angle(float(pose_arr[2]) - base_heading)
        current_s_unwrapped = self._unwrap_current_s(current_s)
        offsets = self._target_offsets(current_s)
        candidates = [
            self._generate_candidate(
                i, float(offset), current_s, current_s_unwrapped,
                current_d, heading_error, float(speed_mps),
            )
            for i, offset in enumerate(offsets)
        ]

        if obstacle_points is None:
            points = self._scan_points(pose_arr, ranges, angle_min, angle_increment)
        else:
            points = np.asarray(list(obstacle_points), dtype=float).reshape(-1, 2)
            points = points[np.all(np.isfinite(points), axis=1)]
        for candidate in candidates:
            self._check_collision(candidate, points)

        collision_flags = np.asarray(
            [candidate.collision or not candidate.feasible for candidate in candidates],
            dtype=bool,
        )
        safety = self._safety_costs(collision_flags)
        cfg = self.config
        for candidate, safety_cost in zip(candidates, safety):
            candidate.safety_cost = float(safety_cost)
            candidate.consistency_cost = self._consistency_cost(candidate)
            candidate.total_cost = (
                cfg.safety_weight * candidate.safety_cost
                + cfg.smoothness_weight * candidate.smoothness_cost
                + cfg.consistency_weight * candidate.consistency_cost
            )

        free = [candidate for candidate in candidates if candidate.feasible and not candidate.collision]
        emergency = not free
        if free:
            selected = min(free, key=lambda candidate: candidate.total_cost)
        else:
            feasible = [candidate for candidate in candidates if candidate.feasible]
            pool = feasible or candidates
            selected = min(
                pool,
                key=lambda candidate: (-candidate.collision_free_length_m, candidate.total_cost),
            )

        target_speed = self._target_speed(selected, current_s, emergency)
        self.previous_path = selected
        return PlanResult(
            selected=selected,
            candidates=candidates,
            target_speed_mps=target_speed,
            emergency=emergency,
            current_s=current_s,
            current_d=current_d,
            heading_error=heading_error,
        )


class OffRoadController:
    """Direct-control adapter around :class:`OffRoadPlanner`."""

    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self.planner = OffRoadPlanner(config)
        self.last_result: Optional[PlanResult] = None

    @property
    def config(self) -> PlannerConfig:
        return self.planner.config

    def update_config(self, **values) -> None:
        self.planner.update_config(**values)

    def set_base_frame(self, *args, **kwargs) -> None:
        self.planner.set_base_frame(*args, **kwargs)

    def reset_history(self) -> None:
        self.last_result = None
        self.planner.reset_history()

    def _pure_pursuit(self, pose: Sequence[float], speed_mps: float, path: CandidatePath) -> float:
        cfg = self.config
        pose_arr = np.asarray(pose, dtype=float).reshape(-1)
        lookahead = cfg.lookahead_min_m + cfg.lookahead_speed_gain_s * abs(speed_mps)
        path_distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(path.x), np.diff(path.y)))]
        target_index = int(np.searchsorted(path_distance, lookahead, side="left"))
        target_index = min(target_index, len(path.x) - 1)
        dx = float(path.x[target_index] - pose_arr[0])
        dy = float(path.y[target_index] - pose_arr[1])
        cy = math.cos(float(pose_arr[2]))
        sy = math.sin(float(pose_arr[2]))
        local_x = cy * dx + sy * dy
        local_y = -sy * dx + cy * dy
        distance_sq = max(local_x * local_x + local_y * local_y, _EPS)
        command_curvature = 2.0 * local_y / distance_sq
        steer = math.atan(cfg.wheelbase_m * command_curvature)
        return float(np.clip(steer, -cfg.max_steer_rad, cfg.max_steer_rad))

    def process(
        self,
        pose: Sequence[float],
        speed_mps: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> tuple[float, float]:
        result = self.planner.plan(
            pose, speed_mps, ranges, angle_min, angle_increment,
        )
        self.last_result = result
        steering = self._pure_pursuit(pose, speed_mps, result.selected)
        return result.target_speed_mps, steering

    def command_from_last(self, pose: Sequence[float], speed_mps: float) -> tuple[float, float]:
        """Track the last 20 Hz plan from a faster low-level control loop."""
        if self.last_result is None:
            return 0.0, 0.0
        steering = self._pure_pursuit(pose, speed_mps, self.last_result.selected)
        return self.last_result.target_speed_mps, steering
