#!/usr/bin/env python3
"""Dynamic Window Approach (DWA) for the emergency direct-control state.

This implements the algorithm in ``08_Local_path_planning_-_practice_DWA.pdf``:

* construct velocity and steering dynamic windows from the current command;
* roll out constant ``(v, delta)`` controls with a kinematic bicycle model;
* reject trajectories whose asymmetric vehicle footprint intersects a scan point;
* minimize normalized heading, obstacle-distance and velocity costs.

Planning and collision checks are performed in the vehicle frame.  The selected
trajectories are transformed into ``map`` only for goal selection and RViz output,
so a map/localization offset cannot move a LiDAR wall away from the vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np


_EPS = 1.0e-9


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _sample_interval(low: float, high: float, resolution: float, *required: float) -> np.ndarray:
    """Sample a closed interval and retain its boundaries/current command."""
    if high < low:
        low = high = 0.5 * (low + high)
    count = max(1, int(math.floor((high - low) / resolution)))
    values = low + np.arange(count + 1, dtype=float) * resolution
    values = values[values <= high + _EPS]
    values = np.r_[values, high]
    for value in required:
        if low - _EPS <= value <= high + _EPS:
            values = np.r_[values, float(np.clip(value, low, high))]
    return np.unique(np.round(values, decimals=10))


@dataclass(frozen=True)
class DWAConfig:
    """Vehicle limits and the three cost terms shown in the practice PDF."""

    min_speed_mps: float = 0.0
    max_speed_mps: float = 2.0
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 4.0
    max_steer_rad: float = 0.47
    max_steer_rate_radps: float = 3.0
    velocity_resolution_mps: float = 0.10
    steer_resolution_rad: float = 0.04
    predict_time_s: float = 1.5
    prediction_dt_s: float = 0.10
    dynamic_window_dt_s: float = 0.05
    wheelbase_m: float = 0.33
    goal_lookahead_m: float = 2.5
    heading_weight: float = 2.0
    obstacle_weight: float = 4.0
    velocity_weight: float = 1.0
    safety_radius_m: float = 0.35
    obstacle_cost_scale_m: float = 0.20
    vehicle_front_m: float = 0.45
    vehicle_rear_m: float = 0.10
    vehicle_width_m: float = 0.31
    collision_margin_m: float = 0.06
    scan_min_range_m: float = 0.03
    scan_max_range_m: float = 8.0
    lidar_offset_x_m: float = 0.0
    lidar_offset_y_m: float = 0.0
    lidar_yaw_rad: float = 0.0

    def validated(self) -> "DWAConfig":
        positive = {
            "max_speed_mps": self.max_speed_mps,
            "max_accel_mps2": self.max_accel_mps2,
            "max_decel_mps2": self.max_decel_mps2,
            "max_steer_rad": self.max_steer_rad,
            "max_steer_rate_radps": self.max_steer_rate_radps,
            "velocity_resolution_mps": self.velocity_resolution_mps,
            "steer_resolution_rad": self.steer_resolution_rad,
            "predict_time_s": self.predict_time_s,
            "prediction_dt_s": self.prediction_dt_s,
            "dynamic_window_dt_s": self.dynamic_window_dt_s,
            "wheelbase_m": self.wheelbase_m,
            "goal_lookahead_m": self.goal_lookahead_m,
            "obstacle_cost_scale_m": self.obstacle_cost_scale_m,
            "vehicle_front_m": self.vehicle_front_m,
            "vehicle_rear_m": self.vehicle_rear_m,
            "vehicle_width_m": self.vehicle_width_m,
            "scan_max_range_m": self.scan_max_range_m,
        }
        bad = [name for name, value in positive.items()
               if not np.isfinite(value) or value <= 0.0]
        if bad:
            raise ValueError(f"DWA parameters must be positive: {', '.join(bad)}")

        nonnegative = {
            "min_speed_mps": self.min_speed_mps,
            "heading_weight": self.heading_weight,
            "obstacle_weight": self.obstacle_weight,
            "velocity_weight": self.velocity_weight,
            "safety_radius_m": self.safety_radius_m,
            "collision_margin_m": self.collision_margin_m,
            "scan_min_range_m": self.scan_min_range_m,
        }
        bad = [name for name, value in nonnegative.items()
               if not np.isfinite(value) or value < 0.0]
        if bad:
            raise ValueError(f"DWA parameters must be nonnegative: {', '.join(bad)}")
        if self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("DWA min_speed_mps must be smaller than max_speed_mps")
        if self.prediction_dt_s > self.predict_time_s:
            raise ValueError("DWA prediction_dt_s must not exceed predict_time_s")
        if self.scan_min_range_m >= self.scan_max_range_m:
            raise ValueError("DWA scan_min_range_m must be smaller than scan_max_range_m")
        return self


@dataclass
class DWACandidate:
    index: int
    speed_mps: float
    steering_rad: float
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    collision: bool
    collision_free_distance_m: float
    min_clearance_m: float
    heading_cost: float
    obstacle_cost: float
    velocity_cost: float
    total_cost: float = math.inf


@dataclass
class DWAResult:
    selected: DWACandidate
    candidates: list[DWACandidate]
    target_speed_mps: float
    steering_rad: float
    emergency: bool
    goal_x_m: float
    goal_y_m: float
    dynamic_window: tuple[float, float, float, float]


class DWAPlanner:
    """Pure DWA planner with no ROS dependencies."""

    def __init__(self, config: Optional[DWAConfig] = None) -> None:
        self.config = (config or DWAConfig()).validated()
        self._reference_path: Optional[np.ndarray] = None

    def update_config(self, **values) -> None:
        self.config = replace(self.config, **values).validated()

    def set_reference_path(self, x: Sequence[float], y: Sequence[float]) -> None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size != y_arr.size or x_arr.size < 2:
            raise ValueError("DWA reference path needs equally sized x/y arrays with at least two points")
        path = np.column_stack((x_arr, y_arr))
        if not np.all(np.isfinite(path)):
            raise ValueError("DWA reference path contains non-finite coordinates")
        self._reference_path = path

    def dynamic_window(self, speed_mps: float, steering_rad: float) -> tuple[float, float, float, float]:
        cfg = self.config
        speed = float(np.clip(speed_mps, cfg.min_speed_mps, cfg.max_speed_mps))
        steer = float(np.clip(steering_rad, -cfg.max_steer_rad, cfg.max_steer_rad))
        speed_min = max(cfg.min_speed_mps, speed - cfg.max_decel_mps2 * cfg.dynamic_window_dt_s)
        speed_max = min(cfg.max_speed_mps, speed + cfg.max_accel_mps2 * cfg.dynamic_window_dt_s)
        steer_step = cfg.max_steer_rate_radps * cfg.dynamic_window_dt_s
        steer_min = max(-cfg.max_steer_rad, steer - steer_step)
        steer_max = min(cfg.max_steer_rad, steer + steer_step)
        return speed_min, speed_max, steer_min, steer_max

    def _goal(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one map-frame goal and the same point in the current vehicle frame."""
        cfg = self.config
        if self._reference_path is None:
            local = np.array([cfg.goal_lookahead_m, 0.0], dtype=float)
            cy, sy = math.cos(pose[2]), math.sin(pose[2])
            world = np.array([
                pose[0] + cy * local[0] - sy * local[1],
                pose[1] + sy * local[0] + cy * local[1],
            ])
            return world, local

        path = self._reference_path
        nearest = int(np.argmin(np.sum((path - pose[:2]) ** 2, axis=1)))
        travelled = 0.0
        goal_index = nearest
        for _ in range(len(path)):
            nxt = (goal_index + 1) % len(path)
            travelled += float(np.linalg.norm(path[nxt] - path[goal_index]))
            goal_index = nxt
            if travelled >= cfg.goal_lookahead_m:
                break
        world = path[goal_index].copy()
        dx, dy = world - pose[:2]
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        local = np.array([cy * dx + sy * dy, -sy * dx + cy * dy])
        return world, local

    def _scan_points(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> np.ndarray:
        cfg = self.config
        ranges_arr = np.asarray(ranges, dtype=float).reshape(-1)
        angles = float(angle_min) + np.arange(ranges_arr.size) * float(angle_increment)
        valid = (
            np.isfinite(ranges_arr)
            & (ranges_arr >= cfg.scan_min_range_m)
            & (ranges_arr < cfg.scan_max_range_m)
        )
        if not np.any(valid):
            return np.empty((0, 2), dtype=float)
        scan_x = ranges_arr[valid] * np.cos(angles[valid])
        scan_y = ranges_arr[valid] * np.sin(angles[valid])
        cy, sy = math.cos(cfg.lidar_yaw_rad), math.sin(cfg.lidar_yaw_rad)
        vehicle_x = cfg.lidar_offset_x_m + cy * scan_x - sy * scan_y
        vehicle_y = cfg.lidar_offset_y_m + sy * scan_x + cy * scan_y
        return np.column_stack((vehicle_x, vehicle_y))

    def _predict(self, speed_mps: float, steering_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        count = int(math.ceil(cfg.predict_time_s / cfg.prediction_dt_s)) + 1
        x = np.zeros(count, dtype=float)
        y = np.zeros(count, dtype=float)
        yaw = np.zeros(count, dtype=float)
        yaw_rate = speed_mps * math.tan(steering_rad) / cfg.wheelbase_m
        for index in range(1, count):
            x[index] = x[index - 1] + speed_mps * math.cos(yaw[index - 1]) * cfg.prediction_dt_s
            y[index] = y[index - 1] + speed_mps * math.sin(yaw[index - 1]) * cfg.prediction_dt_s
            yaw[index] = yaw[index - 1] + yaw_rate * cfg.prediction_dt_s
        return x, y, yaw

    @staticmethod
    def _surface_clearance(
        local_x: np.ndarray,
        local_y: np.ndarray,
        front: float,
        rear: float,
        half_width: float,
    ) -> np.ndarray:
        outside_x = np.maximum.reduce((local_x - front, -rear - local_x, np.zeros_like(local_x)))
        outside_y = np.maximum(np.abs(local_y) - half_width, 0.0)
        return np.hypot(outside_x, outside_y)

    def _collision_metrics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yaw: np.ndarray,
        speed_mps: float,
        obstacles: np.ndarray,
    ) -> tuple[bool, float, float]:
        cfg = self.config
        if obstacles.size == 0:
            return False, speed_mps * cfg.predict_time_s, math.inf

        front_hard = cfg.vehicle_front_m + cfg.collision_margin_m
        rear_hard = cfg.vehicle_rear_m + cfg.collision_margin_m
        half_hard = 0.5 * cfg.vehicle_width_m + cfg.collision_margin_m
        min_clearance = math.inf
        for index, (px, py, psi) in enumerate(zip(x, y, yaw)):
            dx = obstacles[:, 0] - px
            dy = obstacles[:, 1] - py
            cy, sy = math.cos(psi), math.sin(psi)
            local_x = cy * dx + sy * dy
            local_y = -sy * dx + cy * dy
            hard_hit = (
                (local_x <= front_hard)
                & (local_x >= -rear_hard)
                & (np.abs(local_y) <= half_hard)
            )
            clearance = self._surface_clearance(
                local_x, local_y,
                cfg.vehicle_front_m, cfg.vehicle_rear_m,
                0.5 * cfg.vehicle_width_m,
            )
            min_clearance = min(min_clearance, float(np.min(clearance)))
            if np.any(hard_hit):
                free_distance = max(0.0, (index - 1) * speed_mps * cfg.prediction_dt_s)
                return True, free_distance, min_clearance
        return False, speed_mps * cfg.predict_time_s, min_clearance

    @staticmethod
    def _to_map(
        pose: np.ndarray,
        x_local: np.ndarray,
        y_local: np.ndarray,
        yaw_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        x_map = pose[0] + cy * x_local - sy * y_local
        y_map = pose[1] + sy * x_local + cy * y_local
        yaw_map = np.asarray([_wrap_angle(pose[2] + value) for value in yaw_local])
        return x_map, y_map, yaw_map

    @staticmethod
    def _normalized(values: np.ndarray) -> np.ndarray:
        low = float(np.min(values))
        high = float(np.max(values))
        if high - low <= _EPS:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    def plan(
        self,
        pose: Sequence[float],
        speed_mps: float,
        steering_rad: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> DWAResult:
        cfg = self.config
        pose_arr = np.asarray(pose, dtype=float).reshape(-1)
        if pose_arr.size < 3 or not np.all(np.isfinite(pose_arr[:3])):
            raise ValueError("DWA pose must contain finite x, y and yaw")
        if not np.isfinite(speed_mps) or not np.isfinite(steering_rad):
            raise ValueError("DWA speed and steering must be finite")

        pose_arr = pose_arr[:3]
        goal_map, goal_local = self._goal(pose_arr)
        obstacles = self._scan_points(ranges, angle_min, angle_increment)
        window = self.dynamic_window(speed_mps, steering_rad)
        speeds = _sample_interval(
            window[0], window[1], cfg.velocity_resolution_mps,
            float(np.clip(speed_mps, window[0], window[1])),
        )
        steers = _sample_interval(
            window[2], window[3], cfg.steer_resolution_rad,
            float(np.clip(steering_rad, window[2], window[3])), 0.0,
        )

        candidates: list[DWACandidate] = []
        for candidate_speed in speeds:
            for candidate_steer in steers:
                x_local, y_local, yaw_local = self._predict(candidate_speed, candidate_steer)
                collision, free_distance, clearance = self._collision_metrics(
                    x_local, y_local, yaw_local, candidate_speed, obstacles,
                )
                goal_bearing = math.atan2(
                    goal_local[1] - y_local[-1],
                    goal_local[0] - x_local[-1],
                )
                heading_cost = abs(_wrap_angle(goal_bearing - yaw_local[-1])) / math.pi
                if math.isinf(clearance):
                    obstacle_cost = 0.0
                else:
                    penetration = max(0.0, cfg.safety_radius_m - clearance)
                    obstacle_cost = math.expm1(min(20.0, penetration / cfg.obstacle_cost_scale_m))
                velocity_cost = (
                    cfg.max_speed_mps - candidate_speed
                ) / (cfg.max_speed_mps - cfg.min_speed_mps)
                x_map, y_map, yaw_map = self._to_map(
                    pose_arr, x_local, y_local, yaw_local,
                )
                candidates.append(DWACandidate(
                    index=len(candidates),
                    speed_mps=float(candidate_speed),
                    steering_rad=float(candidate_steer),
                    x=x_map,
                    y=y_map,
                    yaw=yaw_map,
                    collision=collision,
                    collision_free_distance_m=float(free_distance),
                    min_clearance_m=float(clearance),
                    heading_cost=float(heading_cost),
                    obstacle_cost=float(obstacle_cost),
                    velocity_cost=float(velocity_cost),
                ))

        free = [candidate for candidate in candidates if not candidate.collision]
        emergency = not free
        if free:
            heading = self._normalized(np.asarray([item.heading_cost for item in free]))
            obstacle = self._normalized(np.asarray([item.obstacle_cost for item in free]))
            velocity = self._normalized(np.asarray([item.velocity_cost for item in free]))
            for item, h_cost, o_cost, v_cost in zip(free, heading, obstacle, velocity):
                item.total_cost = float(
                    cfg.heading_weight * h_cost
                    + cfg.obstacle_weight * o_cost
                    + cfg.velocity_weight * v_cost
                )
            selected = min(
                free,
                key=lambda item: (item.total_cost, -item.min_clearance_m, -item.speed_mps),
            )
            target_speed = selected.speed_mps
            target_steer = selected.steering_rad
        else:
            # The PDF initializes the optimum command to zero. Keep the most
            # collision-free rollout selected only so RViz still explains why.
            selected = max(
                candidates,
                key=lambda item: (item.collision_free_distance_m, item.min_clearance_m),
            )
            target_speed = 0.0
            target_steer = 0.0

        return DWAResult(
            selected=selected,
            candidates=candidates,
            target_speed_mps=float(target_speed),
            steering_rad=float(target_steer),
            emergency=emergency,
            goal_x_m=float(goal_map[0]),
            goal_y_m=float(goal_map[1]),
            dynamic_window=window,
        )


class DWAController:
    """State-owning adapter for the controller manager's direct-control seam."""

    def __init__(self, config: Optional[DWAConfig] = None) -> None:
        self.planner = DWAPlanner(config)
        self.last_result: Optional[DWAResult] = None
        self._last_command = (0.0, 0.0)

    @property
    def config(self) -> DWAConfig:
        return self.planner.config

    def update_config(self, **values) -> None:
        self.planner.update_config(**values)

    def set_reference_path(self, x: Sequence[float], y: Sequence[float]) -> None:
        self.planner.set_reference_path(x, y)

    def reset_history(self) -> None:
        self.last_result = None
        self._last_command = (0.0, 0.0)

    def process(
        self,
        pose: Sequence[float],
        speed_mps: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> tuple[float, float]:
        result = self.planner.plan(
            pose, speed_mps, self._last_command[1], ranges,
            angle_min, angle_increment,
        )
        self.last_result = result
        self._last_command = (result.target_speed_mps, result.steering_rad)
        return self._last_command

    def command_from_last(self) -> tuple[float, float]:
        return self._last_command
