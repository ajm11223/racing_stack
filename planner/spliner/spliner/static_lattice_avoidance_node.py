#!/usr/bin/env python3
"""Static-obstacle Frenet lattice planner.

This node intentionally lives beside ``static_avoidance_node.py`` so the
existing single-apex planner remains available.  It builds lateral apex layers,
promotes additional blocking obstacles, generates a spline for every complete
apex combination, and selects a collision-free path with the safety,
smoothness, and consistency costs proposed by Chu et al. (2012).
"""

from dataclasses import dataclass
from itertools import product
import copy
import math
import time
from typing import List, Sequence, Tuple

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from f110_msgs.msg import (
    BehaviorStrategy,
    Obstacle,
    ObstacleArray,
    OTWpntArray,
)

from spliner.static_avoidance_node import ObstacleSpliner


@dataclass(frozen=True)
class LatticeApexCandidate:
    """One admissible lateral apex candidate at an obstacle layer."""

    obstacle_id: int
    s: float
    d: float
    side: str


@dataclass
class LatticePathCandidate:
    """One obstacle-checked path generated from an apex combination."""

    apexes: Tuple[LatticeApexCandidate, ...]
    s: np.ndarray
    d: np.ndarray
    xy: np.ndarray


@dataclass
class LatticeExpansionResult:
    """Candidate state after collision-driven obstacle promotion terminates."""

    planning_obstacles: List[Obstacle]
    promoted_obstacle_ids: List[int]
    layers: List[List[LatticeApexCandidate]]
    combinations: List[Tuple[LatticeApexCandidate, ...]]
    generated_paths: List[LatticePathCandidate]
    collision_free_paths: List[LatticePathCandidate]


@dataclass(frozen=True)
class LatticePathScore:
    """Paper-inspired normalized costs for one feasible path."""

    safety: float
    smoothness: float
    consistency: float
    total: float
    max_abs_curvature: float
    minimum_clearance: float


class StaticLatticeAvoidancePlanner(ObstacleSpliner):
    """Generate adaptive left/right Frenet apex samples for up to 3 obstacles."""

    def __init__(self):
        # These defaults must exist before ObstacleSpliner.__init__ calls the
        # virtual parameter callback and creates the BehaviorStrategy subscriber.
        self.obstacles_in_interest: List[Obstacle] = []
        self.tracked_static_obstacles: List[Obstacle] = []
        self.behavior_state = ""
        self.lattice_d_resolution = 0.30
        self.lattice_track_boundary_margin = 0.225
        self.lattice_obstacle_boundary_margin = 0.30
        self.lattice_max_samples_per_side = 5
        self.lattice_min_free_width = 0.40
        self.lattice_max_obstacles = 3
        self.lattice_obstacle_horizon = 10.0
        self.lattice_longitudinal_margin = 0.30
        self.lattice_weight_safety = 0.55
        self.lattice_weight_smoothness = 0.30
        self.lattice_weight_consistency = 0.15
        self.lattice_safety_sigma = 0.25
        self.lattice_max_curvature = 1.28
        self.lattice_consistency_scale = 0.50
        self.lattice_use_legacy_output = False
        self.lattice_speed_from_curvature = True
        self.lattice_max_lat_acc = 4.5
        self.lattice_max_lon_dec = 5.0
        self.lattice_min_speed = 1.5
        # LiDAR bounds replace the corresponding global side when observed.
        # A side with no usable return falls back to the global corridor.
        self.lattice_use_lidar_bounds = False
        self.lattice_lidar_scan_timeout = 0.20
        self.lattice_lidar_min_range = 0.05
        self.lattice_lidar_max_range = 8.0
        self.lattice_lidar_offset_x = 0.0
        self.lattice_lidar_offset_y = 0.0
        self.lattice_lidar_yaw = 0.0
        self.lattice_lidar_wall_band = 0.80
        self.lattice_lidar_s_window = 0.35
        self.lattice_lidar_min_wall_points = 2
        self.lattice_lidar_bound_padding = 0.05
        self.lattice_lidar_path_clearance = 0.25
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.lidar_left_s = np.asarray([], dtype=float)
        self.lidar_left_d = np.asarray([], dtype=float)
        self.lidar_right_s = np.asarray([], dtype=float)
        self.lidar_right_d = np.asarray([], dtype=float)
        self.lidar_left_xy = np.empty((0, 2), dtype=float)
        self.lidar_right_xy = np.empty((0, 2), dtype=float)
        self.lidar_path_points_xy = np.empty((0, 2), dtype=float)
        self.lidar_path_tree = None
        self._lidar_interval_cache = {}
        self.last_candidate_layers: List[List[LatticeApexCandidate]] = []
        self.last_candidate_combinations: List[Tuple[LatticeApexCandidate, ...]] = []
        self.last_generated_paths: List[LatticePathCandidate] = []
        self.last_collision_free_paths: List[LatticePathCandidate] = []
        self.last_promoted_obstacle_ids: List[int] = []
        self.last_selected_path = None
        self.previous_selected_path = None
        self.last_selected_score = None
        self.last_path_rejections = {}

        super().__init__()

        self.name = "static_lattice_avoidance_planner"
        self.lattice_d_resolution = max(
            float(self.get_parameter("lattice_d_resolution").value),
            1.0e-3,
        )
        self.lattice_track_boundary_margin = max(
            0.0,
            float(self.get_parameter("lattice_track_boundary_margin").value),
        )
        self.lattice_obstacle_boundary_margin = max(
            0.0,
            float(self.get_parameter(
                "lattice_obstacle_boundary_margin"
            ).value),
        )
        self.lattice_max_samples_per_side = max(
            1,
            int(self.get_parameter("lattice_max_samples_per_side").value),
        )
        self.lattice_min_free_width = max(
            0.0,
            float(self.get_parameter("lattice_min_free_width").value),
        )
        self.lattice_max_obstacles = min(
            3,
            max(1, int(self.get_parameter("lattice_max_obstacles").value)),
        )
        self.lattice_obstacle_horizon = max(
            0.0,
            float(self.get_parameter("lattice_obstacle_horizon").value),
        )
        self.lattice_longitudinal_margin = max(
            0.0,
            float(self.get_parameter("lattice_longitudinal_margin").value),
        )
        self.lattice_weight_safety = max(
            0.0, float(self.get_parameter("lattice_weight_safety").value)
        )
        self.lattice_weight_smoothness = max(
            0.0, float(self.get_parameter("lattice_weight_smoothness").value)
        )
        self.lattice_weight_consistency = max(
            0.0, float(self.get_parameter("lattice_weight_consistency").value)
        )
        self.lattice_safety_sigma = max(
            1.0e-3, float(self.get_parameter("lattice_safety_sigma").value)
        )
        self.lattice_max_curvature = max(
            1.0e-3, float(self.get_parameter("lattice_max_curvature").value)
        )
        self.lattice_consistency_scale = max(
            1.0e-3,
            float(self.get_parameter("lattice_consistency_scale").value),
        )
        self.lattice_use_legacy_output = bool(
            self.get_parameter("lattice_use_legacy_output").value
        )
        self.lattice_speed_from_curvature = bool(
            self.get_parameter("lattice_speed_from_curvature").value
        )
        self.lattice_max_lat_acc = max(
            1.0e-2, float(self.get_parameter("lattice_max_lat_acc_mps2").value)
        )
        self.lattice_max_lon_dec = max(
            1.0e-2, float(self.get_parameter("lattice_max_lon_dec_mps2").value)
        )
        self.lattice_min_speed = max(
            0.0, float(self.get_parameter("lattice_min_speed_mps").value)
        )
        self.dyn_param_cb(self.get_parameters([
            "lattice_use_lidar_bounds",
            "lattice_lidar_scan_timeout_s",
            "lattice_lidar_min_range_m",
            "lattice_lidar_max_range_m",
            "lattice_lidar_offset_x_m",
            "lattice_lidar_offset_y_m",
            "lattice_lidar_yaw_rad",
            "lattice_lidar_wall_band_m",
            "lattice_lidar_s_window_m",
            "lattice_lidar_min_wall_points",
            "lattice_lidar_bound_padding_m",
            "lattice_lidar_path_clearance_m",
        ]))

        self.lattice_candidates_pub = self.create_publisher(
            MarkerArray,
            "/planner/avoidance/lattice_candidates",
            10,
        )
        self.lidar_bounds_pub = self.create_publisher(
            MarkerArray,
            "/planner/avoidance/lidar_track_bounds",
            10,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_cb,
            qos_profile_sensor_data,
        )
        # BehaviorStrategy currently exposes only the closest overtaking target.
        # Subscribe to the tracking array as well so one planning pass can build
        # candidate layers for every simultaneous static obstacle (up to three).
        self.create_subscription(
            ObstacleArray,
            "/tracking/obstacles",
            self.tracking_obstacles_cb,
            10,
        )
        self.get_logger().info(
            "[static_lattice_avoidance_planner] paper-based safety/smoothness/"
            "consistency path selection enabled"
        )

    def declare_all_parameters(self):
        """Declare legacy spline parameters plus lattice sampling parameters."""
        super().declare_all_parameters()
        self.declare_parameter("lattice_d_resolution", 0.30)
        self.declare_parameter("lattice_track_boundary_margin", 0.225)
        self.declare_parameter("lattice_obstacle_boundary_margin", 0.30)
        self.declare_parameter("lattice_max_samples_per_side", 5)
        self.declare_parameter("lattice_min_free_width", 0.40)
        self.declare_parameter("lattice_max_obstacles", 3)
        self.declare_parameter("lattice_obstacle_horizon", 10.0)
        self.declare_parameter("lattice_longitudinal_margin", 0.30)
        self.declare_parameter("lattice_weight_safety", 0.55)
        self.declare_parameter("lattice_weight_smoothness", 0.30)
        self.declare_parameter("lattice_weight_consistency", 0.15)
        self.declare_parameter("lattice_safety_sigma", 0.25)
        self.declare_parameter("lattice_max_curvature", 1.28)
        self.declare_parameter("lattice_consistency_scale", 0.50)
        self.declare_parameter("lattice_use_legacy_output", False)
        self.declare_parameter("lattice_speed_from_curvature", True)
        self.declare_parameter("lattice_max_lat_acc_mps2", 4.5)
        self.declare_parameter("lattice_max_lon_dec_mps2", 5.0)
        self.declare_parameter("lattice_min_speed_mps", 1.5)
        self.declare_parameter("lattice_use_lidar_bounds", False)
        self.declare_parameter("lattice_lidar_scan_timeout_s", 0.20)
        self.declare_parameter("lattice_lidar_min_range_m", 0.05)
        self.declare_parameter("lattice_lidar_max_range_m", 8.0)
        self.declare_parameter("lattice_lidar_offset_x_m", 0.0)
        self.declare_parameter("lattice_lidar_offset_y_m", 0.0)
        self.declare_parameter("lattice_lidar_yaw_rad", 0.0)
        self.declare_parameter("lattice_lidar_wall_band_m", 0.80)
        self.declare_parameter("lattice_lidar_s_window_m", 0.35)
        self.declare_parameter("lattice_lidar_min_wall_points", 2)
        self.declare_parameter("lattice_lidar_bound_padding_m", 0.05)
        self.declare_parameter("lattice_lidar_path_clearance_m", 0.25)

    def dyn_param_cb(self, params: List[Parameter]) -> SetParametersResult:
        """Apply lattice sampling updates and preserve legacy parameter handling."""
        for param in params:
            if param.name == "lattice_d_resolution":
                self.lattice_d_resolution = max(float(param.value), 1.0e-3)
            elif param.name == "lattice_track_boundary_margin":
                self.lattice_track_boundary_margin = max(
                    0.0, float(param.value)
                )
            elif param.name == "lattice_obstacle_boundary_margin":
                self.lattice_obstacle_boundary_margin = max(
                    0.0, float(param.value)
                )
            elif param.name == "lattice_max_samples_per_side":
                self.lattice_max_samples_per_side = max(1, int(param.value))
            elif param.name == "lattice_min_free_width":
                self.lattice_min_free_width = max(0.0, float(param.value))
            elif param.name == "lattice_max_obstacles":
                self.lattice_max_obstacles = min(3, max(1, int(param.value)))
            elif param.name == "lattice_obstacle_horizon":
                self.lattice_obstacle_horizon = max(0.0, float(param.value))
            elif param.name == "lattice_longitudinal_margin":
                self.lattice_longitudinal_margin = max(0.0, float(param.value))
            elif param.name == "lattice_weight_safety":
                self.lattice_weight_safety = max(0.0, float(param.value))
            elif param.name == "lattice_weight_smoothness":
                self.lattice_weight_smoothness = max(0.0, float(param.value))
            elif param.name == "lattice_weight_consistency":
                self.lattice_weight_consistency = max(0.0, float(param.value))
            elif param.name == "lattice_safety_sigma":
                self.lattice_safety_sigma = max(1.0e-3, float(param.value))
            elif param.name == "lattice_max_curvature":
                self.lattice_max_curvature = max(1.0e-3, float(param.value))
            elif param.name == "lattice_consistency_scale":
                self.lattice_consistency_scale = max(
                    1.0e-3, float(param.value)
                )
            elif param.name == "lattice_use_legacy_output":
                self.lattice_use_legacy_output = bool(param.value)
            elif param.name == "lattice_speed_from_curvature":
                self.lattice_speed_from_curvature = bool(param.value)
            elif param.name == "lattice_max_lat_acc_mps2":
                self.lattice_max_lat_acc = max(1.0e-2, float(param.value))
            elif param.name == "lattice_max_lon_dec_mps2":
                self.lattice_max_lon_dec = max(1.0e-2, float(param.value))
            elif param.name == "lattice_min_speed_mps":
                self.lattice_min_speed = max(0.0, float(param.value))
            elif (
                param.name == "lattice_use_lidar_bounds"
                or param.name.startswith("lattice_lidar_")
            ):
                self._apply_lidar_parameter(param.name, param.value)
        return super().dyn_param_cb(params)

    def _apply_lidar_parameter(self, name: str, value) -> None:
        """Validate one live LiDAR-bound parameter and clear derived caches."""
        if name == "lattice_use_lidar_bounds":
            self.lattice_use_lidar_bounds = bool(value)
        elif name == "lattice_lidar_scan_timeout_s":
            self.lattice_lidar_scan_timeout = max(0.01, float(value))
        elif name == "lattice_lidar_min_range_m":
            self.lattice_lidar_min_range = max(0.0, float(value))
            self.lattice_lidar_max_range = max(
                self.lattice_lidar_max_range,
                self.lattice_lidar_min_range + 0.01,
            )
        elif name == "lattice_lidar_max_range_m":
            self.lattice_lidar_max_range = max(
                self.lattice_lidar_min_range + 0.01,
                float(value),
            )
        elif name == "lattice_lidar_offset_x_m":
            self.lattice_lidar_offset_x = float(value)
        elif name == "lattice_lidar_offset_y_m":
            self.lattice_lidar_offset_y = float(value)
        elif name == "lattice_lidar_yaw_rad":
            self.lattice_lidar_yaw = float(value)
        elif name == "lattice_lidar_wall_band_m":
            self.lattice_lidar_wall_band = max(0.05, float(value))
        elif name == "lattice_lidar_s_window_m":
            self.lattice_lidar_s_window = max(0.05, float(value))
        elif name == "lattice_lidar_min_wall_points":
            self.lattice_lidar_min_wall_points = max(1, int(value))
        elif name == "lattice_lidar_bound_padding_m":
            self.lattice_lidar_bound_padding = max(0.0, float(value))
        elif name == "lattice_lidar_path_clearance_m":
            self.lattice_lidar_path_clearance = max(0.0, float(value))
        self._lidar_interval_cache = {}

    def behavior_cb(self, data: BehaviorStrategy):
        """Keep strategy targets as the authoritative active-obstacle seed."""
        self.behavior_state = str(data.state)
        self.obstacles_in_interest = [
            copy.deepcopy(obs)
            for obs in data.overtaking_targets
            if obs.is_static
        ]
        # Retain the base-class field for the temporary legacy output seam.
        self.obs_in_interest = (
            copy.deepcopy(self.obstacles_in_interest[0])
            if self.obstacles_in_interest
            else None
        )

    def tracking_obstacles_cb(self, data: ObstacleArray):
        """Cache all currently tracked static obstacles for multi-layer planning."""
        self.tracked_static_obstacles = [
            copy.deepcopy(obs)
            for obs in data.obstacles
            if obs.is_static
        ]

    def scan_cb(self, data: LaserScan):
        """Cache the newest scan; conversion is synchronized in the planning loop."""
        self.latest_scan = data
        self.latest_scan_received_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )

    def _clear_lidar_track_bounds(self) -> None:
        self.lidar_left_s = np.asarray([], dtype=float)
        self.lidar_left_d = np.asarray([], dtype=float)
        self.lidar_right_s = np.asarray([], dtype=float)
        self.lidar_right_d = np.asarray([], dtype=float)
        self.lidar_left_xy = np.empty((0, 2), dtype=float)
        self.lidar_right_xy = np.empty((0, 2), dtype=float)
        self.lidar_path_points_xy = np.empty((0, 2), dtype=float)
        self.lidar_path_tree = None
        self._lidar_interval_cache = {}

    def _refresh_lidar_track_bounds(self, gb_wpnts) -> bool:
        """Convert a fresh scan into conservative local Frenet wall samples."""
        self._clear_lidar_track_bounds()
        if not self.lattice_use_lidar_bounds or self.latest_scan is None:
            return False
        if self.latest_scan_received_sec is None:
            return False
        now_sec = self.get_clock().now().nanoseconds * 1.0e-9
        if now_sec - self.latest_scan_received_sec > self.lattice_lidar_scan_timeout:
            return False
        if (
            self.cur_x is None
            or self.cur_y is None
            or self.cur_yaw is None
            or self.cur_s is None
            or self.gb_max_s is None
            or getattr(self, "converter", None) is None
            or len(gb_wpnts) < 2
        ):
            return False

        scan = self.latest_scan
        ranges = np.asarray(scan.ranges, dtype=float).reshape(-1)
        angles = (
            float(scan.angle_min)
            + np.arange(ranges.size, dtype=float) * float(scan.angle_increment)
        )
        scan_min_range = float(getattr(scan, "range_min", 0.0))
        scan_max_range = float(getattr(scan, "range_max", math.inf))
        minimum_range = max(self.lattice_lidar_min_range, scan_min_range)
        maximum_range = min(self.lattice_lidar_max_range, scan_max_range)
        valid = (
            np.isfinite(ranges)
            & (ranges >= minimum_range)
            & (ranges < maximum_range)
        )
        if not np.any(valid):
            return False

        scan_x = ranges[valid] * np.cos(angles[valid])
        scan_y = ranges[valid] * np.sin(angles[valid])
        lidar_cos = math.cos(self.lattice_lidar_yaw)
        lidar_sin = math.sin(self.lattice_lidar_yaw)
        base_x = (
            self.lattice_lidar_offset_x
            + lidar_cos * scan_x
            - lidar_sin * scan_y
        )
        base_y = (
            self.lattice_lidar_offset_y
            + lidar_sin * scan_x
            + lidar_cos * scan_y
        )
        yaw_cos = math.cos(float(self.cur_yaw))
        yaw_sin = math.sin(float(self.cur_yaw))
        map_x = float(self.cur_x) + yaw_cos * base_x - yaw_sin * base_y
        map_y = float(self.cur_y) + yaw_sin * base_x + yaw_cos * base_y

        try:
            scan_s, scan_d = self.converter.get_frenet(map_x, map_y)
        except (RuntimeError, TypeError, ValueError, FloatingPointError):
            return False
        scan_s = np.asarray(scan_s, dtype=float).reshape(-1)
        scan_d = np.asarray(scan_d, dtype=float).reshape(-1)
        if scan_s.size != map_x.size or scan_d.size != map_x.size:
            return False

        signed_forward = (
            scan_s - float(self.cur_s) + self.gb_max_s / 2.0
        ) % self.gb_max_s - self.gb_max_s / 2.0
        horizon = min(
            float(self.lattice_obstacle_horizon),
            float(self.lattice_lidar_max_range) + 0.5,
        )
        local = (
            np.isfinite(scan_s)
            & np.isfinite(scan_d)
            & (signed_forward >= -0.50)
            & (signed_forward <= horizon)
        )
        if not np.any(local):
            return False

        scan_s = scan_s[local] % self.gb_max_s
        scan_d = scan_d[local]
        points_xy = np.column_stack((map_x[local], map_y[local]))
        self.lidar_path_points_xy = points_xy
        self.lidar_path_tree = cKDTree(points_xy)

        wpnt_dist = float(gb_wpnts[1].s_m - gb_wpnts[0].s_m)
        if not math.isfinite(wpnt_dist) or wpnt_dist <= 0.0:
            self._clear_lidar_track_bounds()
            return False
        indices = np.round(scan_s / wpnt_dist).astype(int) % len(gb_wpnts)
        fixed_left = np.asarray(
            [float(gb_wpnts[index].d_left) for index in indices],
            dtype=float,
        )
        fixed_right = np.asarray(
            [-float(gb_wpnts[index].d_right) for index in indices],
            dtype=float,
        )
        band = float(self.lattice_lidar_wall_band)
        left_mask = (
            (scan_d > 0.0)
            & np.isfinite(fixed_left)
            & (np.abs(scan_d - fixed_left) <= band)
        )
        right_mask = (
            (scan_d < 0.0)
            & np.isfinite(fixed_right)
            & (np.abs(scan_d - fixed_right) <= band)
        )
        self.lidar_left_s = scan_s[left_mask]
        self.lidar_left_d = scan_d[left_mask]
        self.lidar_left_xy = points_xy[left_mask]
        self.lidar_right_s = scan_s[right_mask]
        self.lidar_right_d = scan_d[right_mask]
        self.lidar_right_xy = points_xy[right_mask]
        return bool(np.any(left_mask) or np.any(right_mask))

    def _effective_track_interval(
        self,
        s_value: float,
        fixed_min: float,
        fixed_max: float,
    ) -> Tuple[float, float]:
        """Return measured LiDAR bounds, falling back per side to global widths."""
        if (
            not getattr(self, "lattice_use_lidar_bounds", False)
            or getattr(self, "gb_max_s", None) is None
        ):
            return float(fixed_min), float(fixed_max)

        cache_scale = max(float(self.lattice_lidar_s_window), 0.05)
        key = (
            int(round((float(s_value) % self.gb_max_s) / cache_scale)),
            round(float(fixed_min), 3),
            round(float(fixed_max), 3),
        )
        cached = self._lidar_interval_cache.get(key)
        if cached is not None:
            return cached

        lower = float(fixed_min)
        upper = float(fixed_max)
        padding = float(self.lattice_lidar_bound_padding)
        minimum_points = int(self.lattice_lidar_min_wall_points)

        if self.lidar_left_s.size:
            left_delta = (
                self.lidar_left_s - float(s_value) + self.gb_max_s / 2.0
            ) % self.gb_max_s - self.gb_max_s / 2.0
            left_values = self.lidar_left_d[
                np.abs(left_delta) <= self.lattice_lidar_s_window
            ]
            if left_values.size >= minimum_points:
                # Use the inner-side quantile: an approaching wall corner must
                # shrink the corridor before the rest of the wall catches up.
                measured_left = float(np.quantile(left_values, 0.20)) - padding
                upper = measured_left

        if self.lidar_right_s.size:
            right_delta = (
                self.lidar_right_s - float(s_value) + self.gb_max_s / 2.0
            ) % self.gb_max_s - self.gb_max_s / 2.0
            right_values = self.lidar_right_d[
                np.abs(right_delta) <= self.lattice_lidar_s_window
            ]
            if right_values.size >= minimum_points:
                measured_right = float(np.quantile(right_values, 0.80)) + padding
                lower = measured_right

        result = (float(lower), float(upper))
        self._lidar_interval_cache[key] = result
        return result

    def _lidar_path_residual_clearance(self, path: LatticePathCandidate) -> float:
        """Distance from a completed path to the current raw scan safety circle."""
        if (
            not getattr(self, "lattice_use_lidar_bounds", False)
            or getattr(self, "lidar_path_tree", None) is None
            or not hasattr(path, "xy")
        ):
            return math.inf
        xy = np.asarray(path.xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.size == 0:
            return -math.inf
        distances, _ = self.lidar_path_tree.query(xy, k=1)
        return float(np.min(distances) - self.lattice_lidar_path_clearance)

    def _lidar_bound_markers(self) -> MarkerArray:
        markers = self._delete_all_markers()
        stamp = self.get_clock().now().to_msg()
        for marker_id, points, red, green, blue in (
            (1, self.lidar_left_xy, 1.0, 0.1, 1.0),
            (2, self.lidar_right_xy, 0.1, 1.0, 1.0),
        ):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "lidar_track_bounds"
            marker.id = marker_id
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.a = 1.0
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.pose.orientation.w = 1.0
            marker.points = [
                Point(x=float(x), y=float(y), z=0.03)
                for x, y in points
            ]
            markers.markers.append(marker)
        return markers

    def loop(self):
        """Build and visualize lattice layers, then publish the integration output."""
        if self.measuring:
            start = time.perf_counter()

        gb_wpnts = self.gb_scaled_wpnts.wpnts
        self._refresh_lidar_track_bounds(gb_wpnts)
        self.lidar_bounds_pub.publish(self._lidar_bound_markers())
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        path_markers = MarkerArray()

        constraint_obstacles = self._ordered_planning_obstacles()
        initial_obstacles = self._initial_planning_obstacles(
            constraint_obstacles
        )
        if initial_obstacles:
            expansion = self._expand_planning_obstacles(
                initial_obstacles,
                constraint_obstacles,
                gb_wpnts,
            )
            self.last_candidate_layers = expansion.layers
            self.last_candidate_combinations = expansion.combinations
            self.last_generated_paths = expansion.generated_paths
            self.last_collision_free_paths = expansion.collision_free_paths
            self.last_promoted_obstacle_ids = expansion.promoted_obstacle_ids

            selected_path, selected_score = self._select_best_path(
                expansion.collision_free_paths,
                constraint_obstacles,
                gb_wpnts,
            )
            self.last_selected_path = selected_path
            self.last_selected_score = selected_score
            self.lattice_candidates_pub.publish(self._candidate_markers(
                expansion.layers,
                expansion.generated_paths,
                expansion.collision_free_paths,
                selected_path,
            ))

            counts = [len(layer) for layer in expansion.layers]
            planning_ids = [obs.id for obs in expansion.planning_obstacles]
            constraint_summary = [
                (
                    int(obs.id),
                    round(float(obs.s_center), 2),
                    round(float(obs.d_center), 2),
                )
                for obs in constraint_obstacles
            ]
            score_text = (
                "none"
                if selected_score is None
                else (
                    f"J={selected_score.total:.3f} "
                    f"(safe={selected_score.safety:.3f}, "
                    f"smooth={selected_score.smoothness:.3f}, "
                    f"consistent={selected_score.consistency:.3f})"
                )
            )
            self.get_logger().info(
                f"[{self.name}] planning_obstacles={planning_ids}, "
                f"constraints={constraint_summary}, "
                f"tracked_static={len(self.tracked_static_obstacles)}, "
                f"promoted={expansion.promoted_obstacle_ids}, "
                f"layer_candidates={counts}, "
                f"combinations={len(expansion.combinations)}, "
                f"collision_free={len(expansion.collision_free_paths)}, "
                f"selected={score_text}, "
                f"rejected={self.last_path_rejections}",
                throttle_duration_sec=2,
            )

            if self.lattice_use_legacy_output:
                wpnts, path_markers = self.do_spline(
                    obs=copy.deepcopy(initial_obstacles[0]),
                    gb_wpnts=gb_wpnts,
                )
            elif selected_path is not None:
                wpnts, path_markers = self._selected_path_output(
                    selected_path,
                    gb_wpnts,
                )
                planned_ids = sorted({
                    int(obstacle.id)
                    for obstacle in expansion.planning_obstacles
                    if int(obstacle.id) >= 0
                })
                wpnts.ot_line = "lattice:" + ",".join(
                    str(obstacle_id) for obstacle_id in planned_ids
                )
                self.previous_selected_path = selected_path
        else:
            self.last_candidate_layers = []
            self.last_candidate_combinations = []
            self.last_generated_paths = []
            self.last_collision_free_paths = []
            self.last_promoted_obstacle_ids = []
            self.last_selected_path = None
            self.previous_selected_path = None
            self.last_selected_score = None
            self.last_path_rejections = {}
            self.lattice_candidates_pub.publish(self._delete_all_markers())
            path_markers = self._delete_all_markers()

        if self.measuring:
            self.latency_pub.publish(
                self._latency_message(time.perf_counter() - start)
            )
        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(path_markers)

    def _latency_message(self, latency: float):
        # Imported lazily to keep the lattice-specific imports small.
        from std_msgs.msg import Float32

        return Float32(data=float(latency))

    def _ordered_planning_obstacles(self) -> List[Obstacle]:
        """Return at most three forward static obstacles ordered by lap distance."""
        if self.cur_s is None or self.gb_max_s is None:
            return []
        # A strategy target activates the planner before overtaking.  Once the
        # state machine is already in OVERTAKE, keep replanning from tracking even
        # if it temporarily publishes no overtaking target; this is the seam that
        # handles a newly visible obstacle while the car is on an avoidance path.
        if not self.obstacles_in_interest and self.behavior_state != "OVERTAKE":
            return []

        # Merge tracking and BehaviorStrategy by obstacle id.  Strategy wins for
        # matching ids because it is the state machine's active target.  For an
        # invalid/negative id, include quantized s/d in the key to avoid collapsing
        # distinct detections.
        merged = {}
        for obs in self.tracked_static_obstacles:
            merged[self._obstacle_key(obs)] = obs
        for obs in self.obstacles_in_interest:
            merged[self._obstacle_key(obs)] = obs

        forward = []
        for obs in merged.values():
            distance = (obs.s_center - self.cur_s) % self.gb_max_s
            # This is a sensing horizon, independent of the path's post-apex
            # return distance.  Do not claim collision knowledge beyond LiDAR.
            horizon = min(
                float(self.lattice_obstacle_horizon),
                self.gb_max_s / 2.0,
            )
            if 0.5 <= distance <= horizon:
                forward.append((distance, obs))
        forward.sort(key=lambda item: item[0])
        return [
            copy.deepcopy(obs)
            for _, obs in forward[: self.lattice_max_obstacles]
        ]

    def _initial_planning_obstacles(
        self,
        constraint_obstacles: Sequence[Obstacle],
    ) -> List[Obstacle]:
        """Select active targets and every obstacle sharing their s layer."""
        if not self.obstacles_in_interest and self.behavior_state != "OVERTAKE":
            return []

        constraints_by_key = {
            self._obstacle_key(obs): obs
            for obs in constraint_obstacles
        }
        initial = []
        seen = set()
        for target in self.obstacles_in_interest:
            key = self._obstacle_key(target)
            if key in seen:
                continue
            obstacle = constraints_by_key.get(key, target)
            distance = (obstacle.s_center - self.cur_s) % self.gb_max_s
            if 0.5 <= distance <= self.gb_max_s / 2.0:
                initial.append(copy.deepcopy(obstacle))
                seen.add(key)

        initial.sort(
            key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
        )
        # During OVERTAKE the original strategy target may already be behind the
        # car (or the state machine may emit an empty target list).  Seed a new
        # planning pass with the nearest currently tracked forward obstacle so a
        # newly entering LiDAR detection can trigger a complete replan.
        if not initial and self.behavior_state == "OVERTAKE" and constraint_obstacles:
            initial.append(copy.deepcopy(constraint_obstacles[0]))

        # Obstacles whose inflated longitudinal collision intervals overlap
        # cannot become separate spline control points: equal (or nearly equal)
        # s values violate the strictly increasing CubicHermiteSpline domain.
        # Include those peers immediately so candidate generation can subtract
        # all of their lateral occupied intervals in one cross-section layer.
        initial_keys = {
            self._obstacle_key(obstacle)
            for obstacle in initial
        }
        changed = True
        while changed and len(initial) < self.lattice_max_obstacles:
            changed = False
            for obstacle in constraint_obstacles:
                key = self._obstacle_key(obstacle)
                if key in initial_keys:
                    continue
                if any(
                    self._obstacles_share_longitudinal_layer(
                        obstacle,
                        selected,
                    )
                    for selected in initial
                ):
                    initial.append(copy.deepcopy(obstacle))
                    initial_keys.add(key)
                    changed = True
                    if len(initial) >= self.lattice_max_obstacles:
                        break

        initial.sort(
            key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
        )
        return initial[: self.lattice_max_obstacles]

    @staticmethod
    def _obstacle_key(obs: Obstacle):
        obstacle_id = int(obs.id)
        if obstacle_id >= 0:
            return ("id", obstacle_id)
        return (
            "pose",
            round(float(obs.s_center), 2),
            round(float(obs.d_center), 2),
        )

    def _build_candidate_layers(
        self,
        obstacles: Sequence[Obstacle],
        gb_wpnts,
    ) -> List[List[LatticeApexCandidate]]:
        """Create one candidate layer per longitudinal obstacle group."""
        if len(gb_wpnts) < 2:
            return []

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        if not math.isfinite(wpnt_dist) or wpnt_dist <= 0.0:
            return []

        groups = self._group_longitudinal_obstacles(
            obstacles[: self.lattice_max_obstacles]
        )
        return [
            self._candidates_for_obstacle_group(group, gb_wpnts, wpnt_dist)
            for group in groups
        ]

    def _group_longitudinal_obstacles(
        self,
        obstacles: Sequence[Obstacle],
    ) -> List[List[Obstacle]]:
        """Group connected overlapping longitudinal collision intervals."""
        ordered = sorted(
            obstacles,
            key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s,
        )
        groups: List[List[Obstacle]] = []
        for obstacle in ordered:
            if groups and any(
                self._obstacles_share_longitudinal_layer(obstacle, member)
                for member in groups[-1]
            ):
                groups[-1].append(obstacle)
            else:
                groups.append([obstacle])
        return groups

    def _obstacles_share_longitudinal_layer(
        self,
        first: Obstacle,
        second: Obstacle,
    ) -> bool:
        """Whether two inflated obstacle s intervals overlap on the lap."""
        delta_s = (
            float(first.s_center)
            - float(second.s_center)
            + self.gb_max_s / 2.0
        ) % self.gb_max_s - self.gb_max_s / 2.0
        first_half = (
            self._obstacle_longitudinal_half_extent(first)
            + self.lattice_longitudinal_margin
        )
        second_half = (
            self._obstacle_longitudinal_half_extent(second)
            + self.lattice_longitudinal_margin
        )
        return abs(delta_s) <= first_half + second_half + 1.0e-6

    def _candidates_for_obstacle_group(
        self,
        obstacles: Sequence[Obstacle],
        gb_wpnts,
        wpnt_dist: float,
    ) -> List[LatticeApexCandidate]:
        """Sample the complement of every inflated obstacle in one s layer."""
        if not obstacles:
            return []

        # Use the mean unwrapped longitudinal position as the single spline
        # control point for the group. Exact same-s obstacles retain that s.
        forward_distances = np.asarray([
            (float(obstacle.s_center) - self.cur_s) % self.gb_max_s
            for obstacle in obstacles
        ])
        apex_s = float(
            (self.cur_s + float(np.mean(forward_distances))) % self.gb_max_s
        )

        # Track bounds are intersected across all members, making the vehicle
        # center interval conservative when track width changes through a group.
        # Candidate-space construction intentionally uses the raw track and
        # obstacle boundaries; hard-safety margins are checked on the completed
        # spline later.
        track_mins = []
        track_maxs = []
        for obstacle in obstacles:
            obs_idx = int(round(obstacle.s_center / wpnt_dist)) % len(gb_wpnts)
            gb_wp = gb_wpnts[obs_idx]
            effective_min, effective_max = self._effective_track_interval(
                float(obstacle.s_center),
                -float(gb_wp.d_right),
                float(gb_wp.d_left),
            )
            track_mins.append(effective_min)
            track_maxs.append(effective_max)
        track_min = max(track_mins)
        track_max = min(track_maxs)
        if not (math.isfinite(track_min) and math.isfinite(track_max)):
            return []
        if track_max <= track_min:
            return []

        occupied = []
        for obstacle in obstacles:
            obs_min, obs_max = self._obstacle_lateral_bounds(obstacle)
            lo = max(track_min, obs_min)
            hi = min(track_max, obs_max)
            if hi >= lo:
                occupied.append((lo, hi))
        occupied.sort(key=lambda interval: interval[0])

        merged = []
        for lo, hi in occupied:
            if merged and lo <= merged[-1][1] + 1.0e-9:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        free_intervals = []
        cursor = track_min
        for lo, hi in merged:
            if lo > cursor:
                free_intervals.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < track_max:
            free_intervals.append((cursor, track_max))

        group_center = float(np.mean([
            float(obstacle.d_center)
            for obstacle in obstacles
        ]))
        representative_id = int(obstacles[0].id)
        candidates = []
        for d_min, d_max in free_intervals:
            for d in self._sample_free_interval(d_min, d_max):
                candidates.append(LatticeApexCandidate(
                    representative_id,
                    apex_s,
                    d,
                    "right" if d < group_center else "left",
                ))
        return candidates

    def _candidates_for_obstacle(
        self,
        obs: Obstacle,
        gb_wp,
    ) -> List[LatticeApexCandidate]:
        """Subtract the raw obstacle interval from the raw track interval."""
        track_min, track_max = self._effective_track_interval(
            float(obs.s_center),
            -float(gb_wp.d_right),
            float(gb_wp.d_left),
        )
        if not (math.isfinite(track_min) and math.isfinite(track_max)):
            return []
        if track_max <= track_min:
            return []

        raw_obs_min, raw_obs_max = self._obstacle_lateral_bounds(obs)
        obs_min = raw_obs_min
        obs_max = raw_obs_max

        # Candidate positions are vehicle-center positions. Clamp each free
        # interval to the raw track interval; final paths still go through the
        # existing obstacle- and track-clearance checks.
        right_lo = track_min
        right_hi = min(obs_min, track_max)
        left_lo = max(obs_max, track_min)
        left_hi = track_max

        right_ds = self._sample_free_interval(right_lo, right_hi)
        left_ds = self._sample_free_interval(left_lo, left_hi)

        candidates = [
            LatticeApexCandidate(obs.id, float(obs.s_center), d, "right")
            for d in right_ds
        ]
        candidates.extend(
            LatticeApexCandidate(obs.id, float(obs.s_center), d, "left")
            for d in left_ds
        )
        return candidates

    @staticmethod
    def _obstacle_lateral_bounds(obs: Obstacle) -> Tuple[float, float]:
        """Return conservative obstacle d bounds, using size only as a fallback."""
        half_size = max(float(obs.size), 0.0) / 2.0
        center_min = float(obs.d_center) - half_size
        center_max = float(obs.d_center) + half_size

        d_right = float(obs.d_right)
        d_left = float(obs.d_left)
        explicit_bounds_valid = (
            math.isfinite(d_right)
            and math.isfinite(d_left)
            and abs(d_left - d_right) > 1.0e-6
        )
        if not explicit_bounds_valid:
            return center_min, center_max

        return (
            min(d_right, d_left, center_min),
            max(d_right, d_left, center_max),
        )

    def _sample_free_interval(self, d_min: float, d_max: float) -> List[float]:
        """Place 1/3/5 centered samples according to the raw free-space width."""
        width = float(d_max - d_min)
        if not math.isfinite(width) or width < self.lattice_min_free_width:
            return []

        if width < 1.0:
            count = 1
            spacing = self.lattice_d_resolution
        elif width < 2.0:
            count = 3
            spacing = 0.20
        else:
            count = 5
            spacing = 0.30
        count = min(self.lattice_max_samples_per_side, count)

        # Center the sample group in the free interval. Adjacent candidates
        # stay 0.20 m apart for three samples and 0.30 m apart for five.
        center = (float(d_min) + float(d_max)) / 2.0
        start = center - (count - 1) * spacing / 2.0
        return [
            float(start + index * spacing)
            for index in range(count)
        ]

    @staticmethod
    def _candidate_combinations(
        layers: Sequence[Sequence[LatticeApexCandidate]],
    ) -> List[Tuple[LatticeApexCandidate, ...]]:
        """Enumerate complete apex choices; an empty layer means no feasible path."""
        if not layers or any(len(layer) == 0 for layer in layers):
            return []
        return list(product(*layers))

    def _expand_planning_obstacles(
        self,
        initial_obstacles: Sequence[Obstacle],
        constraint_obstacles: Sequence[Obstacle],
        gb_wpnts,
    ) -> LatticeExpansionResult:
        """Promote a constraint obstacle when it blocks any current path.

        Every collider that rejects at least one candidate receives its own apex
        layer in the same expansion step.  Thus obstacles 2 and 3 are evaluated
        independently against the paths made for obstacle 1, then promoted
        together before the lattice is regenerated.  Promotion remains capped by
        the configured three planning obstacles.
        """
        planning = self._deduplicate_obstacles(initial_obstacles)
        planning.sort(
            key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
        )
        promoted_ids: List[int] = []
        layers: List[List[LatticeApexCandidate]] = []
        combinations: List[Tuple[LatticeApexCandidate, ...]] = []
        generated_paths: List[LatticePathCandidate] = []
        collision_free: List[LatticePathCandidate] = []

        while planning and len(planning) <= self.lattice_max_obstacles:
            layers = self._build_candidate_layers(planning, gb_wpnts)
            combinations = self._candidate_combinations(layers)
            generated_paths = []
            collision_free = []
            collision_counts = {}

            for apexes in combinations:
                path = self._build_candidate_samples(apexes, gb_wpnts)
                if path is None:
                    continue
                generated_paths.append(path)
                collision_path = self._path_with_output_tail(path, gb_wpnts)
                collisions = self._find_path_collisions(
                    collision_path,
                    constraint_obstacles,
                )
                if not collisions:
                    if self._lidar_path_residual_clearance(
                        collision_path
                    ) >= 0.0:
                        collision_free.append(path)
                    continue

                planning_keys = {
                    self._obstacle_key(obs)
                    for obs in planning
                }
                for obstacle in collisions:
                    key = self._obstacle_key(obstacle)
                    if key not in planning_keys:
                        collision_counts[key] = collision_counts.get(key, 0) + 1

            if len(planning) >= self.lattice_max_obstacles:
                break
            if not generated_paths or not collision_counts:
                break

            promoted_obstacles = self._obstacles_to_promote(
                collision_counts,
                constraint_obstacles,
                planning,
            )
            if not promoted_obstacles:
                break

            planning.extend(
                copy.deepcopy(obstacle)
                for obstacle in promoted_obstacles
            )
            planning = self._deduplicate_obstacles(planning)
            planning.sort(
                key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
            )
            promoted_ids.extend(
                int(obstacle.id)
                for obstacle in promoted_obstacles
            )

        return LatticeExpansionResult(
            planning_obstacles=planning,
            promoted_obstacle_ids=promoted_ids,
            layers=layers,
            combinations=combinations,
            generated_paths=generated_paths,
            collision_free_paths=collision_free,
        )

    def _build_candidate_samples(
        self,
        apexes: Tuple[LatticeApexCandidate, ...],
        gb_wpnts,
    ):
        """Build a monotonic Frenet d(s) path through one apex combination."""
        if not apexes or len(gb_wpnts) < 2:
            return None
        if self.cur_s is None or self.cur_d is None:
            return None

        wpnt_dist = float(gb_wpnts[1].s_m - gb_wpnts[0].s_m)
        if not math.isfinite(wpnt_dist) or wpnt_dist <= 0.0:
            return None

        control_s = [float(self.cur_s)]
        control_d = [float(self.cur_d)]
        for apex in apexes:
            forward_distance = (apex.s - self.cur_s) % self.gb_max_s
            unwrapped_s = float(self.cur_s + forward_distance)
            if unwrapped_s <= control_s[-1] + 1.0e-6:
                return None
            control_s.append(unwrapped_s)
            control_d.append(float(apex.d))

        # Preserve the legacy spliner rule and variable naming: use the
        # first obstacle's pre_dist as the post_dist after the final apex,
        # bounded by the configured post min/max distances.
        pre_dist = (
            apexes[0].s - self.cur_s
        ) % self.gb_max_s
        post_dist = min(
            min(
                max(float(pre_dist), float(self.post_min_dist)),
                float(self.post_max_dist),
            ),
            self.gb_max_s / 2.0,
        )
        if not math.isfinite(post_dist) or post_dist <= 0.0:
            return None
        control_s.append(control_s[-1] + post_dist)
        control_d.append(0.0)

        control_s_array = np.asarray(control_s, dtype=float)
        control_d_array = np.asarray(control_d, dtype=float)
        derivatives = np.zeros_like(control_d_array)
        lateral_spline = CubicHermiteSpline(
            control_s_array,
            control_d_array,
            derivatives,
        )

        sample_count = max(
            2,
            int(math.ceil(
                (control_s_array[-1] - control_s_array[0]) / wpnt_dist
            )) + 1,
        )
        sample_s_unwrapped = np.linspace(
            control_s_array[0],
            control_s_array[-1],
            sample_count,
        )
        sample_s = sample_s_unwrapped % self.gb_max_s
        sample_d = np.asarray(lateral_spline(sample_s_unwrapped), dtype=float)
        cartesian = np.asarray(
            self.converter.get_cartesian(sample_s, sample_d),
            dtype=float,
        )
        if cartesian.shape != (2, sample_count):
            return None
        xy = cartesian.T
        if not (
            np.isfinite(sample_s).all()
            and np.isfinite(sample_d).all()
            and np.isfinite(xy).all()
        ):
            return None

        return LatticePathCandidate(
            apexes=apexes,
            s=sample_s,
            d=sample_d,
            xy=xy,
        )

    def _path_with_output_tail(
        self,
        path: LatticePathCandidate,
        gb_wpnts,
    ) -> LatticePathCandidate:
        """Append the same GB tail published by ``_selected_path_output``.

        Candidate splines end shortly after their final apex, while the
        controller output continues for another 100 global waypoints.  Collision
        checks must cover that continuation or a downstream obstacle can be
        absent from every collision count and never receive an apex layer.

        Test doubles and malformed inputs are returned unchanged so expansion
        can still reject them through its existing checks.
        """
        if (
            len(gb_wpnts) < 2
            or not hasattr(path, "s")
            or not hasattr(path, "d")
            or not hasattr(path, "xy")
        ):
            return path

        path_s = np.asarray(path.s, dtype=float)
        path_d = np.asarray(path.d, dtype=float)
        path_xy = np.asarray(path.xy, dtype=float)
        if (
            path_s.size == 0
            or path_d.shape != path_s.shape
            or path_xy.shape != (path_s.size, 2)
        ):
            return path

        wpnt_dist = float(gb_wpnts[1].s_m - gb_wpnts[0].s_m)
        if not math.isfinite(wpnt_dist) or wpnt_dist <= 0.0:
            return path

        tail_start = int(round(float(path_s[-1]) / wpnt_dist)) % len(gb_wpnts)
        tail = [
            gb_wpnts[(tail_start + offset) % len(gb_wpnts)]
            for offset in range(1, min(101, len(gb_wpnts)))
        ]
        if not tail:
            return path

        tail_s = np.asarray([wpnt.s_m for wpnt in tail], dtype=float)
        tail_d = np.asarray([wpnt.d_m for wpnt in tail], dtype=float)
        tail_xy = np.asarray(
            [(wpnt.x_m, wpnt.y_m) for wpnt in tail],
            dtype=float,
        )
        if not (
            np.isfinite(tail_s).all()
            and np.isfinite(tail_d).all()
            and np.isfinite(tail_xy).all()
        ):
            return path

        return LatticePathCandidate(
            apexes=path.apexes,
            s=np.concatenate((path_s, tail_s)),
            d=np.concatenate((path_d, tail_d)),
            xy=np.vstack((path_xy, tail_xy)),
        )

    def _find_path_collisions(
        self,
        path: LatticePathCandidate,
        obstacles: Sequence[Obstacle],
    ) -> List[Obstacle]:
        """Return every obstacle intersecting the sampled vehicle-center path."""
        clearance = self.lattice_obstacle_boundary_margin
        collisions = []
        seen = set()

        for obstacle in obstacles:
            obs_min, obs_max = self._obstacle_lateral_bounds(obstacle)
            obs_min -= clearance
            obs_max += clearance
            half_s = (
                self._obstacle_longitudinal_half_extent(obstacle)
                + self.lattice_longitudinal_margin
            )

            delta_s = (
                path.s
                - float(obstacle.s_center)
                + self.gb_max_s / 2.0
            ) % self.gb_max_s - self.gb_max_s / 2.0
            longitudinal_overlap = np.abs(delta_s) <= half_s
            lateral_overlap = (path.d >= obs_min) & (path.d <= obs_max)
            if np.any(longitudinal_overlap & lateral_overlap):
                key = self._obstacle_key(obstacle)
                if key not in seen:
                    collisions.append(obstacle)
                    seen.add(key)
        return collisions

    def _obstacle_longitudinal_half_extent(self, obstacle: Obstacle) -> float:
        """Estimate the obstacle half-length in s, robust to lap wraparound."""
        half_extent = max(float(obstacle.size), 0.0) / 2.0
        s_start = float(getattr(obstacle, "s_start", obstacle.s_center))
        s_end = float(getattr(obstacle, "s_end", obstacle.s_center))
        if math.isfinite(s_start) and math.isfinite(s_end):
            span = (s_end - s_start) % self.gb_max_s
            if 1.0e-6 < span < self.gb_max_s / 2.0:
                half_extent = max(half_extent, span / 2.0)
        return half_extent

    def _obstacles_to_promote(
        self,
        collision_counts,
        constraint_obstacles: Sequence[Obstacle],
        planning_obstacles: Sequence[Obstacle],
    ) -> List[Obstacle]:
        """Return every new obstacle that blocks at least one candidate path."""
        planning_keys = {
            self._obstacle_key(obs)
            for obs in planning_obstacles
        }
        promotable = [
            obs
            for obs in constraint_obstacles
            if self._obstacle_key(obs) not in planning_keys
            and self._obstacle_key(obs) in collision_counts
        ]
        promotable.sort(
            key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
        )
        available_slots = max(
            0,
            self.lattice_max_obstacles - len(planning_obstacles),
        )
        return promotable[:available_slots]

    def _select_best_path(
        self,
        paths: Sequence[LatticePathCandidate],
        obstacles: Sequence[Obstacle],
        gb_wpnts,
    ):
        """Select the minimum paper-inspired cost among feasible paths.

        Collision and vehicle-curvature limits are hard constraints, as in the
        paper.  The remaining candidates are ranked by normalized physical
        safety, squared-curvature smoothness, and previous-path consistency.
        """
        rejections = {
            "collision": 0,
            "map": 0,
            "geometry": 0,
            "curvature": 0,
            "clearance": 0,
        }
        worst_curvature = None
        worst_clearance = None
        scored = []
        for path in paths:
            collision_path = self._path_with_output_tail(path, gb_wpnts)
            if self._find_path_collisions(collision_path, obstacles):
                rejections["collision"] += 1
                continue
            if hasattr(self, "map_filter") and any(
                not self.map_filter.is_point_inside(float(x), float(y))
                for x, y in path.xy
            ):
                rejections["map"] += 1
                continue

            geometry = self._path_geometry(path.xy)
            if geometry is None:
                rejections["geometry"] += 1
                continue
            _, curvature, path_length = geometry
            max_abs_curvature = float(np.max(np.abs(curvature)))
            if max_abs_curvature > self.lattice_max_curvature:
                rejections["curvature"] += 1
                worst_curvature = (
                    max_abs_curvature
                    if worst_curvature is None
                    else max(worst_curvature, max_abs_curvature)
                )
                continue

            minimum_clearance = self._minimum_path_clearance(
                collision_path,
                obstacles,
                gb_wpnts,
            )
            if not math.isfinite(minimum_clearance) or minimum_clearance <= 0.0:
                rejections["clearance"] += 1
                if math.isfinite(minimum_clearance):
                    worst_clearance = (
                        minimum_clearance
                        if worst_clearance is None
                        else min(worst_clearance, minimum_clearance)
                    )
                continue

            safety = math.exp(
                -0.5
                * (minimum_clearance / self.lattice_safety_sigma) ** 2
            )
            if path_length <= 1.0e-6:
                continue
            smoothness = float(np.trapezoid(
                (curvature / self.lattice_max_curvature) ** 2,
                self._path_arc_lengths(path.xy),
            ) / path_length)
            consistency = self._consistency_cost(
                path,
                self.previous_selected_path,
            )
            total = (
                self.lattice_weight_safety * safety
                + self.lattice_weight_smoothness * smoothness
                + self.lattice_weight_consistency * consistency
            )
            score = LatticePathScore(
                safety=float(safety),
                smoothness=float(smoothness),
                consistency=float(consistency),
                total=float(total),
                max_abs_curvature=max_abs_curvature,
                minimum_clearance=float(minimum_clearance),
            )
            scored.append((score, path))

        self.last_path_rejections = {
            **rejections,
            "max_kappa": (
                None if worst_curvature is None else round(worst_curvature, 3)
            ),
            "min_clearance": (
                None if worst_clearance is None else round(worst_clearance, 3)
            ),
        }
        if not scored:
            return None, None
        score, path = min(
            scored,
            key=lambda item: (
                item[0].total,
                item[0].safety,
                item[0].smoothness,
                tuple(abs(apex.d) for apex in item[1].apexes),
            ),
        )
        return path, score

    @staticmethod
    def _path_arc_lengths(xy: np.ndarray):
        """Return cumulative Cartesian arc length for an open sampled path."""
        points = np.asarray(xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            return np.asarray([], dtype=float)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    @classmethod
    def _path_geometry(cls, xy: np.ndarray):
        """Return tangent heading, curvature, and length for sampled XY points."""
        points = np.asarray(xy, dtype=float)
        arc = cls._path_arc_lengths(points)
        if len(arc) < 2 or not np.isfinite(arc).all():
            return None
        if np.any(np.diff(arc) <= 1.0e-6):
            return None

        edge_order = 2 if len(points) >= 3 else 1
        dx = np.gradient(points[:, 0], arc, edge_order=edge_order)
        dy = np.gradient(points[:, 1], arc, edge_order=edge_order)
        heading = np.unwrap(np.arctan2(dy, dx))
        if len(points) < 3:
            curvature = np.zeros(len(points), dtype=float)
        else:
            ddx = np.gradient(dx, arc, edge_order=edge_order)
            ddy = np.gradient(dy, arc, edge_order=edge_order)
            denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-9)
            curvature = (dx * ddy - dy * ddx) / denominator
        if not (np.isfinite(heading).all() and np.isfinite(curvature).all()):
            return None
        return heading, curvature, float(arc[-1])

    def _minimum_path_clearance(
        self,
        path: LatticePathCandidate,
        obstacles: Sequence[Obstacle],
        gb_wpnts,
    ) -> float:
        """Return minimum residual clearance to inflated obstacles and track."""
        if len(gb_wpnts) < 2:
            return float("-inf")
        wpnt_dist = float(gb_wpnts[1].s_m - gb_wpnts[0].s_m)
        if not math.isfinite(wpnt_dist) or wpnt_dist <= 0.0:
            return float("-inf")

        track_margin = self.lattice_track_boundary_margin
        obstacle_margin = self.lattice_obstacle_boundary_margin
        indices = np.round(path.s / wpnt_dist).astype(int) % len(gb_wpnts)
        track_intervals = [
            self._effective_track_interval(
                float(s_value),
                -float(gb_wpnts[index].d_right),
                float(gb_wpnts[index].d_left),
            )
            for s_value, index in zip(path.s, indices)
        ]
        track_min = np.asarray(
            [lower + track_margin for lower, _ in track_intervals]
        )
        track_max = np.asarray(
            [upper - track_margin for _, upper in track_intervals]
        )
        minimum = float(np.min(np.minimum(path.d - track_min, track_max - path.d)))
        minimum = min(minimum, self._lidar_path_residual_clearance(path))

        for obstacle in obstacles:
            obs_min, obs_max = self._obstacle_lateral_bounds(obstacle)
            obs_min -= obstacle_margin
            obs_max += obstacle_margin
            half_s = (
                self._obstacle_longitudinal_half_extent(obstacle)
                + self.lattice_longitudinal_margin
            )
            delta_s = (
                path.s
                - float(obstacle.s_center)
                + self.gb_max_s / 2.0
            ) % self.gb_max_s - self.gb_max_s / 2.0
            mask = np.abs(delta_s) <= half_s
            if not np.any(mask):
                continue
            lateral = path.d[mask]
            gaps = np.where(
                lateral < obs_min,
                obs_min - lateral,
                np.where(lateral > obs_max, lateral - obs_max, 0.0),
            )
            minimum = min(minimum, float(np.min(gaps)))
        return minimum

    def _consistency_cost(
        self,
        path: LatticePathCandidate,
        previous_path,
    ) -> float:
        """Mean lateral difference over the current/previous overlap (paper Eq. 18)."""
        if previous_path is None:
            return 0.0

        current_s = (np.asarray(path.s) - self.cur_s) % self.gb_max_s
        previous_s = (
            np.asarray(previous_path.s) - self.cur_s
        ) % self.gb_max_s
        current_mask = current_s <= self.gb_max_s / 2.0
        previous_mask = previous_s <= self.gb_max_s / 2.0
        if np.count_nonzero(current_mask) < 2 or np.count_nonzero(previous_mask) < 2:
            return 0.0

        current_order = np.argsort(current_s[current_mask])
        previous_order = np.argsort(previous_s[previous_mask])
        cs = current_s[current_mask][current_order]
        cd = np.asarray(path.d)[current_mask][current_order]
        ps = previous_s[previous_mask][previous_order]
        pd = np.asarray(previous_path.d)[previous_mask][previous_order]
        overlap_start = max(float(cs[0]), float(ps[0]))
        overlap_end = min(float(cs[-1]), float(ps[-1]))
        if overlap_end <= overlap_start + 1.0e-6:
            return 0.0

        sample_count = max(
            2,
            int(math.ceil(
                (overlap_end - overlap_start) / self.lattice_d_resolution
            )) + 1,
        )
        sample_s = np.linspace(overlap_start, overlap_end, sample_count)
        difference = np.abs(
            np.interp(sample_s, cs, cd) - np.interp(sample_s, ps, pd)
        )
        return float(np.clip(
            np.trapezoid(difference, sample_s)
            / (overlap_end - overlap_start)
            / self.lattice_consistency_scale,
            0.0,
            1.0,
        ))

    def _curvature_limited_speeds(self, gb_speeds, gb_kappa, curvature, xy):
        """Cap the avoidance path's speed by its own curvature, then make the
        result reachable by braking.

        The raceline velocity profile was optimised for the raceline's curvature.
        An avoidance spline is strictly sharper than the line it replaces, so
        copying that profile onto it demands lateral acceleration the tyres
        cannot deliver -- on a narrow section the car simply runs wide.

        The lateral budget is taken from the raceline itself
        (``v_gb^2 * |kappa_gb|``, floored at the configured value) rather than
        from the configured value alone. On this stack the raceline profile
        already exceeds ggv's ay_max by a large factor in places, so a fixed
        budget would cut speed wherever the *raceline* is tight, obstacle or
        not -- which is the speed-sector behaviour this is meant to avoid.
        Calibrating against the raceline makes the cap a no-op wherever the
        path curvature matches the raceline's, so only the extra curvature the
        avoidance adds costs anything.

        This is deliberately not a track-wide speed sector: nothing is published
        when there is no obstacle, so a clean lap keeps the full raceline
        profile.

        Two passes:
          1. pointwise grip limit  v <= sqrt(a_budget / |kappa|)
          2. backward pass         v[i] <= sqrt(v[i+1]^2 + 2*a_dec*ds)
        Pass 2 is what makes pass 1 useful. A cap applied only at the apex is
        decorative: the car arrives there still carrying raceline speed because
        nothing told it to start braking earlier.
        """
        gb_speeds = np.asarray(gb_speeds, dtype=float)
        if not self.lattice_speed_from_curvature or gb_speeds.size == 0:
            return gb_speeds

        # Clip at the existing hard curvature limit: anything above it would
        # already have been rejected, so a numerical spike from the finite
        # difference in _path_geometry cannot collapse the whole profile.
        kappa = np.minimum(
            np.abs(np.asarray(curvature, dtype=float)),
            self.lattice_max_curvature,
        )
        budget = np.maximum(
            self.lattice_max_lat_acc,
            gb_speeds ** 2 * np.abs(np.asarray(gb_kappa, dtype=float)),
        )
        grip = np.sqrt(budget / np.maximum(kappa, 1.0e-3))
        speeds = np.minimum(gb_speeds, np.maximum(grip, self.lattice_min_speed))

        arc = self._path_arc_lengths(xy)
        if arc.size != speeds.size:
            return speeds
        for i in range(speeds.size - 2, -1, -1):
            ds = float(arc[i + 1] - arc[i])
            if ds <= 0.0:
                continue
            reachable = math.sqrt(
                speeds[i + 1] ** 2 + 2.0 * self.lattice_max_lon_dec * ds
            )
            if reachable < speeds[i]:
                speeds[i] = reachable
        return speeds

    def _selected_path_output(self, path: LatticePathCandidate, gb_wpnts):
        """Wrap the selected spline and a raceline tail as controller waypoints."""
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        markers = self._delete_all_markers()

        geometry = self._path_geometry(path.xy)
        if geometry is None or len(gb_wpnts) < 2:
            return wpnts, markers
        heading, curvature, _ = geometry
        wpnt_dist = float(gb_wpnts[1].s_m - gb_wpnts[0].s_m)

        gb_idx = [
            int(round(float(s) / wpnt_dist)) % len(gb_wpnts) for s in path.s
        ]
        gb_speeds = np.asarray([float(gb_wpnts[i].vx_mps) for i in gb_idx])
        gb_kappa = np.asarray([float(gb_wpnts[i].kappa_radpm) for i in gb_idx])
        speeds = self._curvature_limited_speeds(
            gb_speeds, gb_kappa, curvature, path.xy
        )

        for index, (s, d, xy) in enumerate(zip(path.s, path.d, path.xy)):
            velocity = float(speeds[index])
            wpnts.wpnts.append(self.xyv_to_wpnts(
                s=float(s),
                d=float(d),
                x=float(xy[0]),
                y=float(xy[1]),
                v=velocity,
                psi=float(heading[index]),
                kappa=float(curvature[index]),
                wpnts=wpnts,
            ))
            markers.markers.append(self.xyv_to_markers(
                x=float(xy[0]),
                y=float(xy[1]),
                v=velocity,
                mrks=markers,
            ))

        tail_start = int(round(float(path.s[-1]) / wpnt_dist)) % len(gb_wpnts)
        for offset in range(1, min(101, len(gb_wpnts))):
            gb_wpnt = gb_wpnts[(tail_start + offset) % len(gb_wpnts)]
            wpnts.wpnts.append(self.xyv_to_wpnts(
                s=gb_wpnt.s_m,
                d=gb_wpnt.d_m,
                x=gb_wpnt.x_m,
                y=gb_wpnt.y_m,
                v=gb_wpnt.vx_mps,
                psi=gb_wpnt.psi_rad,
                kappa=gb_wpnt.kappa_radpm,
                wpnts=wpnts,
            ))

        selected_side = path.apexes[0].side
        wpnts.ot_side = selected_side
        wpnts.ot_line = "lattice"
        wpnts.side_switch = self.last_ot_side != selected_side
        wpnts.last_switch_time = self.last_switch_time
        if wpnts.side_switch:
            self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = selected_side
        return wpnts, markers

    def _deduplicate_obstacles(
        self,
        obstacles: Sequence[Obstacle],
    ) -> List[Obstacle]:
        unique = {}
        for obstacle in obstacles:
            unique[self._obstacle_key(obstacle)] = obstacle
        return [copy.deepcopy(obstacle) for obstacle in unique.values()]

    def _candidate_markers(
        self,
        layers: Sequence[Sequence[LatticeApexCandidate]],
        generated_paths: Sequence[LatticePathCandidate] = (),
        collision_free_paths: Sequence[LatticePathCandidate] = (),
        selected_path=None,
    ) -> MarkerArray:
        """Visualize apex samples and every generated candidate spline.

        Candidate splines are red when they collide, cyan after collision
        filtering, and thick green when selected by the final cost function.
        Keeping final-filter rejects visible is intentional: it explains the
        common ``collision_free=1, selected=none`` case directly in RViz.
        """
        markers = self._delete_all_markers()
        marker_id = 0
        for layer_idx, layer in enumerate(layers):
            for candidate in layer:
                xy = self.converter.get_cartesian(
                    np.asarray([candidate.s]),
                    np.asarray([candidate.d]),
                )
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"lattice_layer_{layer_idx}"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(xy[0, 0])
                marker.pose.position.y = float(xy[1, 0])
                marker.pose.position.z = 0.12
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.18
                marker.scale.y = 0.18
                marker.scale.z = 0.18
                marker.color.a = 0.95
                if candidate.side == "left":
                    marker.color.r = 1.0
                    marker.color.g = 0.85
                    marker.color.b = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 0.85
                    marker.color.b = 1.0
                markers.markers.append(marker)
                marker_id += 1

        collision_free_ids = {id(path) for path in collision_free_paths}
        for path_index, path in enumerate(generated_paths):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lattice_candidate_splines"
            marker.id = path_index
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.035

            if path is selected_path:
                marker.scale.x = 0.075
                marker.color.r = 0.05
                marker.color.g = 1.0
                marker.color.b = 0.10
                marker.color.a = 1.0
            elif id(path) in collision_free_ids:
                marker.color.r = 0.0
                marker.color.g = 0.75
                marker.color.b = 1.0
                marker.color.a = 0.80
            else:
                marker.color.r = 1.0
                marker.color.g = 0.10
                marker.color.b = 0.05
                marker.color.a = 0.55

            for x, y in path.xy:
                marker.points.append(Point(
                    x=float(x),
                    y=float(y),
                    z=0.075,
                ))
            markers.markers.append(marker)
        return markers

    def _delete_all_markers(self) -> MarkerArray:
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        return markers


def main(args=None):
    rclpy.init(args=args)
    planner = StaticLatticeAvoidancePlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
