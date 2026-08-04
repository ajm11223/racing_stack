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
from scipy.interpolate import BPoly
from geometry_msgs.msg import Point
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
        self.lattice_d_resolution_three_samples = 0.20
        self.lattice_d_resolution = 0.30
        self.lattice_track_boundary_margin = 0.225
        self.lattice_obstacle_boundary_margin = 0.30
        self.lattice_max_samples_per_side = 3
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
        self.lattice_d_resolution_three_samples = max(
            float(self.get_parameter(
                "lattice_d_resolution_three_samples"
            ).value),
            1.0e-3,
        )
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

        self.lattice_candidates_pub = self.create_publisher(
            MarkerArray,
            "/planner/avoidance/lattice_candidates",
            10,
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
        self.declare_parameter(
            "lattice_d_resolution_three_samples",
            0.20,
        )
        self.declare_parameter("lattice_d_resolution", 0.30)
        self.declare_parameter("lattice_track_boundary_margin", 0.225)
        self.declare_parameter("lattice_obstacle_boundary_margin", 0.30)
        self.declare_parameter("lattice_max_samples_per_side", 3)
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

    def dyn_param_cb(self, params: List[Parameter]) -> SetParametersResult:
        """Apply lattice sampling updates and preserve legacy parameter handling."""
        for param in params:
            if param.name == "lattice_d_resolution_three_samples":
                self.lattice_d_resolution_three_samples = max(
                    float(param.value),
                    1.0e-3,
                )
            elif param.name == "lattice_d_resolution":
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
        return super().dyn_param_cb(params)

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

    def loop(self):
        """Build and visualize lattice layers, then publish the integration output."""
        if self.measuring:
            start = time.perf_counter()

        gb_wpnts = self.gb_scaled_wpnts.wpnts
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
        # s values violate the strictly increasing lattice control stations.
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
        """Sample raw connected free corridors in one longitudinal layer."""
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

        # Intersect raw track bounds across every member because track width can
        # change across one longitudinal layer.
        track_mins = []
        track_maxs = []
        for obstacle in obstacles:
            obs_idx = int(round(obstacle.s_center / wpnt_dist)) % len(gb_wpnts)
            gb_wp = gb_wpnts[obs_idx]
            track_mins.append(-float(gb_wp.d_right))
            track_maxs.append(float(gb_wp.d_left))
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
        """Sample raw obstacle-to-track corridors; hard filters keep margins."""
        track_min = -float(gb_wp.d_right)
        track_max = float(gb_wp.d_left)
        if not (math.isfinite(track_min) and math.isfinite(track_max)):
            return []
        if track_max <= track_min:
            return []

        obs_min, obs_max = self._obstacle_lateral_bounds(obs)

        # Candidate positions are vehicle-centre positions in the raw corridor.
        # The later collision/track hard filters still apply configured margins.
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
        """Place 0/1/3/5 midpoint-centred samples from the raw corridor width."""
        width = float(d_max - d_min)
        if (
            not math.isfinite(width)
            or width + 1.0e-9 < self.lattice_min_free_width
        ):
            return []

        if width < 1.0 - 1.0e-9:
            count = 1
        elif width < 2.0 - 1.0e-9:
            count = 3
        else:
            count = 5

        center = (float(d_min) + float(d_max)) / 2.0
        half_count = count // 2
        resolution = (
            self.lattice_d_resolution_three_samples
            if count == 3
            else self.lattice_d_resolution
        )
        return [
            float(
                center
                + (index - half_count) * resolution
            )
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
        """Promote a constraint obstacle only when every current path collides.

        A collision affecting only some paths rejects those paths.  If at least
        one collision-free path remains, no new apex layer is introduced.  When
        all paths are blocked, the unavoidable/most frequent nearest collider is
        promoted and the lattice is regenerated, up to three planning obstacles.
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
                collisions = self._find_path_collisions(
                    path,
                    constraint_obstacles,
                )
                if not collisions:
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

            if collision_free:
                break
            if len(planning) >= self.lattice_max_obstacles:
                break
            if not generated_paths or not collision_counts:
                break

            promoted = self._select_obstacle_to_promote(
                collision_counts,
                constraint_obstacles,
                planning,
                len(generated_paths),
            )
            if promoted is None:
                break

            planning.append(copy.deepcopy(promoted))
            planning = self._deduplicate_obstacles(planning)
            planning.sort(
                key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
            )
            promoted_ids.append(int(promoted.id))

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
        """Build a Cartesian Hermite path through one apex combination.

        Match the legacy static spliner at the vehicle end: seed the spline at
        the measured Cartesian pose and constrain its first tangent to the
        vehicle heading. The remaining control-point tangents follow the
        global trajectory heading.
        """
        if not apexes or len(gb_wpnts) < 2:
            return None
        if self.cur_s is None or self.cur_d is None:
            return None
        vehicle_pose = np.asarray(
            [self.cur_x, self.cur_y, self.cur_yaw],
            dtype=float,
        )
        if not np.isfinite(vehicle_pose).all():
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

        converted = np.asarray(
            self.converter.get_cartesian(
                control_s_array[1:] % self.gb_max_s,
                control_d_array[1:],
            ),
            dtype=float,
        )
        if converted.shape != (2, len(control_s_array) - 1):
            return None
        control_xy = np.vstack((vehicle_pose[:2], converted.T))

        control_indices = (
            np.round((control_s_array[1:] % self.gb_max_s) / wpnt_dist)
            .astype(int)
            % len(gb_wpnts)
        )
        tangents = [[math.cos(self.cur_yaw), math.sin(self.cur_yaw)]]
        tangents.extend(
            [
                math.cos(float(gb_wpnts[index].psi_rad)),
                math.sin(float(gb_wpnts[index].psi_rad)),
            ]
            for index in control_indices
        )
        tangents = np.asarray(tangents, dtype=float) * float(self.spline_scale)

        chord_lengths = np.linalg.norm(np.diff(control_xy, axis=0), axis=1)
        spline_parameter = np.concatenate(([0.0], np.cumsum(chord_lengths)))
        spline_length = float(spline_parameter[-1])
        if (
            not np.isfinite(control_xy).all()
            or not np.isfinite(tangents).all()
            or not np.isfinite(spline_length)
            or spline_length <= 0.0
            or spline_length > self.gb_max_s
            or np.any(chord_lengths <= 1.0e-6)
        ):
            return None

        sample_count = max(
            2,
            int(math.ceil(spline_length / wpnt_dist)) + 1,
        )
        sample_parameter = np.linspace(0.0, spline_length, sample_count)
        xy = np.empty((sample_count, 2), dtype=float)
        for dimension in range(2):
            derivatives = [
                [control_xy[index, dimension], tangents[index, dimension]]
                for index in range(len(control_xy))
            ]
            xy[:, dimension] = BPoly.from_derivatives(
                spline_parameter,
                derivatives,
            )(sample_parameter)

        sample_s, sample_d = self.converter.get_frenet(xy[:, 0], xy[:, 1])
        sample_s = np.asarray(sample_s, dtype=float) % self.gb_max_s
        sample_d = np.asarray(sample_d, dtype=float)
        if sample_s.shape != (sample_count,) or sample_d.shape != (sample_count,):
            return None
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
            lateral_overlap = (
                (path.d >= obs_min - 1.0e-9)
                & (path.d <= obs_max + 1.0e-9)
            )
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

    def _select_obstacle_to_promote(
        self,
        collision_counts,
        constraint_obstacles: Sequence[Obstacle],
        planning_obstacles: Sequence[Obstacle],
        generated_paths: int,
    ):
        """Choose an unavoidable collider, otherwise the most frequent nearest one."""
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
        if not promotable:
            return None

        unavoidable = [
            obs
            for obs in promotable
            if collision_counts[self._obstacle_key(obs)] >= generated_paths
        ]
        if unavoidable:
            return min(
                unavoidable,
                key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s,
            )

        return min(
            promotable,
            key=lambda obs: (
                -collision_counts[self._obstacle_key(obs)],
                (obs.s_center - self.cur_s) % self.gb_max_s,
            ),
        )

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
            if self._find_path_collisions(path, obstacles):
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
                path,
                obstacles,
                gb_wpnts,
            )
            if (
                not math.isfinite(minimum_clearance)
                or minimum_clearance <= 1.0e-9
            ):
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
        track_min = np.asarray(
            [-float(gb_wpnts[index].d_right) + track_margin for index in indices]
        )
        track_max = np.asarray(
            [float(gb_wpnts[index].d_left) - track_margin for index in indices]
        )
        minimum = float(np.min(np.minimum(path.d - track_min, track_max - path.d)))

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

        for index, (s, d, xy) in enumerate(zip(path.s, path.d, path.xy)):
            gb_index = int(round(float(s) / wpnt_dist)) % len(gb_wpnts)
            velocity = float(gb_wpnts[gb_index].vx_mps)
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
