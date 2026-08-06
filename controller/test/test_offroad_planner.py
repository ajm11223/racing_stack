import math

import numpy as np
import pytest

from controller.offroad_planner.offroad_planner import OffRoadController, OffRoadPlanner, PlannerConfig


def straight_frame(planner, *, width=None, speed=2.0):
    x = np.linspace(0.0, 20.0, 201)
    y = np.zeros_like(x)
    bounds = None if width is None else np.full_like(x, width)
    planner.set_base_frame(x, y, bounds, bounds, np.full_like(x, speed))


def branch_displacement(path):
    index = int(np.argmax(np.abs(path.q)))
    return float(path.q[index])


def test_cubic_lateral_maneuver_satisfies_paper_boundary_conditions():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        path_horizon_m=4.0,
        path_resolution_m=0.01,
        min_maneuver_m=2.0,
        speed_horizon_gain_s=0.0,
        max_curvature=10.0,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)
    heading_error = 0.12
    candidate = planner._generate_candidate(
        0, 0.6, 1.0, 1.0, -0.2, heading_error, 0.0,
    )

    assert candidate.q[0] == pytest.approx(-0.2)
    assert candidate.q[-1] == pytest.approx(0.6)
    assert (candidate.q[1] - candidate.q[0]) / config.path_resolution_m \
        == pytest.approx(math.tan(heading_error), abs=6.0e-3)
    assert np.max(np.abs(candidate.q[candidate.s_unwrapped >= 3.0] - 0.6)) < 1.0e-12


def test_quintic_fan_starts_with_sampled_steering_and_rejoins_raceline_pose():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        path_horizon_m=5.0,
        path_resolution_m=0.01,
        fan_goal_distance_m=4.0,
        max_curvature=3.0,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)
    initial_steer = 0.3
    candidate = planner._generate_fan_candidate(
        0, initial_steer, 1.0, 1.0, 0.1, 0.12,
    )

    assert candidate.generation_mode == "quintic_fan"
    assert candidate.q[0] == pytest.approx(0.1)
    assert candidate.yaw[0] == pytest.approx(0.12, abs=2.0e-3)
    assert candidate.curvature[0] == pytest.approx(
        math.tan(initial_steer) / config.wheelbase_m, rel=2.0e-3)
    assert candidate.q[-1] == pytest.approx(0.0, abs=1.0e-10)
    assert candidate.yaw[-1] == pytest.approx(0.0, abs=1.0e-8)
    assert candidate.curvature[-1] == pytest.approx(0.0, abs=1.0e-8)


def test_fan_samples_bounded_steering_and_legacy_mode_remains_available():
    fan = OffRoadPlanner(PlannerConfig(
        use_waypoint_bounds=False,
        fan_steering_samples=7,
        max_curvature=3.0,
    ))
    straight_frame(fan)
    fan_result = fan.plan((1.0, 0.0, 0.0), 1.0, [], 0.0, 0.01)

    fan_candidates = [
        path for path in fan_result.candidates
        if path.generation_mode == "quintic_fan"
    ]
    assert len(fan_candidates) == 7
    assert any(path.generation_mode == "cubic_offset" for path in fan_result.candidates)
    assert max(abs(path.initial_steer_rad) for path in fan_candidates) \
        <= fan.config.max_steer_rad + 1.0e-12
    assert any(abs(path.initial_steer_rad) < 1.0e-12 for path in fan_candidates)

    legacy = OffRoadPlanner(PlannerConfig(
        use_quintic_fan=False,
        use_waypoint_bounds=False,
        lateral_span_m=1.0,
    ))
    straight_frame(legacy)
    legacy_result = legacy.plan((1.0, 0.0, 0.0), 1.0, [], 0.0, 0.01)
    assert all(path.generation_mode == "cubic_offset" for path in legacy_result.candidates)
    assert len({path.final_offset_m for path in legacy_result.candidates}) > 1


def test_clear_straight_selects_base_frame_and_full_speed():
    planner = OffRoadPlanner(PlannerConfig())
    straight_frame(planner, width=1.0)

    result = planner.plan((1.0, 0.0, 0.0), 1.0, [], 0.0, 0.01)

    assert result.emergency is False
    assert result.selected.final_offset_m == pytest.approx(0.0, abs=1.0e-9)
    assert result.selected.smoothness_cost == pytest.approx(0.0, abs=1.0e-8)
    assert result.target_speed_mps == pytest.approx(2.0, rel=1.0e-5)


def test_static_obstacle_collision_blur_selects_a_clear_offset_path():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        lateral_span_m=2.4,
        speed_horizon_gain_s=0.0,
        min_maneuver_m=1.5,
        max_curvature=3.0,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)

    result = planner.plan(
        (1.0, 0.0, 0.0), 0.5, [], 0.0, 0.01,
        obstacle_points=[(4.0, 0.0)],
    )
    center = min(result.candidates, key=lambda path: np.max(np.abs(path.q)))

    assert center.collision is True
    assert result.emergency is False
    assert result.selected.collision is False
    assert np.max(np.abs(result.selected.q)) >= 0.3
    assert result.selected.safety_cost < center.safety_cost
    assert result.target_speed_mps < config.max_speed_mps


def test_laserscan_is_converted_to_the_cell_obstacle_map():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        lateral_span_m=2.4,
        speed_horizon_gain_s=0.0,
        min_maneuver_m=1.5,
        max_curvature=3.0,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)
    ranges = np.full(1081, np.inf)
    ranges[540] = 3.0

    result = planner.plan(
        (1.0, 0.0, 0.0), 0.5, ranges,
        -math.pi / 2.0, math.pi / 1080.0,
    )
    center = min(result.candidates, key=lambda path: np.max(np.abs(path.q)))

    assert center.collision is True
    assert result.selected.collision is False
    assert np.max(np.abs(result.selected.q)) >= 0.3


def test_collision_footprint_is_asymmetric_about_base_link():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        vehicle_front_m=0.45,
        vehicle_rear_m=0.10,
        collision_margin_m=0.0,
        max_curvature=3.0,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)

    front_path = planner._generate_candidate(
        0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    )
    planner._check_collision(front_path, np.array([[1.44, 0.0]]))
    assert front_path.collision is True

    rear_path = planner._generate_candidate(
        0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    )
    planner._check_collision(rear_path, np.array([[0.89, 0.0]]))
    assert rear_path.collision is False


def test_consistency_cost_suppresses_a_sudden_return_to_center():
    common = dict(
        use_waypoint_bounds=False,
        lateral_span_m=2.4,
        speed_horizon_gain_s=0.0,
        min_maneuver_m=1.5,
        max_curvature=3.0,
    )
    without_consistency = OffRoadPlanner(PlannerConfig(**common, consistency_weight=0.0))
    with_consistency = OffRoadPlanner(PlannerConfig(**common, consistency_weight=2.0))
    straight_frame(without_consistency)
    straight_frame(with_consistency)

    first_results = []
    for planner in (without_consistency, with_consistency):
        first = planner.plan(
            (1.0, 0.0, 0.0), 0.5, [], 0.0, 0.01,
            obstacle_points=[(4.0, 0.0)],
        )
        assert abs(branch_displacement(first.selected)) > 0.3
        first_results.append(first)

    no_consistency_result = without_consistency.plan(
        (1.1, 0.0, 0.0), 0.5, [], 0.0, 0.01, obstacle_points=[])
    consistency_result = with_consistency.plan(
        (1.1, 0.0, 0.0), 0.5, [], 0.0, 0.01, obstacle_points=[])

    assert branch_displacement(no_consistency_result.selected) == pytest.approx(0.0, abs=1.0e-9)
    assert abs(branch_displacement(consistency_result.selected)) > 0.1
    assert math.copysign(1.0, branch_displacement(consistency_result.selected)) \
        == math.copysign(1.0, branch_displacement(first_results[1].selected))


def test_all_colliding_candidates_use_longest_free_path_and_emergency_speed():
    config = PlannerConfig(
        use_waypoint_bounds=False,
        lateral_span_m=2.4,
        speed_horizon_gain_s=0.0,
        min_maneuver_m=1.5,
        max_curvature=3.0,
        emergency_speed_mps=0.6,
    )
    planner = OffRoadPlanner(config)
    straight_frame(planner)
    wall = [(2.0, y) for y in np.linspace(-1.2, 1.2, 25)]

    result = planner.plan(
        (1.0, 0.0, 0.0), 0.5, [], 0.0, 0.01,
        obstacle_points=wall,
    )
    feasible = [path for path in result.candidates if path.feasible]

    assert result.emergency is True
    assert all(path.collision for path in feasible)
    assert result.selected.collision_free_length_m == pytest.approx(
        max(path.collision_free_length_m for path in feasible)
    )
    assert result.target_speed_mps <= config.emergency_speed_mps


def test_direct_adapter_returns_bounded_ackermann_command():
    controller = OffRoadController(PlannerConfig())
    straight_frame(controller, width=1.0)

    speed, steer = controller.process((1.0, 0.1, 0.0), 0.5, [], 0.0, 0.01)

    assert 0.0 <= speed <= controller.config.max_speed_mps
    assert abs(steer) <= controller.config.max_steer_rad
    assert controller.last_result is not None


def test_closed_base_frame_keeps_consistency_across_the_lap_boundary():
    planner = OffRoadPlanner(PlannerConfig(
        lateral_span_m=1.0,
        use_waypoint_bounds=False,
        max_curvature=3.0,
    ))
    angle = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=False)
    radius = 5.0
    planner.set_base_frame(radius * np.cos(angle), radius * np.sin(angle))
    before = 2.0 * math.pi - 0.03
    after = 0.03

    first = planner.plan(
        (radius * math.cos(before), radius * math.sin(before), before + math.pi / 2.0),
        1.0, [], 0.0, 0.01,
    )
    second = planner.plan(
        (radius * math.cos(after), radius * math.sin(after), after + math.pi / 2.0),
        1.0, [], 0.0, 0.01,
    )

    assert second.selected.s_unwrapped[0] > first.selected.s_unwrapped[0]
    assert np.isfinite(second.selected.consistency_cost)
