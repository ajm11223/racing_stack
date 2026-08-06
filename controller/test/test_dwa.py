import math

import numpy as np
import pytest

from controller.dwa.dwa import DWAConfig, DWAController, DWAPlanner


def test_dynamic_window_is_limited_by_acceleration_and_steering_rate():
    planner = DWAPlanner(DWAConfig(
        dynamic_window_dt_s=0.1,
        max_accel_mps2=2.0,
        max_decel_mps2=3.0,
        max_steer_rate_radps=1.5,
    ))

    window = planner.dynamic_window(speed_mps=1.0, steering_rad=0.1)

    assert window == pytest.approx((0.7, 1.2, -0.05, 0.25))


def test_bicycle_rollout_matches_straight_and_constant_curvature_motion():
    config = DWAConfig(predict_time_s=1.0, prediction_dt_s=0.1)
    planner = DWAPlanner(config)

    x, y, yaw = planner._predict(1.0, 0.0)
    assert x[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(0.0)
    assert yaw[-1] == pytest.approx(0.0)

    _, _, turning_yaw = planner._predict(1.0, 0.2)
    assert turning_yaw[-1] == pytest.approx(math.tan(0.2) / config.wheelbase_m)


def test_clear_space_prefers_fast_goal_aligned_candidate():
    planner = DWAPlanner(DWAConfig())
    planner.set_reference_path([0.0, 5.0, 10.0], [0.0, 0.0, 0.0])

    result = planner.plan([0.0, 0.0, 0.0], 1.0, 0.0, [], -math.pi, 0.01)

    assert result.emergency is False
    assert result.target_speed_mps == pytest.approx(result.dynamic_window[1])
    assert abs(result.steering_rad) < 1.0e-9
    assert all(not candidate.collision for candidate in result.candidates)


def test_obstacle_on_centerline_rejects_straight_rollouts():
    planner = DWAPlanner(DWAConfig(
        obstacle_weight=8.0,
        safety_radius_m=0.4,
    ))
    planner.set_reference_path([0.0, 5.0, 10.0], [0.0, 0.0, 0.0])

    result = planner.plan([0.0, 0.0, 0.0], 1.0, 0.0, [1.3], 0.0, 0.01)

    straight = [candidate for candidate in result.candidates
                if abs(candidate.steering_rad) < 1.0e-9]
    assert straight
    assert any(candidate.collision for candidate in straight)
    assert result.emergency is False
    assert result.selected.collision is False
    assert abs(result.steering_rad) > 1.0e-3


def test_every_rollout_in_collision_commands_full_stop():
    controller = DWAController(DWAConfig())
    count = 360
    ranges = np.full(count, 0.30)

    speed, steering = controller.process(
        [0.0, 0.0, 0.0], 1.0, ranges, -math.pi, 2.0 * math.pi / count,
    )

    assert controller.last_result.emergency is True
    assert speed == 0.0
    assert steering == 0.0
    assert controller.command_from_last() == (0.0, 0.0)


def test_asymmetric_base_link_footprint_uses_front_and_rear_lengths():
    planner = DWAPlanner(DWAConfig(
        vehicle_front_m=0.45,
        vehicle_rear_m=0.10,
        collision_margin_m=0.0,
    ))
    zeros = np.zeros(1)

    front_hit = planner._collision_metrics(
        zeros, zeros, zeros, 0.0, np.array([[0.44, 0.0]]),
    )[0]
    rear_hit = planner._collision_metrics(
        zeros, zeros, zeros, 0.0, np.array([[-0.44, 0.0]]),
    )[0]

    assert front_hit is True
    assert rear_hit is False


def test_scan_collision_result_is_invariant_to_map_pose_translation():
    planner = DWAPlanner(DWAConfig())
    first = planner.plan([0.0, 0.0, 0.0], 1.0, 0.0, [1.3], 0.0, 0.01)
    second = planner.plan([100.0, -40.0, 0.0], 1.0, 0.0, [1.3], 0.0, 0.01)

    assert [item.collision for item in first.candidates] == [
        item.collision for item in second.candidates
    ]
