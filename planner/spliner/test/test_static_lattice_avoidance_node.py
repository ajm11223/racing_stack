from types import SimpleNamespace
import math

import numpy as np
import pytest

from spliner.static_lattice_avoidance_node import (
    LatticeApexCandidate,
    LatticePathCandidate,
    StaticLatticeAvoidancePlanner,
)


def make_planner():
    planner = StaticLatticeAvoidancePlanner.__new__(
        StaticLatticeAvoidancePlanner
    )
    planner.lattice_d_resolution = 0.20
    planner.lattice_track_boundary_margin = 0.25
    planner.lattice_obstacle_boundary_margin = 0.35
    planner.lattice_max_samples_per_side = 5
    planner.lattice_min_free_width = 0.50
    planner.lattice_max_obstacles = 3
    planner.lattice_obstacle_horizon = 10.0
    planner.lattice_longitudinal_margin = 0.30
    planner.lattice_weight_safety = 0.55
    planner.lattice_weight_smoothness = 0.30
    planner.lattice_weight_consistency = 0.15
    planner.lattice_safety_sigma = 0.25
    planner.lattice_max_curvature = 1.28
    planner.lattice_consistency_scale = 0.50
    planner.previous_selected_path = None
    planner.behavior_state = ""
    planner.post_min_dist = 1.5
    planner.post_max_dist = 5.0
    planner.spline_scale = 0.8
    planner.cur_x = None
    planner.cur_y = None
    planner.cur_yaw = None
    return planner


def make_obstacle(obstacle_id, s, d=0.0):
    return SimpleNamespace(
        id=obstacle_id,
        s_center=float(s),
        d_center=float(d),
        s_start=float(s) - 0.1,
        s_end=float(s) + 0.1,
        d_right=float(d) - 0.1,
        d_left=float(d) + 0.1,
        size=0.2,
        is_static=True,
    )


class StraightConverter:
    def get_cartesian(self, s, d):
        return np.vstack((np.asarray(s), np.asarray(d)))

    def get_frenet(self, x, y):
        return np.asarray(x), np.asarray(y)


def make_path(d_values, obstacle_id=1):
    d_values = np.asarray(d_values, dtype=float)
    s_values = np.linspace(0.0, 10.0, len(d_values))
    return LatticePathCandidate(
        apexes=(LatticeApexCandidate(
            obstacle_id,
            5.0,
            float(d_values[len(d_values) // 2]),
            "left" if np.mean(d_values) >= 0.0 else "right",
        ),),
        s=s_values,
        d=d_values,
        xy=np.column_stack((s_values, d_values)),
    )


def make_global_waypoints(count=1001):
    return [
        SimpleNamespace(
            s_m=index * 0.1,
            d_right=2.0,
            d_left=2.0,
            d_m=0.0,
            x_m=index * 0.1,
            y_m=0.0,
            vx_mps=2.0,
            psi_rad=0.0,
            kappa_radpm=0.0,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("width", "expected_count"),
    [
        (0.49, 0),
        (0.50, 1),
        (0.99, 1),
        (1.00, 3),
        (1.99, 3),
        (2.00, 5),
        (3.00, 5),
    ],
)
def test_free_interval_width_selects_zero_one_three_or_five_samples(
    width,
    expected_count,
):
    planner = make_planner()

    samples = planner._sample_free_interval(0.0, width)

    assert len(samples) == expected_count
    assert all(0.0 < value < width for value in samples)


@pytest.mark.parametrize(
    ("width", "expected_spacing"),
    [
        (1.20, 0.20),
        (1.99, 0.20),
        (2.00, 0.30),
        (2.40, 0.30),
    ],
)
def test_interval_samples_use_spacing_for_candidate_count(
    width,
    expected_spacing,
):
    planner = make_planner()

    samples = planner._sample_free_interval(0.0, width)

    assert np.diff(samples) == pytest.approx(
        np.full(len(samples) - 1, expected_spacing)
    )
    assert np.mean(samples) == pytest.approx(width / 2.0)


def test_candidate_sampling_uses_independent_boundary_margins():
    planner = make_planner()
    obstacle = make_obstacle(1, 5.0, d=0.0)
    planner.lattice_min_free_width = 0.05
    gb_wpnt = SimpleNamespace(d_right=1.0, d_left=1.0)

    candidates = planner._candidates_for_obstacle(obstacle, gb_wpnt)

    assert [candidate.d for candidate in candidates] == pytest.approx([
        -0.60,
        0.60,
    ])


def test_track_clearance_uses_configured_constant_margin():
    planner = make_planner()
    planner.gb_max_s = 100.0
    path = make_path(np.full(101, 0.80))
    gb_wpnts = make_global_waypoints()
    for wpnt in gb_wpnts:
        wpnt.d_right = 1.0
        wpnt.d_left = 1.0

    clearance = planner._minimum_path_clearance(path, [], gb_wpnts)

    assert clearance == pytest.approx(-0.05)


def test_obstacle_clearance_uses_configured_constant_margin():
    planner = make_planner()
    planner.gb_max_s = 100.0
    path = make_path(np.full(101, 0.46))
    obstacle = make_obstacle(1, 5.0, d=0.0)

    clearance = planner._minimum_path_clearance(
        path,
        [obstacle],
        make_global_waypoints(),
    )

    assert clearance == pytest.approx(0.01)


def test_explicit_and_size_bounds_are_combined_conservatively():
    obstacle = SimpleNamespace(
        d_right=-0.15,
        d_left=0.20,
        d_center=0.0,
        size=0.50,
    )

    lower, upper = StaticLatticeAvoidancePlanner._obstacle_lateral_bounds(
        obstacle
    )

    assert lower == pytest.approx(-0.25)
    assert upper == pytest.approx(0.25)


def test_zero_explicit_bounds_fall_back_to_center_and_size():
    obstacle = SimpleNamespace(
        d_right=0.0,
        d_left=0.0,
        d_center=0.8,
        size=0.4,
    )

    lower, upper = StaticLatticeAvoidancePlanner._obstacle_lateral_bounds(
        obstacle
    )

    assert lower == pytest.approx(0.6)
    assert upper == pytest.approx(1.0)


def test_candidate_combinations_cover_every_layer_choice():
    layers = [
        [
            LatticeApexCandidate(1, 5.0, -0.7, "right"),
            LatticeApexCandidate(1, 5.0, 0.7, "left"),
        ],
        [
            LatticeApexCandidate(2, 8.0, -0.8, "right"),
            LatticeApexCandidate(2, 8.0, 0.8, "left"),
            LatticeApexCandidate(2, 8.0, 1.0, "left"),
        ],
    ]

    combinations = StaticLatticeAvoidancePlanner._candidate_combinations(
        layers
    )

    assert len(combinations) == 6


def test_empty_layer_has_no_complete_candidate_path():
    candidate = LatticeApexCandidate(1, 5.0, 0.7, "left")

    combinations = StaticLatticeAvoidancePlanner._candidate_combinations(
        [[candidate], []]
    )

    assert combinations == []


def test_tracking_and_strategy_obstacles_are_merged_and_sorted():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.lookahead = 10.0
    planner.tracked_static_obstacles = [
        make_obstacle(1, 5.0),
        make_obstacle(2, 8.0),
        make_obstacle(3, 12.0),  # beyond the 10 m LiDAR horizon
    ]
    # The strategy copy of id=1 should replace the tracking copy.
    planner.obstacles_in_interest = [make_obstacle(1, 6.0)]

    obstacles = planner._ordered_planning_obstacles()

    assert [obstacle.id for obstacle in obstacles] == [1, 2]
    assert obstacles[0].s_center == pytest.approx(6.0)


def test_tracking_obstacles_do_not_activate_planner_without_strategy_target():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.lookahead = 10.0
    planner.tracked_static_obstacles = [make_obstacle(1, 5.0)]
    planner.obstacles_in_interest = []

    assert planner._ordered_planning_obstacles() == []


def test_overtake_state_keeps_tracking_replanning_active_without_target():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.behavior_state = "OVERTAKE"
    newly_visible = make_obstacle(7, 6.0)
    planner.tracked_static_obstacles = [newly_visible]
    planner.obstacles_in_interest = []

    constraints = planner._ordered_planning_obstacles()
    initial = planner._initial_planning_obstacles(constraints)

    assert [obstacle.id for obstacle in constraints] == [7]
    assert [obstacle.id for obstacle in initial] == [7]


def test_same_s_peer_is_included_in_initial_planning_group():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    active = make_obstacle(1, 5.0, d=-0.5)
    same_s_peer = make_obstacle(2, 5.0, d=0.5)
    planner.obstacles_in_interest = [active]

    initial = planner._initial_planning_obstacles([active, same_s_peer])

    assert [obstacle.id for obstacle in initial] == [1, 2]


def test_same_s_obstacles_create_one_layer_from_combined_free_space():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.gb_max_idx = 1000
    first = make_obstacle(1, 5.0, d=-0.5)
    second = make_obstacle(2, 5.0, d=0.5)

    layers = planner._build_candidate_layers(
        [first, second],
        make_global_waypoints(),
    )

    assert len(layers) == 1
    assert layers[0]
    inflated_bounds = []
    for obstacle in (first, second):
        lo, hi = planner._obstacle_lateral_bounds(obstacle)
        inflated_bounds.append((
            lo - planner.lattice_obstacle_boundary_margin,
            hi + planner.lattice_obstacle_boundary_margin,
        ))
    assert all(
        not any(lo <= candidate.d <= hi for lo, hi in inflated_bounds)
        for candidate in layers[0]
    )
    assert {candidate.s for candidate in layers[0]} == {5.0}


def test_same_s_group_produces_monotonic_spline_control_points():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.cur_d = 0.0
    planner.cur_x = 0.0
    planner.cur_y = 0.0
    planner.cur_yaw = 0.0
    planner.gb_max_s = 100.0
    planner.gb_max_idx = 1000
    planner.converter = StraightConverter()
    first = make_obstacle(1, 5.0, d=-0.5)
    second = make_obstacle(2, 5.0, d=0.5)
    gb_wpnts = make_global_waypoints()

    layers = planner._build_candidate_layers([first, second], gb_wpnts)
    path = planner._build_candidate_samples((layers[0][0],), gb_wpnts)

    assert path is not None
    assert len(path.apexes) == 1
    assert path.apexes[0].s == pytest.approx(5.0)


def test_obstacle_horizon_is_independent_of_post_return_distance():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.post_max_dist = 50.0
    planner.tracked_static_obstacles = [
        make_obstacle(1, 5.0),
        make_obstacle(2, 10.0),
        make_obstacle(3, 10.01),
    ]
    planner.obstacles_in_interest = [make_obstacle(1, 5.0)]

    obstacles = planner._ordered_planning_obstacles()

    assert [obstacle.id for obstacle in obstacles] == [1, 2]


@pytest.mark.parametrize(
    ("first_s", "last_s", "expected_post_dist"),
    [
        (0.8, 4.0, 0.4),
        (3.0, 6.0, 1.5),
        (8.0, 9.0, 4.0),
    ],
)
def test_post_dist_is_half_first_obstacle_pre_dist(
    first_s,
    last_s,
    expected_post_dist,
):
    planner = make_planner()
    planner.cur_s = 0.0
    planner.cur_d = 0.0
    planner.cur_x = 0.0
    planner.cur_y = 0.0
    planner.cur_yaw = 0.0
    planner.gb_max_s = 100.0
    planner.converter = StraightConverter()
    gb_wpnts = make_global_waypoints()
    apexes = (
        LatticeApexCandidate(1, first_s, 0.7, "left"),
        LatticeApexCandidate(2, last_s, 0.8, "left"),
    )

    path = planner._build_candidate_samples(apexes, gb_wpnts)

    assert path is not None
    assert path.s[-1] == pytest.approx(
        last_s + expected_post_dist
    )
    assert path.d[-1] == pytest.approx(0.0)


def test_candidate_spline_starts_at_vehicle_pose_and_heading():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.cur_d = 0.0
    planner.cur_x = 0.0
    planner.cur_y = 0.0
    planner.cur_yaw = math.pi / 4.0
    planner.gb_max_s = 100.0
    planner.converter = StraightConverter()
    apex = LatticeApexCandidate(1, 5.0, 0.7, "left")

    path = planner._build_candidate_samples((apex,), make_global_waypoints())

    assert path is not None
    assert path.xy[0] == pytest.approx([planner.cur_x, planner.cur_y])
    initial_delta = path.xy[1] - path.xy[0]
    initial_heading = math.atan2(initial_delta[1], initial_delta[0])
    assert initial_heading == pytest.approx(planner.cur_yaw, abs=0.08)


def test_partial_collision_rejects_only_that_path_without_promotion():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    active = make_obstacle(1, 5.0)
    side_obstacle = make_obstacle(2, 8.0, d=-0.8)

    planner._build_candidate_layers = lambda obstacles, _: [
        [
            LatticeApexCandidate(obs.id, obs.s_center, -0.7, "right"),
            LatticeApexCandidate(obs.id, obs.s_center, 0.7, "left"),
        ]
        for obs in obstacles
    ]
    planner._build_candidate_samples = lambda apexes, _: SimpleNamespace(
        apexes=apexes
    )
    planner._find_path_collisions = lambda path, _: (
        [side_obstacle] if path.apexes[0].d < 0.0 else []
    )

    result = planner._expand_planning_obstacles(
        [active],
        [active, side_obstacle],
        gb_wpnts=[],
    )

    assert result.promoted_obstacle_ids == []
    assert [obs.id for obs in result.planning_obstacles] == [1]
    assert len(result.generated_paths) == 2
    assert len(result.collision_free_paths) == 1


def test_obstacle_is_promoted_when_every_current_path_collides():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    active = make_obstacle(1, 5.0)
    blocking_side_obstacle = make_obstacle(2, 8.0, d=0.8)

    planner._build_candidate_layers = lambda obstacles, _: [
        [LatticeApexCandidate(obs.id, obs.s_center, 0.7, "left")]
        for obs in obstacles
    ]
    planner._build_candidate_samples = lambda apexes, _: SimpleNamespace(
        apexes=apexes
    )

    def collisions(path, _):
        planned_ids = {apex.obstacle_id for apex in path.apexes}
        return [] if 2 in planned_ids else [blocking_side_obstacle]

    planner._find_path_collisions = collisions

    result = planner._expand_planning_obstacles(
        [active],
        [active, blocking_side_obstacle],
        gb_wpnts=[],
    )

    assert result.promoted_obstacle_ids == [2]
    assert [obs.id for obs in result.planning_obstacles] == [1, 2]
    assert len(result.generated_paths) == 1
    assert len(result.collision_free_paths) == 1


def test_safety_cost_prefers_more_obstacle_clearance():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.lattice_weight_safety = 1.0
    planner.lattice_weight_smoothness = 0.0
    planner.lattice_weight_consistency = 0.0
    obstacle = make_obstacle(1, 5.0, d=0.0)
    near_path = make_path(np.full(101, 0.60))
    far_path = make_path(np.full(101, 1.20))

    selected, score = planner._select_best_path(
        [near_path, far_path],
        [obstacle],
        make_global_waypoints(),
    )

    assert selected is far_path
    assert score.minimum_clearance == pytest.approx(0.55)


def test_squared_curvature_cost_prefers_straight_path():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.lattice_weight_safety = 0.0
    planner.lattice_weight_smoothness = 1.0
    planner.lattice_weight_consistency = 0.0
    s_values = np.linspace(0.0, 10.0, 101)
    straight = make_path(np.full(101, 0.8))
    curved = make_path(0.8 + 0.2 * np.sin(np.pi * s_values / 10.0))

    selected, score = planner._select_best_path(
        [curved, straight],
        [],
        make_global_waypoints(),
    )

    assert selected is straight
    assert score.smoothness == pytest.approx(0.0, abs=1.0e-10)


def test_consistency_cost_keeps_previous_avoidance_side():
    planner = make_planner()
    planner.cur_s = 0.0
    planner.gb_max_s = 100.0
    planner.lattice_weight_safety = 0.0
    planner.lattice_weight_smoothness = 0.0
    planner.lattice_weight_consistency = 1.0
    previous = make_path(np.full(101, 0.8))
    same_side = make_path(np.full(101, 0.8))
    opposite_side = make_path(np.full(101, -0.8))
    planner.previous_selected_path = previous

    selected, score = planner._select_best_path(
        [opposite_side, same_side],
        [],
        make_global_waypoints(),
    )

    assert selected is same_side
    assert score.consistency == pytest.approx(0.0)
