from types import SimpleNamespace

import numpy as np

from spliner.static_avoidance_node import ObstacleSpliner


class StraightConverter:
    def get_cartesian(self, s, d):
        return np.vstack((np.asarray(s), np.asarray(d)))

    def get_frenet(self, x, y):
        return np.asarray(x), np.asarray(y)


def make_obstacle(obs_id, s, d=0.0, size=0.4, is_static=True):
    return SimpleNamespace(
        id=obs_id,
        s_center=float(s),
        d_center=float(d),
        size=float(size),
        is_static=is_static,
    )


def make_node():
    node = ObstacleSpliner.__new__(ObstacleSpliner)
    node.name = "static_avoidance_planner"
    node.cur_s = 0.0
    node.cur_x = 0.0
    node.cur_y = 0.0
    node.cur_yaw = 0.0
    node.gb_max_s = 100.0
    node.gb_max_idx = 100
    node.evasion_dist = 0.6
    node.spline_bound_mindist = 0.3
    node.width_car = 0.3
    node.post_min_dist = 1.5
    node.post_max_dist = 5.0
    node.sampling_dist = 5.0
    node.spline_scale = 0.8
    node.converter = StraightConverter()
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    node._held_wpnts = None
    node._held_samples = None
    node._held_path_horizon = 0.0
    node._held_obstacle_ids = set()
    node.tracked_obstacles = []
    return node


def make_waypoints():
    return [
        SimpleNamespace(
            s_m=float(i),
            x_m=float(i),
            y_m=0.0,
            psi_rad=0.0,
            d_left=2.0,
            d_right=2.0,
            vx_mps=2.0,
        )
        for i in range(100)
    ]


def test_planning_obstacles_keeps_unique_static_obstacles_ahead():
    node = make_node()
    primary = make_obstacle(1, 5.0)
    node.tracked_obstacles = [
        make_obstacle(1, 5.0),
        make_obstacle(2, 7.0, is_static=False),
        make_obstacle(3, 9.0),
        make_obstacle(4, 60.0),
    ]

    result = node._planning_obstacles(primary)

    assert [obstacle.id for obstacle in result] == [1, 3]


def test_collision_check_uses_vehicle_and_safety_clearance():
    node = make_node()
    samples = np.column_stack((np.linspace(0.0, 10.0, 101), np.ones(101)))
    colliding = make_obstacle(2, 5.0, d=1.5)
    clear = make_obstacle(3, 6.0, d=2.0)
    beyond_horizon = make_obstacle(4, 15.0, d=1.0)

    result = node._find_obstacle_collisions(
        samples,
        [clear, beyond_horizon, colliding],
        path_horizon=10.0,
    )

    assert [obstacle.id for obstacle in result] == [2]


def test_candidate_can_pass_through_three_ordered_apexes():
    node = make_node()
    obstacles = [
        make_obstacle(1, 5.0),
        make_obstacle(2, 8.0),
        make_obstacle(3, 11.0),
    ]

    candidate = node._build_candidate_samples(
        obstacles,
        make_waypoints(),
        wpnt_dist=1.0,
    )

    assert candidate is not None
    samples, _, path_horizon = candidate
    assert samples.shape[1] == 2
    assert np.isfinite(samples).all()
    assert path_horizon == node.gb_max_s / 2


def test_different_ids_are_not_merged_even_when_centers_match():
    first = make_obstacle(10, 5.0)
    second = make_obstacle(11, 5.0)

    assert not ObstacleSpliner._same_obstacle(first, second)


def test_candidate_deduplicates_coincident_apex_control_points():
    node = make_node()
    obstacles = [
        make_obstacle(10, 5.0),
        make_obstacle(11, 5.0),
    ]

    candidate = node._build_candidate_samples(
        obstacles,
        make_waypoints(),
        wpnt_dist=1.0,
    )

    assert candidate is not None
    samples, _, _ = candidate
    assert np.isfinite(samples).all()


def test_active_obstacle_is_excluded_but_same_position_other_id_remains():
    node = make_node()
    active = make_obstacle(10, 5.0)
    own_collision = make_obstacle(10, 5.0)
    other_collision = make_obstacle(11, 5.0)

    result = node._unconstrained_collisions(
        [own_collision, other_collision],
        [active],
    )

    assert [obstacle.id for obstacle in result] == [11]


def test_candidate_progress_rejects_backward_hook():
    node = make_node()
    hooked_samples = np.array([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.1, 0.1],
        [0.4, 0.0],
    ])

    assert not node._candidate_progress_is_valid(hooked_samples, wpnt_dist=0.1)


def test_candidate_progress_accepts_forward_path():
    node = make_node()
    samples = np.column_stack((np.linspace(0.0, 2.0, 21), np.zeros(21)))

    assert node._candidate_progress_is_valid(samples, wpnt_dist=0.1)


def set_held_path(node):
    node._held_wpnts = object()
    node._held_samples = np.column_stack((
        np.linspace(0.0, 10.0, 101),
        np.zeros(101),
    ))
    node._held_path_horizon = 10.0
    node._held_obstacle_ids = {1}


def test_held_spline_is_reused_for_same_obstacle_set():
    node = make_node()
    primary = make_obstacle(1, 5.0)
    node.tracked_obstacles = [primary]
    set_held_path(node)

    needs_replan = node._held_spline_needs_replan(primary)

    assert not needs_replan


def test_held_spline_replans_for_new_colliding_obstacle():
    node = make_node()
    primary = make_obstacle(1, 5.0)
    node.tracked_obstacles = [primary, make_obstacle(2, 7.0)]
    set_held_path(node)

    needs_replan = node._held_spline_needs_replan(primary)

    assert needs_replan


def test_held_spline_replans_for_new_behavior_target():
    node = make_node()
    set_held_path(node)

    needs_replan = node._held_spline_needs_replan(make_obstacle(9, 6.0))

    assert needs_replan
