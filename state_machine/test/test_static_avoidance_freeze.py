from types import SimpleNamespace

import numpy as np

from state_machine.state_machine_node import StateMachine


def stamp(value):
    seconds = int(value)
    return SimpleNamespace(
        sec=seconds,
        nanosec=int(round((value - seconds) * 1e9)),
    )


def waypoint(x, y, s, d):
    return SimpleNamespace(x_m=x, y_m=y, s_m=s, d_m=d)


def planner_message(stamp_sec, wpnts):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp(stamp_sec)),
        wpnts=wpnts,
    )


class Cache:
    def __init__(self, stamp_sec=100.0, lateral=0.0):
        self.list = [
            waypoint(0.0, 0.0, 10.0, lateral),
            waypoint(2.0, 0.0, 12.0, lateral),
        ]
        self.array = np.asarray(
            [[wp.x_m, wp.y_m, wp.s_m, wp.d_m] for wp in self.list]
        )
        self.stamp = stamp(stamp_sec)
        self.is_init = True
        self.frozen = False
        self.latest_threshold = 0.1
        self.hyst_timer_sec = 4.0
        self.killing_timer_sec = 10.0
        self.on_spline_front_horizon_thres_m = 0.5
        self.on_spline_min_dist_thres_m = 0.5
        self.last_init_sec = stamp_sec
        self.init_count = 1

    def initialize_traj(self, message):
        self.stamp = message.header.stamp
        self.list = message.wpnts
        self.array = np.asarray(
            [[wp.x_m, wp.y_m, wp.s_m, wp.d_m] for wp in self.list]
        )
        self.is_init = True
        self.last_init_sec = self.node.now_sec()
        self.init_count += 1


def make_state_machine(now=101.55):
    node = StateMachine.__new__(StateMachine)
    node.current_position = [0.0, 0.0, 0.0]
    node.cur_s = 10.0
    node.track_length = 44.0
    node.now_sec = lambda: now
    node.static_overtaking_mode = True
    node.cur_static_avoidance_wpnts = Cache()
    node.cur_static_avoidance_wpnts.node = node
    node._check_free_frenet = lambda _cache: True
    return node


def test_sequential_static_replan_replaces_cache_inside_hysteresis():
    node = make_state_machine()
    new_path = [
        waypoint(0.0, 0.0, 10.0, 0.35),
        waypoint(3.0, 0.0, 13.0, 0.35),
    ]
    node.static_avoidance_wpnts = planner_message(101.50, new_path)

    sustainable = node._check_overtaking_mode_sustainability()

    assert sustainable is True
    assert node.cur_static_avoidance_wpnts.list is new_path
    assert node.cur_static_avoidance_wpnts.stamp.sec == 101
    assert node.cur_static_avoidance_wpnts.init_count == 2


def test_discontinuous_static_replan_is_rejected_atomically():
    node = make_state_machine()
    cache = node.cur_static_avoidance_wpnts
    old_list = cache.list
    old_array = cache.array.copy()
    old_stamp = cache.stamp
    old_init_count = cache.init_count
    disconnected = [
        waypoint(10.0, 10.0, 10.0, 0.8),
        waypoint(12.0, 10.0, 13.0, 0.8),
    ]
    candidate = planner_message(101.50, disconnected)

    accepted = node._check_latest_wpnts(candidate, cache)

    assert accepted is False
    assert cache.list is old_list
    assert np.array_equal(cache.array, old_array)
    assert cache.stamp is old_stamp
    assert cache.init_count == old_init_count


def test_same_static_message_is_not_reinitialized_each_state_loop():
    node = make_state_machine(now=100.05)
    cache = node.cur_static_avoidance_wpnts
    same_message = planner_message(100.0, list(cache.list))

    accepted = node._check_latest_wpnts(same_message, cache)

    assert accepted is True
    assert cache.init_count == 1


def test_one_empty_static_planner_frame_keeps_active_cache():
    node = make_state_machine(now=100.5)
    cache = node.cur_static_avoidance_wpnts
    old_list = cache.list
    old_stamp = cache.stamp
    node.static_avoidance_wpnts = planner_message(100.5, [])

    sustainable = node._check_overtaking_mode_sustainability()

    assert sustainable is True
    assert cache.is_init is True
    assert cache.list is old_list
    assert cache.stamp is old_stamp


def test_empty_static_output_expires_after_killing_timer():
    node = make_state_machine(now=111.0)
    cache = node.cur_static_avoidance_wpnts
    node.static_avoidance_wpnts = planner_message(111.0, [])

    sustainable = node._check_overtaking_mode_sustainability()

    assert sustainable is False
    assert cache.is_init is False
