from types import SimpleNamespace

from state_machine.state_machine_node import StateMachine
from state_machine.state_transitions import ObstacleTransition
from state_machine.states_types import StateType


def make_obstacle(gap, *, is_static):
    return SimpleNamespace(
        s_start=float(gap),
        is_static=bool(is_static),
    )


def make_state_machine(*obstacles):
    return SimpleNamespace(
        cur_s=0.0,
        track_length=100.0,
        gb_dynamic_trailing_distance_m=10.0,
        cur_obstacles_in_interest=list(obstacles),
    )


def check_distance(state_machine, threshold=10.0):
    return StateMachine._check_gb_trailing_distance(
        state_machine,
        threshold,
    )


def run_gb_obstacle_transition(obstacle):
    state_machine = make_state_machine(obstacle)
    state_machine.cur_state = StateType.GB_TRACK
    state_machine.cur_gb_wpnts = object()
    state_machine._check_free_frenet = lambda _wpnts: False
    state_machine._check_overtaking_mode = lambda: False
    state_machine._check_static_overtaking_mode = lambda: False
    state_machine._check_gb_trailing_distance = (
        lambda threshold: check_distance(state_machine, threshold)
    )
    return ObstacleTransition(state_machine, close_to_raceline=True)


def test_nearest_static_obstacle_beyond_ten_metres_blocks_trailing():
    state_machine = make_state_machine(
        make_obstacle(10.01, is_static=True),
    )

    assert check_distance(state_machine) is False


def test_nearest_static_obstacle_at_ten_metres_allows_trailing():
    state_machine = make_state_machine(
        make_obstacle(10.0, is_static=True),
    )

    assert check_distance(state_machine) is True


def test_gb_stays_gb_for_static_obstacle_beyond_ten_metres():
    state, source = run_gb_obstacle_transition(
        make_obstacle(10.01, is_static=True),
    )

    assert state == StateType.GB_TRACK
    assert source == StateType.GB_TRACK


def test_gb_enters_trailing_for_static_obstacle_at_ten_metres():
    state, source = run_gb_obstacle_transition(
        make_obstacle(10.0, is_static=True),
    )

    assert state == StateType.TRAILING
    assert source == StateType.GB_TRACK


def test_nearest_dynamic_obstacle_keeps_existing_trailing_behavior():
    state_machine = make_state_machine(
        make_obstacle(5.0, is_static=False),
        make_obstacle(10.0, is_static=True),
    )

    assert check_distance(state_machine) is True


def test_nearest_dynamic_obstacle_at_ten_metres_allows_trailing():
    state_machine = make_state_machine(
        make_obstacle(10.0, is_static=False),
    )

    assert check_distance(state_machine) is True


def test_nearest_dynamic_obstacle_beyond_ten_metres_blocks_trailing():
    state_machine = make_state_machine(
        make_obstacle(10.01, is_static=False),
    )

    assert check_distance(state_machine) is False


def test_gb_stays_gb_for_dynamic_obstacle_beyond_ten_metres():
    state, source = run_gb_obstacle_transition(
        make_obstacle(10.01, is_static=False),
    )

    assert state == StateType.GB_TRACK
    assert source == StateType.GB_TRACK


def test_gb_enters_trailing_for_dynamic_obstacle_at_ten_metres():
    state, source = run_gb_obstacle_transition(
        make_obstacle(10.0, is_static=False),
    )

    assert state == StateType.TRAILING
    assert source == StateType.GB_TRACK


def test_distance_check_handles_track_wraparound():
    state_machine = make_state_machine(
        make_obstacle(3.0, is_static=True),
    )
    state_machine.cur_s = 98.0

    assert check_distance(state_machine) is True
