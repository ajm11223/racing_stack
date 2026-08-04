from types import SimpleNamespace

import numpy as np
import pytest

from state_machine.state_machine_node import StateMachine


def make_path(s_values, d_values):
    s_values = np.asarray(s_values, dtype=float)
    d_values = np.asarray(d_values, dtype=float)
    return SimpleNamespace(
        is_init=True,
        array=np.column_stack((
            np.zeros_like(s_values),
            np.zeros_like(s_values),
            s_values,
            d_values,
        )),
        track_boundary_margin_m=0.30,
        obstacle_boundary_margin_m=0.30,
        longitudinal_margin_m=0.20,
    )


def make_obstacle(s_center=5.0):
    return SimpleNamespace(
        id=1,
        s_center=s_center,
        s_start=s_center - 0.10,
        s_end=s_center + 0.10,
        d_center=0.0,
        d_right=-0.10,
        d_left=0.10,
        size=0.20,
        is_static=True,
        vs=0.0,
    )


def make_state_machine():
    node = StateMachine.__new__(StateMachine)
    node.max_s = 100.0
    node.wpnt_dist = 1.0
    node.cur_gb_wpnts = SimpleNamespace(
        list=[
            SimpleNamespace(d_right=1.0, d_left=1.0)
            for _ in range(100)
        ]
    )
    return node


def test_static_lattice_track_margin_is_constant():
    node = make_state_machine()

    free, clearance = node._check_static_lattice_track_clearance(
        make_path([5.0], [0.69])
    )
    assert free
    assert clearance == pytest.approx(0.01)

    free, clearance = node._check_static_lattice_track_clearance(
        make_path([5.0], [0.70])
    )
    assert not free
    assert clearance == pytest.approx(0.0)


def test_static_lattice_obstacle_margin_is_constant():
    node = make_state_machine()
    obstacle = make_obstacle()

    blocked, clearance = node._check_static_lattice_obstacle_clearance(
        make_path([5.0], [0.41]),
        obstacle,
    )
    assert not blocked
    assert clearance == pytest.approx(0.01)

    blocked, clearance = node._check_static_lattice_obstacle_clearance(
        make_path([5.0], [0.40]),
        obstacle,
    )
    assert blocked
    assert clearance == pytest.approx(0.0)


def test_static_lattice_longitudinal_margin_limits_where_d_is_checked():
    node = make_state_machine()
    obstacle = make_obstacle()
    # Obstacle half-length is 0.10 m and longitudinal margin is 0.20 m,
    # so lateral overlap is checked within 0.30 m of s_center.
    inside = make_path([4.71], [0.0])
    outside = make_path([4.69], [0.0])

    blocked, _ = node._check_static_lattice_obstacle_clearance(
        inside,
        obstacle,
    )
    assert blocked

    blocked, clearance = node._check_static_lattice_obstacle_clearance(
        outside,
        obstacle,
    )
    assert not blocked
    assert clearance is None


def test_static_lattice_uses_conservative_explicit_obstacle_bounds():
    obstacle = SimpleNamespace(
        d_right=-0.15,
        d_left=0.20,
        d_center=0.0,
        size=0.50,
    )

    lower, upper = StateMachine._static_obstacle_lateral_bounds(obstacle)

    assert lower == pytest.approx(-0.25)
    assert upper == pytest.approx(0.25)

