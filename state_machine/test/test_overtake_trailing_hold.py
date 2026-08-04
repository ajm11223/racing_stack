from types import SimpleNamespace

import numpy as np

from state_machine.state_machine_node import StateMachine
from state_machine.state_transitions import OvertakingTransition
from state_machine.states_types import StateType


class Logger:
    def info(self, *_args, **_kwargs):
        pass


def test_capture_overtake_path_copies_geometry_and_blocking_target():
    blocker = object()
    source = SimpleNamespace(
        is_init=True,
        array=np.asarray([[1.0, 2.0, 3.0, 0.4]]),
        list=[object()],
        stamp=object(),
        closest_target=blocker,
        closest_gap=2.5,
        free_dbg={"is_free": False},
    )
    target = SimpleNamespace(
        stamp=None,
        list=[],
        array=None,
        is_init=False,
        last_init_sec=None,
        init_count=0,
        is_gb_track_wpnts=False,
        closest_target=None,
        closest_gap=None,
        free_dbg=None,
        frozen=True,
    )
    state_machine = SimpleNamespace(
        _src_cache=lambda _src: source,
        cur_recovery_wpnts=target,
        _check_on_spline=lambda _cache: True,
        now_sec=lambda: 12.5,
        get_logger=lambda: Logger(),
    )

    captured = StateMachine._capture_overtake_path_for_trailing(
        state_machine
    )

    assert captured is True
    assert target.is_init is True
    assert target.list == source.list
    assert target.list is not source.list
    assert np.array_equal(target.array, source.array)
    assert target.array is not source.array
    assert target.closest_target is blocker
    assert target.closest_gap == 2.5
    assert target.frozen is False


def test_overtake_failure_enters_trailing_on_captured_path():
    state_machine = SimpleNamespace(
        _check_overtaking_mode_sustainability=lambda: False,
        _check_enemy_in_front=lambda: True,
        overtaking_ttl_count=3,
        overtaking_ttl_count_threshold=10,
        _capture_overtake_path_for_trailing=lambda: True,
    )

    state, source = OvertakingTransition(state_machine)

    assert state == StateType.TRAILING
    assert source == StateType.RECOVERY
    assert state_machine.overtaking_ttl_count == 0


def test_captured_path_blocker_is_published_as_trailing_and_replan_target():
    blocker = object()
    state_machine = SimpleNamespace(
        local_wpnts_src=StateType.RECOVERY,
        cur_gb_wpnts=SimpleNamespace(closest_target=None),
        cur_recovery_wpnts=SimpleNamespace(closest_target=blocker),
    )

    trailing, source = StateMachine.get_farthest_target(
        state_machine,
        StateType.RECOVERY,
    )
    overtaking = StateMachine.get_overtaking_target(state_machine)

    assert trailing == [blocker]
    assert overtaking == [blocker]
    assert source == StateType.RECOVERY
