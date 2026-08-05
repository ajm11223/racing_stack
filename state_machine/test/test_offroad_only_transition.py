from types import SimpleNamespace

from state_machine.state_transitions import TrailingTransition
from state_machine.states_types import StateType


def test_offroad_direct_controller_has_priority_when_its_trigger_fires():
    state_machine = SimpleNamespace(
        cur_obstacles_in_interest=[object()],
        _check_close_to_raceline=lambda *args: True,
        _check_close_to_raceline_heading=lambda *args: True,
        _check_offroad=lambda: True,
        _check_ftg=lambda: True,
    )

    state, source = TrailingTransition(state_machine)

    assert state == StateType.OFFROADONLY
    assert source == StateType.OFFROADONLY


def test_ftg_remains_available_when_offroad_trigger_is_disabled():
    state_machine = SimpleNamespace(
        cur_obstacles_in_interest=[object()],
        _check_close_to_raceline=lambda *args: True,
        _check_close_to_raceline_heading=lambda *args: True,
        _check_offroad=lambda: False,
        _check_ftg=lambda: True,
    )

    state, source = TrailingTransition(state_machine)

    assert state == StateType.FTGONLY
    assert source == StateType.FTGONLY
