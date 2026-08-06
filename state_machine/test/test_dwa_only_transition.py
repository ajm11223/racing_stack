from types import SimpleNamespace

from state_machine.state_transitions import TrailingTransition
from state_machine.states_types import StateType


def base_state_machine(check_dwa, check_offroad=True):
    return SimpleNamespace(
        cur_obstacles_in_interest=[object()],
        _check_close_to_raceline=lambda *args: True,
        _check_close_to_raceline_heading=lambda *args: True,
        _check_dwa=lambda: check_dwa,
        _check_offroad=lambda: check_offroad,
        _check_ftg=lambda: True,
    )


def test_dwa_has_priority_when_dwa_and_offroad_share_the_trigger():
    state, source = TrailingTransition(base_state_machine(True, True))

    assert state == StateType.DWAONLY
    assert source == StateType.DWAONLY


def test_offroad_remains_selectable_when_dwa_is_disabled():
    state, source = TrailingTransition(base_state_machine(False, True))

    assert state == StateType.OFFROADONLY
    assert source == StateType.OFFROADONLY


def test_dwa_trigger_reuses_static_target_distance_condition():
    obstacle = SimpleNamespace(id=7, s_start=12.5, is_static=True)
    machine = SimpleNamespace(
        dwa_disabled=False,
        cur_state=StateType.TRAILING,
        behavior_strategy=SimpleNamespace(trailing_targets=[obstacle]),
        cur_obstacles_in_interest=[obstacle],
        cur_s=10.0,
        cur_vs=3.0,
        track_length=100.0,
        offroad_distance_m=3.0,
        name="test_state_machine",
        get_logger=lambda: SimpleNamespace(warn=lambda *args, **kwargs: None),
    )

    from state_machine.state_machine_node import StateMachine

    assert StateMachine._check_dwa(machine) is True


def test_dynamic_target_never_triggers_dwa():
    obstacle = SimpleNamespace(id=7, s_start=11.0, is_static=False)
    machine = SimpleNamespace(
        dwa_disabled=False,
        cur_state=StateType.TRAILING,
        behavior_strategy=SimpleNamespace(trailing_targets=[obstacle]),
        cur_obstacles_in_interest=[obstacle],
        cur_s=10.0,
        cur_vs=0.0,
        track_length=100.0,
        offroad_distance_m=3.0,
        name="test_state_machine",
        get_logger=lambda: SimpleNamespace(warn=lambda *args, **kwargs: None),
    )

    from state_machine.state_machine_node import StateMachine

    assert StateMachine._check_dwa(machine) is False
