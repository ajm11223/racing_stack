from types import SimpleNamespace

from state_machine.state_machine_node import StateMachine
from state_machine.states_types import StateType


def make_state_machine():
    node = StateMachine.__new__(StateMachine)
    node.cur_static_avoidance_wpnts = SimpleNamespace(frozen=False)
    node.static_overtaking_mode = True
    node.local_wpnts_src = StateType.OVERTAKE
    return node


def test_static_path_is_frozen_while_it_is_the_active_overtake_source():
    node = make_state_machine()

    node._hold_static_avoidance_freeze()

    assert node.cur_static_avoidance_wpnts.frozen


def test_static_path_is_released_after_leaving_overtake():
    node = make_state_machine()
    node.cur_static_avoidance_wpnts.frozen = True
    node.local_wpnts_src = StateType.GB_TRACK

    node._hold_static_avoidance_freeze()

    assert not node.cur_static_avoidance_wpnts.frozen
