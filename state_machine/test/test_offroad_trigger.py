from types import SimpleNamespace

import pytest

from state_machine.state_machine_node import StateMachine
from state_machine.states_types import StateType


class Logger:
    def warn(self, *_args, **_kwargs):
        pass


def make_state_machine(
    *, target, speed, cur_s=10.0, source=StateType.GB_TRACK,
    published_target=None,
):
    if published_target is None:
        published_target = target
    return SimpleNamespace(
        offroad_disabled=False,
        cur_state=StateType.TRAILING,
        local_wpnts_src=source,
        behavior_strategy=SimpleNamespace(
            trailing_targets=([] if published_target is None else [published_target]),
        ),
        cur_obstacles_in_interest=([] if target is None else [target]),
        # These are cleared immediately before the real transition. Keeping them
        # empty here reproduces the state-machine loop order that exposed the bug.
        cur_gb_wpnts=SimpleNamespace(closest_target=None),
        cur_recovery_wpnts=SimpleNamespace(closest_target=None),
        ot_closest_target=None,
        cur_s=cur_s,
        cur_vs=speed,
        track_length=100.0,
        offroad_distance_m=3.0,
        name="test_state_machine",
        get_logger=lambda: Logger(),
    )


def obstacle(gap, *, is_static=True, cur_s=10.0):
    return SimpleNamespace(
        id=7,
        s_start=(cur_s + gap) % 100.0,
        is_static=is_static,
    )


def check(state_machine):
    return StateMachine._check_offroad(state_machine)


@pytest.mark.parametrize(
    ("gap", "speed"),
    [
        (3.0, 3.0),  # distance boundary
        (2.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_static_trailing_target_triggers_within_distance(gap, speed):
    state_machine = make_state_machine(target=obstacle(gap), speed=speed)

    assert check(state_machine) is True


@pytest.mark.parametrize("speed", [0.0, 1.0, 2.0, 4.0])
def test_static_target_beyond_distance_does_not_trigger_at_any_speed(speed):
    state_machine = make_state_machine(target=obstacle(3.01), speed=speed)

    assert check(state_machine) is False


@pytest.mark.parametrize(("gap", "speed"), [(1.0, 3.0), (8.0, 1.0)])
def test_dynamic_trailing_target_never_triggers(gap, speed):
    state_machine = make_state_machine(
        target=obstacle(gap, is_static=False),
        speed=speed,
    )

    assert check(state_machine) is False


def test_missing_trailing_target_does_not_use_an_unrelated_obstacle():
    state_machine = make_state_machine(target=None, speed=1.0)

    assert check(state_machine) is False


def test_previous_target_must_still_exist_in_current_obstacle_snapshot():
    previous = obstacle(1.0)
    current_other = obstacle(1.0)
    current_other.id = previous.id + 1
    state_machine = make_state_machine(
        target=current_other,
        published_target=previous,
        speed=1.0,
    )

    assert check(state_machine) is False


def test_current_classification_wins_when_target_static_flag_changes():
    previous = obstacle(1.0, is_static=True)
    current = obstacle(1.0, is_static=False)
    state_machine = make_state_machine(
        target=current,
        published_target=previous,
        speed=1.0,
    )

    assert check(state_machine) is False


def test_distance_is_wrap_safe_and_uses_target_front_face():
    state_machine = make_state_machine(
        target=obstacle(2.5, cur_s=99.0),
        speed=3.0,
        cur_s=99.0,
    )

    assert check(state_machine) is True


def test_trigger_is_limited_to_trailing_or_attack():
    state_machine = make_state_machine(target=obstacle(1.0), speed=1.0)
    state_machine.cur_state = StateType.GB_TRACK

    assert check(state_machine) is False
