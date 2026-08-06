from types import SimpleNamespace

import numpy as np

import state_machine.state_machine_node as state_machine_node
from state_machine.state_machine_node import StateMachine


def waypoint(index):
    return SimpleNamespace(
        x_m=float(index),
        y_m=0.0,
        s_m=float(index),
        kappa_radpm=0.0,
        vx_mps=0.0,
        ax_mps2=0.0,
    )


def test_ggv_scale_changes_only_ggv_ax_and_ay(monkeypatch):
    captured = {}

    def fake_calc_vel_profile(**kwargs):
        captured.update(kwargs)
        return np.ones(3)

    monkeypatch.setattr(
        state_machine_node,
        "calc_vel_profile",
        fake_calc_vel_profile,
    )
    monkeypatch.setattr(
        state_machine_node.tph.calc_ax_profile,
        "calc_ax_profile",
        lambda **_kwargs: np.zeros(2),
    )

    original_ggv = np.array([[0.0, 6.0, 4.0], [10.0, 5.0, 8.0]])
    original_ax = np.array([[0.0, 4.0], [10.0, 3.0]])
    original_braking = np.array([[0.0, 8.0], [10.0, 6.0]])
    path = SimpleNamespace(wpnts=[waypoint(i) for i in range(3)])
    node = SimpleNamespace(
        ggv=original_ggv.copy(),
        ax_max_machines=original_ax.copy(),
        b_ax_max_machines=original_braking.copy(),
        gb_wpnts=SimpleNamespace(wpnts=[waypoint(i) for i in range(3)]),
        wpnt_dist=1.0,
        cur_vs=1.0,
        pars={
            "veh_params": {"dragcoeff": 0.0, "mass": 3.0, "v_max": 10.0},
            "vel_calc_opts": {"vel_profile_conv_filt_window": None, "dyn_model_exp": 1.0},
        },
        params=SimpleNamespace(rqt_mu_scale=1.0, rqt_dyn_exp=1.0),
        name="test_state_machine",
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )

    StateMachine.update_velocity(node, path, ggv_scale_factor=0.8)

    np.testing.assert_allclose(captured["ggv"][:, 0], original_ggv[:, 0])
    np.testing.assert_allclose(captured["ggv"][:, 1:3], original_ggv[:, 1:3] * 0.8)
    np.testing.assert_allclose(captured["ax_max_machines"], original_ax)
    np.testing.assert_allclose(captured["b_ax_max_machines"], original_braking)

    # Scaling must not accumulate by mutating any vehicle-dynamics source array.
    np.testing.assert_allclose(node.ggv, original_ggv)
    np.testing.assert_allclose(node.ax_max_machines, original_ax)
    np.testing.assert_allclose(node.b_ax_max_machines, original_braking)


def test_ggv_scale_s_range_uses_local_scale(monkeypatch):
    captured = {}

    def fake_calc_vel_profile(**kwargs):
        captured.update(kwargs)
        return np.ones(5)

    monkeypatch.setattr(state_machine_node, "calc_vel_profile", fake_calc_vel_profile)
    monkeypatch.setattr(
        state_machine_node.tph.calc_ax_profile,
        "calc_ax_profile",
        lambda **_kwargs: np.zeros(4),
    )

    path = SimpleNamespace(wpnts=[waypoint(i) for i in range(5)])
    for wpnt, s_m in zip(path.wpnts, [33.9, 34.0, 37.0, 40.0, 40.1]):
        wpnt.s_m = s_m

    node = SimpleNamespace(
        ggv=np.array([[0.0, 6.0, 4.0], [10.0, 5.0, 8.0]]),
        ax_max_machines=np.array([[0.0, 4.0], [10.0, 3.0]]),
        b_ax_max_machines=np.array([[0.0, 8.0], [10.0, 6.0]]),
        gb_wpnts=SimpleNamespace(wpnts=[waypoint(i) for i in range(50)]),
        wpnt_dist=1.0,
        cur_vs=1.0,
        pars={
            "veh_params": {"dragcoeff": 0.0, "mass": 3.0, "v_max": 10.0},
            "vel_calc_opts": {"vel_profile_conv_filt_window": None, "dyn_model_exp": 1.0},
        },
        params=SimpleNamespace(rqt_mu_scale=1.0, rqt_dyn_exp=1.0),
        name="test_state_machine",
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )

    StateMachine.update_velocity(
        node,
        path,
        ggv_scale_factor=0.3,
        ggv_scale_s_range=(34.0, 40.0, 0.1),
    )

    np.testing.assert_allclose(captured["ggv"], node.ggv)
    np.testing.assert_allclose(captured["mu"], [0.3, 0.1, 0.1, 0.1, 0.3])
    np.testing.assert_allclose(captured["ax_max_machines"], node.ax_max_machines)
