#!/usr/bin/env python3
"""Headless mass-lap tuning: f110_gym stepped at CPU speed + Controller direct.

No ROS graph, no wall clock, no lidar consumers: the gym physics engine (the
SAME one gym_bridge wraps, same dynamics.yaml, same numba-free shim) is stepped
as fast as the CPU allows, with controller/combined/src/Controller.py driven
directly at an exact 50 Hz control rate (2 physics steps @ 100 Hz).

Fidelity gaps vs the full stack (validate top-k in the real sim before
adopting!): no state_machine/local-planner hop, physics integrated at 100 Hz
instead of 80 Hz. The odom->controller->VESC->gym actuation latency IS
modeled (LATENCY_S command-delay line) since 2026-07-23: without it the
optimizer exploited zero-latency tracking and its params went off at the
wall-hugging corner (s~20-24 m) in the real sim.

Usage (conda unicorn env, workspace sourced):
    python fast_tune.py --smoke                 # one evaluation with yaml params
    python fast_tune.py --n-trials 2000 --workers 8
    python fast_tune.py --show-best [--max-d 0.08] [--by lap]

Reuses tuner_config.yaml (params/frozen/objective). Study lives in a
multiprocess-safe Optuna JournalStorage file (journal_fast.log).
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import yaml

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_stack():
    """Locate the stack root, so this runs from a clone at any path.

    Two layouts are supported: tuning/ sitting inside the repo (<stack>/tuning,
    what you get from `git clone`) and tuning/ at the colcon workspace root
    (<ws>/tuning, the layout these scripts were written in). Set UNICORN_STACK
    to override when the tree lives somewhere else entirely.
    """
    parent = os.path.dirname(TUNING_DIR)
    marker = "controller/controller/combined/src/Controller.py"
    for cand in (os.environ.get("UNICORN_STACK"), parent,
                 os.path.join(parent, "src", "unicorn-racing-stack")):
        if cand and os.path.isfile(os.path.join(cand, marker)):
            return os.path.abspath(cand)
    sys.exit(f"[fast_tune] stack root not found next to {TUNING_DIR}\n"
             f"            expected {marker} under it or under ../src/"
             "unicorn-racing-stack\n"
             "            set UNICORN_STACK=/path/to/unicorn-racing-stack")


STACK = _find_stack()
CFG_DEFAULT = os.path.join(TUNING_DIR, "tuner_config.yaml")
JOURNAL = os.path.join(TUNING_DIR, "journal_fast.log")

sys.path.insert(0, f"{STACK}/controller/controller/combined/src")
sys.path.insert(0, f"{STACK}/race_utils/f110_utils/libs/frenet_conversion")

# numba shim + range_libc scan swap; MUST import before gym.make (same as bridge)
from f1tenth_gym_ros import numba_free  # noqa: F401,E402
import gymnasium as gym  # noqa: E402

# node-split stack (2026-07-25): Controller.py = original pure-PP,
# map_controller.py = MAP variant (chord-cap + lookup feedforward). --ctrl
# picks which one is tuned; each has its own yaml and its own search dims.
from Controller import Controller as PPController  # noqa: E402
from map_controller import Controller as MapController  # noqa: E402
from frenet_conversion.frenet_converter import FrenetConverter  # noqa: E402
from steering_lookup.lookup_steer_angle import LookupSteerAngle  # noqa: E402

# map whose raceline + speed_scaling the trials drive. UNICORN_MAP=<name> picks
# another map under stack_master/maps/ (it must carry global_waypoints.json).
MAP_NAME = os.environ.get("UNICORN_MAP", "s")
MAP_DIR = f"{STACK}/stack_master/maps/{MAP_NAME}"
SIM_DIR = f"{STACK}/stack_master/config/SIM"
CTRL_YAMLS = {
    "pp": f"{STACK}/stack_master/config/controller.yaml",
    "map": f"{STACK}/stack_master/config/controller_map.yaml",
}
# search dims that only exist in the MAP controller
MAP_ONLY_DIMS = ("l1_chord_err", "lat_err_steer_coeff")
# dims searched for PP only. In MAP the chord-error cap (l1_chord_err) already
# shortens L1 on previewed curvature, so searching curvature_factor alongside it
# lets the two mechanisms cancel each other out and wastes a dimension. Excluded
# from MAP studies 2026-07-25 (user request); MAP then uses whatever value
# controller_map.yaml carries.
PP_ONLY_DIMS = ("curvature_factor",)
# MAP steering-lookup table for the feedforward. v7b: UNICORN2-0410 (the real
# car's measured table, what the car actually runs) instead of SIM_linear, so
# the MAP tuning sees the same steering response as the vehicle.
MAP_LU_TABLE = "UNICORN2-0410"

PHYS_HZ = 100.0          # physics integration rate (bridge: 80; finer is fine)
CTRL_EVERY = 2           # -> control at exactly 50 Hz, matching the stack
# actuation latency model: commands take effect this long after they are
# computed (odom staleness at the 50 Hz pick-up + compute + DDS + the 80 Hz
# drive-apply step in gym_bridge). Calibrated 2026-07-23 against the real-sim
# replay of v5 #6327 (lap 10.7 s, wall contact at s~21). Rounded to physics
# steps (10 ms each).
LATENCY_S = 0.03
SCALING_KEY = "/speed_sector_tuner:Sector0.scaling"

# args consumed by Controller.__init__ in order (mirror controller_manager)
CTRL_ARGS = [
    "t_clip_min", "t_clip_max", "m_l1", "q_l1", "curvature_factor",
    "KP", "KI", "KD", "heading_error_thres", "steer_gain_for_speed",
    "future_constant", "speed_lookahead", "lat_err_coeff",
    "acc_scaler_for_steer", "dec_scaler_for_steer", "start_scale_speed",
    "end_scale_speed", "downscale_factor", "speed_lookahead_for_steer",
    "trailing_gap", "trailing_vel_gain", "trailing_p_gain", "trailing_i_gain",
    "trailing_d_gain", "blind_trailing_speed",
]


class FastLapSim:
    """One gym env + raceline; evaluates a param dict -> lap metrics."""

    def __init__(self, latency_s=LATENCY_S, ctrl="pp"):
        self.ctrl = ctrl
        self.lat_steps = int(round(latency_s * PHYS_HZ))
        d = json.load(open(f"{MAP_DIR}/global_waypoints.json"))
        wpnts = d["global_traj_wpnts_iqp"]["wpnts"]
        self.track_length = wpnts[-1]["s_m"]
        # [x, y, v, 0, s, kappa, psi, ax, d] like controller_manager builds
        base = np.array([[w["x_m"], w["y_m"], w["vx_mps"], 0.0, w["s_m"],
                          w["kappa_radpm"], w["psi_rad"], w["ax_mps2"], 0.0]
                         for w in wpnts])
        # tile 2 laps so [idx_nearest:] always has a full lap ahead (the live
        # stack's local-waypoint window does this implicitly)
        second = base.copy()
        second[:, 4] += self.track_length
        self.base_wpnts = np.vstack([base, second])
        self.converter = FrenetConverter(base[:, 0], base[:, 1])
        # start ~1 m past the line (same as the live tuner reset point)
        w0 = next(w for w in wpnts if w["s_m"] >= 1.0)
        self.start_pose = [w0["x_m"], w0["y_m"], w0["psi_rad"]]

        dyn = yaml.safe_load(open(f"{SIM_DIR}/dynamics.yaml"))
        sim_params = {k: float(v) for k, v in dyn.items()}
        self.env = gym.make("f110_gym:f110-v0",
                            map=f"{MAP_DIR}/s", map_ext=".png",
                            params=sim_params, num_agents=1,
                            timestep=1.0 / PHYS_HZ,
                            num_beams=32, scan_fov=4.7,   # scan unused; keep cheap
                            disable_env_checker=True)
        self.yaml_params = yaml.safe_load(open(CTRL_YAMLS[ctrl]))[
            "controller_manager"]["ros__parameters"]
        self.steer_lookup = (LookupSteerAngle(MAP_LU_TABLE, print)
                             if ctrl == "map" else None)

    def _make_controller(self, p):
        """Controller with yaml values overridden by candidate params p."""
        v = dict(self.yaml_params)
        v.update({k: val for k, val in p.items() if not k.startswith("/")})
        common = dict(
            loop_rate=PHYS_HZ / CTRL_EVERY,
            wheelbase=0.321,
            speed_factor_for_lat_err=float(v["speed_factor_for_lat_err"]),
            speed_factor_for_curvature=float(v["speed_factor_for_curvature"]),
            speed_diff_thres=float(v["speed_diff_thres"]),
            start_speed=float(v["start_speed"]),
            start_curvature_factor=float(v["start_curvature_factor"]),
            AEB_thres=float(v["AEB_thres"]),
            converter=self.converter,
            logger_info=lambda *a, **k: None,
            logger_warn=lambda *a, **k: None,
        )
        if self.ctrl == "map":
            return MapController(
                *[float(v[k]) for k in CTRL_ARGS], **common,
                steer_lookup=self.steer_lookup, use_map=True,
                l1_chord_err=float(v["l1_chord_err"]),
                lat_err_steer_coeff=float(v["lat_err_steer_coeff"]),
            )
        return PPController(
            *[float(v[k]) for k in CTRL_ARGS], **common,
            use_chord_cap=bool(v.get("use_chord_cap", False)),
            l1_chord_err=float(v.get("l1_chord_err", 1.0)),
        )

    # n_laps=4 since v6b (was 2): the sim is deterministic per lap, but
    # marginal sets degrade ACROSS laps (slowly building oscillation, lap-3/4
    # departure) - 4 measured laps folds that robustness into the cost.
    @staticmethod
    def _worst_window(x, win):
        """Max over sliding windows of the per-step-delta RMS. Whole-lap RMS
        averages a single bad corner away; this surfaces the WORST local
        oscillation (the C-corner L1-collapse the whole-lap metric missed)."""
        dx = np.diff(np.asarray(x, dtype=float))
        if dx.size < win:
            return float(np.sqrt(np.mean(dx ** 2))) if dx.size else 0.0
        # sliding-window RMS via cumulative sum of squares
        c = np.concatenate([[0.0], np.cumsum(dx ** 2)])
        win_ss = c[win:] - c[:-win]
        return float(np.sqrt(win_ss.max() / win))

    def evaluate(self, p, n_laps=4, max_lap_time=60.0, offtrack_d=0.7,
                 record=None):
        """Standing start + warmup lap + n measured laps. Mirrors the live
        tuner's metrics. Raises RuntimeError(progress_frac) on failure.
        record: optional list; measured-lap samples are appended as
        (lap_no, s, x, y, d, speed, steer_cmd)."""
        scaling = float(p.get(SCALING_KEY, 1.0))
        wp = self.base_wpnts.copy()
        wp[:, 2] *= scaling                      # sector_tuner equivalent
        ctrl = self._make_controller(p)

        obs, _, done, _ = self.env.reset(poses=np.array([self.start_pose]))
        dt_ctrl = CTRL_EVERY / PHYS_HZ
        # actuation delay line: each physics step applies the command issued
        # lat_steps steps earlier (seeded with a standstill command)
        delay_line = deque([(0.0, 0.0)] * self.lat_steps)
        t = 0.0
        deadline = (n_laps + 2) * max_lap_time
        prev_s = None
        laps_done, lap_start_t = -1, 0.0
        lap_d, lap_steer, lap_metrics = [], [], []
        progressed, stuck_t, stuck_s = 0.0, 0.0, 0.0
        steer_cmd, speed_cmd = 0.0, 0.0
        acc_hist = np.zeros(10)
        prev_speed = 0.0

        def fail(reason):
            total = (n_laps + 1) * self.track_length
            raise RuntimeError(f"{reason}|{min(progressed / total, 1.0):.4f}")

        while True:
            # ---- control tick (50 Hz) ----
            x, y = obs["poses_x"][0], obs["poses_y"][0]
            yaw = obs["poses_theta"][0]
            speed = obs["linear_vels_x"][0]
            ctrl.yaw_rate = obs["ang_vels_z"][0]
            acc_hist[1:] = acc_hist[:-1]
            acc_hist[0] = (speed - prev_speed) / dt_ctrl
            prev_speed = speed
            s, d = self.converter.get_frenet(np.array([x]), np.array([y]))
            s, d = float(s[0]), float(d[0])

            if obs["collisions"][0]:
                fail("collision")
            if abs(d) > offtrack_d:
                fail("offtrack")
            if t > deadline:
                fail("timeout")

            # lap accounting on s wrap
            if prev_s is not None and (prev_s - s) > self.track_length / 2:
                if laps_done == -1:
                    laps_done, lap_start_t = 0, t
                    lap_d, lap_steer = [], []
                else:
                    dsg = np.asarray(lap_d)
                    st = np.asarray(lap_steer)
                    # ~1 s window (50 samples @ 50 Hz) = roughly one corner
                    win = int(round(1.0 / dt_ctrl))
                    lap_metrics.append((
                        t - lap_start_t,
                        float(np.mean(np.abs(dsg))), float(np.max(np.abs(dsg))),
                        float(np.sqrt(np.mean(np.diff(dsg) ** 2))),
                        float(np.sqrt(np.mean(np.diff(st) ** 2))),
                        self._worst_window(st, win),   # worst-corner steer osc
                        self._worst_window(dsg, win),  # worst-corner d weave
                    ))
                    laps_done += 1
                    lap_start_t = t
                    lap_d, lap_steer = [], []
                    if laps_done >= n_laps:
                        break
            prev_s = s
            if laps_done >= 0:
                lap_d.append(d)
                lap_steer.append(steer_cmd)
                if record is not None:
                    record.append((laps_done, s, x, y, d, speed, steer_cmd))

            # stuck watchdog (4 s, mirrors live tuner)
            progressed += max(speed, 0.0) * dt_ctrl
            if t - stuck_t > 4.0:
                if progressed - stuck_s < 0.3:
                    fail("stuck")
                stuck_t, stuck_s = t, progressed

            # controller step (same call the manager makes)
            idx = int(np.argmin(np.hypot(wp[:len(wp)//2, 0] - x,
                                         wp[:len(wp)//2, 1] - y)))
            local = wp[idx:idx + len(wp)//2]     # rolling full-lap window
            out = ctrl.main_loop("GB_TRACK", np.array([[x, y, yaw]]), local,
                                 speed, None, [s, d, speed], acc_hist,
                                 self.track_length)
            speed_cmd, steer_cmd = out[0], out[3]

            # ---- physics (2 x 100 Hz, commands through the delay line) ----
            for _ in range(CTRL_EVERY):
                delay_line.append((steer_cmd, speed_cmd))
                st_cmd, sp_cmd = delay_line.popleft()
                obs, _, done, _ = self.env.step(
                    np.array([[st_cmd, sp_cmd]]))
            t += dt_ctrl

        return {
            "lap_times": [m[0] for m in lap_metrics],
            "mean_abs_d": float(np.mean([m[1] for m in lap_metrics])),
            "max_abs_d": float(np.max([m[2] for m in lap_metrics])),
            "d_weave_rms": float(np.mean([m[3] for m in lap_metrics])),
            "dsteer_rms": float(np.mean([m[4] for m in lap_metrics])),
            # worst local (per-corner) oscillation, max across laps
            "dsteer_worst": float(np.max([m[5] for m in lap_metrics])),
            "d_weave_worst": float(np.max([m[6] for m in lap_metrics])),
        }


# ---------------------------------------------------------------------------
def pick_best(study, max_d=None, by="cost"):
    ok = [t for t in study.trials
          if t.value is not None and t.user_attrs.get("lap_times")]
    if max_d is not None:
        ok = [t for t in ok if t.user_attrs.get("mean_abs_d", 9e9) <= max_d]
    if not ok:
        return None
    if by == "lap":
        return min(ok, key=lambda t: float(np.mean(t.user_attrs["lap_times"])))
    return min(ok, key=lambda t: t.value)


def plot_best(study, cfg, max_d=None, by="cost", ctrl="pp",
              out=os.path.join(TUNING_DIR, "best_lap_analysis.png")):
    """Re-drive the best trial's params headless, then graph the driven lap
    against the global raceline. Colors: validated palette slots 1 (blue) and
    2 (orange), reference geometry in neutral gray."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pick_best(study, max_d=max_d, by=by)
    if t is None:
        print("no qualifying trial to plot.")
        return
    print(f"re-driving trial #{t.number} (cost {t.value:.2f}) for the plot...")
    sim = FastLapSim(ctrl=ctrl)
    rec = []
    m = sim.evaluate(dict(t.params), record=rec)
    rec = np.array(rec)                       # lap, s, x, y, d, v, steer
    lap_times = m["lap_times"]
    fastest = int(np.argmin(lap_times))
    lap = rec[rec[:, 0] == fastest]
    s_ord = np.argsort(lap[:, 1])
    lap = lap[s_ord]

    scaling = float(t.params.get(SCALING_KEY, 1.0))
    base = sim.base_wpnts[:len(sim.base_wpnts) // 2]
    d_budget = cfg["objective"].get("d_hard_limit", None)

    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9a94"
    INK, MUT = "#333330", "#6b6b64"
    plt.rcParams.update({"font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUT,
                         "ytick.color": MUT, "axes.edgecolor": "#d9d9d3",
                         "axes.grid": True, "grid.color": "#ecece6",
                         "grid.linewidth": 0.6})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(
        f"Best lap vs global raceline — trial #{t.number}   "
        f"lap {lap_times[fastest]:.2f}s   mean|d| {m['mean_abs_d']:.3f}m   "
        f"scaling {scaling:.2f}", fontsize=11, color=INK)

    ax = axs[0][0]                                     # track map
    ax.plot(base[:, 0], base[:, 1], "--", color=GRAY, lw=1.4,
            label="global raceline")
    ax.plot(lap[:, 2], lap[:, 3], color=BLUE, lw=2.0, label="driven path")
    ax.plot(lap[0, 2], lap[0, 3], "o", color=BLUE, ms=8, mec="white", mew=1.5)
    ax.set_aspect("equal")
    ax.set_title("Track map", color=INK)
    ax.legend(frameon=False, loc="best")

    ax = axs[0][1]                                     # lateral deviation
    ax.axhline(0, color=GRAY, lw=1.0)
    if d_budget:
        ax.axhspan(-d_budget, d_budget, color=BLUE, alpha=0.06, lw=0)
    ax.plot(lap[:, 1], lap[:, 4], color=BLUE, lw=2.0)
    ax.set_title(f"Lateral deviation d(s)  (band = ±{d_budget} m budget)",
                 color=INK)
    ax.set_xlabel("s [m]"); ax.set_ylabel("d [m]")

    ax = axs[1][0]                                     # speed vs profile
    ax.plot(base[:, 4], base[:, 2] * scaling, color=ORANGE, lw=2.0,
            label=f"profile x {scaling:.2f}")
    ax.plot(lap[:, 1], lap[:, 5], color=BLUE, lw=2.0, label="actual")
    ax.set_title("Speed", color=INK)
    ax.set_xlabel("s [m]"); ax.set_ylabel("v [m/s]")
    ax.legend(frameon=False, loc="best")

    ax = axs[1][1]                                     # steering
    ax.axhline(0, color=GRAY, lw=1.0)
    ax.plot(lap[:, 1], lap[:, 6], color=BLUE, lw=2.0)
    ax.set_title(f"Steering command  (weave {m['d_weave_rms']:.4f})", color=INK)
    ax.set_xlabel("s [m]"); ax.set_ylabel("steer [rad]")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


def load_cfg(path):
    return yaml.safe_load(open(path))


def make_objective(cfg, ctrl="pp"):
    ob = cfg["objective"]
    space = {k: v for k, v in cfg["params"].items() if v.get("enabled")
             and not (ctrl == "pp" and k in MAP_ONLY_DIMS)
             and not (ctrl == "map" and k in PP_ONLY_DIMS)}
    frozen = {k: v for k, v in (cfg.get("frozen") or {}).items()
              if not k.startswith("/")}       # node-level frozen keys are N/A here
    sim = FastLapSim(ctrl=ctrl)

    def objective(trial):
        p = dict(frozen)
        for name, b in space.items():
            p[name] = trial.suggest_float(name, b["low"], b["high"],
                                          step=b.get("step"))
        try:
            m = sim.evaluate(p)
        except RuntimeError as e:
            reason, frac = str(e).rsplit("|", 1)
            cost = ob["fail_penalty"] - ob["progress_bonus"] * float(frac)
            trial.set_user_attr("fail", reason)
            return cost
        mean_lap = float(np.mean(m["lap_times"]))
        d_over = max(0.0, m["mean_abs_d"] - ob.get("d_hard_limit", float("inf")))
        cost = (mean_lap
                + ob["w_lat_err"] * m["mean_abs_d"]
                + ob.get("w_d_over", 0.0) * d_over
                + ob["w_osc"] * m["dsteer_rms"]
                + ob.get("w_weave", 0.0) * m["d_weave_rms"]
                # worst-corner oscillation: penalizes a single bad corner the
                # whole-lap RMS averages away (the C-corner L1 collapse)
                + ob.get("w_osc_worst", 0.0) * m["dsteer_worst"]
                + ob.get("w_weave_worst", 0.0) * m["d_weave_worst"])
        for k in ("lap_times", "mean_abs_d", "dsteer_rms", "d_weave_rms",
                  "dsteer_worst", "d_weave_worst"):
            trial.set_user_attr(k, m[k])
        return cost

    return objective


def get_storage():
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(JOURNAL))


def run_worker(study_name, cfg, n_trials, seed, ctrl="pp"):
    import optuna
    import warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # multivariate: joint KDE over the (heavily correlated) L1/scaling dims
    # instead of per-dim independent densities.
    # constant_liar: workers see each other's RUNNING trials (with a pessimistic
    # placeholder cost) so 12 parallel samplers don't pile onto the same point.
    study = optuna.load_study(study_name=study_name, storage=get_storage(),
                              sampler=optuna.samplers.TPESampler(
                                  seed=seed, multivariate=True,
                                  constant_liar=True))
    study.optimize(make_objective(cfg, ctrl), n_trials=n_trials)


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_DEFAULT)
    # fast_v3: +6 stage-1b scaler dims (19 total). History: v1 KP pinned at
    # 0.6 -> v2 widened to 1.2 (best hit 0.89) -> v3 adds the scaler knobs.
    # fast_v4: oscillation weights x4 (w_osc 32, w_weave 1000) - same search
    # space, new objective -> fresh study so v3 costs don't mix into ranking.
    # fast_v5: x2 more (w_osc 64, w_weave 2000) - v4 top-10 was still ranked
    # by lap time; v5 makes smoothness dominate the fine ranking.
    # fast_v6: 30 ms actuation-latency model added (calibrated: v5 best
    # reproduced its real-sim wall crash, live-tuned v3b reproduced its real
    # 8.2 s laps). v5's zero-latency results are NOT comparable -> fresh study.
    # fast_v6b: 4 measured laps per trial (was 2) - long-horizon stability
    # now in the cost; 2-lap costs aren't comparable -> fresh study.
    # v7 2026-07-25: NEW PHYSICS (proportional servo, sv_max 10 = B210-class)
    # + node-split stack -> --ctrl picks pure-PP or MAP controller. Studies
    # named per mode; all pre-v7 results used the old bang-bang plant and are
    # NOT comparable.
    ap.add_argument("--ctrl", choices=["pp", "map"], default="pp",
                    help="which controller to tune: pp = original pure-"
                         "Pursuit (controller.yaml), map = MAP variant "
                         "(controller_map.yaml)")
    ap.add_argument("--study", default=None,
                    help="default: controller_fast_v7c_<ctrl>")
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--smoke", action="store_true",
                    help="single evaluation with current yaml params")
    ap.add_argument("--show-best", action="store_true")
    ap.add_argument("--no-apply", action="store_true",
                    help="with --show-best: only print/snapshot, do NOT write "
                         "the tuned values into controller[_map].yaml")
    ap.add_argument("--plot-best", action="store_true",
                    help="re-drive the best trial and graph it vs the raceline")
    ap.add_argument("--max-d", type=float, default=None)
    ap.add_argument("--by", choices=["cost", "lap"], default="cost")
    args = ap.parse_args()
    if args.study is None:
        args.study = f"controller_fast_v7c_{args.ctrl}"
    cfg = load_cfg(args.config)

    if args.smoke:
        sim = FastLapSim(ctrl=args.ctrl)
        scaling = yaml.safe_load(open(
            f"{MAP_DIR}/speed_scaling.yaml"))["speed_sector_tuner"][
            "ros__parameters"]["Sector0"]["scaling"]
        t0 = time.monotonic()
        m = sim.evaluate({SCALING_KEY: scaling})
        el = time.monotonic() - t0
        print(f"smoke ({args.ctrl} yaml params, scaling {scaling}):")
        print(f"  laps {['%.2f' % x for x in m['lap_times']]}  "
              f"|d| {m['mean_abs_d']:.3f}  weave {m['d_weave_rms']:.4f}  "
              f"osc {m['dsteer_rms']:.4f}")
        print(f"  wall time {el:.1f}s for 3 laps -> {3/el:.1f} laps/s/core")
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=args.study, storage=get_storage(),
                                direction="minimize", load_if_exists=True)

    if args.plot_best:
        plot_best(study, cfg, max_d=args.max_d, by=args.by, ctrl=args.ctrl)
        return

    if args.show_best:
        sys.path.insert(0, TUNING_DIR)
        from tune_controller import print_best
        # apply into the live config for this controller (pp->controller.yaml,
        # map->controller_map.yaml) unless --no-apply
        apply_to = None if args.no_apply else CTRL_YAMLS[args.ctrl]
        print_best(study, cfg, max_d=args.max_d, by=args.by,
                   out_name=args.study, apply_to=apply_to)
        return

    # baseline: current yaml + saved scaling
    states = [t.state.name for t in study.trials]
    if "COMPLETE" not in states and "WAITING" not in states:
        yp = yaml.safe_load(open(CTRL_YAMLS[args.ctrl]))["controller_manager"]["ros__parameters"]
        sc = yaml.safe_load(open(f"{MAP_DIR}/speed_scaling.yaml"))[
            "speed_sector_tuner"]["ros__parameters"]["Sector0"]["scaling"]
        enabled = {k: v for k, v in cfg["params"].items() if v.get("enabled")
                   and not (args.ctrl == "pp" and k in MAP_ONLY_DIMS)
                   and not (args.ctrl == "map" and k in PP_ONLY_DIMS)}
        baseline = {}
        for k, b in enabled.items():
            raw = sc if k == SCALING_KEY else yp.get(k)
            if raw is None:
                baseline = None
                break
            if b.get("step"):
                raw = round(round(raw / b["step"]) * b["step"], 10)
            baseline[k] = float(np.clip(raw, b["low"], b["high"]))
        if baseline:
            study.enqueue_trial(baseline)
            print(f"baseline enqueued: {baseline}")

    print(f"study '{args.study}' ({args.ctrl}): {args.n_trials} trials x {args.workers} workers")
    import multiprocessing as mp
    per = max(args.n_trials // args.workers, 1)
    procs = [mp.Process(target=run_worker,
                        args=(args.study, cfg, per, 1000 + i, args.ctrl))
             for i in range(args.workers)]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\ninterrupted - journal saved, rerun to resume.")
        for p in procs:
            p.terminate()
    el = time.monotonic() - t0
    done = len([t for t in optuna.load_study(study_name=args.study,
                storage=get_storage()).trials if t.value is not None])
    print(f"\n{done} trials in study after {el/60:.1f} min")
    sys.path.insert(0, TUNING_DIR)
    from tune_controller import print_best
    print_best(optuna.load_study(study_name=args.study, storage=get_storage()),
               cfg, max_d=args.max_d, by=args.by, out_name=args.study)


if __name__ == "__main__":
    main()
