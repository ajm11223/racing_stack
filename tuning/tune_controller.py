#!/usr/bin/env python3
"""Closed-loop controller tuning with Optuna (stage 1: solo racing).

Runs against an ALREADY RUNNING sim stack:
    ros2 launch stack_master race.launch.xml sim:=true map:=s [use_map:=true]

Per trial: set candidate params live on /controller_manager, teleport the car
to the raceline start via /initialpose, let it complete a standing-start
warmup lap, then time k laps while recording tracking error (|d| from frenet
odom) and steering oscillation (RMS of per-cycle steer delta). Crash / stuck /
off-track trials fail early with a penalty.

Usage (conda unicorn env, NO /opt/ros/jazzy):
    python tune_controller.py [--config tuner_config.yaml] [--n-trials N]
    python tune_controller.py --show-best        # print best params and exit

The study is stored in SQLite and resumes across invocations. Ctrl-C is safe.
"""

import argparse
import math
import os
import threading
import time

import numpy as np
import optuna
import rclpy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node

CFG_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuner_config.yaml")


class TrialFailed(Exception):
    def __init__(self, reason, progress_frac):
        super().__init__(reason)
        self.reason = reason
        self.progress_frac = progress_frac  # [0,1] fraction of the trial distance covered


class TunerNode(Node):
    def __init__(self, run_cfg):
        super().__init__("controller_tuner")
        self.cfg = run_cfg
        self.lock = threading.Lock()

        # live state
        self.s = None
        self.d = None
        self.last_odom_wall = 0.0
        self.track_length = None
        self.start_pose = None  # (x, y, psi)

        # per-trial recording buffers (guarded by lock)
        self.recording = False
        self.samples = []  # (t_stamp, s, d)
        self.steers = []   # steering commands

        self.create_subscription(Odometry, run_cfg["odom_frenet_topic"], self._odom_cb, 10)
        self.create_subscription(AckermannDriveStamped, run_cfg["drive_topic"], self._drive_cb, 10)
        self.create_subscription(WpntArray, run_cfg["global_wpnts_topic"], self._wpnts_cb, 10)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, run_cfg["initialpose_topic"], 10)
        # per-node parameter service clients, created lazily. Param keys may be
        # plain names (-> run.node_name) or "/other_node:param_name" to tune a
        # different node (e.g. "/speed_sector_tuner:Sector0.scaling").
        self._set_clis = {}
        self._get_clis = {}

    # ------------------------------ callbacks ------------------------------
    def _odom_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.s = msg.pose.pose.position.x
            self.d = msg.pose.pose.position.y
            self.last_odom_wall = time.monotonic()
            if self.recording:
                self.samples.append((t, self.s, self.d))

    def _drive_cb(self, msg):
        with self.lock:
            if self.recording:
                self.steers.append(msg.drive.steering_angle)

    def _wpnts_cb(self, msg):
        if self.track_length is not None or len(msg.wpnts) < 2:
            return
        # reset target: ~1 m PAST the start line, not s=0 — a parked car exactly
        # on the lap boundary makes its frenet s jitter across the wrap, which
        # spams lap_analyser (and is generally ambiguous).
        w0 = next((w for w in msg.wpnts if w.s_m >= 1.0), msg.wpnts[0])
        self.track_length = msg.wpnts[-1].s_m
        self.start_pose = (w0.x_m, w0.y_m, w0.psi_rad)
        self.get_logger().info(
            f"track_length={self.track_length:.1f} m, "
            f"reset point=({w0.x_m:.2f}, {w0.y_m:.2f}) @ s={w0.s_m:.2f}")

    # ------------------------------ actions --------------------------------
    def wait_ready(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.s is not None and self.track_length is not None:
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError(
                "no odom/waypoints - is the sim stack running? "
                "(ros2 launch stack_master race.launch.xml sim:=true map:=...)")
        # preflight: prove the manager's parameter services actually answer us.
        # Topics working but services timing out usually means an RMW/env mismatch
        # between the sim terminal and this one.
        t0 = time.monotonic()
        try:
            self.get_params(["m_l1"], timeout=8.0)
        except RuntimeError as e:
            raise RuntimeError(
                f"param service preflight FAILED ({e}). Topics work but services "
                "don't -> check both terminals use the same env "
                "(RMW_IMPLEMENTATION, ROS_DOMAIN_ID, same setup.bash). "
                "Quick check in this terminal: ros2 param get "
                f"{self.cfg['node_name']} m_l1")
        print(f"param service preflight OK ({time.monotonic()-t0:.2f}s round-trip)")

    def _call(self, client, req, what, timeout):
        fut = client.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"{what} timed out after {timeout:.0f}s")
            time.sleep(0.02)
        return fut.result()

    def _split_key(self, key):
        """'/node:param' -> (node, param); bare 'param' -> (default node, param)."""
        if key.startswith("/") and ":" in key:
            node, name = key.split(":", 1)
            return node, name
        return self.cfg["node_name"], key

    def _set_cli(self, node):
        if node not in self._set_clis:
            self._set_clis[node] = self.create_client(SetParameters, node + "/set_parameters")
        return self._set_clis[node]

    def _get_cli(self, node):
        if node not in self._get_clis:
            self._get_clis[node] = self.create_client(GetParameters, node + "/get_parameters")
        return self._get_clis[node]

    def get_params(self, keys, timeout=8.0):
        by_node = {}
        for k in keys:
            node, name = self._split_key(k)
            by_node.setdefault(node, []).append((k, name))
        out = {}
        for node, pairs in by_node.items():
            cli = self._get_cli(node)
            if not cli.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"{node}/get_parameters not available")
            req = GetParameters.Request(names=[name for _, name in pairs])
            res = self._call(cli, req, f"get_parameters({node})", timeout)
            for (k, _), v in zip(pairs, res.values):
                out[k] = v.double_value if v.type == ParameterType.PARAMETER_DOUBLE \
                    else float(v.integer_value)
        return out

    def set_params(self, values: dict, timeout=8.0, attempts=3):
        by_node = {}
        for k, val in values.items():
            node, name = self._split_key(k)
            by_node.setdefault(node, {})[name] = val
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                for node, params in by_node.items():
                    cli = self._set_cli(node)
                    if not cli.wait_for_service(timeout_sec=timeout):
                        raise RuntimeError(f"{node}/set_parameters not available")
                    req = SetParameters.Request()
                    for name, val in params.items():
                        req.parameters.append(Parameter(name=name, value=ParameterValue(
                            type=ParameterType.PARAMETER_DOUBLE, double_value=float(val))))
                    res = self._call(cli, req, f"set_parameters({node})", timeout)
                    for name, r in zip(params, res.results):
                        if not r.successful:
                            raise RuntimeError(f"{node} rejected {name}: {r.reason}")
                return
            except RuntimeError as e:
                last_err = e
                print(f"  set_params attempt {attempt}/{attempts} failed: {e}")
                time.sleep(0.5)
        raise RuntimeError(f"set_parameters failed after {attempts} attempts: {last_err}")

    def reset_car(self):
        x, y, psi = self.start_pose
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(psi / 2.0)
        msg.pose.pose.orientation.w = math.cos(psi / 2.0)
        self.reset_pub.publish(msg)

    # ------------------------------ trial ----------------------------------
    def run_trial(self):
        """Warmup lap from standstill, then measure cfg laps. Returns metrics."""
        c = self.cfg
        L = self.track_length
        n_laps = int(c["laps_per_trial"])
        total_dist = (n_laps + 1) * L  # incl. warmup lap, for progress fraction

        with self.lock:
            self.samples, self.steers = [], []
            self.recording = True
        try:
            self.reset_car()
            time.sleep(c["settle_after_reset_s"])

            laps_done = -1          # -1 = in warmup lap
            lap_start_t = None
            lap_metrics = []        # (lap_time, mean_abs_d, max_abs_d)
            lap_samples_lo = 0      # index into self.samples where current lap began
            prev_s = None
            progressed = 0.0
            watch_t0 = time.monotonic()
            watch_s = None
            # standing start is slower: give warmup 2x budget
            deadline = time.monotonic() + c["max_lap_time_s"] * (n_laps + 2)

            while True:
                time.sleep(0.05)
                with self.lock:
                    if not self.samples:
                        continue
                    t, s, d = self.samples[-1]
                    n_samples = len(self.samples)
                    stale = time.monotonic() - self.last_odom_wall
                if stale > 3.0:
                    raise TrialFailed("odom stopped (sim died?)", progressed / total_dist)
                if time.monotonic() > deadline:
                    raise TrialFailed("trial timeout", progressed / total_dist)
                if abs(d) > c["offtrack_d_max_m"]:
                    raise TrialFailed(f"off track |d|={abs(d):.2f}", progressed / total_dist)

                if prev_s is not None:
                    ds = s - prev_s
                    if ds < -L / 2:            # s wrapped -> lap boundary
                        ds += L
                        if laps_done == -1:    # warmup finished
                            laps_done = 0
                            lap_start_t = t
                            lap_samples_lo = n_samples
                        else:
                            with self.lock:
                                lap_d_signed = np.array(
                                    [smp[2] for smp in self.samples[lap_samples_lo:n_samples]])
                            lap_d = np.abs(lap_d_signed)
                            # weave = RMS of per-sample SIGNED d increments:
                            # captures the body swaying across the line (sign
                            # flips included), unlike mean|d| (bias) or
                            # dsteer_rms (command jitter).
                            weave = float(np.sqrt(np.mean(np.diff(lap_d_signed) ** 2))) \
                                if len(lap_d_signed) > 2 else 0.0
                            lap_metrics.append(
                                (t - lap_start_t, float(np.mean(lap_d)), float(np.max(lap_d)),
                                 weave))
                            laps_done += 1
                            lap_start_t = t
                            lap_samples_lo = n_samples
                            if laps_done >= n_laps:
                                break
                    progressed += max(ds, 0.0)
                prev_s = s

                # stuck watchdog
                if watch_s is None or time.monotonic() - watch_t0 > c["stuck_window_s"]:
                    if watch_s is not None and progressed - watch_s < c["stuck_min_progress_m"]:
                        raise TrialFailed("stuck / crashed", progressed / total_dist)
                    watch_s, watch_t0 = progressed, time.monotonic()

            with self.lock:
                steers = np.asarray(self.steers, dtype=float)
            dsteer_rms = float(np.sqrt(np.mean(np.diff(steers) ** 2))) if len(steers) > 10 else 0.0
            return {
                "lap_times": [m[0] for m in lap_metrics],
                "mean_abs_d": float(np.mean([m[1] for m in lap_metrics])),
                "max_abs_d": float(np.max([m[2] for m in lap_metrics])),
                "dsteer_rms": dsteer_rms,
                "d_weave_rms": float(np.mean([m[3] for m in lap_metrics])),
            }
        finally:
            with self.lock:
                self.recording = False


# ---------------------------------------------------------------------------
def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_objective(node, cfg):
    ob = cfg["objective"]
    space = {k: v for k, v in cfg["params"].items() if v.get("enabled")}
    dead = {"n": 0}  # consecutive sim-dead failures (unattended-run guard)

    def objective(trial):
        values = dict(cfg.get("frozen") or {})
        for name, b in space.items():
            # step: quantize suggestions for params whose node enforces a
            # FloatingPointRange step (e.g. sector_tuner's 0.01 grid)
            values[name] = trial.suggest_float(name, b["low"], b["high"], step=b.get("step"))

        try:
            node.set_params(values)
            m = node.run_trial()
        except TrialFailed as e:
            cost = ob["fail_penalty"] - ob["progress_bonus"] * e.progress_frac
            print(f"  trial {trial.number}: FAILED ({e.reason}) -> {cost:.2f}")
            # sim gone (not a bad-parameter crash): don't burn the whole night
            # on instant failures — stop the study, it resumes on rerun.
            if "odom stopped" in e.reason or "timeout" in e.reason:
                dead["n"] += 1
                if dead["n"] >= 3:
                    print("sim looks dead (3 consecutive) - stopping study; "
                          "restart sim then rerun tune to resume.")
                    trial.study.stop()
            else:
                dead["n"] = 0
            return cost
        except Exception as e:
            # ROS-level fatality (rclpy context invalidated, service gone after
            # retries): stop the study cleanly instead of crashing mid-night.
            # Everything so far is in the DB; rerun resumes.
            print(f"  trial {trial.number}: FATAL ({type(e).__name__}: {e}) - stopping study")
            trial.study.stop()
            return ob["fail_penalty"]
        dead["n"] = 0

        mean_lap = float(np.mean(m["lap_times"]))
        # hard lateral-error budget: below d_hard_limit only the soft weight
        # applies; every meter over it costs w_d_over -> "fastest lap that
        # STAYS on the line", instead of trading line for speed.
        d_over = max(0.0, m["mean_abs_d"] - ob.get("d_hard_limit", float("inf")))
        cost = (mean_lap
                + ob["w_lat_err"] * m["mean_abs_d"]
                + ob.get("w_d_over", 0.0) * d_over
                + ob["w_osc"] * m["dsteer_rms"]
                + ob.get("w_weave", 0.0) * m["d_weave_rms"])
        trial.set_user_attr("lap_times", m["lap_times"])
        trial.set_user_attr("mean_abs_d", m["mean_abs_d"])
        trial.set_user_attr("dsteer_rms", m["dsteer_rms"])
        trial.set_user_attr("d_weave_rms", m["d_weave_rms"])
        print(f"  trial {trial.number}: lap={mean_lap:.2f}s |d|={m['mean_abs_d']:.3f}m "
              f"osc={m['dsteer_rms']:.4f} weave={m['d_weave_rms']:.4f} -> {cost:.2f}")
        return cost

    return objective


def print_best(study, cfg, max_d=None, by="cost"):
    """Pick the best trial. Default: lowest cost. With --max-d, only trials
    whose mean |d| stayed under the threshold qualify; --by lap ranks by raw
    lap time instead of the weighted cost."""
    done = [t for t in study.trials if t.value is not None]
    if not done:
        print("no completed trials yet.")
        return
    cands = done
    if max_d is not None:
        cands = [t for t in cands if t.user_attrs.get("mean_abs_d", float("inf")) <= max_d]
        if not cands:
            worst = min(t.user_attrs.get("mean_abs_d", float("inf")) for t in done)
            print(f"no trial with mean|d| <= {max_d} (tightest so far: {worst:.3f} m)")
            return
    if by == "lap":
        cands = [t for t in cands if t.user_attrs.get("lap_times")]
        t = min(cands, key=lambda t: float(np.mean(t.user_attrs["lap_times"])))
    else:
        t = min(cands, key=lambda t: t.value)
    lap = float(np.mean(t.user_attrs["lap_times"])) if t.user_attrs.get("lap_times") else None
    d = t.user_attrs.get("mean_abs_d")
    print(f"\n=== best trial #{t.number}  cost={t.value:.3f}"
          + (f"  lap={lap:.2f}s" if lap is not None else "")
          + (f"  mean|d|={d:.3f}m" if d is not None else "")
          + (f"  (filter: mean|d|<={max_d}, by {by})" if max_d is not None else "")
          + " ===")
    for k, v in sorted(t.params.items()):
        print(f"  {k:28s} {v:.4f}")
    print("\napply live:")
    for k, v in sorted(t.params.items()):
        if k.startswith("/") and ":" in k:
            node, name = k.split(":", 1)
        else:
            node, name = cfg["run"]["node_name"], k
        print(f"  ros2 param set {node} {name} {v:.4f}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"best_params_{cfg['study']['name']}.yaml")
    with open(out, "w") as f:
        yaml.dump({"cost": float(t.value), "trial": t.number,
                   "params": {k: float(v) for k, v in t.params.items()},
                   "user_attrs": dict(t.user_attrs)}, f, sort_keys=False)
    print(f"\nsaved -> {out}\n(then: ros2 param set {cfg['run']['node_name']} save_params true)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_DEFAULT)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--show-best", action="store_true")
    ap.add_argument("--max-d", type=float, default=None,
                    help="only consider trials with mean |lateral error| <= this [m]")
    ap.add_argument("--by", choices=["cost", "lap"], default="cost",
                    help="ranking metric among qualifying trials")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    study = optuna.create_study(
        study_name=cfg["study"]["name"], storage=cfg["study"]["storage"],
        direction="minimize", load_if_exists=True)

    if args.show_best:
        print_best(study, cfg, max_d=args.max_d, by=args.by)
        return

    rclpy.init()
    node = TunerNode(cfg["run"])
    try:
        spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin.start()
        node.wait_ready()

        # baseline trial = the controller's CURRENT values (bounds-clipped), read
        # via the get_parameters service. Enqueued until one baseline completes.
        states = [t.state.name for t in study.trials]
        if "COMPLETE" not in states and "WAITING" not in states:
            try:
                enabled = [k for k, v in cfg["params"].items() if v.get("enabled")]
                current = node.get_params(enabled)
                baseline = {k: float(np.clip(current[k], cfg["params"][k]["low"],
                                             cfg["params"][k]["high"])) for k in enabled}
                study.enqueue_trial(baseline)
                print(f"baseline enqueued: {baseline}")
            except RuntimeError as e:
                print(f"baseline skipped ({e})")

        n = args.n_trials if args.n_trials is not None else cfg["study"]["n_trials"]
        try:
            study.optimize(make_objective(node, cfg), n_trials=n)
        except KeyboardInterrupt:
            print("\ninterrupted - study saved, rerun to resume.")
        print_best(study, cfg)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        time.sleep(0.3)  # let the spin thread wind down before interpreter exit


if __name__ == "__main__":
    main()
