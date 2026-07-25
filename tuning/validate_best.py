#!/usr/bin/env python3
"""Validate a saved best-trial param set in the REAL sim stack.

fast_tune.py explores headless (no ROS, 100 Hz physics, controller called
directly); this script closes the loop the docstring demands ("validate top-k
in the real sim before adopting!"): apply the saved params live on the full
stack (gym_bridge 80 Hz + planner/state_machine hop + odom latency) and
measure the same metrics for an apples-to-apples comparison.

Needs an ALREADY RUNNING sim stack:
    ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true

Usage (conda unicorn env, NO /opt/ros/jazzy):
    python validate_best.py [--params best_params_controller_stage1_v3c.yaml]
                            [--laps 2]

Deliberately reads the saved best_params_*.yaml snapshot, NOT the live Optuna
journal - the running fast_tune workers stay untouched.
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
import yaml

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))

from tune_controller import TunerNode, TrialFailed, load_cfg, CFG_DEFAULT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",
                    default=os.path.join(TUNING_DIR,
                                         "best_params_controller_stage1_v3c.yaml"))
    ap.add_argument("--config", default=CFG_DEFAULT)
    ap.add_argument("--laps", type=int, default=None,
                    help="measured laps (default: run.laps_per_trial)")
    ap.add_argument("--no-reset", action="store_true",
                    help="skip the standing-start teleport: keep the car "
                         "flowing and measure from the next lap boundary")
    args = ap.parse_args()

    saved = yaml.safe_load(open(args.params))
    params = {k: float(v) for k, v in saved["params"].items()}
    ref = saved.get("user_attrs", {})
    cfg = load_cfg(args.config)
    run_cfg = dict(cfg["run"])
    if args.laps:
        run_cfg["laps_per_trial"] = args.laps

    print(f"validating trial #{saved.get('trial')} "
          f"(headless cost {saved.get('cost', float('nan')):.3f}) "
          f"from {os.path.basename(args.params)}")

    rclpy.init()
    node = TunerNode(run_cfg)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.wait_ready()
        # frozen values FIRST, one by one in yaml order: global_limit 2.0 must
        # land before Sector0.scaling 1.64 or sector_tuner clips it to the
        # yaml-default limit 1.2.
        frozen = dict(cfg.get("frozen") or {})
        if frozen:
            print(f"applying frozen params: {frozen}")
            node.set_params(frozen)
        print(f"applying {len(params)} trial params...")
        node.set_params(params)

        if args.no_reset:
            node.reset_car = lambda: None   # keep flowing; warmup = current lap

        try:
            m = node.run_trial()
        except TrialFailed as e:
            print(f"\nTRIAL FAILED: {e.reason} "
                  f"(progress {e.progress_frac * 100:.0f}%)")
            with node.lock:
                tail = node.samples[-15:]
            if tail:
                print("last samples (t, s, d):")
                for t, s, d in tail:
                    print(f"  t={t:9.2f}  s={s:6.2f}  d={d:+.3f}")
            return 1

        laps = ", ".join(f"{x:.2f}" for x in m["lap_times"])
        print(f"\n=== real-sim result: laps [{laps}] s ===")
        rows = [("mean lap [s]", float(np.mean(m["lap_times"])),
                 float(np.mean(ref["lap_times"])) if ref.get("lap_times") else None),
                ("mean |d| [m]", m["mean_abs_d"], ref.get("mean_abs_d")),
                ("dsteer_rms",   m["dsteer_rms"], ref.get("dsteer_rms")),
                ("d_weave_rms",  m["d_weave_rms"], ref.get("d_weave_rms"))]
        print(f"{'metric':14s} {'real sim':>10s} {'headless':>10s} {'delta':>10s}")
        for name, real, pred in rows:
            if pred is None:
                print(f"{name:14s} {real:10.4f} {'-':>10s} {'-':>10s}")
            else:
                print(f"{name:14s} {real:10.4f} {pred:10.4f} {real - pred:>+10.4f}")
        return 0
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        time.sleep(0.3)  # let the spin thread wind down (mirrors tune_controller)


if __name__ == "__main__":
    raise SystemExit(main())
