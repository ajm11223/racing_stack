#!/usr/bin/env python3
"""Actuation latency + steering slew rate from a bang-bang steering bag.

Fills the gap the ForzaETH pipeline leaves: analyse_steering/analyse_tires fit
the STATIC steering map, nothing fits the DYNAMICS. The two numbers the gym
simulator needs are

  LATENCY_S  (tuning/fast_tune.py) - transport dead time before the plant sees
             a command. Currently 0.03, calibrated sim-vs-sim, never measured.
  sv_max     (config/CAR|SIM/dynamics.yaml) - steering rate limit. Currently 10,
             estimated from the AGFRC B210 datasheet, never measured.

There is no steering encoder on the car, so the steering angle is observed
through the yaw rate. This fits a first-order-plus-dead-time model

    r(t) = K * lowpass_tau( delta_cmd(t - L) )

by grid search over (L, tau) with K by least squares, on a resampled uniform
grid. L is the dead time -> LATENCY_S. tau is the steering lag; the slew rate
implied by the commanded step is reported as the sv_max candidate.

The software-only part of the delay (controller publish -> servo command
published) is measured separately by cross-correlating the two topics, so the
mechanical part can be separated from the ROS/DDS part.

Record the bag with the car ON BLOCKS-FREE FLAT GROUND, experiment 7:

    ros2 launch id_controller id_controller.launch.py experiment:=7
    ros2 bag record -o steerstep /vesc/ackermann_cmd \
        /vesc/high_level/ackermann_cmd_mux/input/nav_1 \
        /vesc/commands/servo/position /vesc/sensors/imu /vesc/odom

then

    python3 analyse_actuation_latency.py steerstep/
"""
import argparse
import sys

import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

CMD_TOPIC = "/vesc/ackermann_cmd"          # mux output = what the car executes
SERVO_TOPIC = "/vesc/commands/servo/position"
IMU_TOPIC = "/vesc/sensors/imu"
ODOM_TOPIC = "/vesc/odom"


def read_series(bag, topic, getter):
    """-> (t[s], value[]) from `topic`, using header stamps when present."""
    t, v = [], []
    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            avail = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic} not in bag.\navailable: {avail}")
        for conn, stamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            hdr = getattr(msg, "header", None)
            ns = (int(hdr.stamp.sec) * 10**9 + int(hdr.stamp.nanosec)
                  if hdr is not None and hdr.stamp.sec else int(stamp_ns))
            t.append(ns * 1e-9)
            v.append(getter(msg))
    return np.asarray(t), np.asarray(v, dtype=float)


def zoh(t_src, v_src, t_grid):
    """Zero-order hold resample - commands are held, not interpolated."""
    idx = np.searchsorted(t_src, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(v_src) - 1)
    return v_src[idx]


def lowpass(x, tau, dt):
    """First-order lag, forward Euler (tau=0 -> passthrough)."""
    if tau <= 0:
        return x.copy()
    a = dt / (tau + dt)
    y = np.empty_like(x)
    acc = x[0]
    for i, xi in enumerate(x):
        acc += a * (xi - acc)
        y[i] = acc
    return y


def xcorr_delay(t_grid, a, b, max_lag_s):
    """Lag (s) that best aligns b to a, positive = b lags a."""
    dt = t_grid[1] - t_grid[0]
    a = a - a.mean()
    b = b - b.mean()
    n = int(round(max_lag_s / dt))
    best, best_lag = -np.inf, 0.0
    for k in range(0, n + 1):
        aa, bb = a[:len(a) - k], b[k:]
        denom = np.linalg.norm(aa) * np.linalg.norm(bb)
        if denom == 0:
            continue
        c = float(aa @ bb) / denom
        if c > best:
            best, best_lag = c, k * dt
    return best_lag, best


def fit_fopdt(t_grid, cmd, yaw_rate, max_lag_s=0.40, max_tau_s=0.40):
    """Grid search dead time L and lag tau; gain K by least squares."""
    dt = t_grid[1] - t_grid[0]
    best = None
    lags = np.arange(0, int(round(max_lag_s / dt)) + 1)
    taus = np.arange(0, max_tau_s + 1e-9, 0.005)
    for tau in taus:
        filt = lowpass(cmd, tau, dt)
        for k in lags:
            u = filt[:len(filt) - k] if k else filt
            y = yaw_rate[k:]
            uu = float(u @ u)
            if uu == 0:
                continue
            K = float(u @ y) / uu
            resid = y - K * u
            sse = float(resid @ resid)
            if best is None or sse < best[0]:
                best = (sse, k * dt, float(tau), K, float(y @ y))
    sse, L, tau, K, sst = best
    return L, tau, K, 1.0 - sse / sst if sst else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="ros2 bag directory (or .db3/.mcap/.bag)")
    ap.add_argument("--dt", type=float, default=0.002, help="resample grid [s]")
    ap.add_argument("--cmd-topic", default=CMD_TOPIC)
    ap.add_argument("--servo-topic", default=SERVO_TOPIC)
    ap.add_argument("--imu-topic", default=IMU_TOPIC)
    ap.add_argument("--odom-topic", default=ODOM_TOPIC)
    ap.add_argument("--wheelbase", type=float, default=0.321)
    a = ap.parse_args()

    t_cmd, cmd = read_series(a.bag, a.cmd_topic,
                             lambda m: m.drive.steering_angle)
    t_srv, srv = read_series(a.bag, a.servo_topic, lambda m: m.data)
    t_imu, gyro = read_series(a.bag, a.imu_topic,
                              lambda m: m.angular_velocity.z)
    t_od, vx = read_series(a.bag, a.odom_topic,
                           lambda m: m.twist.twist.linear.x)

    t0 = max(t_cmd[0], t_srv[0], t_imu[0], t_od[0])
    t1 = min(t_cmd[-1], t_srv[-1], t_imu[-1], t_od[-1])
    grid = np.arange(t0, t1, a.dt)
    if len(grid) < 100:
        raise SystemExit("bag too short / topics do not overlap in time")

    c = zoh(t_cmd, cmd, grid)
    s = zoh(t_srv, srv, grid)
    r = np.interp(grid, t_imu, gyro)
    v = np.interp(grid, t_od, vx)

    print(f"bag span {grid[-1]-grid[0]:.1f} s, grid {a.dt*1e3:.0f} ms")
    print(f"rates: cmd {len(t_cmd)/(t1-t0):.0f} Hz  servo "
          f"{len(t_srv)/(t1-t0):.0f} Hz  imu {len(t_imu)/(t1-t0):.0f} Hz")
    print(f"speed during run: mean {v.mean():.2f} m/s, "
          f"steering command +-{np.abs(c).max():.3f} rad\n")

    # --- 1. software leg: controller command -> servo command published ---
    sw_lag, sw_corr = xcorr_delay(grid, c, s, max_lag_s=0.10)
    print(f"[software] cmd -> servo topic : {sw_lag*1e3:5.0f} ms  (corr {sw_corr:.3f})")

    # --- 2. full leg: command -> yaw-rate response ---
    L, tau, K, r2 = fit_fopdt(grid, c, r)
    print(f"[full]     cmd -> yaw rate    : dead time {L*1e3:5.0f} ms, "
          f"lag tau {tau*1e3:5.0f} ms, R^2 {r2:.3f}")
    K_kin = v.mean() / a.wheelbase
    print(f"           fitted gain K {K:.2f} rad/s per rad "
          f"(kinematic v/L = {K_kin:.2f}, ratio {K/K_kin:.2f})")

    # --- 3. slew rate implied by the commanded step ---
    step = np.abs(np.diff(c))
    big = step[step > 0.5 * step.max()] if step.max() > 0 else []
    if len(big):
        d_step = float(np.mean(big))
        # 10-90% of a first-order lag takes 2.2*tau
        rise = 2.2 * tau
        sv = d_step / rise if rise > 0 else float("inf")
        print(f"\n[slew]     commanded step {d_step:.3f} rad, "
              f"10-90% rise {rise*1e3:.0f} ms -> {sv:.1f} rad/s")
    else:
        sv = float("nan")
        print("\n[slew]     no clean steering step found (use experiment 7)")

    print("\n--- suggested sim values ---")
    print(f"fast_tune.py  LATENCY_S = {L:.3f}      # was 0.03")
    print(f"dynamics.yaml sv_max    = {sv:.1f}       # was 10.0")
    print("note: tau also covers tire force build-up, so sv_max from it is a\n"
          "      lower bound; cross-check against the servo datasheet.")


if __name__ == "__main__":
    sys.exit(main())
