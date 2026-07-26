#!/usr/bin/env python3
"""Skidpad limit detection: did the run actually reach the grip limit?

A constant-steer circle at speed v on radius R pulls a_lat = v^2 / R. While the
tires have margin that relation is exact, so a_lat rises with v^2 on a straight
line of slope 1/R. Once the tires saturate a_lat stops rising - the car pushes
wide instead. THE KNEE IS THE MEASUREMENT: a_y_max is the plateau value, and
mu = a_y_max / g.

If a_lat is still climbing linearly at the fastest sample, the limit was NOT
reached and the run only gives a LOWER BOUND on grip - which is what a ggv
built from such runs records. This script says which case you are in instead
of leaving it to the eye.

Cross-check on the same data: the measured yaw rate is compared against the
kinematic prediction v*tan(delta)/L. Understeer (ratio dropping below 1) is the
independent signature of the same saturation.

Record one bag per (steering angle, speed) run, or one bag covering a speed
sweep at fixed steering:

    ros2 bag record -o skidpad_d20 /car_state/odom /vesc/sensors/imu \
        /vesc/ackermann_cmd /vesc/odom

    python3 analyse_skidpad.py skidpad_d20/ [more_bags/ ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

G = 9.81
ODOM_TOPIC = "/car_state/odom"
IMU_TOPIC = "/vesc/sensors/imu"
CMD_TOPIC = "/vesc/ackermann_cmd"


def read_series(bag, topic, getter, optional=False):
    t, v = [], []
    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            if optional:
                return None, None
            avail = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic} not in {bag}.\navailable: {avail}")
        for conn, stamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            hdr = getattr(msg, "header", None)
            ns = (int(hdr.stamp.sec) * 10**9 + int(hdr.stamp.nanosec)
                  if hdr is not None and hdr.stamp.sec else int(stamp_ns))
            t.append(ns * 1e-9)
            v.append(getter(msg))
    return np.asarray(t), np.asarray(v, dtype=float)


def moving_rms_stable(t, x, window_s, tol):
    """Boolean mask: |x - local mean| stays under tol over `window_s`."""
    dt = np.median(np.diff(t))
    n = max(int(round(window_s / dt)), 3)
    kern = np.ones(n) / n
    mean = np.convolve(x, kern, mode="same")
    return np.abs(x - mean) < tol


def load_run(bag, dt=0.01):
    t_od, vx = read_series(bag, ODOM_TOPIC, lambda m: m.twist.twist.linear.x)
    t_r, r = read_series(bag, ODOM_TOPIC, lambda m: m.twist.twist.angular.z)
    # VESC IMU sits yaw +90 deg vs base_link: lateral accel is on imu x
    # (same convention analyse_tires.py uses).
    t_im, a_lat = read_series(bag, IMU_TOPIC,
                              lambda m: m.linear_acceleration.x)
    t_cm, delta = read_series(bag, CMD_TOPIC,
                              lambda m: m.drive.steering_angle, optional=True)

    t0 = max(t_od[0], t_im[0])
    t1 = min(t_od[-1], t_im[-1])
    grid = np.arange(t0, t1, dt)
    v = np.interp(grid, t_od, vx)
    yaw = np.interp(grid, t_r, r)
    a = np.abs(np.interp(grid, t_im, a_lat))
    d = (np.interp(grid, t_cm, delta) if t_cm is not None
         else np.full_like(grid, np.nan))
    return grid, v, yaw, a, d


def analyse(bag, wheelbase, dt=0.01):
    t, v, yaw, a, d = load_run(bag, dt)

    # keep only quasi-steady cornering: moving, turning, speed settled
    steady = (moving_rms_stable(t, v, 0.5, 0.15)
              & (v > 1.0) & (np.abs(yaw) > 0.3))
    if steady.sum() < 50:
        print(f"  {Path(bag).name}: no steady cornering found "
              f"({steady.sum()} samples) - skipped")
        return None

    v, yaw, a, d = v[steady], yaw[steady], a[steady], d[steady]

    # bin by speed and take the steady a_lat in each bin
    edges = np.arange(np.floor(v.min() * 2) / 2, v.max() + 0.25, 0.25)
    vb, ab, rb, nb = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() < 20:
            continue
        vb.append(v[m].mean())
        ab.append(np.median(a[m]))
        # understeer ratio: measured yaw rate vs kinematic prediction
        if np.isfinite(d[m]).all() and np.abs(d[m]).mean() > 1e-3:
            kin = v[m] * np.tan(np.abs(d[m])) / wheelbase
            rb.append(np.median(np.abs(yaw[m]) / kin))
        else:
            rb.append(np.nan)
        nb.append(int(m.sum()))
    if len(vb) < 3:
        print(f"  {Path(bag).name}: too few speed bins ({len(vb)}) - "
              f"sweep a wider speed range")
        return None
    vb, ab, rb = np.asarray(vb), np.asarray(ab), np.asarray(rb)

    # linear region: fit a_lat = k * v^2 on the slowest third
    k_lo = max(2, len(vb) // 3)
    k = float(np.sum(ab[:k_lo] * vb[:k_lo]**2) / np.sum(vb[:k_lo]**4))
    predicted = k * vb**2
    shortfall = 1.0 - ab / predicted        # 0 = on the line, >0 = saturating

    print(f"\n=== {Path(bag).name} ===")
    print(f"  implied radius from low-speed fit: {1/k:.2f} m"
          f"   (steering {np.nanmean(np.abs(d)):.3f} rad)")
    print("   v [m/s]   a_lat   v^2/R 예상   부족분   yaw/kin   n")
    for i in range(len(vb)):
        rr = f"{rb[i]:6.2f}" if np.isfinite(rb[i]) else "     -"
        print(f"  {vb[i]:7.2f}  {ab[i]:6.2f}  {predicted[i]:10.2f}  "
              f"{100*shortfall[i]:7.1f}%  {rr}  {nb[i]:4d}")

    saturated = shortfall > 0.10
    if saturated.any():
        i = int(np.argmax(saturated))
        a_y_max = float(np.max(ab))
        print(f"\n  >>> 한계 도달: {vb[i]:.2f} m/s 부터 포화 시작")
        print(f"  >>> a_y_max = {a_y_max:.2f} m/s^2   mu = {a_y_max/G:.2f}")
        return a_y_max
    print(f"\n  >>> 한계 미도달: 최고 속도 {vb[-1]:.2f} m/s 까지 "
          f"a_lat 이 v^2 에 계속 선형")
    print(f"  >>> 그립은 최소 {ab[-1]:.2f} m/s^2 이상 (하한값일 뿐, "
          f"ggv 에 쓰면 보수적)")
    print(f"  >>> 더 빠르게 또는 더 작은 반경(큰 조향)으로 재측정 필요")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+", help="skidpad bag directories")
    ap.add_argument("--wheelbase", type=float, default=0.33)
    a = ap.parse_args()

    peaks = [(Path(b).name, p)
             for b, p in ((b, analyse(b, a.wheelbase)) for b in a.bags)
             if p is not None]
    if not peaks:
        print("\n=== 어떤 런도 한계에 도달하지 못했습니다 ===")
        print("    조향을 키워 반경을 줄이면 훨씬 낮은 속도에서 미끄러집니다")
        return

    vals = [p for _, p in peaks]
    lo, hi = min(vals), max(vals)
    print(f"\n=== 종합 ({len(peaks)} 런 한계 도달) ===")
    for name, p in peaks:
        print(f"    {name:28s} a_y_max {p:5.2f}  mu {p/G:.2f}")
    # the raceline has corners both ways, so the weaker direction binds.
    print(f"\n    ggv.csv 의 ay_max 에는 {lo:.2f} 를 넣으세요 "
          f"(약한 쪽 기준, mu {lo/G:.2f})")
    if hi - lo > 0.5:
        print(f"    주의: 런별 편차 {hi-lo:.2f} m/s^2. 조향 비대칭은 반경만 바꾸고\n"
              f"    a_y_max 는 바꾸지 않습니다 - 좌우 차이라면 하중 배분이나\n"
              f"    타이어 편마모를 의심하세요.")


if __name__ == "__main__":
    sys.exit(main())
