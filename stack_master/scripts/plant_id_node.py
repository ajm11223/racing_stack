#!/usr/bin/env python3
"""Plant identification: steering chirp superimposed on a steady circle.

WHY: the headless sim reproduces the car at long lookahead (m_l1 0.48) but not
at short (0.42), where the real car shows ~2.5x more deterministic path
roughness. Actuation transport delay is already correct (measured 113 ms vs
sim 108 ms), so what is missing is a LAG whose character cannot be separated
from racing laps - the controller only excites a narrow band there. Two
candidates remain:

    tyre relaxation length : time constant = sigma_relax / v
                             -> corner frequency RISES with speed
    servo dynamics         : time constant fixed
                             -> corner frequency UNCHANGED with speed

So: sweep a steering chirp at several speeds and compare the steer -> yaw-rate
frequency response. Whether the -3 dB point moves with speed is the answer.

WHY A CIRCLE, NOT A STRAIGHT: a straight-line step test at 7 m/s needs ~170 m
of run to collect enough transients. Driving a circle keeps the car inside a
fixed area (2R + margin) no matter how long the test runs or how fast it goes,
and gives 25 s of continuous excitation per speed instead of one step per pass.
The steering bias means the identification happens around a cornering operating
point - which is where the controller actually lives anyway.

The command chain (mux output, servo position) is logged too, so a delay
introduced by the plumbing rather than the physics can be ruled out.

USAGE (real car, ~8 x 8 m clear area - the TRACK IS NOT NEEDED):

    # low_level (VESC driver) must be up. Do NOT run the controller:
    # it publishes to the same topic and would fight this node.
    python3 plant_id_node.py

    # smaller area: radius 2.5 m, drop the top speed
    python3 plant_id_node.py --ros-args -p radius:=2.5 -p speeds:="[2.5, 3.5]"

SAFETY
    - Starts with an 8 s WARMUP circle at 1.5 m/s so you can SEE the actual
      radius before it speeds up. The real circle runs WIDER than commanded
      (understeer: the measured understeer gradient implies roughly +20% at
      5 m/s^2), so budget margin. Abort during warmup if it looks too big.
    - Ctrl-C stops the car immediately and closes the bag cleanly.
    - Keep the physical kill switch in hand: this node does not use
      localization and has no idea where the walls are.
    - a_y = v^2 / R is printed per speed. Above ~70% of mu*g (10.3 m/s^2) the
      tyres start sliding and the identification is no longer linear.
    - Every phase is time-bounded; the run aborts after max_run_s.

OUTPUT (both, under out_dir)
    plant_id_<stamp>.csv   one row per 50 Hz tick, command chain + response
    plant_id_<stamp>_bag/  rosbag of the raw topics (record_bag:=true, default)
"""
import csv
import math
import os
import signal
import subprocess
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

WHEELBASE = 0.3302          # lf + lr from SIM/dynamics.yaml
STEER_LIMIT = 0.40          # rad; s_max is 0.4189, stay just inside it
RATE_HZ = 50.0              # match the controller's publish rate
MU_G = 1.0489 * 9.81        # friction ceiling from dynamics.yaml

BAG_TOPICS = [
    '/vesc/high_level/ackermann_cmd',   # what this node commands
    '/vesc/ackermann_cmd',              # after simple_mux
    '/vesc/commands/servo/position',    # after ackermann_to_vesc (VESC input)
    '/vesc/sensors/imu',                # yaw rate / accelerations (response)
    '/vesc/odom',                       # wheel speed
    '/vesc/sensors/core',               # current, duty, voltage
]

CSV_HEADER = [
    't_sec', 'phase', 'speed_target', 'chirp_freq_hz',
    'steer_cmd',            # this node -> /vesc/high_level/ackermann_cmd
    'steer_bias', 'steer_chirp',        # the two components, for the fit
    'steer_mux',            # /vesc/ackermann_cmd  (chain check)
    'servo_position',       # /vesc/commands/servo/position (chain check)
    'speed_cmd', 'speed_mux', 'speed_meas',
    'yaw_rate', 'accel_x', 'accel_y',
    'radius_meas',          # v / yaw_rate: what circle it is actually running
    'motor_current', 'duty_cycle', 'voltage',
]


class PlantId(Node):
    def __init__(self):
        super().__init__('plant_id')
        self.declare_parameter('radius', 3.0)             # commanded circle [m]
        self.declare_parameter('speeds', [2.5, 3.5, 4.5])
        self.declare_parameter('chirp_f0', 0.2)           # Hz
        self.declare_parameter('chirp_f1', 6.0)           # Hz
        self.declare_parameter('chirp_s', 25.0)           # sweep duration
        self.declare_parameter('chirp_lat_acc', 1.0)      # excitation amplitude [m/s^2]
        self.declare_parameter('warmup_speed', 1.5)
        self.declare_parameter('warmup_s', 8.0)
        self.declare_parameter('settle_s', 3.0)           # steady circle before chirp
        self.declare_parameter('ramp_s', 2.5)
        self.declare_parameter('speed_step', True)        # longitudinal test on the circle
        self.declare_parameter('max_run_s', 400.0)
        self.declare_parameter('out_dir', os.path.expanduser('~/runs'))
        self.declare_parameter('record_bag', True)
        self.declare_parameter('drive_topic', '/vesc/high_level/ackermann_cmd')

        g = self.get_parameter
        self.R = float(g('radius').value)
        self.speeds = list(g('speeds').value)
        self.f0 = float(g('chirp_f0').value)
        self.f1 = float(g('chirp_f1').value)
        self.chirp_dur = float(g('chirp_s').value)
        self.chirp_ay = float(g('chirp_lat_acc').value)
        self.warm_v = float(g('warmup_speed').value)
        self.warm_s = float(g('warmup_s').value)
        self.settle = float(g('settle_s').value)
        self.ramp = float(g('ramp_s').value)
        self.max_run = float(g('max_run_s').value)
        self.out_dir = g('out_dir').value

        self.bias = math.atan(WHEELBASE / self.R)   # kinematic steer for radius R

        self.pub = self.create_publisher(
            AckermannDriveStamped, g('drive_topic').value, 10)
        # command-chain observers: prove the commands reach the VESC unchanged,
        # so any lag we measure is physics and not the plumbing
        self.create_subscription(AckermannDriveStamped, '/vesc/ackermann_cmd',
                                 self.mux_cb, 10)
        self.create_subscription(Float64, '/vesc/commands/servo/position',
                                 self.servo_cb, 10)
        self.create_subscription(Imu, '/vesc/sensors/imu', self.imu_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/vesc/odom', self.odom_cb, 10)
        try:
            from vesc_msgs.msg import VescStateStamped
            self.create_subscription(VescStateStamped, '/vesc/sensors/core',
                                     self.core_cb, 10)
        except ImportError:
            self.get_logger().warn('vesc_msgs not importable; core state skipped')

        self.steer_mux = float('nan')
        self.speed_mux = float('nan')
        self.servo = float('nan')
        self.yaw_rate = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.speed = 0.0
        self.current = float('nan')
        self.duty = float('nan')
        self.voltage = float('nan')
        self.rows = []

        self.stamp = time.strftime('%Y%m%d_%H%M%S')
        self.bag_proc = None
        if bool(g('record_bag').value):
            self.start_bag()

        # ---- schedule ----
        self.plan = [dict(kind='warmup', v=self.warm_v, dur=self.warm_s)]
        for v in self.speeds:
            self.plan.append(dict(kind='ramp', v=v, dur=self.ramp))
            self.plan.append(dict(kind='circle', v=v, dur=self.settle))
            self.plan.append(dict(kind='chirp', v=v, dur=self.chirp_dur))
        if bool(g('speed_step').value) and len(self.speeds) >= 2:
            lo, hi = min(self.speeds), max(self.speeds)
            # longitudinal step, still on the circle: measures what a_max /
            # v_switch claim. The track showed the car reaching only ~94.7% of
            # the commanded profile speed and that was never measured directly.
            self.plan.append(dict(kind='circle', v=lo, dur=3.0))
            self.plan.append(dict(kind='vstep', v=hi, dur=4.0))
            self.plan.append(dict(kind='vstep', v=lo, dur=4.0))
        self.plan.append(dict(kind='stop', v=0.0, dur=3.0))

        total = sum(p['dur'] for p in self.plan)
        self.get_logger().info(
            f'circle R={self.R:.2f} m, steer bias {self.bias:.4f} rad, '
            f'~{total:.0f} s total')
        self.get_logger().info(
            f'  space needed ~{2*self.R + 1.8:.1f} x {2*self.R + 1.8:.1f} m '
            f'(circle runs WIDER than commanded - understeer)')
        for v in self.speeds:
            ay = v * v / self.R
            amp = self.chirp_ay * WHEELBASE / (v * v)
            flag = '  << above 70% of mu*g, tyres may slide' if ay > 0.7 * MU_G else ''
            self.get_logger().info(
                f'  v={v:.1f} -> a_y {ay:.2f} ({100*ay/MU_G:.0f}% of limit), '
                f'chirp amp {amp:.4f} rad{flag}')
        self.get_logger().info('starting in 5 s - Ctrl-C aborts and stops the car')

        self.t0 = None
        self.idx = -1              # -1 = pre-roll countdown
        self.phase_t0 = 0.0
        self.timer = self.create_timer(1.0 / RATE_HZ, self.tick)

    # ---- bag ----
    def start_bag(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f'plant_id_{self.stamp}_bag')
            self.bag_proc = subprocess.Popen(
                ['ros2', 'bag', 'record', '-o', path] + BAG_TOPICS,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info(f'recording bag -> {path}')
            time.sleep(1.0)        # let the recorder subscribe before we move
        except Exception as e:
            self.get_logger().error(f'bag record failed to start: {e}')
            self.bag_proc = None

    def stop_bag(self):
        if self.bag_proc is None:
            return
        try:
            self.bag_proc.send_signal(signal.SIGINT)   # clean close
            self.bag_proc.wait(timeout=10)
            self.get_logger().info('bag closed')
        except Exception:
            try:
                self.bag_proc.kill()
            except Exception:
                pass
        self.bag_proc = None

    # ---- callbacks ----
    def mux_cb(self, m):
        self.steer_mux = m.drive.steering_angle
        self.speed_mux = m.drive.speed

    def servo_cb(self, m):
        self.servo = m.data

    def imu_cb(self, m):
        self.yaw_rate = -m.angular_velocity.z   # vesc mounted rotated 90 deg
        self.accel_x = -m.linear_acceleration.x
        self.accel_y = m.linear_acceleration.y

    def odom_cb(self, m):
        self.speed = m.twist.twist.linear.x

    def core_cb(self, m):
        try:
            self.current = m.state.current_motor
            self.duty = m.state.duty_cycle
            self.voltage = m.state.voltage_input
        except AttributeError:
            pass

    def publish(self, speed, steer):
        steer = max(-STEER_LIMIT, min(STEER_LIMIT, steer))
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.pub.publish(msg)

    def chirp(self, pt, v):
        """log-swept sine: even energy per octave, phase-continuous.
        Amplitude is set in lateral acceleration so the excitation (and the
        signal-to-noise of the fit) is the same at every speed."""
        T, f0, f1 = self.chirp_dur, self.f0, self.f1
        k = math.log(f1 / f0)
        phase = 2 * math.pi * f0 * T / k * (math.exp(pt / T * k) - 1.0)
        freq = f0 * math.exp(pt / T * k)
        amp = self.chirp_ay * WHEELBASE / (v * v)
        return amp * math.sin(phase), freq

    # ---- main loop ----
    def tick(self):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        if t > self.max_run:
            self.get_logger().error('max_run_s exceeded - stopping')
            self.finish()
            return

        if self.idx < 0:                      # countdown, car stationary
            self.publish(0.0, 0.0)
            if t >= 5.0:
                self.idx = 0
                self.phase_t0 = t
                self.get_logger().info('GO - warmup circle, check the radius')
            return

        if self.idx >= len(self.plan):
            self.finish()
            return

        p = self.plan[self.idx]
        pt = t - self.phase_t0
        if pt >= p['dur']:
            self.idx += 1
            self.phase_t0 = t
            if self.idx < len(self.plan):
                q = self.plan[self.idx]
                r_meas = (self.speed / self.yaw_rate
                          if abs(self.yaw_rate) > 0.05 else float('nan'))
                self.get_logger().info(
                    f"[{self.idx+1}/{len(self.plan)}] {q['kind']} v={q['v']:.1f}"
                    f"   (measured radius {r_meas:.2f} m)")
            return

        freq = float('nan')
        chirp = 0.0
        if p['kind'] == 'warmup':
            speed, bias = p['v'], self.bias
        elif p['kind'] == 'ramp':
            speed = self.warm_v + (p['v'] - self.warm_v) * min(pt / max(p['dur'], 1e-3), 1.0)
            bias = self.bias
        elif p['kind'] == 'circle':
            speed, bias = p['v'], self.bias
        elif p['kind'] == 'chirp':
            speed, bias = p['v'], self.bias
            chirp, freq = self.chirp(pt, p['v'])
        elif p['kind'] == 'vstep':
            speed, bias = p['v'], self.bias      # STEP, not ramp: that is the point
        else:                                    # stop
            speed, bias = 0.0, 0.0

        steer = bias + chirp
        self.publish(speed, steer)
        r_meas = (self.speed / self.yaw_rate
                  if abs(self.yaw_rate) > 0.05 else float('nan'))
        self.rows.append([
            round(t, 4), p['kind'], p['v'], round(freq, 4),
            round(steer, 5), round(bias, 5), round(chirp, 5),
            round(self.steer_mux, 5), round(self.servo, 5),
            round(speed, 4), round(self.speed_mux, 4), round(self.speed, 4),
            round(self.yaw_rate, 5), round(self.accel_x, 4), round(self.accel_y, 4),
            round(r_meas, 3), round(self.current, 3), round(self.duty, 4),
            round(self.voltage, 2),
        ])

    def finish(self):
        self.publish(0.0, 0.0)
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.save()
        self.stop_bag()
        self.get_logger().info('done - car stopped')
        rclpy.shutdown()

    def save(self):
        if not self.rows:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f'plant_id_{self.stamp}.csv')
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                w.writerows(self.rows)
            self.get_logger().info(f'saved {len(self.rows)} rows -> {path}')
        except Exception as e:
            self.get_logger().error(f'CSV save failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PlantId()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('interrupted - stopping car')
        for _ in range(10):          # spam zero so one gets through
            node.publish(0.0, 0.0)
            time.sleep(0.02)
        node.save()
        node.stop_bag()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
