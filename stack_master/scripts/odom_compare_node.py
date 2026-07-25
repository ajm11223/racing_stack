#!/usr/bin/env python3
"""Live 3-way odometry comparison: /vesc/odom vs /tracked_pose/odom vs /car_state/odom.

Logs x, y, yaw, vx, vy, yaw_rate from all three sources — the two EKF inputs
(VESC wheel odometry, Cartographer tracked pose) and the EKF output — one row
per report tick, to see how the fused estimate tracks each input.

NOTE on frames: /vesc/odom pose (x, y, yaw) is dead-reckoned in its own odom
frame, so it drifts and is NOT directly comparable to the map-frame poses of
/tracked_pose/odom and /car_state/odom. Twist (vx, vy, yaw_rate) is body-frame
for all three and directly comparable. /tracked_pose/odom carries pose only
(twist is zero — pose_to_odom_node.py does not fill it).

Each source is sampled at its own message rate; the report timer pairs the
most recent value from each. The terminal prints a vertical block per tick;
on shutdown every logged row is saved as a timestamped CSV under `csv_dir`
(default ~/Desktop) and the path is printed.

Usage (alongside race.launch / base_system, real car):

    ros2 run stack_master odom_compare_node.py
    ros2 run stack_master odom_compare_node.py --ros-args -p period:=0.5 -p csv_dir:=/tmp
"""
import csv
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

SOURCES = [
    # (csv/log prefix, default topic)
    ('vesc', '/vesc/odom'),
    ('tracked', '/tracked_pose/odom'),
    ('ekf', '/car_state/odom'),
]

FIELDS = ['x_m', 'y_m', 'yaw_deg', 'vx_ms', 'vy_ms', 'yaw_rate_dps']

CSV_HEADER = ['t_sec'] + [f'{p}_{f}' for p, _ in SOURCES for f in FIELDS]


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def extract(msg: Odometry):
    yaw = quat_yaw(msg.pose.pose.orientation)
    return (
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        math.degrees(yaw),
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        math.degrees(msg.twist.twist.angular.z),
    )


class OdomCompare(Node):
    def __init__(self):
        super().__init__('odom_compare')
        self.declare_parameter('period', 0.2)
        self.declare_parameter('csv_dir', os.path.expanduser('~/Desktop'))

        self.latest = {prefix: None for prefix, _ in SOURCES}
        self.rows = []
        self.t0 = None                # wall time of the first logged row

        for prefix, topic in SOURCES:
            self.declare_parameter(f'{prefix}_topic', topic)
            topic = self.get_parameter(f'{prefix}_topic').value
            self.create_subscription(
                Odometry, topic,
                lambda msg, p=prefix: self.latest.__setitem__(p, msg), 10)
        self.create_timer(self.get_parameter('period').value, self.report)
        self.get_logger().info(
            'comparing ' + ' | '.join(t for _, t in SOURCES))

    def report(self):
        missing = [p for p, m in self.latest.items() if m is None]
        if missing:
            self.get_logger().warn('waiting for ' + ' '.join(missing),
                                   throttle_duration_sec=2.0)
            return
        vals = {p: extract(m) for p, m in self.latest.items()}

        if self.t0 is None:
            self.t0 = time.time()
        row = [time.time() - self.t0]
        for prefix, _ in SOURCES:
            row.extend(vals[prefix])
        self.rows.append(tuple(row))

        lines = []
        for prefix, _ in SOURCES:
            x, y, yaw, vx, vy, yaw_rate = vals[prefix]
            lines.append(
                f'  {prefix:>7}  x {x:8.3f} | y {y:8.3f} | yaw {yaw:8.2f} deg'
                f' | vx {vx:6.2f} | vy {vy:6.2f} m/s | yaw_rate {yaw_rate:8.2f} deg/s')
        self.get_logger().info('\n' + '\n'.join(lines))

    def save_csv(self):
        if not self.rows:
            self.get_logger().info('no data collected — no CSV written')
            return
        out_dir = self.get_parameter('csv_dir').value
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, time.strftime('odom_compare_%Y%m%d_%H%M%S.csv'))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows([f'{v:.4f}' for v in row] for row in self.rows)
        self.get_logger().info(f'saved {len(self.rows)} rows -> {path}')


def main():
    rclpy.init()
    node = OdomCompare()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.save_csv()


if __name__ == '__main__':
    main()
