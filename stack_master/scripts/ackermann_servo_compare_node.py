#!/usr/bin/env python3
"""Live comparison of the ackermann command chain vs the actual servo command:
/vesc/high_level/ackermann_cmd (mux input, from planner/controller) vs
/vesc/ackermann_cmd (mux output, what ackermann_to_vesc consumes) vs
/vesc/commands/servo/position (the resulting normalized servo position sent
to the VESC).

This traces one command through the low_level pipeline:
  high_level/ackermann_cmd --[simple_mux_node]--> ackermann_cmd
  ackermann_cmd            --[ackermann_to_vesc]--> commands/servo/position

Each source is sampled at its own message rate; the report timer pairs the
most recent value from each. The terminal prints a vertical block per tick;
on shutdown every logged row is saved as a timestamped CSV under `csv_dir`
(default ~/Desktop) and the path is printed.

Usage (alongside race.launch / low_level, real car):

    ros2 run stack_master ackermann_servo_compare_node.py
    ros2 run stack_master ackermann_servo_compare_node.py --ros-args -p period:=0.5 -p csv_dir:=/tmp
"""
import csv
import os
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float64

CSV_HEADER = [
    't_sec',
    'hl_steer_rad', 'hl_speed_ms',
    'll_steer_rad', 'll_speed_ms',
    'servo_position',
    'steer_diff_rad',   # ll_steer - hl_steer
    'speed_diff_ms',    # ll_speed - hl_speed
]


class AckermannServoCompare(Node):
    def __init__(self):
        super().__init__('ackermann_servo_compare')
        self.declare_parameter('high_level_topic', '/vesc/high_level/ackermann_cmd')
        self.declare_parameter('low_level_topic', '/vesc/ackermann_cmd')
        self.declare_parameter('servo_topic', '/vesc/commands/servo/position')
        self.declare_parameter('period', 0.1)
        self.declare_parameter('csv_dir', os.path.expanduser('~/Desktop'))

        self.high_level = None   # latest ackermann_msgs/AckermannDriveStamped
        self.low_level = None    # latest ackermann_msgs/AckermannDriveStamped
        self.servo = None        # latest std_msgs/Float64
        self.rows = []
        self.t0 = None           # wall time of the first logged row

        hl_topic = self.get_parameter('high_level_topic').value
        ll_topic = self.get_parameter('low_level_topic').value
        servo_topic = self.get_parameter('servo_topic').value
        self.create_subscription(AckermannDriveStamped, hl_topic, self.high_level_cb, 10)
        self.create_subscription(AckermannDriveStamped, ll_topic, self.low_level_cb, 10)
        self.create_subscription(Float64, servo_topic, self.servo_cb, 10)
        self.create_timer(self.get_parameter('period').value, self.report)
        self.get_logger().info(
            f'comparing high_level ({hl_topic}) vs low_level ({ll_topic}) '
            f'vs servo ({servo_topic})')

    def high_level_cb(self, msg: AckermannDriveStamped):
        self.high_level = msg

    def low_level_cb(self, msg: AckermannDriveStamped):
        self.low_level = msg

    def servo_cb(self, msg: Float64):
        self.servo = msg

    def report(self):
        if self.high_level is None or self.low_level is None or self.servo is None:
            self.get_logger().warn(
                'waiting for '
                + ('high_level ' if self.high_level is None else '')
                + ('low_level ' if self.low_level is None else '')
                + ('servo' if self.servo is None else ''),
                throttle_duration_sec=2.0)
            return

        hl_steer = self.high_level.drive.steering_angle
        hl_speed = self.high_level.drive.speed
        ll_steer = self.low_level.drive.steering_angle
        ll_speed = self.low_level.drive.speed
        servo_pos = self.servo.data
        steer_diff = ll_steer - hl_steer
        speed_diff = ll_speed - hl_speed

        if self.t0 is None:
            self.t0 = time.time()
        self.rows.append((
            time.time() - self.t0,
            hl_steer, hl_speed,
            ll_steer, ll_speed,
            servo_pos,
            steer_diff, speed_diff,
        ))
        self.get_logger().info(
            '\n'
            f'  high_level  steer {hl_steer:7.4f} rad | speed {hl_speed:7.3f} m/s\n'
            f'  low_level   steer {ll_steer:7.4f} rad | speed {ll_speed:7.3f} m/s\n'
            f'  servo position {servo_pos:7.4f}\n'
            f'  diff (low_level - high_level) steer {steer_diff:8.4f} rad | speed {speed_diff:8.4f} m/s')

    def save_csv(self):
        if not self.rows:
            self.get_logger().info('no data collected — no CSV written')
            return
        out_dir = self.get_parameter('csv_dir').value
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, time.strftime('ackermann_servo_compare_%Y%m%d_%H%M%S.csv'))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows([f'{v:.4f}' for v in row] for row in self.rows)
        self.get_logger().info(f'saved {len(self.rows)} rows -> {path}')


def main():
    rclpy.init()
    node = AckermannServoCompare()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.save_csv()


if __name__ == '__main__':
    main()
