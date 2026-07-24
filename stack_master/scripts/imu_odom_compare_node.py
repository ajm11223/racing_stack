#!/usr/bin/env python3
"""Live full-IMU vs wheel-odom comparison: /vesc/sensors/imu vs /vesc/odom.

Unlike yaw_rate_compare_node.py (yaw rate only), this logs EVERY IMU output —
linear acceleration x/y/z, gyro x/y/z, orientation roll/pitch/yaw — alongside
the wheel odometry state (vx, yaw, yaw rate), one row per report tick.
The directly comparable pair (gyro z vs odom twist.angular.z) also gets a
diff column, same meaning as in yaw_rate_compare_node.py.

NOTE on axes: the VESC IMU is mounted yaw +90 deg vs base_link (see
config/CAR/static_transforms.yaml), so imu x/y axes are ROTATED relative to
the car: car longitudinal accel appears on imu -y, lateral on imu x.
Gyro z (yaw rate) is unaffected by that rotation. Values are logged in the
sensor's own frame, unmodified.

Each source is sampled at its own message rate; the report timer pairs the
most recent value from each. The terminal prints a vertical block per tick;
on shutdown every logged row is saved as a timestamped CSV under `csv_dir`
(default ~/Desktop) and the path is printed.

Usage (alongside race.launch / base_system, real car):

    ros2 run stack_master imu_odom_compare_node.py
    ros2 run stack_master imu_odom_compare_node.py --ros-args -p period:=0.5 -p csv_dir:=/tmp
"""
import csv
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

CSV_HEADER = [
    't_sec',
    'imu_acc_x_ms2', 'imu_acc_y_ms2', 'imu_acc_z_ms2',
    'imu_gyro_x_dps', 'imu_gyro_y_dps', 'imu_gyro_z_dps',
    'imu_roll_deg', 'imu_pitch_deg', 'imu_yaw_deg',
    'odom_vx_ms', 'odom_yaw_deg', 'odom_yaw_rate_dps',
    'yaw_rate_diff_dps',   # odom_yaw_rate - imu_gyro_z
]


def quat_rpy(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class ImuOdomCompare(Node):
    def __init__(self):
        super().__init__('imu_odom_compare')
        self.declare_parameter('odom_topic', '/vesc/odom')
        self.declare_parameter('imu_topic', '/vesc/sensors/imu')
        self.declare_parameter('period', 0.1)
        self.declare_parameter('csv_dir', os.path.expanduser('~/Desktop'))

        self.imu = None               # latest sensor_msgs/Imu
        self.odom = None              # latest nav_msgs/Odometry
        self.rows = []
        self.t0 = None                # wall time of the first logged row

        odom_topic = self.get_parameter('odom_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(Imu, imu_topic, self.imu_cb,
                                 qos_profile_sensor_data)
        self.create_timer(self.get_parameter('period').value, self.report)
        self.get_logger().info(
            f'comparing full IMU ({imu_topic}) vs odom ({odom_topic})')

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def imu_cb(self, msg: Imu):
        self.imu = msg

    def report(self):
        if self.imu is None or self.odom is None:
            self.get_logger().warn(
                'waiting for '
                + ('imu ' if self.imu is None else '')
                + ('odom' if self.odom is None else ''),
                throttle_duration_sec=2.0)
            return
        deg = math.degrees
        acc = self.imu.linear_acceleration
        gyr = self.imu.angular_velocity
        i_roll, i_pitch, i_yaw = quat_rpy(self.imu.orientation)
        vx = self.odom.twist.twist.linear.x
        o_yaw_rate = self.odom.twist.twist.angular.z
        _, _, o_yaw = quat_rpy(self.odom.pose.pose.orientation)
        rate_diff = o_yaw_rate - gyr.z

        if self.t0 is None:
            self.t0 = time.time()
        self.rows.append((
            time.time() - self.t0,
            acc.x, acc.y, acc.z,
            deg(gyr.x), deg(gyr.y), deg(gyr.z),
            deg(i_roll), deg(i_pitch), deg(i_yaw),
            vx, deg(o_yaw), deg(o_yaw_rate),
            deg(rate_diff),
        ))
        self.get_logger().info(
            '\n'
            f'  imu  acc  [m/s2] x {acc.x:8.3f} | y {acc.y:8.3f} | z {acc.z:8.3f}\n'
            f'  imu  gyro [deg/s] x {deg(gyr.x):7.2f} | y {deg(gyr.y):8.2f} | z {deg(gyr.z):8.2f}\n'
            f'  imu  rpy  [deg]  r {deg(i_roll):8.2f} | p {deg(i_pitch):8.2f} | y {deg(i_yaw):8.2f}\n'
            f'  odom vx {vx:6.2f} m/s | yaw {deg(o_yaw):8.2f} deg | yaw_rate {deg(o_yaw_rate):8.2f} deg/s\n'
            f'  yaw_rate diff (odom - gyro z) {deg(rate_diff):8.2f} deg/s')

    def save_csv(self):
        if not self.rows:
            self.get_logger().info('no data collected — no CSV written')
            return
        out_dir = self.get_parameter('csv_dir').value
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, time.strftime('imu_odom_compare_%Y%m%d_%H%M%S.csv'))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows([f'{v:.4f}' for v in row] for row in self.rows)
        self.get_logger().info(f'saved {len(self.rows)} rows -> {path}')


def main():
    rclpy.init()
    node = ImuOdomCompare()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.save_csv()


if __name__ == '__main__':
    main()
