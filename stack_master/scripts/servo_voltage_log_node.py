#!/usr/bin/env python3
"""Log battery voltage and servo steering command over time.

Records input voltage from /vesc/sensors/core and the commanded servo
position from /vesc/sensors/servo_position_command — one row per report
tick — to see how steering behaves as the battery sags over a run.

Each source is sampled at its own message rate; the report timer pairs the
most recent value from each. The terminal prints one line per tick; on
shutdown every logged row is saved as a timestamped CSV under `csv_dir`
(default ~/Desktop) and the path is printed.

steering_deg is derived from the servo command with the
steering_angle_to_servo_gain/offset parameters (defaults mirror
CAR/vehicle_config.yaml — override if the config changes).

Usage (alongside race.launch / low_level, real car):

    ros2 run stack_master servo_voltage_log_node.py
    ros2 run stack_master servo_voltage_log_node.py --ros-args -p period:=0.05 -p csv_dir:=/tmp
"""
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped

CSV_HEADER = ['t_sec', 'voltage_v', 'servo_pos', 'steering_deg']


class ServoVoltageLog(Node):
    def __init__(self):
        super().__init__('servo_voltage_log')
        self.declare_parameter('period', 0.1)
        self.declare_parameter('csv_dir', os.path.expanduser('~/Desktop'))
        self.declare_parameter('core_topic', '/vesc/sensors/core')
        self.declare_parameter('servo_topic', '/vesc/sensors/servo_position_command')
        # servo -> steering angle conversion (defaults from CAR/vehicle_config.yaml)
        self.declare_parameter('steering_angle_to_servo_gain', 0.49)
        self.declare_parameter('steering_angle_to_servo_offset', 0.47892)

        self.servo_gain = self.get_parameter('steering_angle_to_servo_gain').value
        self.servo_offset = self.get_parameter('steering_angle_to_servo_offset').value

        self.voltage = None
        self.servo = None
        self.rows = []
        self.t0 = None                # wall time of the first logged row

        self.create_subscription(
            VescStateStamped, self.get_parameter('core_topic').value,
            self.core_cb, 10)
        self.create_subscription(
            Float64, self.get_parameter('servo_topic').value,
            self.servo_cb, 10)
        self.create_timer(self.get_parameter('period').value, self.report)
        self.get_logger().info(
            f"logging {self.get_parameter('core_topic').value} (voltage) | "
            f"{self.get_parameter('servo_topic').value} (servo)")

    def core_cb(self, msg):
        self.voltage = msg.state.voltage_input

    def servo_cb(self, msg):
        self.servo = msg.data

    def report(self):
        missing = [n for n, v in (('voltage', self.voltage), ('servo', self.servo))
                   if v is None]
        if missing:
            self.get_logger().warn('waiting for ' + ' '.join(missing),
                                   throttle_duration_sec=2.0)
            return

        steering_deg = math.degrees(
            (self.servo - self.servo_offset) / self.servo_gain)

        if self.t0 is None:
            self.t0 = time.time()
        self.rows.append(
            (time.time() - self.t0, self.voltage, self.servo, steering_deg))

        self.get_logger().info(
            f'V {self.voltage:6.2f} V | servo {self.servo:7.4f}'
            f' | steer {steering_deg:7.2f} deg')

    def save_csv(self):
        if not self.rows:
            self.get_logger().info('no data collected — no CSV written')
            return
        out_dir = self.get_parameter('csv_dir').value
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, time.strftime('servo_voltage_%Y%m%d_%H%M%S.csv'))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows([f'{v:.4f}' for v in row] for row in self.rows)
        self.get_logger().info(f'saved {len(self.rows)} rows -> {path}')


def main():
    rclpy.init()
    node = ServoVoltageLog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.save_csv()


if __name__ == '__main__':
    main()
