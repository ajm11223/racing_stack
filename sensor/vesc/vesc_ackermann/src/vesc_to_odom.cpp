// Copyright 2020 F1TENTH Foundation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//   * Neither the name of the {copyright_holder} nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// -*- mode:c++; fill-column: 100; -*-

#include "vesc_ackermann/vesc_to_odom.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <vesc_msgs/msg/vesc_state_stamped.hpp>

namespace vesc_ackermann
{

using geometry_msgs::msg::TransformStamped;
using nav_msgs::msg::Odometry;
using std::placeholders::_1;
using std_msgs::msg::Float64;
using vesc_msgs::msg::VescStateStamped;

VescToOdom::VescToOdom(const rclcpp::NodeOptions & options)
: Node("vesc_to_odom_node", options),
  odom_frame_("odom"),
  base_frame_("base_link"),
  use_servo_cmd_(true),
  publish_tf_(false),
  x_(0.0),
  y_(0.0),
  yaw_(0.0),
  imu_received_(false),
  filtered_ax_(0.0)
{
  // get ROS parameters
  odom_frame_ = declare_parameter("odom_frame", odom_frame_);
  base_frame_ = declare_parameter("base_frame", base_frame_);
  use_servo_cmd_ = declare_parameter("use_servo_cmd_to_calc_angular_velocity", use_servo_cmd_);

  speed_to_erpm_gain_ = declare_parameter<double>("speed_to_erpm_gain");
  speed_to_erpm_offset_ = declare_parameter<double>("speed_to_erpm_offset");

  // Adaptive wheel odometry parameters. With acceleration_gain = 0 and
  // gain_offset = 0 (or enabled = false) the output is identical to the
  // fixed-gain conversion.
  adaptive_enabled_ = declare_parameter("adaptive_odom.enabled", false);
  adaptive_acceleration_gain_ = declare_parameter("adaptive_odom.acceleration_gain", 0.0);
  adaptive_gain_offset_ = declare_parameter("adaptive_odom.gain_offset", 0.0);
  adaptive_lpf_alpha_ = declare_parameter("adaptive_odom.low_pass_filter_alpha", 0.1);
  adaptive_minimum_gain_ = declare_parameter("adaptive_odom.minimum_gain", 1000.0);
  adaptive_max_abs_accel_ = declare_parameter("adaptive_odom.maximum_abs_acceleration", 20.0);
  adaptive_imu_timeout_ = declare_parameter("adaptive_odom.imu_timeout", 0.5);

  if (use_servo_cmd_) {
    steering_to_servo_gain_ =
      declare_parameter<double>("steering_angle_to_servo_gain");
    steering_to_servo_offset_ =
      declare_parameter<double>("steering_angle_to_servo_offset");
    wheelbase_ = declare_parameter<double>("wheelbase");
  }

  publish_tf_ = declare_parameter("publish_tf", publish_tf_);

  // create odom publisher
  odom_pub_ = create_publisher<Odometry>("odom", 10);

  // debug: effective adaptive gain + filtered accel, for live monitoring/tuning
  // (relative topics -> /vesc/debug/adaptive_gain, /vesc/debug/filtered_accel)
  adaptive_gain_pub_ = create_publisher<Float64>("debug/adaptive_gain", 10);
  filtered_accel_pub_ = create_publisher<Float64>("debug/filtered_accel", 10);

  // create tf broadcaster
  if (publish_tf_) {
    tf_pub_.reset(new tf2_ros::TransformBroadcaster(this));
  }

  // subscribe to vesc state and. optionally, servo command
  vesc_state_sub_ = create_subscription<VescStateStamped>(
    "sensors/core", 10, std::bind(&VescToOdom::vescStateCallback, this, _1));

  if (use_servo_cmd_) {
    servo_sub_ = create_subscription<Float64>(
      "sensors/servo_position_command", 10, std::bind(&VescToOdom::servoCmdCallback, this, _1));
  }

  // VESC-internal IMU (relative topic -> /vesc/sensors/imu under the vesc
  // namespace) drives the adaptive gain via longitudinal acceleration.
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "sensors/imu", rclcpp::SensorDataQoS(), std::bind(&VescToOdom::imuCallback, this, _1));

  // apply gain/offset changes at runtime (ros2 param set, or the vesc_calibration
  // node's apply), so the odom conversion tracks a retuned mapping without restart.
  param_cb_handle_ = add_on_set_parameters_callback(
    std::bind(&VescToOdom::onSetParameters, this, _1));
}

rcl_interfaces::msg::SetParametersResult VescToOdom::onSetParameters(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  // validate before applying anything, so a rejected batch leaves state untouched
  for (const auto & p : params) {
    const auto & name = p.get_name();
    if (name == "adaptive_odom.low_pass_filter_alpha") {
      const double v = p.as_double();
      if (!std::isfinite(v) || v <= 0.0 || v > 1.0) {
        result.successful = false;
        result.reason = "adaptive_odom.low_pass_filter_alpha must be in (0, 1]";
        return result;
      }
    } else if (name == "adaptive_odom.minimum_gain") {
      const double v = p.as_double();
      if (!std::isfinite(v) || v <= 0.0) {
        result.successful = false;
        result.reason = "adaptive_odom.minimum_gain must be > 0";
        return result;
      }
    } else if (name == "adaptive_odom.acceleration_gain" ||
      name == "adaptive_odom.gain_offset" ||
      name == "adaptive_odom.maximum_abs_acceleration" ||
      name == "adaptive_odom.imu_timeout")
    {
      if (!std::isfinite(p.as_double())) {
        result.successful = false;
        result.reason = name + " must be finite";
        return result;
      }
    }
  }

  for (const auto & p : params) {
    const auto & name = p.get_name();
    if (name == "speed_to_erpm_gain") {
      speed_to_erpm_gain_ = p.as_double();
    } else if (name == "speed_to_erpm_offset") {
      speed_to_erpm_offset_ = p.as_double();
    } else if (name == "steering_angle_to_servo_gain") {
      steering_to_servo_gain_ = p.as_double();
    } else if (name == "steering_angle_to_servo_offset") {
      steering_to_servo_offset_ = p.as_double();
    } else if (name == "adaptive_odom.enabled") {
      adaptive_enabled_ = p.as_bool();
    } else if (name == "adaptive_odom.acceleration_gain") {
      adaptive_acceleration_gain_ = p.as_double();
    } else if (name == "adaptive_odom.gain_offset") {
      adaptive_gain_offset_ = p.as_double();
    } else if (name == "adaptive_odom.low_pass_filter_alpha") {
      adaptive_lpf_alpha_ = p.as_double();
    } else if (name == "adaptive_odom.minimum_gain") {
      adaptive_minimum_gain_ = p.as_double();
    } else if (name == "adaptive_odom.maximum_abs_acceleration") {
      adaptive_max_abs_accel_ = p.as_double();
    } else if (name == "adaptive_odom.imu_timeout") {
      adaptive_imu_timeout_ = p.as_double();
    }
  }
  return result;
}

void VescToOdom::imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu)
{
  // VESC IMU is mounted with +X along the vehicle heading; verify the sign on
  // the car by accelerating forward (ax must go positive) before tuning
  // acceleration_gain. See ADAPTIVE_WHEEL_ODOMETRY_PORTING_GUIDE.md.
  double ax = imu->linear_acceleration.x;
  if (!std::isfinite(ax)) {
    return;
  }
  ax = std::max(-adaptive_max_abs_accel_, std::min(ax, adaptive_max_abs_accel_));

  if (!imu_received_) {
    // seed the filter with the first valid sample to avoid a transient from 0
    filtered_ax_ = ax;
    imu_received_ = true;
  } else {
    filtered_ax_ = adaptive_lpf_alpha_ * ax + (1.0 - adaptive_lpf_alpha_) * filtered_ax_;
  }
  last_imu_time_ = now();
}

void VescToOdom::vescStateCallback(const VescStateStamped::SharedPtr state)
{
  // 변경 전(07.15): 첫 servo 명령이 올 때까지 odom을 아예 발행하지 않았음.
  // mux가 engage 전까지 idle이라 부팅 직후 /vesc/odom이 침묵
  // → cartographer(use_odometry=true)가 odom 큐를 기다리며 스캔 처리를 멈추는 데드락.
  //   if (use_servo_cmd_ && !last_servo_cmd_) {
  //     return;
  //   }
  // 변경 후: servo 명령 수신 전에는 조향각 0으로 간주하고 즉시 발행
  // (부팅 직후엔 정지 상태라 각속도 기여가 없어 안전).

  // convert to engineering units
  // Adaptive wheel odometry: gain = base + offset + sigma * filtered_ax, so
  // acceleration-proportional slip inflates the divisor instead of the speed.
  // Falls back to the fixed gain when disabled or the IMU is missing/stale.
  double gain = speed_to_erpm_gain_;
  if (adaptive_enabled_ && imu_received_) {
    const bool imu_fresh = adaptive_imu_timeout_ <= 0.0 ||
      (now() - last_imu_time_).seconds() < adaptive_imu_timeout_;
    if (imu_fresh) {
      gain = std::max(
        speed_to_erpm_gain_ + adaptive_gain_offset_ +
        adaptive_acceleration_gain_ * filtered_ax_,
        adaptive_minimum_gain_);
    }
  }
  double current_speed = (state->state.speed - speed_to_erpm_offset_) / gain;

  if (rclcpp::ok()) {
    Float64 gain_msg;
    gain_msg.data = gain;
    adaptive_gain_pub_->publish(gain_msg);
    Float64 accel_msg;
    accel_msg.data = filtered_ax_;
    filtered_accel_pub_->publish(accel_msg);
  }
  if (std::fabs(current_speed) < 0.05) {
    current_speed = 0.0;
  }
  double current_steering_angle(0.0), current_angular_velocity(0.0);
  if (use_servo_cmd_ && last_servo_cmd_) {  // 변경(07.15): && last_servo_cmd_ 조건 추가
    current_steering_angle =
      (last_servo_cmd_->data - steering_to_servo_offset_) / steering_to_servo_gain_;
    current_angular_velocity = current_speed * tan(current_steering_angle) / wheelbase_;
  }

  // use current state as last state if this is our first time here
  if (!last_state_) {
    last_state_ = state;
  }

  // calc elapsed time
  auto dt = rclcpp::Time(state->header.stamp) - rclcpp::Time(last_state_->header.stamp);

  /** @todo could probably do better propigating odometry, e.g. trapezoidal integration */

  // propigate odometry
  double x_dot = current_speed * cos(yaw_);
  double y_dot = current_speed * sin(yaw_);
  x_ += x_dot * dt.seconds();
  y_ += y_dot * dt.seconds();
  if (use_servo_cmd_) {
    yaw_ += current_angular_velocity * dt.seconds();
  }

  // save state for next time
  last_state_ = state;

  // publish odometry message
  Odometry odom;
  odom.header.frame_id = odom_frame_;
  odom.header.stamp = state->header.stamp;
  odom.child_frame_id = base_frame_;

  // Position
  odom.pose.pose.position.x = x_;
  odom.pose.pose.position.y = y_;
  odom.pose.pose.orientation.x = 0.0;
  odom.pose.pose.orientation.y = 0.0;
  odom.pose.pose.orientation.z = sin(yaw_ / 2.0);
  odom.pose.pose.orientation.w = cos(yaw_ / 2.0);

  // Position uncertainty
  /** @todo Think about position uncertainty, perhaps get from parameters? */
  odom.pose.covariance[0] = 0.2;   ///< x
  odom.pose.covariance[7] = 0.2;   ///< y
  odom.pose.covariance[35] = 0.4;  ///< yaw

  // Velocity ("in the coordinate frame given by the child_frame_id")
  odom.twist.twist.linear.x = current_speed;
  odom.twist.twist.linear.y = 0.0;
  odom.twist.twist.angular.z = current_angular_velocity;

  // Longitudinal velocity covariance (vx).
  odom.twist.covariance[0] = 0.0001;

  if (publish_tf_) {
    TransformStamped tf;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id = base_frame_;
    tf.header.stamp = now();
    tf.transform.translation.x = x_;
    tf.transform.translation.y = y_;
    tf.transform.translation.z = 0.0;
    tf.transform.rotation = odom.pose.pose.orientation;

    if (rclcpp::ok()) {
      tf_pub_->sendTransform(tf);
    }
  }

  if (rclcpp::ok()) {
    odom_pub_->publish(odom);
  }
}

void VescToOdom::servoCmdCallback(const Float64::SharedPtr servo)
{
  last_servo_cmd_ = servo;
}

}  // namespace vesc_ackermann

#include "rclcpp_components/register_node_macro.hpp"  // NOLINT

RCLCPP_COMPONENTS_REGISTER_NODE(vesc_ackermann::VescToOdom)
