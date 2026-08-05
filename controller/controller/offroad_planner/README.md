# Chu–Lee–Sunwoo off-road local planner

This module implements *Local Path Planning for Off-Road Autonomous Driving
With Avoidance of Static Obstacles* (IEEE T-ITS, 2012) behind the independent
`OFFROADONLY` control state.

Implemented paper stages:

1. arc-length reparameterized cubic-spline base frame and Newton localization;
2. cubic lateral-offset candidates with the paper's boundary conditions and
   speed-dependent maneuver length;
3. curvilinear/Cartesian transformation, nonholonomic curvature rejection, and
   footprint collision checks against a cell-quantized local LiDAR map;
4. Gaussian collision-risk, integrated squared-curvature, and previous-path
   consistency costs;
5. minimum-cost collision-free selection, longest-free-path fallback, and
   curvature/risk target-speed constraints.

The paper does not specify the low-level path-tracking controller. A Pure
Pursuit adapter is therefore kept outside `OffRoadPlanner` in
`OffRoadController`; it converts the selected path and paper target speed into
the `(speed, steering_angle)` interface used by `controller_manager`.

The A1 experiment's numerical weights, vehicle dimensions, and heuristic gains
are not published in the paper. Every such value is exposed under the
`offroad_*` namespace in `stack_master/config/controller.yaml` and
`controller_map.yaml`. The checked-in defaults are scaled for the F1TENTH car
and must be validated on the target map and vehicle before driving at speed.

Runtime selection is controlled by `offroad_active`, `offroad_speed_mps`, and
`offroad_timer_sec` in `state_machine_params.yaml`. FTG remains implemented and
can be restored independently by disabling `offroad_active` and enabling
`ftg_active`.
