# Chu–Lee–Sunwoo off-road local planner

This module implements *Local Path Planning for Off-Road Autonomous Driving
With Avoidance of Static Obstacles* (IEEE T-ITS, 2012) behind the independent
`OFFROADONLY` control state.

Implemented paper stages:

1. arc-length reparameterized cubic-spline base frame and Newton localization;
2. early-curvature quintic lateral-offset candidates that spread sooner and
   join their target offsets with continuous terminal curvature; the paper's
   cubic generator remains available as a fallback;
3. an optional quintic curvature fan that augments the selected offset family: every
   branch starts with the measured vehicle heading, samples a bounded initial
   steering angle, and rejoins a common base-frame position, heading, and
   curvature;
4. curvilinear/Cartesian transformation, nonholonomic curvature rejection, and
   footprint collision checks against a cell-quantized local LiDAR map;
5. Gaussian collision-risk, integrated squared-curvature, and previous-path
   consistency costs;
6. minimum-cost collision-free selection, longest-free-path fallback, and
   curvature/risk target-speed constraints.

The road-boundary rejection in stage 4 treats the vehicle as a rectangle
carried at the path's heading rather than as a point on the path centreline:
its lateral half-extent grows with the heading error, and `d_left`/`d_right`
are taken as the tightest values inside the footprint's longitudinal span. A
yawed vehicle can otherwise put a corner over the track bound while a
centreline-only check still passes.

The offset family uses early-curvature quintics when
`offroad_use_early_curvature_quintic` is true. Their initial curvature deviation
from the legacy cubic is scaled by `offroad_early_curvature_gain`; values above
one spread the branches earlier. Set the boolean false to restore the paper's
cubic offsets.

The curvature fan is enabled with `offroad_use_quintic_fan`. It is additive to
whichever offset family is selected. The fan size and terminal pose are
controlled by `offroad_fan_steering_samples`,
`offroad_fan_goal_distance_m`, and `offroad_fan_goal_offset_m`. The steering
sample count must be an odd integer so the zero-steering branch is present.

`offroad_planner.py.backup_pre_quintic_fan_20260806` is the source backup made
immediately before this extension was applied.

The paper does not specify the low-level path-tracking controller. A Pure
Pursuit adapter is therefore kept outside `OffRoadPlanner` in
`OffRoadController`; it converts the selected path and paper target speed into
the `(speed, steering_angle)` interface used by `controller_manager`.

The A1 experiment's numerical weights, vehicle dimensions, and heuristic gains
are not published in the paper. Every such value is exposed under the
`offroad_*` namespace in `stack_master/config/controller.yaml` and
`controller_map.yaml`. The checked-in defaults are scaled for the F1TENTH car
and must be validated on the target map and vehicle before driving at speed.

Runtime selection is controlled by `offroad_active` and `offroad_distance_m` in
`state_machine_params.yaml`. It is limited to a static TRAILING target and
triggers immediately when the target's front face is within the configured
distance.
FTG remains implemented and can be restored independently by disabling
`offroad_active` and enabling `ftg_active`.
