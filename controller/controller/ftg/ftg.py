#!/usr/bin/env python3
"""Follow-The-Gap controller with angle-based, beam-count-independent tuning.

Per scan: sanitize NaN->max, window and smooth the front scan, split usable
returns into gaps at close returns or range disparities, then score each gap by
its width, steering continuity, and global-raceline alignment. Aim at the chosen
gap centre and EMA-smooth the steer; if no gap exists, turn toward the side with
more aggregate clearance.
"""
import math
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


class FTG:
    # speed-scheduling thresholds on |steer| [rad]
    STRAIGHTS_STEERING_ANGLE = np.pi / 18   # 10 deg
    MILD_CURVE_ANGLE = np.pi / 6            # 30 deg
    ULTRASTRAIGHTS_ANGLE = np.pi / 60       # 3 deg

    # gap tunables: DEFAULTS, overridden from controller.yaml (ftg_*)
    # and live-tunable via rqt.
    FRONT_FOV = math.radians(45.0)    # HALF-angle [rad] = ftg_front_fov_deg/2 (total 90 deg)
    SMOOTH_RAD = math.radians(1.0)    # scan smoothing window [rad]     (ftg_smooth_deg)
    DISP_THRESH = 1.0                 # range jump = discontinuity [m]  (ftg_disp_thresh)
    BUBBLE_M = 0.30                   # accepted for config compatibility (unused)
    PREV_ANGLE_PENALTY_GAIN = 0.6     # score penalty per beam of target change
    RACELINE_PENALTY_GAIN = 0.4        # gap score penalty per beam from raceline
    RACELINE_LOOKAHEAD_M = 1.5         # global-raceline lookahead distance [m]
    NO_GAP_SPEED = 1.2                # fallback speed with no usable gap [m/s]
    STEER_EMA = 0.0                   # steer low-pass a: s=a*prev+(1-a)*new (ftg_steer_ema)
    MAX_STEER = 0.4                   # steering clip [rad]             (ftg_max_steer)
    SPEED_SCALE = 1.0                 # overall speed multiplier

    def __init__(self, node=None, mapping=False, debug=False,
                 safety_radius=None, max_lidar_dist=None, max_speed=1.5,
                 range_offset=None, track_width=2.0,
                 front_fov_deg=None, smooth_deg=None, disp_thresh=None,
                 bubble_m=None, steer_ema=None, max_steer=None,
                 prev_angle_penalty_gain=None, raceline_penalty_gain=None,
                 raceline_lookahead_m=None, no_gap_speed=None) -> None:
        self.node = node
        self.mapping = mapping

        self.DEBUG = debug
        self.SAFETY_RADIUS = safety_radius          # accepted for compat (unused)
        self.range_offset = range_offset            # accepted for compat (unused)
        self.MAX_LIDAR_DIST = max_lidar_dist if max_lidar_dist else 10.0
        self.MAX_SPEED = max_speed
        self.track_width = track_width

        # override class-default tunables from yaml when provided
        if front_fov_deg is not None:
            self.FRONT_FOV = math.radians(float(front_fov_deg) / 2.0)  # param = TOTAL front FOV
        if smooth_deg is not None:
            self.SMOOTH_RAD = math.radians(float(smooth_deg))
        if disp_thresh is not None:
            self.DISP_THRESH = float(disp_thresh)
        if bubble_m is not None:
            self.BUBBLE_M = float(bubble_m)
        if steer_ema is not None:
            self.STEER_EMA = float(steer_ema)
        if max_steer is not None:
            self.MAX_STEER = float(max_steer)
        if prev_angle_penalty_gain is not None:
            self.PREV_ANGLE_PENALTY_GAIN = float(prev_angle_penalty_gain)
        if raceline_penalty_gain is not None:
            self.RACELINE_PENALTY_GAIN = float(raceline_penalty_gain)
        if raceline_lookahead_m is not None:
            self.RACELINE_LOOKAHEAD_M = float(raceline_lookahead_m)
        if no_gap_speed is not None:
            self.NO_GAP_SPEED = float(no_gap_speed)

        self.recompute_speeds()

        self.velocity = 0.0
        self._steer_prev = None
        self._prev_gap_angle = None
        self.angle_min = -0.75 * np.pi
        self.angle_inc = None
        self.radians_per_elem = None

        self.best_pnt = self.scan_pub = self.best_gap = None
        if self.node is not None:
            self.best_pnt = self.node.create_publisher(Marker, '/best_points/marker', 10)
            self.scan_pub = self.node.create_publisher(MarkerArray, '/scan_proc/markers', 10)
            self.best_gap = self.node.create_publisher(MarkerArray, '/best_gap/markers', 10)

    def recompute_speeds(self) -> None:
        s = self.SPEED_SCALE
        self.CORNERS_SPEED = 0.3 * self.MAX_SPEED * s
        self.MILD_CORNERS_SPEED = 0.45 * self.MAX_SPEED * s
        self.STRAIGHTS_SPEED = 0.8 * self.MAX_SPEED * s
        self.ULTRASTRAIGHTS_SPEED = self.MAX_SPEED * s

    def set_vel(self, velocity) -> None:
        self.velocity = velocity

    def reset_history(self) -> None:
        """Forget steering/gap history when starting a new FTG session."""
        self._steer_prev = None
        self._prev_gap_angle = None

    def _now(self):
        return self.node.get_clock().now().to_msg() if self.node is not None else None

    def process_lidar(self, ranges, angle_min=None, angle_increment=None,
                      raceline_angle=None) -> tuple:
        """Returns (speed, steering_angle). angle_min/angle_increment from the
        LaserScan make it beam/FOV independent; if omitted a 270-deg FOV is assumed."""
        n = len(ranges)
        if angle_increment is not None and angle_increment > 0.0:
            self.angle_inc = float(angle_increment)
            self.angle_min = float(angle_min) if angle_min is not None else -(n - 1) * self.angle_inc / 2.0
        else:
            self.angle_inc = (1.5 * np.pi) / n
            self.angle_min = -(n - 1) * self.angle_inc / 2.0
        self.radians_per_elem = self.angle_inc

        # NaN/inf == no return == open -> max range, then clip
        r = np.asarray(ranges, dtype=float)
        r = np.where(np.isfinite(r), r, self.MAX_LIDAR_DIST)
        r = np.clip(r, 0.0, self.MAX_LIDAR_DIST)

        # front FOV window (angle-based)
        i_lo = max(0, int(math.ceil((-self.FRONT_FOV - self.angle_min) / self.angle_inc)))
        i_hi = min(n, int(math.floor((self.FRONT_FOV - self.angle_min) / self.angle_inc)) + 1)
        if i_hi - i_lo < 3:
            return self._no_gap_command(np.empty(0), np.empty(0))
        proc = r[i_lo:i_hi].copy()
        base_angle = self.angle_min + i_lo * self.angle_inc

        # smoothing window sized in radians
        w = max(1, int(round(self.SMOOTH_RAD / self.angle_inc)))
        if w > 1:
            proc = np.convolve(proc, np.ones(w) / w, 'same')

        angles = base_angle + np.arange(len(proc)) * self.angle_inc

        # Close returns are excluded, and each range jump starts a new gap.
        # This intentionally does not apply the old disparity safety bubble.
        gaps = self._find_gaps_by_abs_diff_split(proc, self.track_width / 2.0)
        gl, gr = self._select_gap_with_angle_penalty(
            gaps, angles, raceline_angle=raceline_angle)
        if gr <= gl:
            # Match the source controller: a real no-gap result forgets the old
            # target before selecting the clearer side. An empty FOV retains it.
            self._prev_gap_angle = None
            return self._no_gap_command(proc, angles)
        mid = (gl + gr) // 2
        self._prev_gap_angle = float(angles[mid])

        # steer toward the gap centre (0 = forward, + = left), then EMA-smooth
        raw_steer = float(np.clip(base_angle + mid * self.angle_inc,
                                  -self.MAX_STEER, self.MAX_STEER))
        if self._steer_prev is None:
            steer = raw_steer
        else:
            a = self.STEER_EMA
            steer = a * self._steer_prev + (1.0 - a) * raw_steer
        self._steer_prev = steer

        if self.DEBUG:
            self._publish_debug(proc, base_angle, gl, gr, mid)
        return self._speed_for(steer), steer

    def raceline_target_angle(self, waypoints, pose):
        """Return the bearing to a forward global-raceline lookahead point."""
        if waypoints is None or pose is None:
            return None
        pts = np.asarray(waypoints, dtype=float)
        car = np.asarray(pose, dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2 or car.size < 3:
            return None
        if not np.all(np.isfinite(car[:3])) or not np.all(np.isfinite(pts[:, :2])):
            return None

        car_xy = car[:2]
        nearest = int(np.argmin(np.linalg.norm(pts[:, :2] - car_xy, axis=1)))
        target = nearest
        travelled = 0.0
        lookahead = max(0.0, self.RACELINE_LOOKAHEAD_M)
        for _ in range(len(pts)):
            if travelled >= lookahead:
                break
            nxt = (target + 1) % len(pts)
            travelled += float(np.linalg.norm(pts[nxt, :2] - pts[target, :2]))
            target = nxt

        delta = pts[target, :2] - car_xy
        if float(np.linalg.norm(delta)) < 1e-6:
            return None
        angle = math.atan2(float(delta[1]), float(delta[0])) - float(car[2])
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return float(np.clip(angle, -self.FRONT_FOV, self.FRONT_FOV))

    def _find_gaps_by_abs_diff_split(self, ranges, min_dist) -> list:
        """Return usable ``[start, end)`` runs split at range disparities."""
        gaps = []
        start = None
        n = len(ranges)
        for i in range(n):
            if float(ranges[i]) <= min_dist:
                if start is not None and i > start:
                    gaps.append((start, i))
                start = None
                continue

            if start is None:
                start = i

            if i < n - 1 and abs(float(ranges[i + 1] - ranges[i])) > self.DISP_THRESH:
                gaps.append((start, i + 1))
                start = None

        if start is not None and n > start:
            gaps.append((start, n))
        return gaps

    def _select_gap_with_angle_penalty(self, gaps, angles,
                                       raceline_angle=None) -> tuple:
        """Score gap width against steering continuity and global raceline."""
        if not gaps:
            return 0, 0
        if len(angles) < 2:
            return max(gaps, key=lambda gap: gap[1] - gap[0])

        use_prev = self._prev_gap_angle is not None
        use_raceline = raceline_angle is not None and np.isfinite(raceline_angle)
        if not use_prev and not use_raceline:
            return max(gaps, key=lambda gap: gap[1] - gap[0])

        angle_step = max(abs(float(angles[1] - angles[0])), 1e-6)

        def score(gap):
            start, end = gap
            center_angle = float(angles[(start + end) // 2])
            value = float(end - start)
            if use_prev:
                prev_diff_beams = abs(center_angle - self._prev_gap_angle) / angle_step
                value -= self.PREV_ANGLE_PENALTY_GAIN * prev_diff_beams
            if use_raceline:
                race_diff_beams = abs(center_angle - raceline_angle) / angle_step
                value -= self.RACELINE_PENALTY_GAIN * race_diff_beams
            return value

        return max(gaps, key=score)

    def _no_gap_command(self, ranges, angles) -> tuple:
        """Use the previous target or turn toward the side with more clearance."""
        if self._prev_gap_angle is not None:
            steer = float(np.clip(self._prev_gap_angle, -self.MAX_STEER, self.MAX_STEER))
        elif len(ranges) > 0:
            left_clearance = float(np.sum(ranges[angles > 0.0]))
            right_clearance = float(np.sum(ranges[angles < 0.0]))
            steer = self.MAX_STEER if left_clearance >= right_clearance else -self.MAX_STEER
        else:
            steer = self.MAX_STEER
        self._steer_prev = steer
        return self.NO_GAP_SPEED, steer

    def _speed_for(self, steering_angle) -> float:
        if self.mapping:
            return 1.5
        a = abs(steering_angle)
        if a > self.MILD_CURVE_ANGLE:
            return self.CORNERS_SPEED
        if a > self.STRAIGHTS_STEERING_ANGLE:
            return self.MILD_CORNERS_SPEED
        if a > self.ULTRASTRAIGHTS_ANGLE:
            return self.STRAIGHTS_SPEED
        return self.ULTRASTRAIGHTS_SPEED

    def _publish_debug(self, proc, base_angle, gl, gr, mid) -> None:
        clr = MarkerArray()
        m = Marker(); m.header.frame_id = 'laser'; m.header.stamp = self._now()
        m.action = Marker.DELETEALL
        clr.markers.append(m)
        self.best_gap.publish(clr)

        gap_markers = MarkerArray()
        for i in range(gl, gr):
            ang = base_angle + i * self.angle_inc
            rng = float(proc[i]) if i < len(proc) else 1.0
            mrk = Marker()
            mrk.header.frame_id = 'laser'; mrk.header.stamp = self._now()
            mrk.type = mrk.SPHERE
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.05
            mrk.color.a = 1.0; mrk.color.r = 1.0; mrk.color.g = 1.0
            mrk.id = i - gl
            mrk.pose.position.x = math.cos(ang) * rng
            mrk.pose.position.y = math.sin(ang) * rng
            mrk.pose.orientation.w = 1.0
            gap_markers.markers.append(mrk)
        self.best_gap.publish(gap_markers)

        ang = base_angle + mid * self.angle_inc
        rng = float(proc[mid]) if mid < len(proc) else 1.0
        bm = Marker()
        bm.header.frame_id = 'laser'; bm.header.stamp = self._now()
        bm.type = bm.SPHERE
        bm.scale.x = bm.scale.y = bm.scale.z = 0.2
        bm.color.a = 1.0; bm.color.b = 1.0; bm.color.g = 1.0
        bm.id = 0
        bm.pose.position.x = math.cos(ang) * rng
        bm.pose.position.y = math.sin(ang) * rng
        bm.pose.orientation.w = 1.0
        self.best_pnt.publish(bm)

        sm = MarkerArray()
        step = max(1, len(proc) // 360)
        for i in range(0, len(proc), step):
            ang = base_angle + i * self.angle_inc
            mrk = Marker()
            mrk.header.frame_id = 'laser'; mrk.header.stamp = self._now()
            mrk.type = mrk.SPHERE
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.05
            mrk.color.a = 1.0; mrk.color.r = 1.0; mrk.color.b = 1.0
            mrk.id = i
            mrk.pose.position.x = math.cos(ang) * float(proc[i])
            mrk.pose.position.y = math.sin(ang) * float(proc[i])
            mrk.pose.orientation.w = 1.0
            sm.markers.append(mrk)
        self.scan_pub.publish(sm)
