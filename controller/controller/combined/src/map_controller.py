#!/usr/bin/env python3

import logging

import numpy as np

from visualization_msgs.msg import Marker, MarkerArray


def parse_l1_sectors(flat):
    """Flat param list -> [(s_start, s_end, t_clip_min), ...].

    The yaml side is a plain double array so it stays live-tunable through the
    normal ROS2 param path: [s0_start, s0_end, s0_floor, s1_start, ...].
    Values that don't complete a triple are dropped; a triple with floor <= 0
    is a disabled sector. s_start > s_end means the sector wraps past s = 0.
    """
    if not flat:
        return []
    vals = [float(v) for v in flat]
    return [tuple(vals[i:i + 3]) for i in range(0, len(vals) - 2, 3)
            if vals[i + 2] > 0.0]


class Controller:
    """This class implements the L1 / Pure-Pursuit controller for autonomous driving.
    Input and output topics are managed by the controller manager.

    ROS2 port note: the ROS1 MAP (steering-lookup) branch is back behind the
    `use_map` flag (steer_lookup instance injected by the manager); Pure-Pursuit
    remains the default and the safety fallback. The marker publisher and loggers
    are injected by the manager so this class stays ROS-node-free.
    """

    def __init__(self,
                t_clip_min,
                t_clip_max,
                m_l1,
                q_l1,

                curvature_factor,

                KP,
                KI,
                KD,
                heading_error_thres,
                steer_gain_for_speed,

                future_constant,

                speed_lookahead,
                lat_err_coeff,
                acc_scaler_for_steer,
                dec_scaler_for_steer,
                start_scale_speed,
                end_scale_speed,
                downscale_factor,
                speed_lookahead_for_steer,

                trailing_gap,
                trailing_vel_gain,
                trailing_p_gain,
                trailing_i_gain,
                trailing_d_gain,
                blind_trailing_speed,

                loop_rate,
                wheelbase,

                speed_factor_for_lat_err,
                speed_factor_for_curvature,

                speed_diff_thres,
                start_speed,
                start_curvature_factor,

                AEB_thres,

                converter,

                steer_lookup=None,   # MAP: LookupSteerAngle instance (None -> PP only)
                use_map=False,       # MAP on/off; live-toggled by the manager

                l1_chord_err=1.0,          # v3 chord-error cap [m]; >=1.0 ~ off
                lat_err_steer_coeff=0.6931,  # v3; ln2 = legacy exp(ln2*d)
                map_speed_blend=0.0,       # MAP lookup speed: 0 = reference, 1 = actual

                ot_sectors=None,     # per-map s sectors where an OVERTAKE gets a
                                     # shorter L1 floor: flat [start, end, floor, ...]
                                     # (see parse_l1_sectors); empty -> feature off
                ot_l1_hold_sec=0.0,     # keep the avoidance L1 floor this long after
                                        # OVERTAKE/OFFROADONLY ends
                ot_l1_release_sec=0.0,  # then ramp it back to t_clip_min over this

                predict_pub=None,
                logger_info=logging.info,
                logger_warn=logging.warning,
            ):

        # Parameters from manager
        self.t_clip_min = t_clip_min
        self.t_clip_max = t_clip_max
        self.m_l1 = m_l1
        self.q_l1 = q_l1
        self.speed_lookahead = speed_lookahead
        self.lat_err_coeff = lat_err_coeff
        self.acc_scaler_for_steer = acc_scaler_for_steer
        self.dec_scaler_for_steer = dec_scaler_for_steer
        self.start_scale_speed = start_scale_speed
        self.end_scale_speed = end_scale_speed
        self.downscale_factor = downscale_factor
        self.speed_lookahead_for_steer = speed_lookahead_for_steer

        # marker publisher injected by the manager (ROS2: a Node-created publisher)
        self.predict_pub = predict_pub

        # L1 dist calc param
        self.curvature_factor = curvature_factor

        self.speed_factor_for_lat_err = speed_factor_for_lat_err
        self.speed_factor_for_curvature = speed_factor_for_curvature

        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.heading_error_thres = heading_error_thres
        self.steer_gain_for_speed = steer_gain_for_speed

        self.future_constant = future_constant

        self.trailing_gap = trailing_gap
        self.trailing_vel_gain = trailing_vel_gain
        self.trailing_p_gain = trailing_p_gain
        self.trailing_i_gain = trailing_i_gain
        self.trailing_d_gain = trailing_d_gain
        self.blind_trailing_speed = blind_trailing_speed

        self.loop_rate = loop_rate
        self.AEB_thres = AEB_thres
        self.converter = converter

        # MAP steering lookup (None / use_map=False -> pure-pursuit only)
        self.steer_lookup = steer_lookup
        self.use_map = use_map

        # v3 lateral-control refinements (defaults reproduce legacy behaviour)
        self.l1_chord_err = l1_chord_err
        self.lat_err_steer_coeff = lat_err_steer_coeff
        self.map_speed_blend = map_speed_blend

        # per-map OVERTAKE L1 floor sectors (see calc_future_L1_point). Stored
        # parsed; the manager re-parses on a live param update.
        self.ot_sectors = parse_l1_sectors(ot_sectors)
        self.ot_l1_hold_sec = ot_l1_hold_sec
        self.ot_l1_release_sec = ot_l1_release_sec
        # Latched avoidance floor + ticks since OVERTAKE/OFFROADONLY ended.
        # Attribute names stay ot_l1_* for parameter compatibility.
        self.ot_l1_floor = None
        self.ot_l1_ticks = 0

        # Parameters in the controller
        self.curr_steering_angle = 0
        self.idx_nearest_waypoint = None  # index of nearest waypoint to car
        self.track_length = None
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0
        self.trailing_command = 2
        self.speed_command = None
        self.last_valid_speed = 0
        self.curvature_waypoints = 0
        self.current_steer_command = 0
        self.yaw_rate = 0

        self.logger_info = logger_info
        self.logger_warn = logger_warn

        self.speed_diff_thres = speed_diff_thres
        self.start_speed = start_speed
        self.start_curvature_factor = start_curvature_factor

        self.wheelbase = wheelbase

        self.start_mode = False
        self.future_lat_err = 0.0
        # Per-tick trace of the intermediate values behind the steering command.
        # controller_manager publishes it on /controller/debug when measure:=true;
        # fast_tune ignores it. Written every tick, so a stale key means that
        # branch did not run this tick (e.g. the chord cap never engaged).
        self.dbg = {}
        self.future_lat_e_norm = 0.0
        self.boost_mode = False

    def main_loop(self, state, position_in_map, waypoint_array_in_map, speed_now, opponent, position_in_map_frenet, acc_now, track_length):
        # Updating parameters from manager
        self.state = state
        self.position_in_map = position_in_map

        #-------------------------------Future Position-----------------------------
        self.future_position = np.zeros((1, 3))
        #-------------------------------Future Position-----------------------------

        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.acc_now = acc_now
        self.track_length = track_length

        ## PREPROCESS ##
        # speed vector
        yaw = self.position_in_map[0, 2]

        v = [np.cos(yaw)*self.speed_now, np.sin(yaw)*self.speed_now]

        #-------------------------------Future Position-----------------------------

        self.future_position = self.calc_future_position(self.future_constant)

        #-------------------------------Future Position-----------------------------

        self.idx_nearest_waypoint = self.nearest_waypoint(self.position_in_map[0, :2], self.waypoint_array_in_map[:, :2])

        # if all waypoints are equal set self.idx_nearest_waypoint to 0
        if np.isnan(self.idx_nearest_waypoint):
            self.idx_nearest_waypoint = 0

        if len(self.waypoint_array_in_map[self.idx_nearest_waypoint:]) > 2:
            # calculate curvature of global optimizer waypoints
            self.curvature_waypoints = np.mean(abs(self.waypoint_array_in_map[self.idx_nearest_waypoint+10:self.idx_nearest_waypoint+20, 5]))

        # calculate future lateral error and future lateral error norm

        self.future_lat_e_norm, self.future_lat_err = self.calc_future_lateral_error_norm()

        ### LONGITUDINAL CONTROL ###

        #-----------------------------------------Future-------------------------------------------
        self.speed_command = self.calc_speed_command(v, self.future_lat_e_norm)
        #-----------------------------------------Future-------------------------------------------

        self.speed_command = self.speed_adjust_heading(self.speed_command)

        # POSTPROCESS for acceleration/speed decision

        # Only advance the last-valid anchor on a finite command; np.isfinite
        # keeps a NaN/inf speed (e.g. from an empty curvature slice when the car
        # runs off a stale frozen local path) from reaching VESC/sim, which has
        # no guard and would poison odom. Mirrors the steering safety net below.
        if self.speed_command is not None and np.isfinite(self.speed_command):
            speed = max(self.speed_command, 0)
            self.last_valid_speed = speed
            acceleration = 0
            jerk = 0

        else:
            speed = self.last_valid_speed
            jerk = 0
            acceleration = 0
            self.logger_warn("[Controller] non-finite/none speed; holding last valid speed")

        ### LATERAL CONTROL ###

        steering_angle = None
        self.future_idx_nearest_waypoint = self.nearest_waypoint(self.future_position[0, :2], self.waypoint_array_in_map[:, :2])

        #-----------------------------------------Future-------------------------------------------
        L1_point, L1_distance = self.calc_future_L1_point(self.future_lat_err)
        #-----------------------------------------Future-------------------------------------------

        if L1_point.any() is not None:

            #-----------------------------------------Future-------------------------------------------
            steering_angle = self.calc_steering_angle_for_future(L1_point, L1_distance, yaw, self.future_lat_e_norm, v)
            #-----------------------------------------Future-------------------------------------------

            self.current_steer_command = steering_angle

        else:
            raise Exception("L1_point is None")

        # Final safety net: never emit a non-finite steering command. A NaN/inf
        # here (e.g. from a degenerate local trajectory during overtaking) would
        # otherwise propagate into calc_future_position next cycle and poison the
        # controller permanently, and downstream (VESC / sim) has no guard. Hold
        # the last valid steer instead so the node stays alive and recovers once
        # clean waypoints return; current_steer_command is kept finite for the
        # future-position feedback.
        if not np.isfinite(steering_angle):
            self.logger_warn("[Controller] non-finite steering; holding last valid steer")
            steering_angle = self.curr_steering_angle
            self.current_steer_command = steering_angle

        speed = self.AEB_for_weird_local_wpnt(speed)

        return speed, acceleration, jerk, steering_angle, L1_point, L1_distance, self.idx_nearest_waypoint, self.curvature_waypoints, self.future_position

    def AEB_for_weird_local_wpnt(self, speed):
        nearest_local_wpnt = self.waypoint_array_in_map[self.idx_nearest_waypoint, :2]

        local_wpnt_dist = np.sqrt((self.position_in_map[0, 0] - nearest_local_wpnt[0])**2 + (self.position_in_map[0, 1] - nearest_local_wpnt[1])**2)

        if local_wpnt_dist >= self.AEB_thres:
            return 2.0
        else:
            return speed

    def calc_steering_angle_for_future(self, future_L1_point, L1_distance, yaw, furture_lat_e_norm, v):
        """
        The purpose of this function is to calculate the steering angle based on the L1 point, desired lateral acceleration and velocity

        Inputs:
            future_L1_point: future_L1_point in frenet coordinates at L1 distance in front of the car
            L1_distance: distance of the L1 point to the car
            yaw: yaw angle of the car
            furture_lat_e_norm: future normed lateral error
            v : future speed vector

        Returns:
            steering_angle: calculated steering angle


        """
        marks = MarkerArray()
        for i in range(1):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.3
            mrk.scale.y = 0.3
            mrk.scale.z = 0.3
            mrk.color.a = 1.0
            mrk.color.b = 1.0

            mrk.id = i
            mrk.pose.position.x = self.future_position[0, 0]
            mrk.pose.position.y = self.future_position[0, 1]
            mrk.pose.orientation.w = 1.0
            marks.markers.append(mrk)

        if self.predict_pub is not None:
            self.predict_pub.publish(marks)

        if (self.state == "TRAILING") and (self.opponent is not None):
            speed_la_for_lu = self.speed_now
        else:
            adv_ts_st = self.speed_lookahead_for_steer
            la_position_steer = [self.future_position[0, 0] + v[0]*adv_ts_st, self.future_position[0, 1] + v[1]*adv_ts_st]
            idx_future_la_steer = self.nearest_waypoint(la_position_steer, self.waypoint_array_in_map[:, :2])
            speed_la_for_lu = self.waypoint_array_in_map[idx_future_la_steer, 2]

        speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, furture_lat_e_norm)

        Future_L1_vector = np.array([future_L1_point[0] - self.future_position[0, 0], future_L1_point[1] - self.future_position[0, 1]])

        L1_norm = np.linalg.norm(Future_L1_vector)
        if L1_norm < 1e-6:
            self.logger_warn("[Controller] norm of L1 vector was ~0, eta is set to 0")
            eta = 0
        else:
            # clip to [-1, 1]: the ratio is analytically bounded but can exceed
            # 1 by fp rounding (esp. on sharp evasion where L1 is ~normal to the
            # heading), and arcsin(>1)=NaN would poison steering. Mirrors the
            # guarded arcsin in calc_future_position.
            sin_eta = np.dot([-np.sin(yaw), np.cos(yaw)], Future_L1_vector) / L1_norm
            eta = np.arcsin(np.clip(sin_eta, -1.0, 1.0))

        # Pure-Pursuit steering — always computed first so it doubles as the
        # safety fallback when the MAP lookup is disabled, missing, or fails.
        steering_angle = np.arctan(2*self.wheelbase*np.sin(eta)/L1_distance)

        # --- debug trace (see self.dbg) -------------------------------------
        # Only the final steering command leaves this class, so a bag can show
        # WHAT the controller did but never WHY. Recording the intermediate
        # values costs nothing and makes a run attributable: which stage moved
        # the command, and whether the lookup was used at all.
        self.dbg["eta"] = float(eta)
        self.dbg["speed_for_lu"] = float(speed_for_lu)
        self.dbg["future_lat_err"] = float(self.future_lat_err)
        self.dbg["steer_pp"] = float(steering_angle)
        self.dbg["lat_acc"] = float("nan")
        self.dbg["steer_map"] = float("nan")
        self.dbg["map_used"] = 0.0
        # --------------------------------------------------------------------

        if self.use_map and self.steer_lookup is not None:
            # MAP (ROS1 branch restored): lateral acceleration required by the
            # L1 geometry, inverted through the measured steering lookup table
            # so tire slip is compensated at the source. Uses speed_for_lu
            # (lookahead + lat-err adjusted speed), matching the ROS1 original.
            # speed_for_lu is the *reference* waypoint speed, so on a straight the
            # car is still accelerating towards it and lat_acc (~v^2) is inflated,
            # amplifying small eta into steering oscillation. Blend towards the
            # actual speed; 0.0 keeps the legacy behaviour exactly.
            v_lu = speed_for_lu + self.map_speed_blend * (self.speed_now - speed_for_lu)
            lat_acc = 2 * v_lu**2 * np.sin(eta) / L1_distance
            self.dbg["lat_acc"] = float(lat_acc)
            if np.isfinite(lat_acc) and np.isfinite(v_lu):
                try:
                    map_steer = float(self.steer_lookup.lookup_steer_angle(lat_acc, v_lu))
                except Exception as e:
                    map_steer = np.nan
                    self.logger_warn(f"[Controller] MAP lookup failed ({e}); using PP fallback")
                self.dbg["steer_map"] = float(map_steer)
                if np.isfinite(map_steer):
                    steering_angle = map_steer
                    self.dbg["map_used"] = 1.0

        dt = 1.0 / self.loop_rate

        #-------------------------Steering Scaling-----------------------------

        # modifying steer based on heading

        head_corr = self.compute_future_heading_correction(Future_L1_vector, yaw, dt, self.speed_now)
        steering_angle += head_corr
        self.dbg["head_corr"] = float(head_corr)

        # modifying steer based on acceleration
        #########################################
        steering_angle = self.acc_scaling(steering_angle)
        #########################################
        self.dbg["steer_after_acc"] = float(steering_angle)

        # modifying steer based on speed

        steering_angle = self.speed_steer_scaling(steering_angle, speed_for_lu)
        self.dbg["steer_after_speed"] = float(steering_angle)

        # modifying steer based on velocity

        steering_angle *= np.clip(1 + (self.speed_now/10), 1, self.steer_gain_for_speed)

        # modifying steer based on lateral error

        steering_angle = self.steer_scaling_for_lat_err(steering_angle, self.future_lat_err)
        self.dbg["steer_after_laterr"] = float(steering_angle)

        #-------------------------Steering Scaling-----------------------------

        # limit change of steering angle
        threshold = 0.4
        if abs(steering_angle - self.curr_steering_angle) > threshold:
            self.logger_info("steering angle clipped")
        steering_angle = np.clip(steering_angle, self.curr_steering_angle - threshold, self.curr_steering_angle + threshold)
        steering_angle = np.clip(steering_angle, -0.53, 0.53)

        # np.clip passes NaN through unchanged, so guard curr_steering_angle
        # (the rate-limit/feedback anchor) against poisoning: only advance it on
        # a finite result. main_loop's final net then falls back to this value.
        if np.isfinite(steering_angle):
            self.curr_steering_angle = steering_angle

        return steering_angle

    def sector_t_clip_min(self, s):
        """L1 lower bound for the s sector the car is in, else the global one.

        First matching sector wins, so overlapping ranges are resolved by the
        order they are listed in the yaml.
        """
        for start, end, floor in self.ot_sectors:
            inside = start <= s <= end if start <= end else (s >= start or s <= end)
            if inside:
                return floor
        return self.t_clip_min

    def latch_l1_floor(self, floor=None, s=None):
        """Latch a short avoidance lookahead before the post-state release.

        OVERTAKE gets its floor from the configured Frenet-s sector. OFFROADONLY
        bypasses this controller, so its manager supplies the lookahead used by
        the off-road Pure Pursuit controller when that state ends.
        """
        if floor is None:
            if s is None:
                s = self.position_in_map_frenet[0]
            floor = self.sector_t_clip_min(s)
        floor = float(floor)
        self.ot_l1_floor = floor if np.isfinite(floor) else self.t_clip_min
        self.ot_l1_ticks = 0
        return self.ot_l1_floor

    def latch_offroad_l1_floor(
        self, lookahead_min_m, lookahead_speed_gain_s, speed_mps
    ):
        """Latch the lookahead used by OFFROADONLY at handover."""
        floor = (
            float(lookahead_min_m)
            + float(lookahead_speed_gain_s) * abs(float(speed_mps))
        )
        return self.latch_l1_floor(floor=min(self.t_clip_min, floor))

    def ot_l1_floor_now(self):
        """Avoidance L1 floor, held and ramped after its state ends.

        The state machine drops OVERTAKE the moment the obstacle leaves the
        lidar FOV, which is while the car is still alongside/just past it. A
        hard snap back to t_clip_min there yanks the car onto the raceline. So
        the sector floor is latched: held flat for ot_l1_hold_sec, then blended
        linearly to t_clip_min over ot_l1_release_sec. Re-entering OVERTAKE
        re-latches, so a second obstacle restarts the hold. OFFROADONLY uses the
        same release after its direct controller hands control back.
        """
        if self.state == "OVERTAKE":
            return self.latch_l1_floor(s=self.position_in_map_frenet[0])

        if self.ot_l1_floor is None:
            return self.t_clip_min

        self.ot_l1_ticks += 1
        hold = self.ot_l1_hold_sec * self.loop_rate
        ramp = self.ot_l1_release_sec * self.loop_rate
        if self.ot_l1_ticks <= hold:
            return self.ot_l1_floor
        if self.ot_l1_ticks < hold + ramp:
            alpha = (self.ot_l1_ticks - hold) / ramp
            return self.ot_l1_floor + alpha * (self.t_clip_min - self.ot_l1_floor)

        self.ot_l1_floor = None
        return self.t_clip_min

    def calc_future_L1_point(self, future_lateral_error):

        # calculate future L1 guidance

        if self.speed_now < 2.0:

            speed = np.clip(self.speed_command, self.speed_now - 1, self.speed_now + 1)
            speed_scaler = self.m_l1 * speed

        else:

            speed_scaler = self.m_l1 * self.speed_now

        if self.state == "START":
            curvature_scaler = self.start_curvature_factor*self.curvature_waypoints
        else:
            curvature_scaler = self.curvature_factor*self.curvature_waypoints*self.speed_now*self.speed_now

        L1_distance = (speed_scaler - curvature_scaler) + self.q_l1

        # clip lower bound to avoid ultraswerve when far away from mincurv.
        # t_clip_min is tuned for the global line, which is far longer than the
        # short lattice avoidance path needs: measured in map m's sigma section
        # (s 16-24) it pinned L1 at 1.70 m for 100% of the OVERTAKE ticks while
        # the speed term asked for 0.6 m, so the car cut the obstacle. Each
        # ot_sectors triple gives one s range its own shorter floor, so every
        # static obstacle on the track can be tuned independently.
        t_clip_min = self.ot_l1_floor_now()
        self.dbg["l1_floor"] = float(t_clip_min)
        self.dbg["l1_floor_hold"] = float(self.ot_l1_ticks) if self.ot_l1_floor is not None else -1.0
        lower_bound = max(t_clip_min, np.sqrt(2)*future_lateral_error)

        L1_distance = np.clip(L1_distance, lower_bound, self.t_clip_max)

        # NaN guard: curvature_waypoints goes NaN when the +10:+20 curvature
        # slice is empty (stale/short local waypoints right after a teleport
        # reset); np.clip passes NaN through and int(NaN) below would kill
        # the node. Fall back to the shortest safe lookahead until clean
        # waypoints return. Mirrors the finite-guards in main_loop.
        if not np.isfinite(L1_distance):
            L1_distance = t_clip_min

        # --- chord-error cap (v3) ---------------------------------------
        # Aiming a chord at a point L1 ahead on a curve cuts the corner by
        # e ~= kappa*L1^2/8; cap L1 so e stays under l1_chord_err. kappa is
        # previewed over the WHOLE [car, car+L1] span (not the fixed +10:+20
        # window) so the cap tightens continuously BEFORE the L1 point enters
        # the corner - removes the entry L1-collapse step that caused the
        # high-speed corner-entry steering flip. l1_chord_err >= ~1.0
        # effectively disables the cap (legacy behaviour).
        wp = self.waypoint_array_in_map
        idx0 = self.future_idx_nearest_waypoint
        ds = np.diff(wp[idx0:idx0 + 20, 4])
        spacing = float(np.median(ds[ds > 0])) if np.any(ds > 0) else 0.1
        n_ahead = max(int(L1_distance / spacing), 2)
        kappa_preview = float(np.mean(np.abs(wp[idx0:idx0 + n_ahead, 5])))
        self.dbg["kappa_preview"] = kappa_preview
        self.dbg["L1_uncapped"] = float(L1_distance)
        self.dbg["L1_cap"] = float("inf")
        self.dbg["cap_bound"] = 0.0
        if kappa_preview > 1e-4:
            L1_cap = np.sqrt(8.0 * self.l1_chord_err / kappa_preview)
            self.dbg["L1_cap"] = float(L1_cap)
            self.dbg["cap_bound"] = 1.0 if L1_cap < L1_distance else 0.0
            L1_distance = max(min(L1_distance, L1_cap), lower_bound)
        # ----------------------------------------------------------------

        future_L1_point = self.waypoint_at_distance_before_car(L1_distance, self.waypoint_array_in_map[:, :2], self.future_idx_nearest_waypoint)

        return future_L1_point, L1_distance

    def calc_speed_command(self, v, lat_e_norm):
        """
        The purpose of this function is to isolate the speed calculation from the main control_loop

        Inputs:
            v: speed vector
            lat_e_norm: normed lateral error
            curvature_waypoints: -
        Returns:
            speed_command: calculated and adjusted speed, which can be sent to mux
        """

        # lookahead for speed (speed delay incorporation by propagating position)
        adv_ts_sp = self.speed_lookahead
        offset = 2
        la_position = [self.position_in_map[0, 0] + v[0]*adv_ts_sp, self.position_in_map[0, 1] + v[1]*adv_ts_sp]
        idx_la_position = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])
        idx_la_position = np.clip(idx_la_position + offset, 0, len(self.waypoint_array_in_map) - 1)
        global_speed = self.waypoint_array_in_map[idx_la_position, 2]
        cur_speed = self.speed_now

        if cur_speed < 0:
            cur_speed = 0

        if (self.state == "START"
            and self.boost_mode
            and self.waypoint_array_in_map[0, 7] > 0):
            if (global_speed-cur_speed) > 0:
                global_speed = self.start_speed
            elif self.cur_state_speed - cur_speed > 0:
                self.cur_state_speed -= self.speed_diff_thres * (self.cur_state_speed - cur_speed)
                global_speed = self.cur_state_speed
            else:
                self.boost_mode = False
        else:
            self.boost_mode = False

        if ((self.state == "TRAILING") and (self.opponent is not None)):  # Trailing controller
            speed_command = self.trailing_controller(global_speed)
        else:
            self.trailing_speed = global_speed
            self.i_gap = 0
            speed_command = global_speed

        speed_command = self.speed_adjust_lat_err(speed_command, lat_e_norm)

        return speed_command

    def trailing_controller(self, global_speed):
        """
        Adjust the speed of the ego car to trail the opponent at a fixed distance
        Inputs:
            speed_command: velocity of global raceline
            self.opponent: frenet s position and vs velocity of opponent
            self.position_in_map_frenet: frenet s position and vs veloctz of ego car
        Returns:
            trailing_command: reference velocity for trailing
        """

        self.gap = (self.opponent[0] - self.position_in_map_frenet[0]) % self.track_length  # gap to opponent
        self.gap_actual = self.gap
        self.gap_should = self.trailing_vel_gain * self.speed_now + self.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff = self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error/self.loop_rate, -10, 10)

        p_value = self.gap_error * self.trailing_p_gain
        d_value = self.v_diff * self.trailing_d_gain
        i_value = self.i_gap * self.trailing_i_gain

        self.trailing_command = np.clip(self.opponent[2] - p_value - i_value - d_value, 0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            self.trailing_command = max(self.blind_trailing_speed, self.trailing_command)

        return self.trailing_command

    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)

    def acc_scaling(self, steer):
        """
        Steer scaling based on acceleration
        increase steer when accelerating
        decrease steer when decelerating

        Returns:
            steer: scaled steering angle based on acceleration
        """

        if self.start_mode:
            return steer

        if np.mean(self.acc_now) >= 1:
            steer *= self.acc_scaler_for_steer
        elif np.mean(self.acc_now) <= -3.0:
            if self.state == "START":
                steer *= 0.7
            else:
                steer *= self.dec_scaler_for_steer

        return steer

    def speed_steer_scaling(self, steer, speed):
        """
        Steer scaling based on speed
        decrease steer when driving fast

        Returns:
            steer: scaled steering angle based on speed
        """
        speed_diff = max(0.1, self.end_scale_speed-self.start_scale_speed)  # to prevent division by zero
        factor = 1 - np.clip((speed - self.start_scale_speed)/(speed_diff), 0.0, 1.0) * self.downscale_factor
        steer *= factor
        return steer

    def steer_scaling_for_lat_err(self, steer, lateral_error):

        if self.start_mode:
            return steer

        # v3: coefficient parameterized (legacy hardcoded ln2 ~= 0.6931).
        # This term is positive feedback (error -> more steer -> overshoot);
        # 0.0 disables it, the optimizer decides how much survives.
        factor = np.exp(self.lat_err_steer_coeff * lateral_error)

        steer *= factor
        return steer

    def calc_future_lateral_error_norm(self):
        """
        Calculates future lateral error

        Returns:
           future lat_e_norm: normalization of the future lateral error
           future lateral_error: future distance from car's position to nearest waypoint
        """
        future_position = self.future_position[0, :2]
        idx_future_local_wpnts = self.nearest_waypoint(future_position, self.waypoint_array_in_map[:, :2])
        future_local_wpnts_d = abs(self.waypoint_array_in_map[idx_future_local_wpnts, 8])
        future_potision_s, future_position_d = self.converter.get_frenet([self.future_position[0, 0]], [self.future_position[0, 1]])
        future_position_d = abs(future_position_d[0])
        future_lat_err = future_position_d - future_local_wpnts_d

        max_lat_e = 1
        min_lat_e = 0.
        lat_e_clip = np.clip(future_lat_err, a_min=min_lat_e, a_max=max_lat_e)
        lat_e_norm = ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e))
        return lat_e_norm, future_lat_err

    def speed_adjust_lat_err(self, global_speed, lat_e_norm):
        """
        Reduce speed from the global_speed based on the lateral error
        and curvature of the track. lat_e_coeff scales the speed reduction:
        lat_e_coeff = 0: no account for lateral error
        lat_e_coaff = 1: maximum accounting

        Returns:
            global_speed: the speed we want to follow
        """
        # scaling down global speed with lateral error and curvature
        lat_e_coeff = self.lat_err_coeff  # must be in [0, 1]
        lat_e_norm *= self.speed_factor_for_lat_err
        curv = np.clip(2*(np.mean(self.curvature_waypoints)/0.8) - 2, a_min=0.0, a_max=1.0)  # 0.8 ca. max curvature mean
        curv *= self.speed_factor_for_curvature
        global_speed *= (1.0 - lat_e_coeff + lat_e_coeff*np.exp(-lat_e_norm*curv))
        return global_speed

    def speed_adjust_heading(self, speed_command):
        """
        Reduce speed from the global_speed based on the heading error.
        If the difference between the map heading and the actual heading
        is larger than 10 degrees, the speed gets scaled down linearly up to 0.5x

        Returns:
            global_speed: the speed we want to follow
        """

        heading = self.position_in_map[0, 2]
        map_heading = self.waypoint_array_in_map[self.idx_nearest_waypoint, 6]
        if abs(heading - map_heading) > np.pi:
            heading_error = 2*np.pi - abs(heading - map_heading)
        else:
            heading_error = abs(heading - map_heading)

        if heading_error < self.heading_error_thres*np.pi/180:  # 10 degrees error is okay
            return speed_command
        elif heading_error < np.pi/2:
            scaler = 1 - 0.5 * heading_error/(np.pi/2)
        else:
            scaler = 0.5
        return speed_command * scaler

    def compute_future_heading_correction(self, L1_vector, yaw, dt, speed,
                               alpha=0.1, v_threshold=15.0,
                               use_pid=True, use_filter=True):

        target_heading = np.arctan2(L1_vector[1], L1_vector[0])
        heading_error = target_heading - yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        if use_filter:
            if not hasattr(self, 'filtered_heading_error'):
                self.filtered_heading_error = heading_error
            self.filtered_heading_error = alpha * heading_error + (1 - alpha) * self.filtered_heading_error
            heading_error = self.filtered_heading_error

        if speed < v_threshold:
            dynamic_gain = self.KP * (speed / v_threshold)
        else:
            dynamic_gain = self.KP

        if self.state == "OVERTAKE":
            dynamic_gain *= 0.65

        if not hasattr(self, 'heading_error_integral'):
            self.heading_error_integral = 0.0
        if not hasattr(self, 'prev_heading_error'):
            self.prev_heading_error = heading_error

        if use_pid:
            self.heading_error_integral += heading_error * dt
            derivative = (heading_error - self.prev_heading_error) / dt if dt > 0 else 0.0
            self.prev_heading_error = heading_error

            correction = dynamic_gain * heading_error + self.KI * self.heading_error_integral + self.KD * derivative
        else:
            correction = dynamic_gain * heading_error

        return correction

    def calc_future_position(self, T):
        """
        Predicts the future vehicle state (position and heading) T seconds ahead
        based on the current vehicle state and updates self.position_in_map[0].

        Inputs:
            T: Prediction time (seconds), e.g., 0.25

        Assumes the following variables exist in self:
            self.position_in_map : 2D array with the first row containing [x, y, psi]
            self.speed_now       : Current vehicle speed (v)
            self.current_steer_command : Current steering input (delta)
            self.yaw_rate        : Current yaw rate from the IMU (rad/s)
            self.wheelbase       : Vehicle wheelbase (distance between front and rear axles)
        """

        x_current = self.position_in_map[0, 0]

        # Extract current state
        x_current = self.position_in_map[0, 0]
        y_current = self.position_in_map[0, 1]
        psi_current = self.position_in_map[0, 2]
        v = self.speed_now
        delta = self.current_steer_command  # Steering input

        # Vehicle geometry parameters.
        # Here, L_f and L_r are assumed to be 52% and 48% of the total wheelbase respectively.
        L_total = self.wheelbase
        L_f = 0.52 * L_total
        L_r = 0.48 * L_total

        # 1. Compute geometric slip angle (basic model)
        beta_model = np.arctan((L_r / (L_f + L_r)) * np.tan(delta))

        # 2. Estimate slip angle indirectly using IMU yaw rate data
        if abs(v) > 2.0:
            # If speed is sufficient, estimate slip angle from IMU yaw rate
            beta_imu = np.arcsin(np.clip(((L_f + L_r) * self.yaw_rate / v), -1.0, 1.0))
        else:
            beta_imu = beta_model  # Maintain basic model when speed is very low

        # 3. Fuse the geometric and IMU-based slip angles using weighted average
        lambda_weight = 1.0
        beta_fused = lambda_weight * beta_model + (1 - lambda_weight) * beta_imu

        # 4. Predict future position using the fused slip angle
        future_x = x_current + v * np.cos(psi_current + beta_fused) * T
        future_y = y_current + v * np.sin(psi_current + beta_fused) * T

        # 5. Predict future heading:
        # Option A: Model-based prediction
        future_psi_model = psi_current + (v / (L_f + L_r)) * np.sin(beta_fused) * T
        # Option B: IMU-based prediction
        future_psi_imu = psi_current + self.yaw_rate * T
        # Fuse the two heading predictions using a weighted average
        gamma_weight = 1.0
        future_psi = gamma_weight * future_psi_model + (1 - gamma_weight) * future_psi_imu
        # Normalize heading to the range [-pi, pi]
        future_psi = np.arctan2(np.sin(future_psi), np.cos(future_psi))

        # Update the global state: overwrite self.position_in_map[0] with the future state.

        future_position = np.zeros((1, 3))

        future_position[0, 0] = future_x
        future_position[0, 1] = future_y
        future_position[0, 2] = future_psi

        return future_position

    def nearest_waypoint(self, position, waypoints):
        """
        Calculates index of nearest waypoint to the car

        Returns:
            index of nearest waypoint to the car
        """
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position)

    def waypoint_at_distance_before_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Calculates the waypoint at a certain frenet distance in front of the car

        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """

        if distance is None:
            distance = self.t_clip_min
        d_distance = distance

        # Extract only waypoints ahead of current index
        waypoints_ahead = waypoints[idx_waypoint_behind_car:]

        # Compute segment-wise distances between waypoints
        deltas = np.diff(waypoints_ahead, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)

        # Compute cumulative distances
        cum_lengths = np.cumsum(seg_lengths)

        # Find the first index where cumulative distance exceeds lookahead
        idx_offset = min(np.searchsorted(cum_lengths, d_distance), len(waypoints_ahead) - 1)

        # v3: return the exact point at d_distance by interpolating on the
        # segment [w_j, w_j+1] instead of snapping to w_j - removes the eta
        # quantization jitter that short corner lookaheads amplify. The
        # result always lies ON the existing path polyline.
        j = idx_offset
        if j >= len(waypoints_ahead) - 1:
            return waypoints_ahead[-1]
        start = cum_lengths[j - 1] if j > 0 else 0.0
        seg = cum_lengths[j] - start
        frac = (d_distance - start) / seg if seg > 1e-9 else 1.0
        frac = min(max(frac, 0.0), 1.0)
        return waypoints_ahead[j] + frac * (waypoints_ahead[j + 1] - waypoints_ahead[j])
