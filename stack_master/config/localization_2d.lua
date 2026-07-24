-- localization_2d.lua — pure localization on a frozen map (.pbstream).
--
-- Includes mapping_2d.lua as the base configuration and overrides only what
-- differs for localization. Anything not listed here behaves exactly as in
-- mapping (sensor topics, IMU usage, online correlative scan matching, ...).

include "mapping_2d.lua"

-- ===== cartographer_ros options: overrides vs mapping =====
options.pose_publish_period_sec = 1e-2        -- mapping: 5e-3
options.trajectory_publish_period_sec = 30e-3 -- mapping: 1e-1

-- ===== performance =====
MAP_BUILDER.num_background_threads = 6        -- mapping: 4

-- ===== front-end: shorter range, cheaper matching on a known track =====
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 0.1 -- mapping: default 40

-- Match on half a scan instead of a full one: correction period 25 ms -> 12.5 ms
-- (UST 40 Hz, 10 subdivisions/scan). At 7 m/s this also halves the per-match
-- displacement (~0.17 m -> ~0.09 m), keeping it inside the online correlative
-- matcher's 0.1 m linear search window. Costs CPU; watch match scores.
-- TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 5 -- mapping: 10 (full scan)

-- ===== back-end: pure localization =====
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}
POSE_GRAPH.optimize_every_n_nodes = 3         -- mapping: 100; should be low for localization

POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 2.  -- default: 7
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(30.)
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3

return options
