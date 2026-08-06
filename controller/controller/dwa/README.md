# DWA direct controller

`DWAONLY` uses the Dynamic Window Approach described in
`png/08_Local_path_planning_-_practice_DWA.pdf`.  It samples reachable velocity
and steering commands, predicts each trajectory with a kinematic bicycle model,
rejects scan collisions with the full asymmetric vehicle footprint, then scores
the remaining candidates by heading, obstacle clearance, and velocity.

The planner uses `/scan` in `base_link` coordinates for collision checks and
uses `/global_waypoints` only to select a forward goal.  RViz topics are:

- `/dwa_planner/candidates`
- `/dwa_planner/selected`
- `/dwa_planner/goal`

If every sampled trajectory collides, the commanded speed and steering are both
zero.  The longest collision-free prefix is still shown on the selected marker
for diagnosis.
