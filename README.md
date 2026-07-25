# UNICORN Racing Stack — ROS 2 Jazzy

A full **F1TENTH autonomous racing stack** on ROS 2 Jazzy — perception → tracking
→ prediction → planning → state machine → control — with an in-repo
**f1tenth_gym** simulator for software-in-the-loop testing. Self-contained: clone,
build, run.

> The ROS 1 (catkin) stack lives on the upstream repo's **`ros1`** branch
> (frozen, for reference) — see [Credits](#credits).

RoboStack (conda) makes the build OS- and arch-agnostic. Verified platforms:

|  | Ubuntu x86_64 | Ubuntu arm64 | macOS arm64 | Windows |
|---|:---:|:---:|:---:|:---:|
| **Status** | ✅ verified | ✅ verified | 🔺 partial | ⬜ untested |
| **Hardware** | NUC, desktop | Jetson (Orin) | Mac mini, MacBook | conda |

**Install support:** the **conda (RoboStack)** path below is the only tested and
supported one. **System ROS 2 Jazzy (apt/rosdep)** and **Docker** are planned —
not yet officially supported.

## Get started

**RoboStack (conda) is the default and verified path** — ROS 2 Jazzy + every
dependency into one conda env (`unicorn`), on **Linux and macOS**, without
touching system ROS. Copy-paste the three blocks below top to bottom.

### 1. conda — skip if you already have it

If `conda` (or `mamba`) is already on your PATH, **skip this step**. Otherwise
install **Miniforge** (recommended — minimal, `conda-forge` by default, no
license terms):

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" && exec $SHELL
conda config --set auto_activate_base false
```

### 2. clone

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone https://github.com/ajm11223/racing_stack.git unicorn-racing-stack
cd unicorn-racing-stack
```

<details><summary>workspace layout</summary>

The colcon **workspace root** is `~/unicorn_ws` (the dir that holds `src/`); this
repo lives at `~/unicorn_ws/src/unicorn-racing-stack`. Everything is vendored in
— no submodules, so a plain `git clone` is enough.
</details>

### 3. install

One script does everything — creates the `unicorn` env, registers the `unicorn`
alias in `~/.bashrc` and `~/.zshrc`, installs the pip layer, fixes `quadprog`,
raises OS socket buffers for CycloneDDS, and builds (Release). Runs from bash or zsh:

```bash
./setup_conda_onLaptop.sh   # sim / laptop: skips hardware-only nodes
# ./setup_conda_onCar.sh    # the car: full build (everything)
```

The laptop build skips only the **hardware-only** packages (`urg_node`,
`vesc_driver`/`vesc_ackermann`, `particle_filter`) and then runs the car script. It
**keeps** `vesc_msgs` (so you can `ros2 topic echo` VESC messages) and the full
cartographer stack incl. `cartographer_rviz` (to view submaps in RViz). Running
`setup_conda_onCar.sh` directly clears those ignores and builds everything.

<details><summary>prefer to run the steps yourself? (same thing, in order)</summary>

```bash
conda env create -f environment.yml                                              # conda layer: ROS 2 Jazzy + deps
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc                      # alias (add to ~/.zshrc too for zsh)
source unicorn.sh                                                                 # enter the env now
pip install -r requirements.txt                                                   # pip layer
pip install -e ./race_utils/unicorn_gym/f1tenth_gym                               # gym core -> f110_gym
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper   # range_libc
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13      # quadprog swap (must be LAST)
cbuild                                                                            # colcon build (Release)
```
</details>

## Quick simulation start

After the install script, open a **new shell** (or `source ~/.bashrc` / `~/.zshrc`), then:

```bash
unicorn   # enter the env: conda + PYTHONNOUSERSITE=1 + CycloneDDS + workspace, all sourced
ros2 launch stack_master race.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

**Then press `a` to arm autonomous driving** — until you do, the car sits still:
the controller publishes, but `simple_mux` forwards nothing while `keyboard_joy`
streams its startup *human* mode. The key hook is global (pynput — any focused
window counts, and typing normally can trigger it):

| key | effect |
|---|---|
| `a` | **auto** — controller drives the car |
| `h` | **human** — manual mode (car stops unless you steer it) |
| arrows / space | manual drive / zero command (human mode) |

So a still car after a clean launch usually just means human mode — check
`ros2 topic echo /joy_keyboard --once` (`buttons[5]=1` is auto) before hunting
deeper. And a stray `h` typed in any window mid-run disarms the car.

`unicorn` also defines helpers: `cbuild [pkgs...]` (colcon build Release + re-source;
no args = whole workspace) and `ros2kill` (kill every ROS 2 node / launcher / daemon).

<details><summary>What <code>unicorn.sh</code> sets — and why you always enter with it</summary>

Always enter with `unicorn` (which **sources** `unicorn.sh`), never a bare
`conda activate`. It works in bash and zsh, and:
- sets **`PYTHONNOUSERSITE=1`** so a stale `~/.local/lib/python*` can't shadow
  the env;
- selects **CycloneDDS** — `RMW_IMPLEMENTATION` must be set *after* `conda
  activate` (which clears it). The default FastDDS busy-spins a core on this
  many-node graph (~22 Hz sim); CycloneDDS idles at ~21% CPU and hits the full
  ~80 Hz. It points `CYCLONEDDS_URI` at the repo's `cyclonedds.xml` (loopback +
  a Wi-Fi interface, multicast on). Adjust `ROS_DOMAIN_ID` (default `1`);
- **resets the ROS env to a clean baseline** so a system ROS / other workspace
  `source`d in your rc file (any distro/path) can't shadow the conda env. Do
  **not** globally `source /opt/ros/<distro>/setup.*` in your rc file;
- sources the colcon workspace and defines `cbuild` / `ros2kill`.

For a fully isolated env immune to any host rc file, use the container
(`.devcontainer` / `.docker`).
</details>

## System ROS 2 (apt / rosdep) — not yet tested

<details><summary>Path B — system ROS 2 Jazzy on Ubuntu 24.04 (unverified)</summary>

> ⚠️ The conda path above is the verified one. This is documented for
> completeness; expect to fix gaps.

```bash
# B1  install ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation.html), then:
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip

# B2  rosdep (ROS + apt deps from every package.xml)
sudo rosdep init    # first time only; ignore "already exists"
rosdep update
rosdep install --from-paths src/unicorn-racing-stack --ignore-src -r -y

# B3  python layer (same requirements.txt as the conda path)
pip install --user -r src/unicorn-racing-stack/requirements.txt
pip install --user -e src/unicorn-racing-stack/race_utils/unicorn_gym/f1tenth_gym

# B4  build
colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Ubuntu 24.04 usually does **not** need the conda path's `setuptools`/`asio` pins
(the distro ships compatible versions). Source with
`source /opt/ros/jazzy/setup.bash && source install/setup.bash`.
</details>

<details><summary>Manual conda bootstrap (without <code>environment.yml</code> / <code>setup_conda_onCar.sh</code>)</summary>

```bash
conda create -n unicorn -c conda-forge -c robostack-jazzy ros-jazzy-desktop -y
conda activate unicorn
conda config --env --add channels conda-forge
conda config --env --add channels robostack-jazzy
conda config --env --set channel_priority strict

conda install -c conda-forge -c robostack-jazzy -y \
  compilers cmake pkg-config make ninja colcon-common-extensions rosdep
conda install -c conda-forge -c robostack-jazzy -y \
  ros-jazzy-ackermann-msgs ros-jazzy-asio-cmake-module ros-jazzy-diagnostic-updater \
  ros-jazzy-foxglove-bridge ros-jazzy-io-context ros-jazzy-nav2-lifecycle-manager \
  ros-jazzy-nav2-map-server ros-jazzy-nav2-msgs ros-jazzy-robot-localization \
  ros-jazzy-rosbag2-storage-mcap ros-jazzy-serial-driver ros-jazzy-tf-transformations \
  ros-jazzy-xacro ros-jazzy-teleop-twist-keyboard ros-jazzy-cartographer-ros \
  ros-jazzy-joint-state-publisher transforms3d opencv matplotlib-base

pip install -r requirements.txt
pip install -e race_utils/unicorn_gym/f1tenth_gym
conda install -c conda-forge -y "setuptools<80" "asio=1.29.0"   # see pin notes above
```
</details>

<details><summary>Hardware-driver build notes (real car only — not needed for sim)</summary>

The **simulation stack builds cleanly** via `setup_conda_onLaptop.sh` (it marks
these trees `COLCON_IGNORE` for you). Only the real-car drivers need care:

- **`vesc_driver`** → `transport_drivers` use `asio::io_service` (removed in asio
  ≥1.30). Fixed by the `asio=1.29.0` pin in `environment.yml`.
- **`urg_node`** (Hokuyo LiDAR) → its siblings (`urg_c`, `laser_proc`,
  `urg_node_msgs`) sit nested inside `urg_node/`; colcon won't descend. Move them up:
  ```bash
  cd src/unicorn-racing-stack/sensor/urg_node && mv urg_c laser_proc urg_node_msgs ../
  ```

For **simulation only**, just ignore the hardware trees:
```bash
touch src/unicorn-racing-stack/sensor/urg_node/COLCON_IGNORE
touch src/unicorn-racing-stack/sensor/vesc/COLCON_IGNORE
touch src/unicorn-racing-stack/stack_master/COLCON_IGNORE
```
</details>

## Controller parameter tuning (Optuna)

Two tuners live in [`tuning/`](tuning). Use them in this order:

1. **`fast_tune.py` — headless mass-lap search.** No ROS graph, no wall clock: the
   f1tenth_gym physics engine is stepped as fast as the CPU allows and the
   controller is called directly at 50 Hz. Thousands of trials per hour. This is
   where the search happens.
2. **`validate_best.py` — real-stack check.** Applies a saved best on the full
   running stack and measures the same metrics. **Always validate before
   adopting** — the headless model is not the full stack (see Caveats).
3. **`tune_controller.py` — closed-loop tuner** against a running sim. Highest
   fidelity, far slower. Optional.

### Files

| path | what it is |
|---|---|
| `tuning/tuner_config.yaml` | **the one file you edit**: search space, frozen values, objective weights |
| `tuning/fast_tune.py` | headless search (main entry point) |
| `tuning/progress.py` | trial count / rate / ETA / best so far of a running study |
| `tuning/validate_best.py` | replay a saved best on the real stack |
| `tuning/tune_controller.py` | closed-loop tuner (needs a running sim) |
| `tuning/overnight.sh` | unattended sim + closed-loop tuner run |
| `tuning/best_params_<study>.yaml` | best trial of a study, written by `--show-best` |
| `tuning/journal_fast.log` | Optuna storage, created on first run (gitignored, grows to 100s of MB) |
| `stack_master/config/controller.yaml` | Pure-Pursuit params — target of `--ctrl pp` |
| `stack_master/config/controller_map.yaml` | MAP params — target of `--ctrl map` |
| `stack_master/maps/<map>/global_waypoints.json` | raceline + speed profile the trials drive |

### 1. Enter the environment

```bash
cd ~/unicorn_ws                                  # the dir that holds src/
source src/unicorn-racing-stack/unicorn.sh
cd src/unicorn-racing-stack/tuning
```

conda (RoboStack) only — do **not** `source /opt/ros/jazzy/setup.bash` first; a
system ROS on the path makes topics work while param services time out.

The scripts locate the stack from their own location, so a clone at any path
works. Overrides:

```bash
export UNICORN_STACK=/path/to/unicorn-racing-stack   # only if the tree moved
export UNICORN_MAP=s                                 # map to tune on (default: s)
```

### 2. Smoke test — always run this first

```bash
python fast_tune.py --ctrl map --smoke
```
```
smoke (map yaml params, scaling 1.0):
  laps ['9.02', '9.02', '9.02', '9.02']  |d| 0.084  weave 0.0064  osc 0.0194
  wall time 2.5s for 3 laps -> 1.2 laps/s/core
```

This evaluates the **current yaml params** once. If it fails or crashes, fix that
before searching. `laps/s/core` × workers is your throughput budget.

### 3. Set up the search — `tuner_config.yaml`

```yaml
params:                     # enabled: true  -> searched within [low, high]
  m_l1:       {enabled: true, low: 0.35, high: 1.00}
  t_clip_min: {enabled: true, low: 0.60, high: 2.50}
frozen:                     # applied every trial, never searched
  "/speed_sector_tuner:Sector0.scaling": 1.0
  KI: 0.0
objective:
  w_lat_err: 4.0            # cost per m of mean |d|
  d_hard_limit: 0.10        # |d| budget; beyond it w_d_over kicks in
  w_osc: 64.0               # per rad RMS of per-cycle steering delta
  w_weave: 2000.0           # per m RMS of signed d increments (body sway)
  w_osc_worst: 40.0         # same, but on the worst ~1 s window (per-corner)
  w_weave_worst: 600.0
  fail_penalty: 120.0       # crash / stuck / offtrack / timeout
```

`cost = lap + w_lat_err·|d| + w_d_over·over + w_osc·osc + w_weave·weave +
w_osc_worst·osc_worst + w_weave_worst·weave_worst` — lower is better.

Current state of the MAP search (17 dims): `Sector0.scaling` frozen at 1.0
(geometry-only tuning), the steer-downscale band `start/end_scale_speed` frozen
at 6.5 / 10.0, and `curvature_factor` excluded for MAP because `l1_chord_err`
already shortens L1 on curvature (`PP_ONLY_DIMS` in `fast_tune.py`).

### 4. Run the search

```bash
STUDY=my_v1_map                       # new name for every new search space
nohup python fast_tune.py --ctrl map --study $STUDY \
      --n-trials 20000 --workers $(( $(nproc) - 2 )) \
      > fast_$STUDY.log 2>&1 &
echo "pid $!"
```

`--n-trials` is the **total**, split across workers (`n_trials // workers` each).
Leave 2 cores free. On a 16-core laptop expect ~70–90 trials/min.

### 5. Watch it

```bash
pgrep -c -f 'python.*fast_tune\.py'            # parent + workers
python progress.py $STUDY 20000                # counts, rate, ETA, best so far
```
```
study my_v1_map
  trials 1101  {'COMPLETE': 1089, 'RUNNING': 12}
  complete 1089 | elapsed 15.7 min | 70 trials/min
  ~11.7 h left to reach 50000
  best #770 cost 21.553  lap 9.180s  |d| 0.0318  osc 0.0147  weave 0.00354
  failed 156 (14%): {'collision': 126, 'offtrack': 30}
```

A 10–30 % failure rate early on is normal (TPE exploring). Rank differently
without touching anything:

```bash
python fast_tune.py --ctrl map --study $STUDY --show-best --no-apply
python fast_tune.py --ctrl map --study $STUDY --show-best --no-apply --max-d 0.05 --by lap
python fast_tune.py --ctrl map --study $STUDY --plot-best        # re-drive + graph
```

Stop / pause / resume (the journal persists, so a run resumes where it stopped):

```bash
pkill       -f 'python.*fast_tune\.py'   # stop
pkill -STOP -f 'python.*fast_tune\.py'   # pause (frees the CPU)
pkill -CONT -f 'python.*fast_tune\.py'   # resume
```

### 6. Adopt a result

```bash
# 1) save the snapshot and WRITE it into controller_map.yaml
python fast_tune.py --ctrl map --study $STUDY --show-best

# 2) pause the workers - the real sim needs the CPU
pkill -STOP -f 'python.*fast_tune\.py'

# 3) terminal 2: bring the stack up, then press 'a' to arm autonomous
#    (see Quick simulation start -- a still car = human mode, not a bug)
ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true

# 4) terminal 1: same metrics, full stack
python validate_best.py --params best_params_$STUDY.yaml --laps 2

# 5) back to searching
pkill -CONT -f 'python.*fast_tune\.py'
```

Live-tune single params on the running stack, then persist them:

```bash
ros2 param set /controller_manager m_l1 0.42
ros2 param set /controller_manager save_params true     # -> controller_map.yaml
```

### 7. Closed-loop tuner (optional)

`tune_controller.py` drives an already-running stack. Its study lives in SQLite
next to the scripts (`tuning/optuna_stage1.db`, gitignored):

```bash
ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true   # terminal 2
python tune_controller.py --n-trials 60                                  # terminal 1
```

`overnight.sh` (unattended sim + tuner, single-instance flock) wraps both and
derives its paths from its own location. Neither needs editing for a clone at
a different path.

### Caveats — read before trusting a result

- **`--show-best` overwrites `controller.yaml` / `controller_map.yaml`.** Pass
  `--no-apply` to only print and snapshot. It writes every param of the winning
  trial plus the `frozen` values.
- **Start a new `--study` name whenever the problem changes** — search space,
  objective weights, raceline, `ggv.csv`, `dynamics.yaml`. Old trials were scored
  under the old physics and would win the ranking on a stale basis. `fast_tune`
  reads the raceline once at startup.
- **Headless ≠ full stack.** Physics runs at 100 Hz (bridge: 80), there is no
  state_machine / local-planner hop, and actuation latency is a 30 ms model
  (`LATENCY_S`). Validate the top few in the real sim before adopting.
- **Steering lookup mismatch.** The headless MAP search uses the car's measured
  table (`MAP_LU_TABLE = UNICORN2-0410`), while `race.launch.xml` forces
  `SIM_linear` whenever `sim:=true` — so step 6 validates against a *different*
  steering response. Change the `<let name="lu_table" ...>` line if you need them
  matched (a CLI `lu_table:=` is overridden by that `let`).
- **CPU contention invalidates real-sim measurements.** Always `SIGSTOP` the
  workers before `validate_best.py` / `tune_controller.py`.
- **`journal_fast.log` grows to hundreds of MB** and every worker replays it at
  startup. Archive it between campaigns:
  `mv journal_fast.log journal_fast.log.$(date +%F)` (old studies become
  unreadable afterwards, but the `best_params_*.yaml` snapshots remain).
- **Some params cannot be scored by the solo headless run** — leaving them
  `enabled` only adds random-walk dimensions: `trailing_*` (needs an opponent and
  a gap-error term), `start_speed` / `speed_diff_thres` / `start_curvature_factor`
  (START state; `fast_tune` always drives `GB_TRACK`), `ftg_*` (FTG state).
- **Never search `AEB_thres`** (safety) **or `KI`** (unbounded integrator →
  windup). `steer_gain_for_speed` is a no-op at 1.0 and becomes a blanket steer
  multiplier above it, which fights the MAP feedforward calibration.
- **`Sector0.scaling` is the only knob that raises the lap-time ceiling.** It is
  frozen at 1.0 so the geometry is tuned against a fixed speed profile. If you
  raise it, re-tune: the steer-downscale band and `downscale_factor` gain real
  authority at higher speeds.
- **Never start two sims / two `overnight.sh` at once.** Two gym instances mix
  odom and every consumer goes insane (0.05 s laps, double lap counters).

## Credits

This stack is a snapshot of work built on top of the
[HMCL-UNIST `unicorn-racing-stack`](https://github.com/hmcl-unist/unicorn-racing-stack)
(ROS 2 Jazzy port and racing pipeline), which itself derives from the
[ForzaETH `race_stack`](https://github.com/ForzaETH/race_stack) (ETH Zürich).
Vendored components originate from their own upstreams:

- `race_utils/raycaster` — [jeongsang-ryu/2D-RayCaster](https://github.com/jeongsang-ryu/2D-RayCaster)
- `race_utils/unicorn_gym` — [jeongsang-ryu/unicorn-gym](https://github.com/jeongsang-ryu/unicorn-gym)
- `state_estimation/kiss_icp_localization` — [freshleesh/kiss_icp_localization](https://github.com/freshleesh/kiss_icp_localization)

This repository is published as a flattened single-commit snapshot; the full
authorship history lives in the upstream repositories above.
