#!/usr/bin/env bash
# overnight.sh — unattended sim + Optuna tuning run.
#
# Usage (or just use the `overnight` shortcut from ~/.bashrc):
#     T=~/unicorn_ws/src/unicorn-racing-stack/tuning
#     nohup bash $T/overnight.sh [n_trials] > $T/overnight.log 2>&1 &
#
# Deliberately NON-interactive: `bash -ic ...` backgrounded from a terminal
# gets SIGTTIN/SIGTTOU from the controlling tty and freezes in T state
# (that is exactly what happened on 2026-07-22). Plain `bash` + inherited
# env + sourcing unicorn.sh directly has no tty interaction at all.
# no `set -u`: conda/RoboStack activate.d scripts reference unbound vars
N_TRIALS="${1:-400}"
# derive both from this script's own location (<stack>/tuning/overnight.sh), so
# the run survives the repo being cloned anywhere
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK="$(dirname "$LOGDIR")"

# single-instance lock: a second `overnight` while one is running must NOT
# start - on 2026-07-22 a double invocation killed the first run's sim
# mid-trial and the two stacks shredded each other. Refuse instead.
exec 9>"$LOGDIR/.overnight.lock"
if ! flock -n 9; then
    echo "[overnight] already running - watch it: tail -f $LOGDIR/tune.log"
    echo "[overnight] to stop it and start fresh: overnight-stop, then overnight"
    exit 1
fi

echo "[overnight] entering unicorn env..."
source "$STACK/unicorn.sh"

# this script OWNS the stack: kill any already-running sim/tuner first.
# Two gym instances mixing odom makes every consumer go insane (lap_analyser
# "0.05s laps", double lap counters) - overnight must start from a clean slate.
if pgrep -f "ros2 launch stack_master" >/dev/null 2>&1 || \
   pgrep -f "unicorn_ws/install" >/dev/null 2>&1; then
    echo "[overnight] found a running stack - stopping it first..."
    pkill -f "tune_controller.py" 2>/dev/null
    pkill -f "ros2 launch stack_master" 2>/dev/null
    sleep 3
    pkill -9 -f "unicorn_ws/install" 2>/dev/null
    sleep 2
fi

# a stale ros2 daemon started under another RMW answers `ros2 node list` with
# nothing -> readiness check below would never pass. Fresh daemon, fresh RMW.
ros2 daemon stop >/dev/null 2>&1

echo "[overnight] starting sim (log: $LOGDIR/sim.log)..."
setsid ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true \
    > "$LOGDIR/sim.log" 2>&1 &
SIM_PID=$!

# wait until the stack is actually up instead of a blind sleep: readiness =
# frenet odometry flowing (daemon-independent, unlike `ros2 node list`)
up=""
for _ in $(seq 1 30); do
    if timeout 5 ros2 topic echo /car_state/odom_frenet --once >/dev/null 2>&1; then
        up=1; break
    fi
done
if [ -z "$up" ]; then
    echo "[overnight] sim never came up - check $LOGDIR/sim.log"; exit 1
fi
sleep 10   # lazy init (waypoints -> controller) settle
echo "[overnight] sim up (pid $SIM_PID). starting tuner: $N_TRIALS trials..."

# sleep inhibitor scoped to exactly the tuning process
cd "$LOGDIR"
systemd-inhibit --what=sleep:idle --why="unicorn overnight tuning" \
    python tune_controller.py --n-trials "$N_TRIALS" > "$LOGDIR/tune.log" 2>&1
RC=$?

echo "[overnight] tuner finished (exit $RC). stopping sim..."
kill -- -"$SIM_PID" 2>/dev/null || kill "$SIM_PID" 2>/dev/null
echo "[overnight] done. results: python tune_controller.py --show-best"
