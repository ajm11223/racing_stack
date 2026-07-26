#!/usr/bin/env python3
"""Progress of fast_tune studies: trial counts, rate, ETA, best so far.

    python progress.py <study> [<study> ...] [target_n_trials]
    python progress.py                       # every study in the journal

Reads the Optuna journal next to this file and does NOT import fast_tune, so it
starts in a couple of seconds (no gym) and never touches the running workers.
The journal is large, so it is opened ONCE and every requested study is read
from that single handle - much faster than invoking this script per study.
"""
import collections
import datetime as dt
import os
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "journal_fast.log")

args = sys.argv[1:]
target = None
if args and args[-1].isdigit():          # trailing number = target trial count
    target = int(args[-1])
    args = args[:-1]

if not os.path.isfile(JOURNAL):
    sys.exit(f"no journal at {JOURNAL} - has a run started yet?")

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = JournalStorage(JournalFileBackend(JOURNAL))
known = sorted(s.study_name for s in
               optuna.get_all_study_summaries(storage=storage))

names = args or known
missing = [n for n in names if n not in known]
if missing:
    sys.exit(f"unknown study/studies: {', '.join(missing)}\n"
             f"known studies: {', '.join(known)}")

# which studies currently have workers on them (marks the live ones)
try:
    import subprocess
    ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True,
                        timeout=5).stdout
    live = {n for n in names if f"--study {n}" in ps}
except Exception:
    live = set()

WINDOW = 200          # trials used for the recent-throughput estimate


def recent_rate(done):
    """Trials/min over the last WINDOW completions.

    NOT total/wall-clock-since-creation: a study that was paused (workers
    stopped and restarted later) keeps accumulating wall time with no trials,
    which drags that number toward zero and makes a healthy run look dead.
    Measuring across recent completions only is immune to the gaps.
    """
    ts = sorted(t.datetime_complete for t in done if t.datetime_complete)
    if len(ts) < 2:
        return 0.0, None
    win = ts[-WINDOW:]
    span = (win[-1] - win[0]).total_seconds() / 60
    return ((len(win) - 1) / span if span > 0 else 0.0), ts[-1]


rows = []
for name in names:
    st = optuna.load_study(study_name=name, storage=storage)
    trials = st.trials
    if not trials:
        rows.append((name, None))
        continue
    done = [t for t in trials if t.value is not None]
    ran = [t for t in done if t.user_attrs.get("lap_times")]
    fails = collections.Counter(t.user_attrs["fail"].split("|")[0]
                                for t in trials if t.user_attrs.get("fail"))
    best = min(ran, key=lambda t: t.value) if ran else None
    rate, last = recent_rate(done)
    # RUNNING trials are only real workers if a process is actually attached;
    # otherwise they are stale rows left behind by killed workers.
    running = sum(1 for t in trials if t.state.name == "RUNNING")
    idle_min = ((dt.datetime.now() - last).total_seconds() / 60
                if last else None)
    rows.append((name, dict(
        n=len(trials), done=len(done), ran=len(ran), running=running,
        stale=(name not in live), rate=rate, idle=idle_min,
        fails=fails, best=best)))

W = max(len(n) for n in names) + 3
hdr = (f"{'study':<{W}} {'trials':>7} {'ran':>6} {'fail%':>6} {'wrk':>5} "
       f"{'/min':>6} {'best':>8} {'lap':>6} {'|d|':>7} {'weave':>8}")
print(hdr)
print("-" * len(hdr))
for name, r in rows:
    tag = ("* " if name in live else "  ") + name
    if r is None:
        print(f"{tag:<{W}} {'(no trials yet)':>7}")
        continue
    failpct = 100 * sum(r["fails"].values()) / r["n"] if r["n"] else 0.0
    b = r["best"]
    if b:
        laps = b.user_attrs["lap_times"]
        cells = (f"{b.value:8.3f} {sum(laps)/len(laps):6.2f} "
                 f"{b.user_attrs.get('mean_abs_d', float('nan')):7.4f} "
                 f"{b.user_attrs.get('d_weave_rms', float('nan')):8.5f}")
    else:
        cells = f"{'-':>8} {'-':>6} {'-':>7} {'-':>8}"
    # a stopped study has no workers; its RUNNING rows are stale leftovers
    wrk = f"{r['running']:d}" if not r["stale"] else f"({r['running']})"
    rate = f"{r['rate']:6.1f}" if not r["stale"] else f"{'idle':>6}"
    print(f"{tag:<{W}} {r['n']:7d} {r['ran']:6d} {failpct:5.0f}% "
          f"{wrk:>5} {rate} {cells}")

print()
for name, r in rows:
    if r is None:
        continue
    if r["stale"]:
        idle = (f"idle {r['idle']:.0f} min" if r["idle"] is not None
                else "idle")
        stale = (f", {r['running']} stale RUNNING rows"
                 if r["running"] else "")
        line = f"{name}: {r['done']} complete, STOPPED ({idle}){stale}"
    else:
        line = f"{name}: {r['done']} complete, {r['rate']:.1f}/min"
        if target and r["rate"]:
            left = max(target - r["done"], 0) / r["rate"] / 60
            line += f", ~{left:.1f} h to {target}"
    if r["best"]:
        line += f", best #{r['best'].number}"
    if r["fails"]:
        line += f"  fails={dict(r['fails'])}"
    print("  " + line)
print(f"\n  * = workers attached   (n) = stale RUNNING rows from killed workers"
      f"   /min = over the last {WINDOW} completions, pauses excluded")
