#!/usr/bin/env python3
"""Progress of a fast_tune study: trial counts, rate, ETA, best so far.

    python progress.py <study_name> [target_n_trials]

Reads the Optuna journal next to this file and does NOT import fast_tune, so it
starts in a couple of seconds (no gym) and never touches the running workers.
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

if len(sys.argv) < 2:
    sys.exit(f"usage: python {os.path.basename(__file__)} "
             "<study_name> [target_n_trials]")
study_name = sys.argv[1]
target = int(sys.argv[2]) if len(sys.argv) > 2 else None

if not os.path.isfile(JOURNAL):
    sys.exit(f"no journal at {JOURNAL} - has a run started yet?")

optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    st = optuna.load_study(study_name=study_name,
                           storage=JournalStorage(JournalFileBackend(JOURNAL)))
except KeyError:
    names = {s.study_name for s in optuna.get_all_study_summaries(
        storage=JournalStorage(JournalFileBackend(JOURNAL)))}
    sys.exit(f"study '{study_name}' not in {JOURNAL}\nknown studies: "
             + ", ".join(sorted(names)))

trials = st.trials
if not trials:
    sys.exit(f"study {study_name}: no trials yet")

counts = dict(collections.Counter(t.state.name for t in trials))
done = [t for t in trials if t.value is not None]
starts = [t.datetime_start for t in trials if t.datetime_start]
elapsed_min = ((dt.datetime.now() - min(starts)).total_seconds() / 60
               if starts else 0.0)
rate = len(done) / elapsed_min if elapsed_min else 0.0

print(f"study {study_name}")
print(f"  trials {len(trials)}  {counts}")
print(f"  complete {len(done)} | elapsed {elapsed_min:.1f} min | "
      f"{rate:.0f} trials/min")
if target and rate:
    print(f"  ~{max(target - len(done), 0) / rate / 60:.1f} h left to reach "
          f"{target}")

best = st.best_trial
laps = best.user_attrs.get("lap_times")
line = f"  best #{best.number} cost {best.value:.3f}"
if laps:
    line += (f"  lap {sum(laps) / len(laps):.3f}s"
             f"  |d| {best.user_attrs.get('mean_abs_d', float('nan')):.4f}"
             f"  osc {best.user_attrs.get('dsteer_rms', float('nan')):.4f}"
             f"  weave {best.user_attrs.get('d_weave_rms', float('nan')):.5f}")
print(line)

fails = collections.Counter(t.user_attrs["fail"].split("|")[0]
                            for t in trials if t.user_attrs.get("fail"))
if fails:
    total = sum(fails.values())
    print(f"  failed {total} ({100 * total / len(trials):.0f}%): {dict(fails)}")
