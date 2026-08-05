#!/usr/bin/env python3
"""Show a map's raceline with s (arc length) annotated, to pick an s window.

The controller's per-map knobs are addressed by s (ot_sectors in
controller_map.yaml, the sector yamls' start/end), but nothing in the stack tells
you which s a given corner actually is. This draws the raceline with s ticks and
prints the same numbers, so a window can be read off a corner instead of guessed.

Usage (no build needed, run by path):
    python3 raceline_s_map.py m                    # plot + table
    python3 raceline_s_map.py m --zone 16 24       # highlight a candidate window
    python3 raceline_s_map.py m --step 1.0         # denser s labels
    python3 raceline_s_map.py m --out /tmp/m.png   # save instead of show
"""
import argparse
import json
import os

import matplotlib
import numpy as np

STACK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(map_name, key="global_traj_wpnts_iqp"):
    path = os.path.join(STACK, "stack_master", "maps", map_name, "global_waypoints.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no global_waypoints.json for map '{map_name}' ({path})")
    with open(path) as stream:
        wpnts = json.load(stream)[key]["wpnts"]
    return np.array([[w["s_m"], w["x_m"], w["y_m"], w["kappa_radpm"], w["vx_mps"],
                      w["d_left"], w["d_right"]] for w in wpnts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--zone", nargs=2, type=float, metavar=("S_START", "S_END"),
                    help="highlight this s window (e.g. a candidate ot_zone)")
    ap.add_argument("--step", type=float, default=2.0, help="s label spacing [m]")
    ap.add_argument("--out", help="save PNG here instead of opening a window")
    args = ap.parse_args()

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = load(args.map)
    s, x, y, kappa, vx, dl, dr = a.T
    track_len = float(s[-1])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [3, 1]})

    ax.plot(x, y, "-", color="0.75", lw=1, zorder=1)
    sc = ax.scatter(x, y, c=vx, cmap="viridis", s=12, zorder=2)
    fig.colorbar(sc, ax=ax, label="raceline speed [m/s]", shrink=0.7)

    if args.zone:
        z0, z1 = args.zone
        # A window is [z0, z1] in s; wrap it the same way the controller does not
        # (single window, no wrap) so what is drawn is what would fire.
        mask = (s >= z0) & (s <= z1)
        ax.plot(x[mask], y[mask], "-", color="crimson", lw=4, alpha=0.6, zorder=3,
                label=f"zone s {z0:g}–{z1:g} m ({mask.sum()} wpnts, "
                      f"{(z1 - z0):g} m, |kappa| max {np.abs(kappa[mask]).max():.2f})")
        ax.legend(loc="best", fontsize=9)

    for s_tick in np.arange(0.0, track_len, args.step):
        i = int(np.argmin(np.abs(s - s_tick)))
        ax.plot(x[i], y[i], "k.", ms=4, zorder=4)
        ax.annotate(f"{s[i]:.0f}", (x[i], y[i]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, zorder=5)

    ax.plot(x[0], y[0], "r*", ms=14, zorder=6)
    ax.annotate("s=0", (x[0], y[0]), textcoords="offset points", xytext=(6, -12),
                color="crimson", fontsize=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_title(f"map '{args.map}'  raceline  length {track_len:.2f} m  "
                 f"{len(s)} wpnts  (labels every {args.step:g} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)

    ax2.plot(s, kappa, label="kappa [1/m]")
    ax2.plot(s, vx / 10.0, label="speed/10 [m/s]", alpha=0.7)
    ax2.plot(s, dl + dr, label="track width [m]", alpha=0.7)
    if args.zone:
        ax2.axvspan(args.zone[0], args.zone[1], color="crimson", alpha=0.15)
    ax2.set_xlabel("s [m]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    print(f"map '{args.map}': length {track_len:.2f} m, {len(s)} waypoints, "
          f"spacing {track_len / len(s):.3f} m")
    print(f"{'s[m]':>6} {'x':>7} {'y':>7} {'kappa':>7} {'v':>5} {'width':>6}")
    for s_tick in np.arange(0.0, track_len, args.step):
        i = int(np.argmin(np.abs(s - s_tick)))
        mark = ""
        # the nearest waypoint to a tick can sit a few cm short of the bound
        if args.zone and args.zone[0] - 0.05 <= s[i] <= args.zone[1] + 0.05:
            mark = "  <- zone"
        print(f"{s[i]:6.1f} {x[i]:7.2f} {y[i]:7.2f} {kappa[i]:+7.2f} "
              f"{vx[i]:5.1f} {dl[i] + dr[i]:6.2f}{mark}")

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
