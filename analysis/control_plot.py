"""Tilt and commanded joint targets for a single drop, clocked from release.

Top panel: front- and rear-body tilt from vertical over the drop.
Bottom panel: the joint targets the policy commanded (roll, pitch, tail),
shared x-axis, so a change in tilt can be read against the command that caused it.

The x-axis is time since release, and each tick is annotated with the free-fall
distance at that instant, d = g t^2 / 2 -- the drop height a run of that length
needs. Release is t[0]: controller.py starts logging once the front accelerometer
reads free fall, so the first sample is the drop, up to one 50 Hz tick of
detection latency.

Rear tilt is RECONSTRUCTED from the front IMU and the three spine encoders
(plot.compute_rear_body_dynamics), not read from B_Q*: the rear IMU reports
identity in every log under raw/, so its quaternion carries no orientation.

Usage:
    python raw/control_plot.py                                  # default run
    python raw/control_plot.py path/to/run.csv -n 30
    python raw/control_plot.py path/to/run.csv --deg --out fig.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Runnable as `python raw/control_plot.py` from the repo root, which does not put
# raw/ on sys.path, and as `python control_plot.py` from inside raw/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot import G, compute_rear_body_dynamics, compute_tilt_angle

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = os.path.join(HERE, "with_tail", "roll", "roll_180deg",
                           "wt_roll_180deg_0810_rep03.csv")

# Commanded joint targets, in the controller's own column names. rot2 is omitted:
# controller.py drives it as -rot1 (Cmd_B2 == -Cmd_F1), so it carries no
# information the roll trace does not already show.
TARGETS = [("Cmd_F1", "roll"), ("Cmd_F2", "pitch"), ("Cmd_B1", "tail")]


def as_vector(path):
    """Figures are always saved as vector art: SVG unless .pdf was asked for."""
    stem, ext = os.path.splitext(path)
    return path if ext.lower() == ".pdf" else stem + ".svg"


def fall_distance(t):
    """Free-fall distance (m) at time t (s) after release."""
    return 0.5 * G * np.asarray(t, dtype=float) ** 2


def load_drop(path, steps):
    """One telemetry CSV -> the first `steps` control steps, clocked from release."""
    df = pd.read_csv(path).iloc[:steps]
    t = df["Time"].to_numpy(dtype=float)
    front_q = df[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy(dtype=float)
    rear_q = compute_rear_body_dynamics(front_q, df["F_M1"], df["F_M2"], df["B_M2"])
    return {
        "t": t - t[0],
        "tilt_front": compute_tilt_angle(front_q),
        "tilt_rear": compute_tilt_angle(rear_q),
        "targets": {name: df[col].to_numpy(dtype=float) for col, name in TARGETS},
    }


def time_distance_ticks(t_max, spacing):
    """Tick positions every `spacing` seconds, labelled "time (distance)"."""
    ticks = np.arange(0.0, t_max + 1e-9, spacing)
    labels = [f"{t:.1f}\n({d:.2f} m)" for t, d in zip(ticks, fall_distance(ticks))]
    return ticks, labels


def plot_drop(run, steps=50, spacing=0.2, degrees=False, out=None):
    d = load_drop(run, steps)
    t = d["t"]

    fig, (ax_tilt, ax_cmd) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [4, 1]})

    ax_tilt.plot(t, d["tilt_front"], label="front body")
    ax_tilt.plot(t, d["tilt_rear"], label="rear body")
    ax_tilt.set_ylabel("tilt angle [deg]")
    ax_tilt.set_title(os.path.splitext(os.path.basename(run))[0])
    ax_tilt.legend()

    scale = np.degrees(1.0) if degrees else 1.0
    for name, series in d["targets"].items():
        ax_cmd.plot(t, series * scale, label=name)
    ax_cmd.set_ylabel("target [deg]" if degrees else "target [rad]")
    ax_cmd.legend(ncol=3, fontsize="small")

    ticks, labels = time_distance_ticks(t[-1], spacing)
    ax_cmd.set_xticks(ticks)
    ax_cmd.set_xticklabels(labels)
    ax_cmd.set_xlim(t[0], t[-1])
    ax_cmd.set_xlabel("time [s] (free-fall distance)")

    fig.tight_layout()
    if out:
        out = as_vector(out)
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=DEFAULT_RUN,
                        help="telemetry CSV to plot (default: %(default)s)")
    parser.add_argument("-n", "--steps", type=int, default=32,
                        help="control steps to plot, from release (default: 50)")
    parser.add_argument("--spacing", type=float, default=0.2,
                        help="x-tick spacing in seconds (default: 0.2)")
    parser.add_argument("--deg", action="store_true",
                        help="plot joint targets in degrees instead of radians")
    parser.add_argument("--out", default=None,
                        help="save here instead of opening a window; written as SVG "
                             "unless the name ends in .pdf")
    args = parser.parse_args()
    plot_drop(args.run, args.steps, args.spacing, args.deg, args.out)
