"""Tilt and commanded joint targets for a single drop, clocked from release.

Top panel: front- and rear-body tilt (|roll| + |pitch|, as in Fig. 4) over the drop.
Bottom panel: the joint targets the policy commanded (roll, pitch, tail),
shared x-axis, so a change in tilt can be read against the command that caused it.

The x-axis is time since release, and each tick is annotated with the free-fall
distance at that instant, d = g t^2 / 2 -- the drop height a run of that length
needs. Release is t[0]: controller.py starts logging once the front accelerometer
reads free fall, so the first sample is the drop, up to one 50 Hz tick of
detection latency.

Rear tilt is RECONSTRUCTED from the front IMU and the three spine encoders
(kinematics.compute_rear_body_quat), not read from B_Q*: the rear IMU reports
identity in every log, so its quaternion carries no orientation.

Usage:
    python analysis/plot_drop.py                                # default trial
    python analysis/plot_drop.py path/to/trial.csv -n 45
    python analysis/plot_drop.py path/to/trial.csv --deg --out drop.svg
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kinematics import fall_distance, load_drop

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (ROOT / "data" / "hardware_trials" / "with_tail" / "roll" / "roll_180deg"
               / "wt_roll_180deg_0810_rep03.csv")

# Commanded joint targets, in the controller's own column names. rot2 is omitted:
# controller.py drives it as -rot1 (Cmd_B2 == -Cmd_F1), so it carries no
# information the roll trace does not already show.
TARGETS = [("Cmd_F1", "roll"), ("Cmd_F2", "pitch"), ("Cmd_B1", "tail")]


def as_vector(path):
    """Figures are always saved as vector art: SVG unless .pdf was asked for."""
    stem, ext = os.path.splitext(path)
    return path if ext.lower() == ".pdf" else stem + ".svg"


def time_distance_ticks(t_max, spacing):
    """Tick positions every `spacing` seconds, labelled "time (distance)"."""
    ticks = np.arange(0.0, t_max + 1e-9, spacing)
    labels = [f"{t:.1f}\n({d:.2f} m)" for t, d in zip(ticks, fall_distance(ticks))]
    return ticks, labels


def plot_drop(run, steps=32, spacing=0.2, degrees=False, out=None):
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
    for col, name in TARGETS:
        ax_cmd.plot(t, d["frame"][col].to_numpy(dtype=float) * scale, label=name)
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
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=str(DEFAULT_RUN),
                        help=f"trial CSV to plot (default: {DEFAULT_RUN.relative_to(ROOT)})")
    parser.add_argument("-n", "--steps", type=int, default=32,
                        help="control steps to plot, from release; the logs hold 49-50 "
                             "(default: %(default)s)")
    parser.add_argument("--spacing", type=float, default=0.2,
                        help="x-tick spacing in seconds (default: %(default)s)")
    parser.add_argument("--deg", action="store_true",
                        help="plot joint targets in degrees instead of radians")
    parser.add_argument("--out", default=None,
                        help="save here instead of opening a window; written as SVG (text "
                             "kept editable) unless the name ends in .pdf, any other "
                             "extension is replaced by .svg")
    args = parser.parse_args()
    plot_drop(args.run, args.steps, args.spacing, args.deg, args.out)
