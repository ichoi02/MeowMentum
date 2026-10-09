"""Paper Fig. 5: whole-cat score per release condition, no tail vs with tail.

(a) Roll sweep at 45/90/180 deg with pitch fixed at 0 deg. (b) Pitch sweep at
0/15/30/45 deg with roll fixed at 180 deg (the 180 deg roll trials appear in both).
Boxes show the median and interquartile range, whiskers extend to the furthest
trial within 1.5 x IQR, and every trial is drawn as a jittered point. Grey = no
tail, hatched blue = with tail.

The scores are the posture_score column of results/trial_metrics.csv, written by
whole_cat_stats.py (so the short drop it excludes is not shown). The y-axis is
broken: the scores span roughly 0.36-0.97, so the upper segment starts at the
tenth below the lowest score and a short lower segment keeps 0 on the axis.

Usage:
    python analysis/fig5_whole_cat.py
    python analysis/fig5_whole_cat.py --metrics /tmp/whole_cat_2p8m/trial_metrics.csv \
        --out /tmp/fig5_2p8m.svg
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from kinematics import PITCH_SWEEP, ROLL_SWEEP

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "results" / "trial_metrics.csv"
DEFAULT_OUT = ROOT / "results" / "figures" / "fig5_whole_cat.svg"

PANELS = (
    (ROLL_SWEEP, ("45°", "90°", "180°"), "Release roll (degrees, pitch fixed at 0°)"),
    (PITCH_SWEEP, ("0°", "15°", "30°", "45°"), "Release pitch (degrees, roll fixed at 180°)"),
)

# (offset from the condition tick, box face, line/point colour, hatch)
MORPHOLOGY_STYLE = {
    "no_tail": (-0.18, "#C8C8C8", "#555555", None),
    "with_tail": (0.18, "#A9CFE8", "#1F6FB4", "///"),
}
MORPHOLOGY_LABELS = {"no_tail": "No tail", "with_tail": "With tail"}
BOX_WIDTH = 0.3
JITTER = 0.05      # half-width of the uniform horizontal jitter, in tick units
SEED = 5           # jitter only; the boxes do not depend on it

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "hatch.linewidth": 1.0, "svg.fonttype": "none",
}


def score_range_bottom(scores, step=0.1):
    """Bottom of the upper axis segment: the multiple of `step` below the lowest score."""
    return float(np.floor(np.min(scores) / step + 1e-9) * step)


def draw_boxes(ax, frame, conditions, rng):
    """Box plot plus jittered trials for each condition and morphology."""
    positions = np.arange(len(conditions))
    for morphology, (offset, face, line, hatch) in MORPHOLOGY_STYLE.items():
        groups = [frame.loc[(frame["condition"] == c) & (frame["morphology"] == morphology),
                            "posture_score"].to_numpy(float) for c in conditions]
        box = ax.boxplot(
            groups, positions=positions + offset, widths=BOX_WIDTH, whis=1.5,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color": line, "linewidth": 2.0, "solid_capstyle": "butt"},
            whiskerprops={"color": line, "linewidth": 1.4},
            capprops={"color": line, "linewidth": 1.4},
            capwidths=BOX_WIDTH * 0.45,
        )
        for patch in box["boxes"]:
            # The edge colour doubles as the hatch colour; linewidth 0 hides the outline.
            patch.set(facecolor=face, edgecolor="white" if hatch else face, linewidth=0,
                      hatch=hatch)
        for position, values in zip(positions + offset, groups):
            ax.scatter(position + rng.uniform(-JITTER, JITTER, len(values)), values, s=16,
                       color=line, edgecolor="white", linewidth=0.6, zorder=3, clip_on=False)
    return positions


def plot_fig5(metrics=DEFAULT_METRICS, out=DEFAULT_OUT):
    frame = pd.read_csv(metrics)
    bottom = score_range_bottom(frame["posture_score"])
    broken = bottom > 0.1
    rng = np.random.default_rng(SEED)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.5, 5.0))
        outer = fig.add_gridspec(len(PANELS), 1, hspace=0.42)
        for i, (conditions, tick_labels, xlabel) in enumerate(PANELS):
            if broken:
                inner = outer[i].subgridspec(2, 1, height_ratios=[9, 1], hspace=0.12)
                ax = fig.add_subplot(inner[0])
                ax_zero = fig.add_subplot(inner[1], sharex=ax)
            else:
                ax = fig.add_subplot(outer[i])
                ax_zero = ax
            positions = draw_boxes(ax, frame, conditions, rng)
            ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
            ax.set_ylim(bottom if broken else 0.0, 1.0)
            ax.set_yticks(np.round(np.arange(bottom if broken else 0.0, 1.0001, 0.1), 1))
            ax.set_ylabel("Whole-cat score (0–1)")
            ax.text(-0.2, 1.0, f"({'ab'[i]})", transform=ax.transAxes, fontweight="bold",
                    fontsize=9, va="bottom")
            if broken:
                # Lower segment: only the 0 tick and the x-axis.
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax_zero.set_ylim(0.0, 0.02)
                ax_zero.set_yticks([0.0], ["0.0"])
                # Break marks: a pair of slashes across the gap in the left spine.
                marks = {"marker": [(-1, -0.5), (1, 0.5)], "markersize": 6,
                         "linestyle": "none", "color": "#333333", "mec": "#333333",
                         "mew": 0.8, "clip_on": False}
                ax.plot([0], [0], transform=ax.transAxes, **marks)
                ax_zero.plot([0], [1], transform=ax_zero.transAxes, **marks)
            ax_zero.set_xticks(positions, tick_labels)
            ax_zero.set_xlabel(xlabel)
            for a in {ax, ax_zero}:
                a.tick_params(axis="both", direction="out", length=3, width=0.8)
            print_summary(frame, conditions, tick_labels, xlabel)

        handles = [Patch(facecolor=face, edgecolor="white" if hatch else face, linewidth=0,
                         hatch=hatch, label=MORPHOLOGY_LABELS[m])
                   for m, (_, face, _, hatch) in MORPHOLOGY_STYLE.items()]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.55, 0.035), handlelength=2.6, columnspacing=2.0)
        save_figure(fig, out)
        plt.close(fig)


def print_summary(frame, conditions, tick_labels, xlabel):
    """Median, IQR and n per box, so the figure can be checked against the table."""
    print(xlabel)
    for cond, label in zip(conditions, tick_labels):
        cells = []
        for morphology in MORPHOLOGY_STYLE:
            s = frame.loc[(frame["condition"] == cond) & (frame["morphology"] == morphology),
                          "posture_score"]
            cells.append(f"{MORPHOLOGY_LABELS[morphology]} median {s.median():.3f} "
                         f"IQR [{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}] n={len(s)}")
        print(f"  {label:>4s}: " + "; ".join(cells))


def save_figure(fig, filename):
    """Save as vector art: SVG unless the name ends in .pdf (any other suffix
    becomes .svg), with text kept as editable <text> nodes. The file carries no
    timestamp and a fixed id salt, so regenerating it gives identical bytes."""
    path = Path(filename)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if path.suffix == ".svg" else {"CreationDate": None}
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "meowmentum"}):
        fig.savefig(path, format=path.suffix[1:], bbox_inches="tight", facecolor="white",
                    metadata=metadata)
    print(f"Saved {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS),
                        help="trial_metrics.csv from whole_cat_stats.py "
                             f"(default: {DEFAULT_METRICS.relative_to(ROOT)})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="figure path; written as SVG (text kept editable) unless the "
                             "name ends in .pdf, any other extension is replaced by .svg "
                             f"(default: {DEFAULT_OUT.relative_to(ROOT)})")
    args = parser.parse_args(argv)
    plot_fig5(args.metrics, args.out)


if __name__ == "__main__":
    main()
