"""Draw paper Fig. 5: whole-cat score per release condition, no tail vs with tail.

Panel (a) is the roll sweep at 0 deg pitch and panel (b) the pitch sweep at 180 deg roll,
so the 180 deg roll trials appear in both. Boxes show the median and interquartile range,
whiskers reach the furthest trial within 1.5 x IQR, and every trial is a jittered point.
The scores are the whole-cat scores at 2.5 m of fall, computed from the drop logs with
whole_cat_stats.score_trials. The y-axis is broken below the lowest score, with a short lower
segment that keeps 0 on the axis.

Usage, from the repository root:
    python -m results.plot_scripts.fig5_whole_cat
    python -m results.plot_scripts.fig5_whole_cat --out fig5.pdf
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from results.plot_scripts.kinematics import PITCH_SWEEP, ROLL_SWEEP
from results.plot_scripts.whole_cat_stats import score_trials

# Default input and output paths
DEFAULT_DATA = Path("results/hardware_trials")
DEFAULT_OUT = Path("results/figures/fig5_whole_cat.svg")

# Each panel's release conditions, their tick labels and its x-axis label
PANELS = (
    (ROLL_SWEEP, ("45°", "90°", "180°"), "Release roll (degrees, pitch fixed at 0°)"),
    (PITCH_SWEEP, ("0°", "15°", "30°", "45°"), "Release pitch (degrees, roll fixed at 180°)"),
)

# Per morphology: offset from the condition tick, box face colour, line and point colour,
# and hatch
MORPHOLOGY_STYLE = {
    "no_tail": (-0.18, "#C8C8C8", "#555555", None),
    "with_tail": (0.18, "#A9CFE8", "#1F6FB4", "///"),
}
MORPHOLOGY_LABELS = {"no_tail": "No tail", "with_tail": "With tail"}

# Box width and the half-width of the uniform horizontal jitter, in tick units; the seed
# only moves the jittered points
BOX_WIDTH = 0.3
JITTER = 0.05
SEED = 5

# Spacing of the score ticks; the upper axis segment starts on one of them
TICK_STEP = 0.1

# Matplotlib style of the paper figure
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "hatch.linewidth": 1.0, "svg.fonttype": "none",
}


def score_range_bottom(scores):
    """Return the bottom of the upper axis segment: the score tick below the lowest score."""
    return float(np.floor(np.min(scores) / TICK_STEP + 1e-9) * TICK_STEP)


def condition_scores(frame, condition, morphology):
    """Return the whole-cat scores of one release condition and morphology."""
    rows = (frame["condition"] == condition) & (frame["morphology"] == morphology)
    return frame.loc[rows, "posture_score"]


def draw_boxes(ax, frame, conditions, rng):
    """Draw a box per condition and morphology with the trials on top; return the ticks."""
    # Draw each morphology's boxes beside the condition ticks
    positions = np.arange(len(conditions))
    for morphology, (offset, face, line, hatch) in MORPHOLOGY_STYLE.items():
        # Draw the boxes, with whiskers at 1.5 x IQR and no outlier markers
        groups = [condition_scores(frame, c, morphology).to_numpy(float) for c in conditions]
        box = ax.boxplot(
            groups, positions=positions + offset, widths=BOX_WIDTH, whis=1.5,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color": line, "linewidth": 2.0, "solid_capstyle": "butt"},
            whiskerprops={"color": line, "linewidth": 1.4},
            capprops={"color": line, "linewidth": 1.4},
            capwidths=BOX_WIDTH * 0.45,
        )

        # Fill the boxes; the edge colour doubles as the hatch colour, and linewidth 0
        # hides the outline
        for patch in box["boxes"]:
            patch.set(facecolor=face, edgecolor="white" if hatch else face, linewidth=0,
                      hatch=hatch)

        # Draw every trial as a horizontally jittered point
        for position, values in zip(positions + offset, groups):
            ax.scatter(position + rng.uniform(-JITTER, JITTER, len(values)), values, s=16,
                       color=line, edgecolor="white", linewidth=0.6, zorder=3, clip_on=False)
    return positions


def add_panel_axes(fig, cell, broken):
    """Add a panel's score axes and the zero segment below it (the same axes if unbroken)."""
    if not broken:
        ax = fig.add_subplot(cell)
        return ax, ax

    inner = cell.subgridspec(2, 1, height_ratios=[9, 1], hspace=0.12)
    ax = fig.add_subplot(inner[0])
    return ax, fig.add_subplot(inner[1], sharex=ax)


def break_axis(ax, ax_zero):
    """Join the upper score segment to the zero segment with break marks."""
    # Leave only the 0 tick and the x-axis to the lower segment
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_zero.set_ylim(0.0, 0.02)
    ax_zero.set_yticks([0.0], ["0.0"])

    # Draw a pair of slashes across the gap in the left spine
    marks = {"marker": [(-1, -0.5), (1, 0.5)], "markersize": 6,
             "linestyle": "none", "color": "#333333", "mec": "#333333",
             "mew": 0.8, "clip_on": False}
    ax.plot([0], [0], transform=ax.transAxes, **marks)
    ax_zero.plot([0], [1], transform=ax_zero.transAxes, **marks)


def plot_fig5(data=DEFAULT_DATA, out=DEFAULT_OUT):
    """Score the drops under `data`, draw Fig. 5 and save it."""
    # Read the scores and break the y-axis when the upper segment starts above the first tick
    frame, _ = score_trials(Path(data))
    bottom = score_range_bottom(frame["posture_score"])
    broken = bottom > TICK_STEP
    y_min = bottom if broken else 0.0
    rng = np.random.default_rng(SEED)

    with plt.rc_context(STYLE):
        # Stack one panel per sweep
        fig = plt.figure(figsize=(3.5, 5.0))
        outer = fig.add_gridspec(len(PANELS), 1, hspace=0.42)
        for i, (conditions, tick_labels, xlabel) in enumerate(PANELS):
            # Draw the boxes and label the score axis
            ax, ax_zero = add_panel_axes(fig, outer[i], broken)
            positions = draw_boxes(ax, frame, conditions, rng)
            ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
            ax.set_ylim(y_min, 1.0)
            ax.set_yticks(np.round(np.arange(y_min, 1.0001, TICK_STEP), 1))
            ax.set_ylabel("Whole-cat score (0–1)")
            ax.text(-0.2, 1.0, f"({'ab'[i]})", transform=ax.transAxes, fontweight="bold",
                    fontsize=9, va="bottom")
            if broken:
                break_axis(ax, ax_zero)

            # Label the conditions under the lowest segment and point the ticks outward
            ax_zero.set_xticks(positions, tick_labels)
            ax_zero.set_xlabel(xlabel)
            ax.tick_params(axis="both", direction="out", length=3, width=0.8)
            if broken:
                ax_zero.tick_params(axis="both", direction="out", length=3, width=0.8)

            # Print the numbers behind the boxes
            print_summary(frame, conditions, tick_labels, xlabel)

        # Add the morphology legend under the panels
        handles = [Patch(facecolor=face, edgecolor="white" if hatch else face, linewidth=0,
                         hatch=hatch, label=MORPHOLOGY_LABELS[m])
                   for m, (_, face, _, hatch) in MORPHOLOGY_STYLE.items()]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.55, 0.035), handlelength=2.6, columnspacing=2.0)

        # Save the figure
        save_figure(fig, out)
        plt.close(fig)


def print_summary(frame, conditions, tick_labels, xlabel):
    """Print the median, IQR and n of each box, to check the figure against the table."""
    print(xlabel)
    for condition, label in zip(conditions, tick_labels):
        cells = []
        for morphology in MORPHOLOGY_STYLE:
            scores = condition_scores(frame, condition, morphology)
            cells.append(f"{MORPHOLOGY_LABELS[morphology]} median {scores.median():.3f} "
                         f"IQR [{scores.quantile(0.25):.3f}, {scores.quantile(0.75):.3f}] "
                         f"n={len(scores)}")
        print(f"  {label:>4s}: " + "; ".join(cells))


def save_figure(fig, filename):
    """Save the figure as vector art with reproducible bytes and return its path.

    The file is SVG unless the name ends in .pdf (any other suffix becomes .svg), with text
    kept as editable <text> nodes. No timestamp and a fixed id salt make reruns identical.
    """
    # Choose the vector format from the file name
    path = Path(filename)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save without a timestamp and with a fixed id salt
    metadata = {"Date": None} if path.suffix == ".svg" else {"CreationDate": None}
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "meowmentum"}):
        fig.savefig(path, format=path.suffix[1:], bbox_inches="tight", facecolor="white",
                    metadata=metadata)
    print(f"Saved {path}")
    return path


def main(argv=None):
    """Parse the command line and draw Fig. 5."""
    # Parse the command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="folder of trial CSVs, searched recursively "
                             f"(default: {DEFAULT_DATA})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="figure path; written as SVG (text kept editable) unless the "
                             "name ends in .pdf, any other extension is replaced by .svg "
                             f"(default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    # Draw the figure
    plot_fig5(args.data, args.out)


if __name__ == "__main__":
    main()
