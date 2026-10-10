"""Draw paper Fig. 4: mean body tilt over the fall, one line per release condition.

Panel (a) is the with-tail robot and panel (b) the no-tail robot. Every trial CSV under
--data is grouped by morphology and release condition. Each trial's front and rear body
tilts (|roll| + |pitch|) are averaged per control step, and each line and band are the
mean and +/-1 population SD (ddof=0) of that series across the condition's trials.
Trials are stacked by control step, not resampled, and the first 33 steps (about 0.64 s,
2 m of fall) are shown. The tilt at 1 m and 2 m of fall, pooled per morphology, is printed
and, with --milestones, saved as CSV.

Usage, from the repository root:
    python -m results.plot_scripts.fig4_tilt
    python -m results.plot_scripts.fig4_tilt --body rear --band ci --out fig4_rear_ci.svg
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy import stats

from results.plot_scripts.kinematics import (
    CONDITION_ORDER, MORPHOLOGIES, ROLL_SWEEP, fall_distance, fall_time, find_trials,
    load_drop, parse_trial_name,
)

# Default input and output paths
DEFAULT_DATA = Path("results/hardware_trials")
DEFAULT_OUT = Path("results/figures/fig4_tilt.svg")

# Plotted window in control steps from release; every log (49 or 50 rows) covers it in full
MAX_STEPS = 33

# Free-fall depths (m) at which tilt is reported, read from the full runs
MILESTONE_DEPTHS = (1.0, 2.0)

# Axis layout: tilt range and ticks (deg), and time tick spacing (s)
YLIM = (0, 225)
YTICKS = np.arange(0, 226, 45)
X_SPACING = 0.2

# Tilt series and axis label for each --body choice
BODIES = {"mean": "tilt_mean", "front": "tilt_front", "rear": "tilt_rear"}
YLABEL = {"mean": "Mean body tilt (degrees)", "front": "Front body tilt (degrees)",
          "rear": "Rear body tilt (degrees)"}

# Panel title of each morphology
PANEL_TITLES = {"with_tail": "With tail", "no_tail": "No tail"}

# Label, colour and line style of each release condition: the roll sweep is cool and
# dashed, the pitch sweep warm and solid, each darker as the swept angle grows
CONDITION_LABELS = {
    "r45": "45° roll, 0° pitch", "r90": "90° roll, 0° pitch",
    "r180": "180° roll, 0° pitch", "r180_p15": "180° roll, 15° pitch",
    "r180_p30": "180° roll, 30° pitch", "r180_p45": "180° roll, 45° pitch",
}
CONDITION_COLORS = {
    "r45": "#6BAED6", "r90": "#2171B5", "r180": "#08306B",
    "r180_p15": "#FEB24C", "r180_p30": "#FC4E2A", "r180_p45": "#B10026",
}
CONDITION_LINESTYLE = {c: "--" if c in ROLL_SWEEP else "-" for c in CONDITION_ORDER}

# Matplotlib style of the paper figure
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "text.usetex": False, "axes.unicode_minus": False,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 7.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "svg.fonttype": "none",
}


def mean_std(values):
    """Return the mean across trials and the +/-1 population SD (ddof=0) band around it."""
    values = np.asarray(values, dtype=float)
    mean, sd = values.mean(axis=0), values.std(axis=0)
    return mean, mean - sd, mean + sd


def mean_ci(values):
    """Return the mean across trials and its 95% t-interval (zero width for one trial)."""
    # Average across trials; a single trial has no interval
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    n = values.shape[0]
    if n < 2:
        return mean, mean, mean

    # Scale the standard error by the two-sided 95% t quantile
    half_width = stats.t.ppf(0.975, n - 1) * stats.sem(values, axis=0)
    return mean, mean - half_width, mean + half_width


# Band statistic for each --band choice
BANDS = {"std": mean_std, "ci": mean_ci}


def find_groups(root):
    """Group the trial CSVs under `root` by (morphology, condition), in path order."""
    groups = {}
    for path in find_trials(root):
        meta = parse_trial_name(path)
        groups.setdefault((meta["morphology"], meta["condition"]), []).append(path)
    return groups


def summarize(paths, steps):
    """Stack the per-step tilts of a group of runs, cut to the shortest run.

    Returns:
        A dict with "n_runs", "n_steps", "t" (the mean timestamp of each step) and, for
        each body in BODIES, an (n_runs, n_steps) tilt array.
    """
    # Load the runs and find their common length
    runs = [load_drop(path, steps) for path in paths]
    n_steps = min(len(run["t"]) for run in runs)

    # Average the timestamps per step and stack each body's tilt
    summary = {"n_runs": len(runs), "n_steps": n_steps,
               "t": np.mean([run["t"][:n_steps] for run in runs], axis=0)}
    for body, key in BODIES.items():
        summary[body] = np.array([run[key][:n_steps] for run in runs])
    return summary


def milestone_tilts(paths, body):
    """Return the tilt at each milestone depth over the full runs, one row per depth.

    Each run contributes its sample nearest the depth's free-fall time rather than an
    interpolated value. "covered" is False when a run ends before that time.
    """
    # Load the runs in full
    runs = [load_drop(path) for path in paths]

    # Read every run at each depth
    rows = []
    for depth in MILESTONE_DEPTHS:
        # Take each run's sample nearest the free-fall time of this depth
        target = fall_time(depth)
        nearest = [int(np.argmin(np.abs(run["t"] - target))) for run in runs]
        values = np.array([run[BODIES[body]][i] for run, i in zip(runs, nearest)])

        # Record the mean and SDs across runs, and whether every run lasted that long
        rows.append({
            "depth_m": depth, "time_s": target, "n": len(values),
            "mean_deg": values.mean(),
            "sd_pop_deg": values.std(ddof=0),
            "sd_sample_deg": values.std(ddof=1),
            "covered": all(run["t"][-1] >= target for run in runs),
        })
    return rows


def time_ticks(x_max):
    """Return the time ticks, their "time (distance)" labels and the upper x bound.

    The axis extends to the next whole tick only when the data reaches 90% of the way
    there; otherwise it ends at the data.
    """
    # Extend the axis to the next tick when the data nearly reaches it
    last = float(np.floor(x_max / X_SPACING + 1e-9) * X_SPACING)
    upper = last + X_SPACING if (x_max - last) >= 0.9 * X_SPACING else x_max

    # Label each tick with the time and the free-fall distance
    ticks = np.arange(0.0, upper + 1e-9, X_SPACING)
    labels = [f"{t:.1f}\n({d:.2f})" for t, d in zip(ticks, fall_distance(ticks))]
    return ticks, labels, upper


def plot_panel(ax, groups, morphology, steps, body, band):
    """Draw one morphology's mean tilt and band per condition; return the last x plotted."""
    # Draw every release condition flown with this morphology
    band_stat = BANDS[band]
    x_max = 0.0
    for condition in CONDITION_ORDER:
        paths = groups.get((morphology, condition))
        if not paths:
            continue

        # Average the condition's runs per step
        summary = summarize(paths, steps)
        x = summary["t"]
        x_max = max(x_max, x[-1])
        mean, low, high = band_stat(summary[body])

        # Draw the mean line and its shaded band
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]
        ax.plot(x, mean, color=color, linestyle=CONDITION_LINESTYLE[condition],
                linewidth=1.4, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)

        # Report the group size and its final tilt
        n_steps = summary["n_steps"]
        note = "" if n_steps == steps else f" (cut to {n_steps} steps by its shortest run)"
        print(f"  {PANEL_TITLES[morphology]}, {label}: n={summary['n_runs']}, "
              f"{n_steps} steps, final {mean[-1]:.1f} deg{note}")
    return x_max


def format_panels(axes, morphologies, body, x_max):
    """Set the limits, ticks, titles, panel letters and fall-depth lines of every panel."""
    # Compute the shared time ticks and the free-fall times of the milestone depths
    ticks, labels, upper = time_ticks(x_max)
    depth_times = [fall_time(depth) for depth in MILESTONE_DEPTHS]

    # Lay out each panel and mark the milestone depths with grey lines
    for i, (ax, morphology) in enumerate(zip(axes, morphologies)):
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        ax.set_ylabel(YLABEL[body])
        ax.set_xlim(0.0, upper)
        ax.tick_params(axis="both", which="both", direction="in", length=3.5, width=0.8)
        ax.set_title(PANEL_TITLES[morphology], fontweight="bold", fontstyle="italic")
        ax.text(-0.17, 1.04, f"({'abcdefgh'[i]})", transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="bottom")
        for t in depth_times:
            if t <= upper:
                ax.axvline(t, color="#BBBBBB", linewidth=1.5, zorder=0)

    # Label the depth lines once, under the top panel
    trans = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for depth, t in zip(MILESTONE_DEPTHS, depth_times):
        if t <= upper:
            axes[0].text(t, -0.03, f"({depth:g} m)", transform=trans, ha="center",
                         va="top", fontsize=7.5, color="#222222")

    # Label the shared time axis on the bottom panel
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Time (s)\n(Free-fall distance (m))")


def add_legend(fig, axes, groups):
    """Lay out the figure and add the shared condition legend under the bottom panel."""
    # Make one legend entry per condition present, in the paper's order
    present = {condition for _, condition in groups}
    handles = [Line2D([], [], color=CONDITION_COLORS[c], linestyle=CONDITION_LINESTYLE[c],
                      linewidth=1.4, label=CONDITION_LABELS[c])
               for c in CONDITION_ORDER if c in present]

    # Reorder the entries so the column-major legend reads row by row: roll sweep, then pitch
    ncol = 3
    rows = [handles[i:i + ncol] for i in range(0, len(handles), ncol)]
    column_major = [row[j] for j in range(ncol) for row in rows if j < len(row)]

    # Fix the layout, then place the legend just under the bottom panel's x-axis label
    fig.tight_layout(h_pad=1.6)
    fig.canvas.draw()
    label_box = axes[-1].xaxis.label.get_window_extent()
    label_bottom = fig.transFigure.inverted().transform((0, label_box.y0))[1]
    fig.legend(handles=column_major, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, label_bottom - 0.01), handlelength=2.2,
               columnspacing=1.0)


def report_milestones(groups, body):
    """Print the tilt at each milestone depth, pooled per morphology, and return the table."""
    # Print the table header
    print(f"\n{YLABEL[body].replace(' (degrees)', '')} at fixed free-fall depth, pooled over "
          f"all trials of each morphology (nearest sample, full runs). Mean +/- population "
          f"SD (ddof=0, as in the paper); sample SD (ddof=1) in brackets:")

    # Report each morphology's trials pooled over all conditions
    rows = []
    for morphology in MORPHOLOGIES:
        # Collect every trial of this morphology
        paths = []
        for (group_morphology, _), group_paths in groups.items():
            if group_morphology == morphology:
                paths.extend(group_paths)
        if not paths:
            continue

        # Print and collect one row per depth
        for row in milestone_tilts(paths, body):
            flag = "" if row["covered"] else "  (not reached by every run)"
            print(f"  {PANEL_TITLES[morphology]:9s} n={row['n']}: {row['depth_m']:g} m "
                  f"({row['time_s']:.2f} s): {row['mean_deg']:.1f} +/- "
                  f"{row['sd_pop_deg']:.1f} deg [{row['sd_sample_deg']:.1f}]{flag}")
            rows.append({"morphology": morphology, "body": body, **row})
    return pd.DataFrame(rows)


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


def plot_fig4(data=DEFAULT_DATA, out=DEFAULT_OUT, steps=MAX_STEPS, body="mean", band="std",
              milestones_out=None):
    """Draw and save Fig. 4 and return the milestone table, saved when `milestones_out` is set."""
    # Group the trials and keep the morphologies that have any
    steps = min(steps, MAX_STEPS)
    groups = find_groups(data)
    present = {morphology for morphology, _ in groups}
    morphologies = [m for m in MORPHOLOGIES if m in present]

    with plt.rc_context(STYLE):
        # Draw one panel per morphology
        fig, axes = plt.subplots(len(morphologies), 1, sharex=True, squeeze=False,
                                 figsize=(3.5, 1.9 * len(morphologies) + 0.9))
        axes = axes[:, 0]
        x_max = 0.0
        for ax, morphology in zip(axes, morphologies):
            x_max = max(x_max, plot_panel(ax, groups, morphology, steps, body, band))

        # Lay out the axes and the shared legend
        format_panels(axes, morphologies, body, x_max)
        add_legend(fig, axes, groups)

        # Report the milestone tilts and save them
        milestones = report_milestones(groups, body)
        if milestones_out and not milestones.empty:
            Path(milestones_out).parent.mkdir(parents=True, exist_ok=True)
            milestones.to_csv(milestones_out, index=False)
            print(f"Saved {milestones_out}")

        # Save the figure
        save_figure(fig, out)
        plt.close(fig)
    return milestones


def main(argv=None):
    """Parse the command line, draw Fig. 4 and return the milestone table."""
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
    parser.add_argument("--milestones", default=None,
                        help="also save the 1 m / 2 m tilt values to this CSV "
                             "(default: print them only)")
    parser.add_argument("-n", "--steps", type=int, default=MAX_STEPS,
                        help=f"control steps to plot, from release (default and max: {MAX_STEPS})")
    parser.add_argument("--body", choices=list(BODIES), default="mean",
                        help="tilt to draw: the per-run average of both bodies, or a "
                             "single body (default: %(default)s)")
    parser.add_argument("--band", choices=list(BANDS), default="std",
                        help="shaded band: +/-1 population SD, or a 95%% t-interval "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)

    # Draw the figure
    print(f"Reading trials from {args.data}")
    return plot_fig4(args.data, args.out, args.steps, args.body, args.band, args.milestones)


if __name__ == "__main__":
    main()
