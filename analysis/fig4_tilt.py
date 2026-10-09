"""Paper Fig. 4: mean body tilt over the fall, one line per release condition.

Panel (a) is the with-tail robot, panel (b) the no-tail robot. Every trial CSV under
--data is used and grouped by morphology and release condition (from its filename).

Each drop is reduced to ONE tilt series before any statistics are taken: the front
and rear body tilts (|roll| + |pitch|, kinematics.compute_tilt_angle) are averaged
per trial, per step, and the line and band are the mean and +/-1 SD across trials of
that quantity. The band is therefore the spread across drops, not the spread between
the two bodies. SDs are population SDs (numpy's default, ddof=0), as in the paper.
`--body front` or `--body rear` plots a single body instead; `--band ci` swaps the
SD band for a 95% t-interval.

Trials are averaged BY CONTROL STEP, not by resampling onto a common clock: row k of
every log is the k-th policy step, so it is the same point in the maneuver even
though the 50 Hz loop jitters by a few ms. Nothing is interpolated. The x position of
step k is the mean of the trials' own timestamps at k. The window is the first 33
steps (~0.64 s, ~2 m of fall), which every log (49-50 rows) covers in full. Release
is t[0]: controller.py starts logging once the front accelerometer reads free fall.

The x-axis is time since release, each tick annotated with the free-fall distance
d = g t^2 / 2. Grey lines mark 1 m and 2 m of fall; the script prints (and writes to
--milestones) the tilt at those depths pooled over all trials of each morphology,
read from the sample nearest each depth's fall time.

All trials are included, also wt_roll_090deg_0816_rep03, which the whole-cat
analysis excludes because its impact came before 2.5 m of fall: its impact (~2.4 m)
comes after both the 33-step window and the 2 m sample.

Rear tilt is RECONSTRUCTED from the front IMU and the three spine encoders
(kinematics.compute_rear_body_quat), not read from B_Q*: the rear IMU reports
identity in every log, so its quaternion carries no orientation.

Usage:
    python analysis/fig4_tilt.py
    python analysis/fig4_tilt.py --body rear --band ci --out /tmp/fig4_rear_ci.svg --milestones ''
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy import stats

from kinematics import (
    MORPHOLOGIES, PITCH_SWEEP, ROLL_SWEEP, fall_distance, fall_time, find_trials,
    load_drop, parse_trial_name,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "hardware_trials"
DEFAULT_OUT = ROOT / "results" / "figures" / "fig4_tilt.svg"
DEFAULT_MILESTONES = ROOT / "results" / "tilt_milestones.csv"

# The plotted window, in control steps from release. Every log is 49 or 50 steps,
# so 33 (~0.64 s) sits inside what every drop covers in full, which is what lets
# runs be stacked by step index with no padding and no interpolation.
MAX_STEPS = 33

# Depths at which tilt is reported alongside the figure (and marked on it). They
# are read from the FULL runs, so the numbers do not move when --steps changes.
MILESTONE_DEPTHS = (1.0, 2.0)

YLIM = (0, 225)
YTICKS = np.arange(0, 226, 45)
X_SPACING = 0.2  # s

BODIES = {"mean": "tilt_mean", "front": "tilt_front", "rear": "tilt_rear"}
YLABEL = {"mean": "Mean body tilt (degrees)", "front": "Front body tilt (degrees)",
          "rear": "Rear body tilt (degrees)"}

PANEL_TITLES = {"with_tail": "With tail", "no_tail": "No tail"}

# Colour encodes the sweep and how far into it a condition sits: the roll sweep runs
# cool (light blue -> navy) and the pitch sweep warm (amber -> red), each ordered
# light to dark with the swept angle. Linestyle repeats the sweep (dashed = roll
# sweep at 0 deg pitch, solid = pitch sweep at 180 deg roll), so the two families
# separate even in greyscale.
CONDITIONS = ("r45", "r90", "r180", "r180_p15", "r180_p30", "r180_p45")
CONDITION_LABELS = {
    "r45": "45° roll, 0° pitch", "r90": "90° roll, 0° pitch",
    "r180": "180° roll, 0° pitch", "r180_p15": "180° roll, 15° pitch",
    "r180_p30": "180° roll, 30° pitch", "r180_p45": "180° roll, 45° pitch",
}
CONDITION_COLORS = {
    "r45": "#6BAED6", "r90": "#2171B5", "r180": "#08306B",            # cool
    "r180_p15": "#FEB24C", "r180_p30": "#FC4E2A", "r180_p45": "#B10026",  # warm
}
CONDITION_LINESTYLE = {c: "--" if c in ROLL_SWEEP else "-" for c in CONDITIONS}
assert set(CONDITIONS) == set(ROLL_SWEEP) | set(PITCH_SWEEP)

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


def mean_std(values, axis=0):
    """Mean and +/-1 population SD (ddof=0) along `axis`."""
    values = np.asarray(values, dtype=float)
    mean, sd = values.mean(axis=axis), values.std(axis=axis)
    return mean, mean - sd, mean + sd


def mean_ci(values, confidence=0.95, axis=0):
    """Mean and a t-based confidence interval along `axis`."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    n = values.shape[axis]
    if n < 2:
        return mean, mean, mean
    half_width = stats.t.ppf((1 + confidence) / 2, n - 1) * stats.sem(values, axis=axis)
    return mean, mean - half_width, mean + half_width


BANDS = {"std": mean_std, "ci": mean_ci}


def find_groups(root):
    """{(morphology, condition): [csv, ...]} for every trial under `root`."""
    groups = {}
    for path in find_trials(root):
        meta = parse_trial_name(path)
        groups.setdefault((meta["morphology"], meta["condition"]), []).append(path)
    return groups


def summarize(paths, steps):
    """Per-step tilt stacks over a group of runs, cut to the shortest run."""
    runs = [load_drop(p, steps) for p in paths]
    k = min(len(r["t"]) for r in runs)
    out = {"n": len(runs), "k": k,
           "t": np.mean([r["t"][:k] for r in runs], axis=0)}
    for body, key in BODIES.items():
        out[body] = np.array([r[key][:k] for r in runs])   # (n_runs, k)
    return out


def milestone_tilts(paths, body, depths=MILESTONE_DEPTHS):
    """Tilt at fixed free-fall depths over the full runs, one row per depth.

    The sample nearest each target time is taken rather than an interpolated value,
    for the same reason the curves themselves are not resampled. A run that ends
    before a depth is reached is reported as uncovered rather than silently
    contributing its last sample.
    """
    runs = [load_drop(p) for p in paths]
    rows = []
    for depth in depths:
        target = fall_time(depth)
        values = np.array([r[BODIES[body]][int(np.argmin(np.abs(r["t"] - target)))]
                           for r in runs])
        rows.append({
            "depth_m": depth, "time_s": target, "n": len(values),
            "mean_deg": values.mean(),
            "sd_pop_deg": values.std(ddof=0),
            "sd_sample_deg": values.std(ddof=1),
            "covered": all(r["t"][-1] >= target for r in runs),
        })
    return rows


def time_ticks(x_max, spacing=X_SPACING, snap=0.9):
    """Ticks every `spacing` s, labelled "time\\n(distance)", plus the axis bound.

    The axis is stretched to the next whole tick only when the data already reaches
    `snap` of the way there; otherwise the bound is the data itself.
    """
    last = float(np.floor(x_max / spacing + 1e-9) * spacing)
    upper = last + spacing if (x_max - last) >= snap * spacing else x_max
    ticks = np.arange(0.0, upper + 1e-9, spacing)
    labels = [f"{t:.1f}\n({d:.2f})" for t, d in zip(ticks, fall_distance(ticks))]
    return ticks, labels, upper


def plot_panel(ax, groups, morphology, steps, body, band):
    """Draw one morphology's conditions; return the last x value plotted."""
    stat = BANDS[band]
    x_max = 0.0
    for cond in CONDITIONS:
        paths = groups.get((morphology, cond))
        if not paths:
            continue
        s = summarize(paths, steps)
        x = s["t"]
        x_max = max(x_max, x[-1])
        mean, lo, hi = stat(s[body])
        ax.plot(x, mean, color=CONDITION_COLORS[cond], linestyle=CONDITION_LINESTYLE[cond],
                linewidth=1.4, label=CONDITION_LABELS[cond])
        ax.fill_between(x, lo, hi, color=CONDITION_COLORS[cond], alpha=0.15, linewidth=0)
        note = "" if s["k"] == steps else f" (cut to {s['k']} steps by its shortest run)"
        print(f"  {PANEL_TITLES[morphology]}, {CONDITION_LABELS[cond]}: n={s['n']}, "
              f"{s['k']} steps, final {mean[-1]:.1f} deg{note}")
    return x_max


def report_milestones(groups, body):
    """Print and return the tilt at each milestone depth, pooled per morphology."""
    rows = []
    print(f"\n{YLABEL[body].replace(' (degrees)', '')} at fixed free-fall depth, pooled over "
          f"all trials of each morphology (nearest sample, full runs). Mean +/- population "
          f"SD (ddof=0, as in the paper); sample SD (ddof=1) in brackets:")
    for morphology in MORPHOLOGIES:
        paths = [p for (m, _), group in groups.items() if m == morphology for p in group]
        if not paths:
            continue
        for row in milestone_tilts(paths, body):
            flag = "" if row["covered"] else "  (not reached by every run)"
            print(f"  {PANEL_TITLES[morphology]:9s} n={row['n']}: {row['depth_m']:g} m "
                  f"({row['time_s']:.2f} s): {row['mean_deg']:.1f} +/- "
                  f"{row['sd_pop_deg']:.1f} deg [{row['sd_sample_deg']:.1f}]{flag}")
            rows.append({"morphology": morphology, "body": body, **row})
    return pd.DataFrame(rows)


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


def plot_fig4(data=DEFAULT_DATA, out=DEFAULT_OUT, steps=MAX_STEPS, body="mean", band="std",
              milestones_out=DEFAULT_MILESTONES):
    steps = min(steps, MAX_STEPS)
    groups = find_groups(data)
    morphologies = [m for m in MORPHOLOGIES if any(k[0] == m for k in groups)]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(morphologies), 1, sharex=True, squeeze=False,
                                 figsize=(3.5, 1.9 * len(morphologies) + 0.9))
        axes = axes[:, 0]
        x_max = 0.0
        for ax, morphology in zip(axes, morphologies):
            x_max = max(x_max, plot_panel(ax, groups, morphology, steps, body, band))

        ticks, labels, upper = time_ticks(x_max)
        depth_times = [fall_time(d) for d in MILESTONE_DEPTHS]
        for i, (ax, morphology) in enumerate(zip(axes, morphologies)):
            ax.set_ylim(*YLIM)
            ax.set_yticks(YTICKS)
            ax.set_ylabel(YLABEL[body])
            ax.set_xlim(0.0, upper)
            ax.tick_params(axis="both", which="both", direction="in", length=3.5, width=0.8)
            ax.set_title(PANEL_TITLES[morphology], fontweight="bold", fontstyle="italic")
            ax.text(-0.17, 1.04, f"({'abcdefgh'[i]})", transform=ax.transAxes,
                    fontweight="bold", fontsize=9, va="bottom")
            for depth, t in zip(MILESTONE_DEPTHS, depth_times):
                if t <= upper:
                    ax.axvline(t, color="#BBBBBB", linewidth=1.5, zorder=0)
        # Label the depth lines once, under the top panel.
        trans = blended_transform_factory(axes[0].transData, axes[0].transAxes)
        for depth, t in zip(MILESTONE_DEPTHS, depth_times):
            if t <= upper:
                axes[0].text(t, -0.03, f"({depth:g} m)", transform=trans, ha="center",
                             va="top", fontsize=7.5, color="#222222")
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(labels)
        axes[-1].set_xlabel("Time (s)\n(Free-fall distance (m))")

        # Shared legend, filled row by row: the roll sweep, then the pitch sweep.
        present = [c for c in CONDITIONS if any(k[1] == c for k in groups)]
        handles = [Line2D([], [], color=CONDITION_COLORS[c], linestyle=CONDITION_LINESTYLE[c],
                          linewidth=1.4, label=CONDITION_LABELS[c]) for c in present]
        ncol = 3
        rows = [handles[i:i + ncol] for i in range(0, len(handles), ncol)]
        column_major = [row[j] for j in range(ncol) for row in rows if j < len(row)]
        fig.tight_layout(h_pad=1.6)
        # Place the legend just under the x-axis label of the bottom panel.
        fig.canvas.draw()
        label_box = axes[-1].xaxis.label.get_window_extent()
        label_bottom = fig.transFigure.inverted().transform((0, label_box.y0))[1]
        fig.legend(handles=column_major, loc="upper center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, label_bottom - 0.01), handlelength=2.2,
                   columnspacing=1.0)

        milestones = report_milestones(groups, body)
        if milestones_out and not milestones.empty:
            Path(milestones_out).parent.mkdir(parents=True, exist_ok=True)
            milestones.to_csv(milestones_out, index=False)
            print(f"Saved {milestones_out}")
        save_figure(fig, out)
        plt.close(fig)
    return milestones


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="folder of trial CSVs, searched recursively "
                             f"(default: {DEFAULT_DATA.relative_to(ROOT)})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="figure path; written as SVG (text kept editable) unless the "
                             "name ends in .pdf, any other extension is replaced by .svg "
                             f"(default: {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--milestones", default=str(DEFAULT_MILESTONES),
                        help="CSV for the 1 m / 2 m tilt values; '' to skip "
                             f"(default: {DEFAULT_MILESTONES.relative_to(ROOT)})")
    parser.add_argument("-n", "--steps", type=int, default=MAX_STEPS,
                        help=f"control steps to plot, from release (default and max: {MAX_STEPS})")
    parser.add_argument("--body", choices=list(BODIES), default="mean",
                        help="tilt to draw: the per-run average of both bodies, or a "
                             "single body (default: %(default)s)")
    parser.add_argument("--band", choices=list(BANDS), default="std",
                        help="shaded band: +/-1 population SD, or a 95%% t-interval "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)
    print(f"Reading trials from {args.data}")
    return plot_fig4(args.data, args.out, args.steps, args.body, args.band, args.milestones)


if __name__ == "__main__":
    main()
