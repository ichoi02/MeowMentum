"""Mean body tilt across a set of drops, one line per condition folder.

Point `--runs-in` at a directory and every CSV underneath it is loaded and grouped
by the CONDITION FOLDER that holds it -- the leaf directory the runs sit in, at
whatever depth -- giving one mean line per condition with a shaded band. So
`--runs-in raw/with_tail/roll_pitch` draws 3 lines, `--runs-in raw/with_tail`
draws 6, and `--runs-in raw` draws 12.

Lines are coloured by release condition and dashed by variant, so a figure that
spans both morphologies stays readable: colour says which release, linestyle says
tail or no tail. Groups are labelled from CONDITION_LABELS, prefixed with the
variant when a figure mixes them.

Each drop is reduced to ONE tilt series before any statistics are taken: the front
and rear body tilts are averaged together per run, per step, and the band is
computed across runs on that combined quantity. The band is therefore the spread
across drops, not the spread between the two bodies. `--body front` or `--body
rear` plots a single body instead; `--band ci` swaps the default +/-1 std for a
95% t-interval.

Runs are averaged BY CONTROL STEP, not by resampling onto a common clock: each
run's k-th row is the k-th policy step, so row k of every run is the same point in
the maneuver even though the 50 Hz loop jitters by a few ms. Interpolating onto a
uniform grid would smear that timing, so nothing here is interpolated. The x
position of step k is the mean of the runs' own timestamps at k. Groups are cut to
their shortest run, so every plotted sample is a real measurement from every run in
the group.

The x-axis is time since release by default, and each tick is annotated with the
free-fall distance at that instant, d = g t^2 / 2 -- the drop height a run of that
length needs. `--x-axis distance` plots against that distance instead, with the time
in brackets; step k then sits at d of its mean timestamp, so the two axes are the
same clock read in different units. Release is t[0]: controller.py starts logging
once the front accelerometer reads free fall, so the first sample is the drop, up to
one 50 Hz tick of detection latency.

Rear tilt is RECONSTRUCTED from the front IMU and the three spine encoders
(plot.compute_rear_body_dynamics), not read from B_Q*: the rear IMU reports
identity in every log under raw/, so its quaternion carries no orientation.

Usage:
    python raw/control_plot_batch.py                                  # with_tail/roll
    python raw/control_plot_batch.py --runs-in raw/no_tail/roll_pitch
    python raw/control_plot_batch.py --x-axis distance --spacing 0.25
    python raw/control_plot_batch.py --runs-in raw --out tilt_all.pdf
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

# Runnable as `python raw/control_plot_batch.py` from the repo root, which does not
# put raw/ on sys.path, and as `python control_plot_batch.py` from inside raw/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot import G, compute_rear_body_dynamics, compute_tilt_angle, fall_time


def find_project_root(start=Path(__file__).resolve().parent):
    for candidate in (start, *start.parents):
        if (candidate / 'raw/plot.py').exists():
            return candidate
    raise FileNotFoundError('Could not locate raw/plot.py')


ROOT = find_project_root()
RAW = ROOT / 'raw'
OUT = ROOT / 'plots' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_ROOT = RAW / 'with_tail' / 'roll'
FIGSIZE = (7.16, 4.0)          # IEEE double-column width

# The plotted window, in control steps. Every run under raw/ is 49 or 50 steps, so 31
# (~0.6 s) sits well inside what every drop covers in full, which is what lets runs be
# stacked by step index with no padding and no interpolation; a group holding a shorter
# run is reported.
MAX_STEPS = 33

# Depths at which tilt is reported alongside the figure. 2 m needs 0.64 s of fall,
# past the 31-step window, so these are read from the FULL runs -- the depth is a
# property of the drop, and the number must not move when --steps changes.
MILESTONE_DEPTHS = (1.0, 2.0)

# Tilt spans a fixed 0-180 deg so figures are comparable across conditions and runs.
YLIM = (0, 225)
YTICKS = np.arange(0, 226, 45)

# "mean" is the per-run average of the two bodies, formed before any averaging
# across runs -- see the module docstring.
BODIES = {'mean': 'tilt_mean', 'front': 'tilt_front', 'rear': 'tilt_rear'}
YLABEL = {'mean': 'mean body tilt (deg)', 'front': 'front body tilt (deg)',
          'rear': 'rear body tilt (deg)'}

# --x-axis options, with each one's default tick spacing in its own unit. Distance
# grows as t^2, so 0.2 m ticks would crowd the ~2.4 m a full window covers.
X_AXES = {
    'time':     {'xlabel': 'time (s), free-fall distance in brackets', 'spacing': 0.2},
    'distance': {'xlabel': 'free-fall distance (m), time in brackets', 'spacing': 0.5},
}

# Colour encodes the sweep and how far into it a condition sits: the roll sweep runs
# cool (light blue -> navy) and the roll+pitch sweep warm (amber -> red), each ordered
# light to dark with the swept angle. The two families separate at a glance even in the
# 12-line figure, warm/cool stays legible under red-green colourblindness, and the
# within-family luminance ramp survives a greyscale print. Variant is carried by
# linestyle, so both variables can be read off one figure.
ROLL_SWEEP = ['roll_045deg', 'roll_090deg', 'roll_180deg']
PITCH_SWEEP = ['roll_180deg_pitch_+015deg', 'roll_180deg_pitch_+030deg',
               'roll_180deg_pitch_+045deg']
CONDITIONS = ROLL_SWEEP + PITCH_SWEEP
CONDITION_LABELS = {
    'roll_045deg': '45° roll', 'roll_090deg': '90° roll', 'roll_180deg': '180° roll',
    'roll_180deg_pitch_+015deg': '180° roll, 15° pitch',
    'roll_180deg_pitch_+030deg': '180° roll, 30° pitch',
    'roll_180deg_pitch_+045deg': '180° roll, 45° pitch',
}
CONDITION_COLORS = {
    **dict(zip(ROLL_SWEEP,  ['#6BAED6', '#2171B5', '#08306B'])),   # cool
    **dict(zip(PITCH_SWEEP, ['#FEB24C', '#FC4E2A', '#B10026'])),   # warm
}

VARIANTS = ['no_tail', 'with_tail']
VARIANT_LABELS = {'no_tail': 'No tail', 'with_tail': 'With tail'}
VARIANT_LINESTYLE = {'no_tail': '--', 'with_tail': '-'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'text.usetex': False, 'axes.unicode_minus': False,
    'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.fontsize': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#222222',
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'svg.fonttype': 'none',
})


def mean_ci(values, confidence=0.95, axis=0):
    """Mean and a t-based confidence interval along `axis`."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    n = values.shape[axis]
    if n < 2:
        return mean, mean, mean
    half_width = stats.t.ppf((1 + confidence) / 2, n - 1) * stats.sem(values, axis=axis)
    return mean, mean - half_width, mean + half_width


def mean_std(values, axis=0):
    """Mean and +/-1 std along `axis`, in the same shape as `mean_ci`."""
    values = np.asarray(values, dtype=float)
    mean, sd = values.mean(axis=axis), values.std(axis=axis)
    return mean, mean - sd, mean + sd


BANDS = {'std': mean_std, 'ci': mean_ci}


VECTOR_FORMATS = {'.svg', '.pdf'}
DEFAULT_FORMAT = '.svg'


def save_figure(fig, filename):
    """Save as vector art. A bare filename lands in plots/paper_figures/.

    SVG by default, written with `svg.fonttype: none` so every label stays an
    editable <text> node in Illustrator rather than a set of outlines. An explicit
    .pdf suffix is honoured; anything else becomes .svg.
    """
    path = Path(filename)
    if not path.is_absolute() and path.parent == Path('.'):
        path = OUT / path
    if path.suffix.lower() not in VECTOR_FORMATS:
        path = path.with_suffix(DEFAULT_FORMAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'svg.fonttype': 'none', 'font.family': 'sans-serif',
                         'text.usetex': False, 'axes.unicode_minus': False}):
        fig.savefig(path, format=path.suffix[1:], bbox_inches='tight',
                    facecolor='white')
    print(f'Saved {path.relative_to(ROOT) if ROOT in path.parents else path}')
    return path


def clean_axis(ax):
    ax.grid(False)
    ax.tick_params(axis='both', which='both', direction='in', length=3.5, width=0.8)


def fall_distance(t):
    """Free-fall distance (m) at time t (s) after release."""
    return 0.5 * G * np.asarray(t, dtype=float) ** 2


def load_drop(path, steps):
    """One telemetry CSV -> the first `steps` control steps, clocked from release."""
    df = pd.read_csv(path).iloc[:steps]
    t = df['Time'].to_numpy(dtype=float)
    front_q = df[['F_Q0', 'F_Q1', 'F_Q2', 'F_Q3']].to_numpy(dtype=float)
    rear_q = compute_rear_body_dynamics(front_q, df['F_M1'], df['F_M2'], df['B_M2'])
    tilt_front = compute_tilt_angle(front_q)
    tilt_rear = compute_tilt_angle(rear_q)
    return {
        't': t - t[0],
        'tilt_front': tilt_front,
        'tilt_rear': tilt_rear,
        'tilt_mean': 0.5 * (tilt_front + tilt_rear),
    }


def find_groups(root):
    """{key: [csv, ...]}, one entry per condition folder found under `root`.

    A condition is any directory that directly contains runs, at any depth, so the
    same call works whether `root` is one condition folder, a variant, or all of
    raw/. Keys are paths relative to `root`, which keeps them unique -- the leaf
    name alone is not (roll_045deg exists under both variants). `label_for` turns
    them into the text that reaches the legend.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f'--runs-in must be a directory: {root}')
    dirs = sorted({p.parent for p in root.rglob('*.csv')})
    if not dirs:
        raise FileNotFoundError(f'no .csv runs found under {root}')
    keys = [str(d.relative_to(root)) for d in dirs]
    keys = [root.name if k == '.' else k for k in keys]
    return {k: sorted(d.glob('*.csv')) for k, d in zip(keys, dirs)}


def split_key(key):
    """Group key -> (variant or None, condition folder name)."""
    parts = Path(key).parts
    variant = next((p for p in parts if p in VARIANT_LABELS), None)
    return variant, parts[-1]


def sort_key(key):
    """Canonical order: variant, then release condition, as the paper lists them."""
    variant, cond = split_key(key)
    return (VARIANTS.index(variant) if variant in VARIANTS else len(VARIANTS),
            CONDITIONS.index(cond) if cond in CONDITIONS else len(CONDITIONS),
            key)


def label_for(key, show_variant):
    variant, cond = split_key(key)
    label = CONDITION_LABELS.get(cond, cond)
    if show_variant and variant:
        label = f'{VARIANT_LABELS[variant]}, {label}'
    return label


def summarize(paths, steps):
    """Per-step tilt stacks over a group of runs, cut to the shortest run."""
    runs = [load_drop(p, steps) for p in paths]
    k = min(len(r['t']) for r in runs)
    out = {'n': len(runs), 'k': k,
           't': np.mean([r['t'][:k] for r in runs], axis=0)}
    for body, key in BODIES.items():
        out[body] = np.array([r[key][:k] for r in runs])   # (n_runs, k)
    return out


def milestone_tilts(paths, body, depths=MILESTONE_DEPTHS):
    """Mean tilt at fixed free-fall depths, over the full runs.

    The sample nearest each target time is taken rather than an interpolated value,
    for the same reason the curves themselves are not resampled. A run that ends
    before a depth is reached is reported as uncovered rather than silently
    contributing its last sample.
    """
    runs = [load_drop(p, None) for p in paths]
    out = {}
    for depth in depths:
        target = fall_time(depth)
        values = [r[BODIES[body]][int(np.argmin(np.abs(r['t'] - target)))]
                  for r in runs]
        covered = all(r['t'][-1] >= target for r in runs)
        out[depth] = (float(np.mean(values)), float(np.std(values)), covered)
    return out


def report_milestones(groups, body, depths=MILESTONE_DEPTHS):
    """Print mean tilt at each milestone depth, pooled over every trial plotted."""
    paths = [p for group in groups.values() for p in group]
    print(f'\n{YLABEL[body]} at fixed free-fall depth, '
          f'{len(paths)} trials pooled (full runs, nearest sample):')
    for depth, (mean, sd, covered) in milestone_tilts(paths, body, depths).items():
        flag = '' if covered else '  (not reached by every run)'
        # Depth with its fall time in brackets, mirroring the x-axis tick labels.
        print(f'  {depth:g} m ({fall_time(depth):.2f} s): '
              f'{mean:.1f} +/- {sd:.1f} deg{flag}')


def axis_ticks(x_max, spacing, x_axis, snap=0.9):
    """Tick positions every `spacing` axis units, labelled with the other unit in brackets.

    A time axis reads "time (distance)", a distance axis "distance (time)". Returns
    the axis upper bound alongside them. The axis is stretched to the next whole tick
    when the data already reaches at least `snap` of the way there, so a window ending
    just short of a tick still gets it drawn -- 30 steps end at 0.584 s and would
    otherwise leave 0.6 outside the limits. Past that fraction the bound is the data
    itself, so a window ending just AFTER a tick (31 steps reach 0.604 s) is not
    padded most of the way to the following one.
    """
    last = float(np.floor(x_max / spacing + 1e-9) * spacing)
    upper = last + spacing if (x_max - last) >= snap * spacing else x_max
    ticks = np.arange(0.0, upper + 1e-9, spacing)
    # As many decimals as the spacing needs, so 0.25 m ticks do not print as 0.2.
    decimals = max(1, len(f'{spacing:g}'.partition('.')[2]))
    if x_axis == 'time':
        labels = [f'{t:.{decimals}f}\n({d:.2f} m)'
                  for t, d in zip(ticks, fall_distance(ticks))]
    else:
        labels = [f'{d:.{decimals}f}\n({fall_time(d):.2f} s)' for d in ticks]
    return ticks, labels, upper


def plot_groups(root, steps=MAX_STEPS, spacing=None, body='mean', band='std', out=None,
                x_axis='time'):
    steps = min(steps, MAX_STEPS)
    if spacing is None:
        spacing = X_AXES[x_axis]['spacing']
    groups = find_groups(root)
    keys = sorted(groups, key=sort_key)
    variants = {split_key(k)[0] for k in keys}
    mixed = len(variants - {None}) > 1
    stat = BANDS[band]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_max = 0.0
    for key in keys:
        s = summarize(groups[key], steps)
        # Distance is taken of the step's mean time rather than averaged over runs, so
        # the distance axis maps one-to-one onto the time axis and its labels invert.
        x = s['t'] if x_axis == 'time' else fall_distance(s['t'])
        x_max = max(x_max, x[-1])
        variant, cond = split_key(key)
        mean, lo, hi = stat(s[body])
        label = label_for(key, show_variant=mixed)
        ax.plot(x, mean, color=CONDITION_COLORS.get(cond),
                linestyle=VARIANT_LINESTYLE.get(variant, '-'), linewidth=1.4,
                label=None if mixed else f'{label} (n={s["n"]})')
        ax.fill_between(x, lo, hi, color=CONDITION_COLORS.get(cond),
                        alpha=0.15, linewidth=0)
        if s['k'] < steps:
            print(f'  note: {key} cut to {s["k"]} steps by its shortest run')
        print(f'  {label}: n={s["n"]}, {s["k"]} steps, '
              f'final {mean[-1]:.1f} deg')

    ax.set_ylabel(YLABEL[body])
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_xlabel(X_AXES[x_axis]['xlabel'])
    ticks, labels, upper = axis_ticks(x_max, spacing, x_axis)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.0, upper)
    clean_axis(ax)
    # Short unlabelled ticks at the depths report_milestones prints, so those numbers
    # can be found on the curves. Where one lands on a major tick (1 m and 2 m on the
    # default distance axis) matplotlib drops it as a duplicate.
    milestones = [d if x_axis == 'distance' else fall_time(d) for d in MILESTONE_DEPTHS]
    ax.set_xticks(milestones, minor=True)
    ax.tick_params(axis='x', which='minor', length=2.0)

    if mixed:
        # Colour and linestyle carry different variables, so they get their own keys.
        cond_keys = dict.fromkeys(split_key(k)[1] for k in keys)
        # ax.legend() replaces the axes' current legend, so the first one has to be
        # re-added as a standalone artist before the second call is made.
        cond_legend = ax.legend(
            handles=[Line2D([], [], color=CONDITION_COLORS.get(c), lw=1.4,
                            label=CONDITION_LABELS.get(c, c)) for c in cond_keys],
            loc='upper right', frameon=False)
        ax.add_artist(cond_legend)
        ax.legend(handles=[Line2D([], [], color='#333333', lw=1.4,
                                  linestyle=VARIANT_LINESTYLE[v],
                                  label=VARIANT_LABELS[v])
                           for v in VARIANTS if v in variants],
                  loc='lower left', frameon=False)
    else:
        ax.legend(frameon=False)

    report_milestones(groups, body)

    fig.tight_layout()
    if out:
        save_figure(fig, out)
    else:
        plt.show()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-in', default=str(DEFAULT_ROOT),
                        help='folder to draw runs from; every CSV underneath is used, '
                             'grouped by the condition folder holding it '
                             '(default: %(default)s)')
    parser.add_argument('-n', '--steps', type=int, default=MAX_STEPS,
                        help=f'control steps to plot, from release '
                             f'(default: %(default)s, max {MAX_STEPS})')
    parser.add_argument('--x-axis', choices=list(X_AXES), default='time',
                        help='plot against time since release or free-fall distance; '
                             'each tick carries the other in brackets '
                             '(default: %(default)s)')
    parser.add_argument('--spacing', type=float, default=None,
                        help=f'x-tick spacing in the --x-axis unit (default: '
                             f'{X_AXES["time"]["spacing"]:g} s for time, '
                             f'{X_AXES["distance"]["spacing"]:g} m for distance)')
    parser.add_argument('--body', choices=list(BODIES), default='mean',
                        help='tilt to draw: the per-run average of both bodies, or a '
                             'single body (default: %(default)s)')
    parser.add_argument('--band', choices=list(BANDS), default='std',
                        help='shaded band: +/-1 std, or a 95%% t-interval '
                             '(default: %(default)s)')
    parser.add_argument('--out', default=None,
                        help='save here instead of opening a window; written as SVG '
                             'unless the name ends in .pdf, and a bare filename lands '
                             'in plots/paper_figures/')
    args = parser.parse_args()
    print(f'Reading runs from {args.runs_in}')
    plot_groups(args.runs_in, args.steps, args.spacing, args.body, args.band, args.out,
                x_axis=args.x_axis)
