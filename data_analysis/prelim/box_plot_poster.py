"""
Morphology comparison box plots (poster version).
Plot 1 : roll angle + pitch angle
Plot 2 : roll rate  + pitch rate
Color  : morphology type
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Parameters — edit here
# =============================================================================

# Morphology colors
MORPHOLOGY_COLORS = {
    'spine+tail': '#800020',   # maroon
    'spine-only': '#214937',   # dark green
}

# Data
MODEL_BASED = ''  # '_mbc'
REPORT_PATH = f'./data_analysis/prelim/report{MODEL_BASED}.csv'

# Per-plot configuration
PLOTS = [
    dict(
        metrics=[
            dict(col='F_roll_2p5m',  label='Roll',  wrap=True),
            dict(col='F_pitch_2p5m', label='Pitch', wrap=True),
        ],
        ylabel = 'Angle at impact (deg)',
        out    = f'./data_analysis/prelim/box_plot_poster_angles{MODEL_BASED}.png',
        ylim   = (-10, 90),
    ),
    dict(
        metrics=[
            dict(col='F_rollrate_2p5m',  label='Roll',  wrap=False),
            dict(col='F_pitchrate_2p5m', label='Pitch', wrap=False),
        ],
        ylabel = 'Angular velocity at impact (deg/s)',
        out    = f'./data_analysis/prelim/box_plot_poster_rates{MODEL_BASED}.png',
        ylim   = (-36, 320),
    ),
]

# Box appearance
BOX_WIDTH      = 0.10   # visual width of each box
BOX_OFFSET     = 0.15   # center-to-center distance between the two morphology boxes
BOX_ALPHA      = 0.85
BOX_EDGE_COLOR = '#333333'
BOX_EDGE_WIDTH = 1
MEDIAN_COLOR   = '#333333'
MEDIAN_WIDTH   = 1.5

# Jitter (individual data points overlaid)
JITTER_ALPHA = 0.45
JITTER_SIZE  = 20
JITTER_COLOR = '#333333'

# Figure
FIGURE_SIZE = (6, 7)
DPI         = 300

# X-axis padding (units on each side beyond the outermost group)
X_MARGIN      = 0.35
# Spacing between group centers (1.0 = default matplotlib spacing)
GROUP_SPACING = 0.55

# Font
FONT_FAMILY = 'Helvetica'
FONT_LABEL  = 25
FONT_TITLE  = 16
FONT_TICK   = 25

# =============================================================================

plt.rcParams['font.family'] = FONT_FAMILY

def wrap_360(v):
    if v > 180:
        return v - 360
    elif v < -180:
        return v + 360
    return v

report = pd.read_csv(REPORT_PATH)
report = report[report['morphology'].isin(MORPHOLOGY_COLORS.keys())]

morphologies = list(MORPHOLOGY_COLORS.keys())
n_morphs     = len(morphologies)
offsets      = np.linspace(-BOX_OFFSET / 2, BOX_OFFSET / 2, n_morphs) if n_morphs > 1 else [0]


def make_plot(cfg):
    metrics = cfg['metrics']
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for g_idx, metric in enumerate(metrics):
        data = report[metric['col']].copy()
        if metric['wrap']:
            data = data.apply(wrap_360)
        data = data.abs()

        for m_idx, morph in enumerate(morphologies):
            y = data[report['morphology'] == morph].dropna().values
            if len(y) == 0:
                continue

            pos   = g_idx * GROUP_SPACING + offsets[m_idx]
            color = MORPHOLOGY_COLORS[morph]

            ax.boxplot(
                y,
                positions=[pos],
                widths=BOX_WIDTH * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, color=BOX_EDGE_COLOR,
                              alpha=BOX_ALPHA, linewidth=BOX_EDGE_WIDTH),
                medianprops=dict(color=MEDIAN_COLOR, linewidth=MEDIAN_WIDTH),
                whiskerprops=dict(color=BOX_EDGE_COLOR, linewidth=BOX_EDGE_WIDTH),
                capprops=dict(color=BOX_EDGE_COLOR, linewidth=BOX_EDGE_WIDTH),
            )

            jitter = np.random.normal(0, BOX_WIDTH * 0.12, size=len(y))
            ax.scatter(
                np.repeat(pos, len(y)) + jitter, y,
                color=JITTER_COLOR, alpha=JITTER_ALPHA, s=JITTER_SIZE, zorder=3,
            )

    group_centers = [g * GROUP_SPACING for g in range(len(metrics))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([m['label'] for m in metrics], fontsize=FONT_TICK)
    ax.set_xlim(group_centers[0] - X_MARGIN, group_centers[-1] + X_MARGIN)
    if 'ylim' in cfg:
        ax.set_ylim(*cfg['ylim'])

    ax.tick_params(axis='y', labelsize=FONT_TICK)

    ax.set_xlabel('', fontsize=FONT_LABEL)
    ax.set_ylabel(cfg.get('ylabel', ''), fontsize=FONT_LABEL)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(cfg['out'], dpi=DPI, bbox_inches='tight')
    print(f"Saved: {cfg['out']}")


for cfg in PLOTS:
    make_plot(cfg)
