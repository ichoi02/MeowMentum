"""
Morphology comparison scatter plots — roll and pitch (poster version).
Generates one figure per axis pair. All parameters are at the top.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# =============================================================================
# Parameters — edit here
# =============================================================================

# Morphology colors
MORPHOLOGY_COLORS = {
    'spine+tail': '#800020',   # maroon
    'spine-only': "#214937",   # dark green
}

# Data
MODEL_BASED = ''  # '_mbc'
REPORT_PATH = f'./data_analysis/prelim/report{MODEL_BASED}.csv'

# Markers per trial type
SHAPE_MARKERS = {
    'r45':       'o',
    'r90':       's',
    'r180':      '^',
    'r180_p15':  'P',
    'r180_p-15': 'H',
}

# Marker appearance
MARKER_SIZE       = 300
MARKER_ALPHA      = 0.9
MARKER_EDGE_COLOR = '#333333'
MARKER_EDGE_WIDTH = 1

# Figure
FIGURE_SIZE = (7, 7)
DPI         = 300

# Axis margins (fraction of data range added on each side for auto limits)
AXIS_MARGIN = 0.08

# Font
FONT_FAMILY = 'Helvetica'
FONT_LABEL  = 25
FONT_TITLE  = 16
FONT_TICK   = 25

# Per-plot configuration
# xlim/ylim: explicit (lo, hi) tuple, or None for auto-symmetric from data
PLOTS = [
    dict(
        x_col  = 'F_roll_2p5m',
        y_col  = 'F_rollrate_2p5m',
        xlabel = 'Roll angle at impact (deg)',
        ylabel = 'Angular velocity at impact (deg/s)',
        title  = 'Righting Performance by Morphology — Roll',
        out    = f'./data_analysis/prelim/morphology_comparison_roll{MODEL_BASED}.png',
        xlim   = (-110, 110),
        ylim   = None,
    ),
    dict(
        x_col  = 'F_pitch_2p5m',
        y_col  = 'F_pitchrate_2p5m',
        xlabel = 'Pitch angle at impact (deg)',
        ylabel = 'Angular velocity at impact (deg/s)',
        title  = 'Righting Performance by Morphology — Pitch',
        out    = f'./data_analysis/prelim/morphology_comparison_pitch{MODEL_BASED}.png',
        xlim   = None,
        ylim   = None,
    ),
]

# =============================================================================

plt.rcParams['font.family'] = FONT_FAMILY

report = pd.read_csv(REPORT_PATH)
report = report[report['morphology'].isin(MORPHOLOGY_COLORS.keys())]


def make_plot(cfg):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for _, row in report.iterrows():
        x = row[cfg['x_col']]
        y = row[cfg['y_col']]
        color  = MORPHOLOGY_COLORS.get(row['morphology'], '#888888')
        marker = SHAPE_MARKERS.get(row['trial'], 'o')
        ax.scatter(x, y, c=[color], s=MARKER_SIZE, marker=marker,
                   edgecolors=MARKER_EDGE_COLOR, linewidths=MARKER_EDGE_WIDTH,
                   zorder=4, alpha=MARKER_ALPHA)

    # ax.scatter(0, 0, marker='*', s=400, color='gold',
    #            edgecolors='#888800', linewidths=1, zorder=6)

    # Axis limits
    if cfg['xlim'] is not None:
        ax.set_xlim(*cfg['xlim'])
    else:
        mx = report[cfg['x_col']].abs().max()
        pad = mx * AXIS_MARGIN
        ax.set_xlim(-mx - pad, mx + pad)

    if cfg['ylim'] is not None:
        ax.set_ylim(*cfg['ylim'])
    else:
        my = report[cfg['y_col']].abs().max()
        pad = my * AXIS_MARGIN
        ax.set_ylim(-my - pad, my + pad)

    ax.axhline(0, color='#aaaaaa', linewidth=0.8, zorder=1)
    ax.axvline(0, color='#aaaaaa', linewidth=0.8, zorder=1)
    # ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(cfg['xlabel'], fontsize=FONT_LABEL, fontweight='medium')
    ax.set_ylabel(cfg['ylabel'], fontsize=FONT_LABEL, fontweight='medium')
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    # ax.set_title(cfg['title'], fontsize=FONT_TITLE, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(cfg['out'], dpi=DPI, bbox_inches='tight')
    print(f"Saved: {cfg['out']}")


for cfg in PLOTS:
    make_plot(cfg)
