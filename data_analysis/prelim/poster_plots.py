"""
Publication figures for morphology ablation study.
Uses the poster box plot style (maroon/dark green, clean axes).

Figure 1 : boxplots_angles.png — roll + pitch angle, overall pooled + per condition
Figure 2 : boxplots_rates.png  — roll + pitch rate,  overall pooled + per condition

Layout per figure:
  For each metric (roll / pitch):
    Left  : overall pooled box (all conditions combined)
    Right : per-condition boxes (5 conditions)
  A dashed divider separates overall from per-condition within each metric.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# =============================================================================
# Parameters — edit here
# =============================================================================

MORPHOLOGY_COLORS = {
    'spine+tail': '#800020',   # maroon
    'spine-only': '#214937',   # dark green
}

# Bonferroni correction: 10 comparisons
# (3 roll conditions x 2 outcomes) + (2 pitch conditions x 2 outcomes)
N_COMPARISONS   = 10
ALPHA_CORRECTED = 0.05 / N_COMPARISONS   # 0.005

REPORT_PATH = './data_analysis/prelim/report.csv'

# Box appearance
BOX_WIDTH      = 0.08
BOX_OFFSET     = 0.08
BOX_ALPHA      = 0.95
BOX_EDGE_COLOR = '#333333'
BOX_EDGE_WIDTH = 1
MEDIAN_COLOR   = '#333333'
MEDIAN_WIDTH   = 1.5

# Jitter
JITTER_ALPHA = 0.45
JITTER_SIZE  = 20
JITTER_COLOR = '#333333'

# Figure
FIGSIZE = (10, 5.5)
DPI          = 300
FONT_FAMILY  = 'Helvetica'
FONT_LABEL   = 13
FONT_TICK    = 11
FONT_SIG     = 11

GROUP_SPACING  = 0.65
X_MARGIN       = 0.40
X_MARGIN_SMALL = 0.25

# Y-axis limits — edit these directly
Y_BOTTOM_FRAC = 0.04   # zero sits 4% of the y-range above the x-axis

YLIM_ANGLES = (-Y_BOTTOM_FRAC * 100, 100)
YLIM_RATES  = (-Y_BOTTOM_FRAC * 480, 480)

plt.rcParams['font.family'] = FONT_FAMILY

# =============================================================================
# Load & prepare
# =============================================================================

def angular_distance(angle):
    return np.abs(((angle + 180) % 360) - 180)

def wrap_angle(angle):
    return ((angle + 180) % 360) - 180

report = pd.read_csv(REPORT_PATH)
report = report[report['morphology'].isin(MORPHOLOGY_COLORS.keys())].copy()

report['abs_roll_2p5m']      = angular_distance(report['F_roll_2p5m'])
report['abs_rollrate_2p5m']  = report['F_rollrate_2p5m'].abs()
report['abs_pitch_2p5m']     = angular_distance(report['F_pitch_2p5m'])
report['abs_pitchrate_2p5m'] = report['F_pitchrate_2p5m'].abs()

morphologies = list(MORPHOLOGY_COLORS.keys())
offsets      = np.linspace(-BOX_OFFSET / 2, BOX_OFFSET / 2, len(morphologies))

ROLL_CONDITION_ORDER  = ['r45', 'r90', 'r180']
ROLL_CONDITION_LABELS = [
    '$45^\\circ$\nRoll',
    '$90^\\circ$\nRoll',
    '$180^\\circ$\nRoll',
]

PITCH_CONDITION_ORDER  = ['r180_p-15', 'r180_p15']
PITCH_CONDITION_LABELS = [
    '$180^\\circ$ Roll\n$-15^\\circ$ Pitch',
    '$180^\\circ$ Roll\n$+15^\\circ$ Pitch',
]

# =============================================================================
# Helpers
# =============================================================================

def draw_box(ax, pos, y, color):
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

def draw_sig(ax, x1, x2, y_top, p, h_frac=0.03):
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    h      = yrange * h_frac
    col    = 'black' if p < ALPHA_CORRECTED else '#AAAAAA'
    marker = ('***' if p < ALPHA_CORRECTED / 10 else
              '**'  if p < ALPHA_CORRECTED / 5  else
              '*'   if p < ALPHA_CORRECTED       else 'ns')
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h, y_top + h, y_top],
            lw=1.0, color=col)
    ax.text((x1 + x2) / 2, y_top + h, marker,
            ha='center', va='bottom', fontsize=FONT_SIG,
            color=col, fontweight='bold' if p < ALPHA_CORRECTED else 'normal')

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

def sig_bar_y(ax, frac=0.87):
    lo, hi = ax.get_ylim()
    return lo + (hi - lo) * frac

# =============================================================================
# Plot configs
# =============================================================================

PLOT_CONFIGS = [
    dict(
        metrics=[
            dict(col='abs_roll_2p5m',  label='Roll',
                 conditions=ROLL_CONDITION_ORDER,  condition_labels=ROLL_CONDITION_LABELS),
            dict(col='abs_pitch_2p5m', label='Pitch $^*$',
                 conditions=PITCH_CONDITION_ORDER, condition_labels=PITCH_CONDITION_LABELS),
        ],
        ylabel='Angle at impact (deg)',
        ylim=YLIM_ANGLES,
        out='./data_analysis/prelim/boxplots_angles.png',
    ),
    dict(
        metrics=[
            dict(col='abs_rollrate_2p5m',  label='Roll',
                 conditions=ROLL_CONDITION_ORDER,  condition_labels=ROLL_CONDITION_LABELS),
            dict(col='abs_pitchrate_2p5m', label='Pitch $^*$',
                 conditions=PITCH_CONDITION_ORDER, condition_labels=PITCH_CONDITION_LABELS),
        ],
        ylabel='Angular velocity at impact (deg/s)',
        ylim=YLIM_RATES,
        out='./data_analysis/prelim/boxplots_rates.png',
    ),
]

# =============================================================================
# Generate figures
# =============================================================================

for cfg in PLOT_CONFIGS:
    metrics  = cfg['metrics']

    # Width ratios depend on number of conditions per metric
    # Layout: [ov_0, pc_0, gap, ov_1, pc_1]
    n0 = len(metrics[0]['conditions'])
    n1 = len(metrics[1]['conditions'])
    w_ratios = [1, n0, 0.25, 1, n1]

    fig = plt.figure(figsize=FIGSIZE)
    gs  = GridSpec(1, len(w_ratios), figure=fig, width_ratios=w_ratios,
                   wspace=0.06, left=0.06, right=0.97, top=0.88, bottom=0.20)

    metric_cols = [(0, 1), (3, 4)]
    first_ax    = None

    for m_idx, (metric, (oc, pc)) in enumerate(zip(metrics, metric_cols)):
        var              = metric['col']
        cond_order       = metric['conditions']
        cond_labels      = metric['condition_labels']
        n_conds          = len(cond_order)
        group_centers    = [g * GROUP_SPACING for g in range(n_conds)]

        # ── Create axes ────────────────────────────────────────────────────
        if first_ax is None:
            ax_ov  = fig.add_subplot(gs[0, oc])
            ax_pc  = fig.add_subplot(gs[0, pc], sharey=ax_ov)
            first_ax = ax_ov
        else:
            ax_ov  = fig.add_subplot(gs[0, oc], sharey=first_ax)
            ax_pc  = fig.add_subplot(gs[0, pc], sharey=first_ax)

        # ── Overall panel ──────────────────────────────────────────────────
        # Pool only the relevant conditions for this metric
        df_metric = report[report['trial'].isin(cond_order)]
        for i, morph in enumerate(morphologies):
            y = df_metric[df_metric['morphology'] == morph][var].dropna().values
            draw_box(ax_ov, offsets[i], y, MORPHOLOGY_COLORS[morph])

        ax_ov.set_ylim(*cfg['ylim'])
        a = df_metric[df_metric['morphology'] == 'spine+tail'][var].dropna().values
        b = df_metric[df_metric['morphology'] == 'spine-only'][var].dropna().values
        _, p_ov = stats.ttest_ind(a, b)
        draw_sig(ax_ov, offsets[0], offsets[-1], sig_bar_y(ax_ov), p_ov)

        ax_ov.set_xticks([0])
        ax_ov.set_xticklabels(['Overall'], fontsize=FONT_TICK)
        ax_ov.set_xlim(-X_MARGIN_SMALL - BOX_OFFSET / 2,
                        X_MARGIN_SMALL + BOX_OFFSET / 2)
        style_ax(ax_ov)
        # ax_ov.set_title(metric['label'], fontsize=FONT_LABEL,
        #                 fontweight='bold', pad=8)

        ax_ov.spines['right'].set_visible(True)
        ax_ov.spines['right'].set_linestyle((0, (4, 4)))
        ax_ov.spines['right'].set_linewidth(0.8)
        ax_ov.spines['right'].set_color('#888888')

        if m_idx == 0:
            ax_ov.set_ylabel(cfg['ylabel'], fontsize=FONT_LABEL)
        else:
            plt.setp(ax_ov.get_yticklabels(), visible=False)
            ax_ov.tick_params(axis='y', left=False)

        # ── Per-condition panel ────────────────────────────────────────────
        for g_idx, cond in enumerate(cond_order):
            df_cond = report[report['trial'] == cond]
            for i, morph in enumerate(morphologies):
                y = df_cond[df_cond['morphology'] == morph][var].dropna().values
                if len(y) > 0:
                    draw_box(ax_pc, group_centers[g_idx] + offsets[i],
                             y, MORPHOLOGY_COLORS[morph])
            a = df_cond[df_cond['morphology'] == 'spine+tail'][var].dropna().values
            b = df_cond[df_cond['morphology'] == 'spine-only'][var].dropna().values
            if len(a) > 1 and len(b) > 1:
                _, p_c = stats.ttest_ind(a, b)
                draw_sig(ax_pc,
                         group_centers[g_idx] + offsets[0],
                         group_centers[g_idx] + offsets[-1],
                         sig_bar_y(ax_pc), p_c)

        ax_pc.set_xticks(group_centers)
        ax_pc.set_xticklabels(cond_labels, fontsize=FONT_TICK - 1)
        ax_pc.set_xlim(group_centers[0]  - X_MARGIN,
                       group_centers[-1] + X_MARGIN)
        style_ax(ax_pc)
        plt.setp(ax_pc.get_yticklabels(), visible=False)
        ax_pc.tick_params(axis='y', left=False)
        ax_pc.spines['left'].set_visible(False)

    # legend on the first overall panel
    legend_patches = [
        mpatches.Patch(facecolor=MORPHOLOGY_COLORS[m], alpha=BOX_ALPHA,
                       edgecolor=BOX_EDGE_COLOR, linewidth=0.8, label=m)
        for m in morphologies
    ]
    # first_ax.legend(handles=legend_patches, fontsize=FONT_TICK,
    #                 loc='upper right', framealpha=0.9)

    fig.text(
        0.5, 0.01,
        ('Significance markers: Bonferroni-corrected $t$-test ($\\alpha_{adj} = 0.005$, $n = 10$ comparisons). '
         '$N = 5$ per group per condition; $N = 25$ per group overall. '
         '$^*$Pitch: interpret with caution (initial pitch confounded, $p = 0.010$).'),
        ha='center', fontsize=8, style='italic'
    )

    plt.savefig(cfg['out'], dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'Saved: {cfg["out"]}')