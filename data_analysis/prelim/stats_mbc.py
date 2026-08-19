"""
RL vs MBC Controller Comparison
================================
Auxiliary analysis comparing RL and model-based controller (MBC) performance,
spine+tail morphology only, across all 5 trial conditions.

Stat approach mirrors the morphology ablation analysis:
  - Diagnostic check: does controller assignment systematically bias initial states?
  - Overall OLS regression controlling for condition + initial states
  - Per-condition t-tests with Bonferroni correction
  - Post-hoc power analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import TTestIndPower
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Parameters — edit here
# =============================================================================

CONTROLLER_COLORS = {
    'RL':  '#1A3A5C',   # dark navy
    'MBC': '#8B4513',   # saddle brown
}

REPORT_MBC_PATH = './data_analysis/prelim/report_mbc.csv'

# Bonferroni correction: 10 comparisons
# (3 roll conditions x 2 outcomes) + (2 pitch conditions x 2 outcomes)
N_COMPARISONS   = 10
ALPHA_CORRECTED = 0.05 / N_COMPARISONS   # 0.005

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
FIGSIZE      = (10, 5.5)
DPI          = 300
FONT_FAMILY  = 'Helvetica'
FONT_LABEL   = 13
FONT_TICK    = 11
FONT_SIG     = 11

GROUP_SPACING  = 0.65
X_MARGIN       = 0.40
X_MARGIN_SMALL = 0.25

Y_BOTTOM_FRAC  = 0.04
YLIM_ANGLES    = (-Y_BOTTOM_FRAC * 100, 100)
YLIM_RATES     = (-Y_BOTTOM_FRAC * 480, 480)

plt.rcParams['font.family'] = FONT_FAMILY

# =============================================================================
# Load & prepare
# =============================================================================

def angular_distance(angle):
    return np.abs(((angle + 180) % 360) - 180)

def wrap_angle(angle):
    return ((angle + 180) % 360) - 180

df = pd.read_csv(REPORT_MBC_PATH)
df = df[df['controller'].isin(CONTROLLER_COLORS.keys())].copy()

df['abs_roll_2p5m']      = angular_distance(df['F_roll_2p5m'])
df['abs_rollrate_2p5m']  = df['F_rollrate_2p5m'].abs()
df['abs_pitch_2p5m']     = angular_distance(df['F_pitch_2p5m'])
df['abs_pitchrate_2p5m'] = df['F_pitchrate_2p5m'].abs()

df['roll_initial']       = wrap_angle(df['F_roll_initial'])
df['pitch_initial']      = wrap_angle(df['F_pitch_initial'])
df['rollrate_initial']   = df['F_rollrate_initial']
df['pitchrate_initial']  = df['F_pitchrate_initial']

controllers = list(CONTROLLER_COLORS.keys())
offsets     = np.linspace(-BOX_OFFSET / 2, BOX_OFFSET / 2, len(controllers))

ROLL_CONDITION_ORDER   = ['r45', 'r90', 'r180']
ROLL_CONDITION_LABELS  = ['$45^\\circ$\nRoll', '$90^\\circ$\nRoll', '$180^\\circ$\nRoll']
PITCH_CONDITION_ORDER  = ['r180_p-15', 'r180_p15']
PITCH_CONDITION_LABELS = ['$180^\\circ$ Roll\n$-15^\\circ$ Pitch', '$180^\\circ$ Roll\n$+15^\\circ$ Pitch']

outcome_vars = [
    ('abs_roll_2p5m',      r'$|\phi|$ at impact (deg)'),
    ('abs_rollrate_2p5m',  r'$|\dot{\phi}|$ at impact (deg/s)'),
    ('abs_pitch_2p5m',     r'$|\theta|$ at impact (deg)'),
    ('abs_pitchrate_2p5m', r'$|\dot{\theta}|$ at impact (deg/s)'),
]

print(f"Loaded {len(df)} drops")
print(pd.crosstab(df['controller'], df['trial']))

# =============================================================================
# Step 1: Diagnostic check
# =============================================================================

print("\n" + "="*80)
print("STEP 1: DIAGNOSTIC CHECK")
print("Does controller assignment systematically bias initial states?")
print("="*80)

initial_vars = [
    ('roll_initial',      'Initial Roll (deg)'),
    ('pitch_initial',     'Initial Pitch (deg)'),
    ('rollrate_initial',  'Initial Roll Rate (deg/s)'),
    ('pitchrate_initial', 'Initial Pitch Rate (deg/s)'),
]

for var, label in initial_vars:
    model       = smf.ols(f"{var} ~ C(controller) + C(trial)", data=df).fit()
    anova_table = anova_lm(model, typ=2)
    f_val = anova_table.loc['C(controller)', 'F']
    p_val = anova_table.loc['C(controller)', 'PR(>F)']
    result = '✅ RANDOM' if p_val >= 0.05 else '⚠️  CONFOUNDED'
    print(f"  {label:30s}: p = {p_val:.4f} → {result}")

# =============================================================================
# Step 2: Overall OLS
# =============================================================================

print("\n" + "="*80)
print("STEP 2: OVERALL OLS REGRESSION")
print("="*80)

formula_base = ("C(controller) + C(trial) + roll_initial + pitch_initial "
                "+ rollrate_initial + pitchrate_initial")

overall_results = []
for var, label in outcome_vars:
    m    = smf.ols(f"{var} ~ {formula_base}", data=df).fit()
    coef = m.params['C(controller)[T.RL]']
    ci   = m.conf_int().loc['C(controller)[T.RL]']
    p    = m.pvalues['C(controller)[T.RL]']
    sig  = '✅' if p < 0.05 else ''
    print(f"  {label:40s}: coef={coef:+7.2f}, 95% CI=[{ci[0]:+7.2f},{ci[1]:+7.2f}], p={p:.4f} {sig}")
    overall_results.append({'label': label, 'coef': coef, 'ci_lo': ci[0], 'ci_hi': ci[1], 'p': p})

# =============================================================================
# Step 3: Per-condition t-tests
# =============================================================================

print("\n" + "="*80)
print("STEP 3: PER-CONDITION t-TESTS (Bonferroni-corrected)")
print(f"N_COMPARISONS = {N_COMPARISONS}, alpha_adj = {ALPHA_CORRECTED:.4f}")
print("="*80)

all_conditions = ROLL_CONDITION_ORDER + PITCH_CONDITION_ORDER

for var, label in outcome_vars:
    print(f"\n{label}")
    for cond in all_conditions:
        dfc = df[df['trial'] == cond]
        a   = dfc[dfc['controller'] == 'RL' ][var].dropna().values
        b   = dfc[dfc['controller'] == 'MBC'][var].dropna().values
        if len(a) < 2 or len(b) < 2:
            print(f"  {cond:15s}: insufficient data")
            continue
        _, p_t = stats.ttest_ind(a, b)
        _, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
        p_adj  = min(p_t * N_COMPARISONS, 1.0)
        sig    = '✅' if p_adj < 0.05 else ''
        print(f"  {cond:15s}: RL={a.mean():6.1f}±{a.std():5.1f}  MBC={b.mean():6.1f}±{b.std():5.1f}"
              f"  p_adj={p_adj:.3f} {sig}  (MWU p_raw={p_u:.3f})")

# =============================================================================
# Step 4: Post-hoc power analysis
# =============================================================================

print("\n" + "="*80)
print("STEP 4: POST-HOC POWER ANALYSIS")
print("="*80)

power_analysis = TTestIndPower()
n_overall    = df[df['controller'] == 'RL']['trial'].count() // len(all_conditions)
n_per_cond   = 5

mde_overall  = power_analysis.solve_power(nobs1=n_overall,  alpha=0.05, power=0.80)
mde_percond  = power_analysis.solve_power(nobs1=n_per_cond, alpha=0.05, power=0.80)
print(f"  MDE overall   (N={n_overall:2d}/group): d = {mde_overall:.3f}")
print(f"  MDE per-cond  (N={n_per_cond:2d}/group): d = {mde_percond:.3f}")

print("\n  Observed Cohen's d (pooled across all conditions):")
for var, label in outcome_vars:
    a  = df[df['controller'] == 'RL' ][var].dropna().values
    b  = df[df['controller'] == 'MBC'][var].dropna().values
    sp = np.sqrt((a.std()**2 + b.std()**2) / 2)
    d  = abs((a.mean() - b.mean()) / sp) if sp > 0 else 0
    pw = power_analysis.solve_power(effect_size=d, nobs1=len(a), alpha=0.05)
    print(f"  {label:40s}: d={d:.3f}, achieved power={pw:.3f}")

# =============================================================================
# Step 5: Plots
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
    ax.scatter(np.repeat(pos, len(y)) + jitter, y,
               color=JITTER_COLOR, alpha=JITTER_ALPHA, s=JITTER_SIZE, zorder=3)

def draw_sig(ax, x1, x2, y_top, p_raw, h_frac=0.03):
    p_adj  = min(p_raw * N_COMPARISONS, 1.0)
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    h      = yrange * h_frac
    col    = 'black' if p_adj < 0.05 else '#AAAAAA'
    marker = ('***' if p_adj < 0.001 else '**' if p_adj < 0.01 else
              '*'   if p_adj < 0.05  else 'ns')
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h, y_top + h, y_top], lw=1.0, color=col)
    ax.text((x1 + x2) / 2, y_top + h, marker,
            ha='center', va='bottom', fontsize=FONT_SIG,
            color=col, fontweight='bold' if p_adj < 0.05 else 'normal')

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

def sig_bar_y(ax, frac=0.87):
    lo, hi = ax.get_ylim()
    return lo + (hi - lo) * frac

PLOT_CONFIGS = [
    dict(
        metrics=[
            dict(col='abs_roll_2p5m',  label='Roll',
                 conditions=ROLL_CONDITION_ORDER,  condition_labels=ROLL_CONDITION_LABELS),
            dict(col='abs_pitch_2p5m', label='Pitch',
                 conditions=PITCH_CONDITION_ORDER, condition_labels=PITCH_CONDITION_LABELS),
        ],
        ylabel='Angle at impact (deg)',
        ylim=YLIM_ANGLES,
        out='./data_analysis/prelim/boxplots_mbc_angles.png',
    ),
    dict(
        metrics=[
            dict(col='abs_rollrate_2p5m',  label='Roll',
                 conditions=ROLL_CONDITION_ORDER,  condition_labels=ROLL_CONDITION_LABELS),
            dict(col='abs_pitchrate_2p5m', label='Pitch',
                 conditions=PITCH_CONDITION_ORDER, condition_labels=PITCH_CONDITION_LABELS),
        ],
        ylabel='Angular velocity at impact (deg/s)',
        ylim=YLIM_RATES,
        out='./data_analysis/prelim/boxplots_mbc_rates.png',
    ),
]

for cfg in PLOT_CONFIGS:
    metrics  = cfg['metrics']
    n0       = len(metrics[0]['conditions'])
    n1       = len(metrics[1]['conditions'])
    w_ratios = [1, n0, 0.25, 1, n1]

    fig      = plt.figure(figsize=FIGSIZE)
    gs       = GridSpec(1, len(w_ratios), figure=fig, width_ratios=w_ratios,
                        wspace=0.06, left=0.06, right=0.97, top=0.88, bottom=0.20)
    metric_cols = [(0, 1), (3, 4)]
    first_ax    = None

    for m_idx, (metric, (oc, pc)) in enumerate(zip(metrics, metric_cols)):
        var           = metric['col']
        cond_order    = metric['conditions']
        cond_labels   = metric['condition_labels']
        n_conds       = len(cond_order)
        group_centers = [g * GROUP_SPACING for g in range(n_conds)]

        if first_ax is None:
            ax_ov = fig.add_subplot(gs[0, oc])
            ax_pc = fig.add_subplot(gs[0, pc], sharey=ax_ov)
            first_ax = ax_ov
        else:
            ax_ov = fig.add_subplot(gs[0, oc], sharey=first_ax)
            ax_pc = fig.add_subplot(gs[0, pc], sharey=first_ax)

        # Overall panel
        df_metric = df[df['trial'].isin(cond_order)]
        for i, ctrl in enumerate(controllers):
            y = df_metric[df_metric['controller'] == ctrl][var].dropna().values
            draw_box(ax_ov, offsets[i], y, CONTROLLER_COLORS[ctrl])

        ax_ov.set_ylim(*cfg['ylim'])
        a = df_metric[df_metric['controller'] == 'RL' ][var].dropna().values
        b = df_metric[df_metric['controller'] == 'MBC'][var].dropna().values
        _, p_ov = stats.ttest_ind(a, b)
        draw_sig(ax_ov, offsets[0], offsets[-1], sig_bar_y(ax_ov), p_ov)

        ax_ov.set_xticks([0])
        ax_ov.set_xticklabels(['Overall'], fontsize=FONT_TICK)
        ax_ov.set_xlim(-X_MARGIN_SMALL - BOX_OFFSET / 2,
                        X_MARGIN_SMALL + BOX_OFFSET / 2)
        style_ax(ax_ov)
        ax_ov.set_title(metric['label'], fontsize=FONT_LABEL, fontweight='bold', pad=8)

        ax_ov.spines['right'].set_visible(True)
        ax_ov.spines['right'].set_linestyle((0, (4, 4)))
        ax_ov.spines['right'].set_linewidth(0.8)
        ax_ov.spines['right'].set_color('#888888')

        if m_idx == 0:
            ax_ov.set_ylabel(cfg['ylabel'], fontsize=FONT_LABEL)
        else:
            plt.setp(ax_ov.get_yticklabels(), visible=False)
            ax_ov.tick_params(axis='y', left=False)

        # Per-condition panel
        for g_idx, cond in enumerate(cond_order):
            dfc = df[df['trial'] == cond]
            for i, ctrl in enumerate(controllers):
                y = dfc[dfc['controller'] == ctrl][var].dropna().values
                if len(y) > 0:
                    draw_box(ax_pc, group_centers[g_idx] + offsets[i],
                             y, CONTROLLER_COLORS[ctrl])
            a = dfc[dfc['controller'] == 'RL' ][var].dropna().values
            b = dfc[dfc['controller'] == 'MBC'][var].dropna().values
            if len(a) > 1 and len(b) > 1:
                _, p_c = stats.ttest_ind(a, b)
                draw_sig(ax_pc,
                         group_centers[g_idx] + offsets[0],
                         group_centers[g_idx] + offsets[-1],
                         sig_bar_y(ax_pc), p_c)

        ax_pc.set_xticks(group_centers)
        ax_pc.set_xticklabels(cond_labels, fontsize=FONT_TICK - 1)
        ax_pc.set_xlim(group_centers[0] - X_MARGIN, group_centers[-1] + X_MARGIN)
        style_ax(ax_pc)
        plt.setp(ax_pc.get_yticklabels(), visible=False)
        ax_pc.tick_params(axis='y', left=False)
        ax_pc.spines['left'].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=CONTROLLER_COLORS[c], alpha=BOX_ALPHA,
                       edgecolor=BOX_EDGE_COLOR, linewidth=0.8, label=c)
        for c in controllers
    ]

    fig.text(
        0.5, 0.01,
        ('Significance markers: Bonferroni-corrected $t$-test '
         '($p_\\mathrm{adj} = p \\times 10$; $\\alpha = 0.05$; ns: not significant). '
         'spine+tail morphology only. $N = 5$ per group per condition; $N = 25$ per group overall.'),
        ha='center', fontsize=8, style='italic'
    )

    plt.savefig(cfg['out'], dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {cfg["out"]}')