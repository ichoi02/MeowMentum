"""
Standalone legend for poster — covers both box plots and scatter plots.
Section 1 : Morphology (color)
Section 2 : Trial type  (marker shape, scatter only)
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# =============================================================================
# Parameters — keep in sync with scatter_plot_poster.py / box_plot_poster.py
# =============================================================================

MORPHOLOGY_COLORS = {
    'spine+tail': '#800020',   # maroon
    'spine-only': '#214937',   # dark green
}

MORPHOLOGY_LABELS = {
    'spine+tail': 'Spine + tail',
    'spine-only': 'Spine only',
}

SHAPE_MARKERS = {
    'r45':       ('o', 'r45'),
    'r90':       ('s', 'r90'),
    'r180':      ('^', 'r180'),
    'r180_p15':  ('P', 'r180_p15'),
    'r180_p-15': ('H', 'r180_p-15'),
}

FONT_FAMILY    = 'Helvetica'
FONT_SIZE      = 20
MARKER_SIZE    = 12
EDGE_COLOR     = '#333333'
EDGE_WIDTH     = 1.2

DPI            = 300
OUT_PATH       = './data_analysis/prelim/legend_poster.png'

# =============================================================================

plt.rcParams['font.family'] = FONT_FAMILY

morphology_handles = [
    mpatches.Patch(facecolor=color, edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH, label=MORPHOLOGY_LABELS[name])
    for name, color in MORPHOLOGY_COLORS.items()
]

shape_handles = [
    mlines.Line2D([], [], color=EDGE_COLOR, marker=marker, linestyle='None',
                  markersize=MARKER_SIZE, markerfacecolor='none',
                  markeredgewidth=EDGE_WIDTH, label=label)
    for marker, label in SHAPE_MARKERS.values()
]

all_handles = morphology_handles + shape_handles

fig, ax = plt.subplots(figsize=(1, 1))
ax.set_axis_off()

legend = ax.legend(
    handles=all_handles,
    loc='center',
    frameon=False,
    fontsize=FONT_SIZE,
    handlelength=1.5,
    handletextpad=0.6,
    labelspacing=0.5,
)

fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches=bbox)
print(f"Saved: {OUT_PATH}")
