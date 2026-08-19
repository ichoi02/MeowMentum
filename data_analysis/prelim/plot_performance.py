"""
Performance scatter plot.

X axis     : |initial roll angle| (deg)
Y axis     : |initial roll rate|  (deg/s)
Left half  : |final roll angle|   at 2.5 m  (Red heatmap - Left Colorbar)
Right half : |final roll rate|    at 2.5 m  (Blue heatmap - Right Colorbar)
Shape      : trial type (r45=circle / r90=square / r180=triangle)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.lines as mlines

from pdb import set_trace as st

MORPHOLOGY = 'spine+tail'


MODEL_BASED = '' # '_mbc'
REPORT_PATH = f'./data_analysis/prelim/report{MODEL_BASED}.csv'
OUT_PATH    = f'./data_analysis/prelim/performance_scatter{MODEL_BASED}_{MORPHOLOGY}.png'


report = pd.read_csv(REPORT_PATH)
report = report[report['morphology'] == MORPHOLOGY]

# ---------- 2. Custom Left/Right Path Generators ----------
def create_half_paths():
    paths = {}
    
    # 1. Circle (r45)
    t_left = np.linspace(np.pi / 2, 3 * np.pi / 2, 40)
    v_left_circ = list(zip(np.cos(t_left), np.sin(t_left))) + [(0, 0)]
    t_right = np.linspace(-np.pi / 2, np.pi / 2, 40)
    v_right_circ = list(zip(np.cos(t_right), np.sin(t_right))) + [(0, 0)]
    circ_codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 39 + [mpath.Path.CLOSEPOLY]
    paths['r45'] = {'left': mpath.Path(v_left_circ, circ_codes), 'right': mpath.Path(v_right_circ, circ_codes), 'edge': 'o'}

    # 2. Square (r90)
    v_left_sq = [(-1, -1), (0, -1), (0, 1), (-1, 1), (-1, -1)]
    v_right_sq = [(0, -1), (1, -1), (1, 1), (0, 1), (0, -1)]
    sq_codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*3 + [mpath.Path.CLOSEPOLY]
    paths['r90'] = {'left': mpath.Path(v_left_sq, sq_codes), 'right': mpath.Path(v_right_sq, sq_codes), 'edge': 's'}

    # 3. Triangle (r180)
    v_left_tri = [(-1, -1), (0, 1), (0, -1), (-1, -1)]
    v_right_tri = [(1, -1), (0, 1), (0, -1), (1, -1)]
    tri_codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    paths['r180'] = {'left': mpath.Path(v_left_tri, tri_codes), 'right': mpath.Path(v_right_tri, tri_codes), 'edge': '^'}

    # 4. Pentagon (r180_p15)
    angle_offset = np.radians(15)
    t_left_pent = np.linspace(np.pi / 2 + angle_offset, 2.5 * np.pi + angle_offset, 5, endpoint=False)
    v_left_pent = list(zip(np.cos(t_left_pent), np.sin(t_left_pent))) + [(0, 0)]
    t_right_pent = np.linspace(-np.pi / 2 + angle_offset, 1.5 * np.pi + angle_offset, 5, endpoint=False)
    v_right_pent = list(zip(np.cos(t_right_pent), np.sin(t_right_pent))) + [(0, 0)]
    pent_codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*4 + [mpath.Path.CLOSEPOLY]
    paths['r180_p15'] = {'left': mpath.Path(v_left_pent, pent_codes), 'right': mpath.Path(v_right_pent, pent_codes), 'edge': 'P'}

    # 5. Hexagon (r180_p-15)
    angle_offset = np.radians(-15)
    t_left_hex = np.linspace(np.pi / 2 + angle_offset, 2.5 * np.pi + angle_offset, 6, endpoint=False)
    v_left_hex = list(zip(np.cos(t_left_hex), np.sin(t_left_hex))) + [(0, 0)]
    t_right_hex = np.linspace(-np.pi / 2 + angle_offset, 1.5 * np.pi + angle_offset, 6, endpoint=False)
    v_right_hex = list(zip(np.cos(t_right_hex), np.sin(t_right_hex))) + [(0, 0)]
    hex_codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*5 + [mpath.Path.CLOSEPOLY]
    paths['r180_p-15'] = {'left': mpath.Path(v_left_hex, hex_codes), 'right': mpath.Path(v_right_hex, hex_codes), 'edge': 'H'}
    
    return paths

SHAPE_PATHS = create_half_paths()

# ---------- 3. Normalization and Colormaps ----------
cmap_roll = plt.cm.Reds
cmap_rate = plt.cm.Blues

max_init_roll = report['F_roll_initial'].abs().max()
if pd.isna(max_init_roll) or max_init_roll == 0:
    max_init_roll = 180

max_init_rate = report['F_rollrate_initial'].abs().max()
if pd.isna(max_init_rate) or max_init_rate == 0:
    max_init_rate = 100

norm_roll = mcolors.Normalize(vmin=0, vmax=max_init_roll)
norm_rate = mcolors.Normalize(vmin=0, vmax=max_init_rate)

# ---------- 4. Plot Initialization ----------
MARKER_SIZE = 700
fig, ax = plt.subplots(figsize=(14, 8))

texts = []

for _, row in report.iterrows():
    # Axes = performance (final state); ideal is bottom-left (0, 0)
    x = abs(row['F_roll_2p5m'])
    y = abs(row['F_rollrate_2p5m'])

    # Color halves = initial conditions
    c_roll = cmap_roll(norm_roll(abs(row['F_roll_initial'])))
    c_rate = cmap_rate(norm_rate(abs(row['F_rollrate_initial'])))
    trial = row['trial']

    # if trial not in SHAPE_PATHS:
    #     print(f"Warning: Trial '{trial}' not recognized. Skipping this data point.")
    #     continue

    left_path  = SHAPE_PATHS[trial]['left']
    right_path = SHAPE_PATHS[trial]['right']
    edge_marker = SHAPE_PATHS[trial]['edge']

    if x > 180:
        x -= 360

    ax.scatter(x, y, c=[c_roll], s=MARKER_SIZE, marker=left_path,  linewidths=0, zorder=4, alpha=0.9)
    ax.scatter(x, y, c=[c_rate], s=MARKER_SIZE, marker=right_path, linewidths=0, zorder=4, alpha=0.9)
    ax.scatter(x, y, c='none',   s=MARKER_SIZE, marker=edge_marker,
               edgecolors='#333333', linewidths=1.5, zorder=5)

    texts.append(ax.text(x, y, f"{trial}-{row['rep']}", fontsize=9, color='#222222', zorder=6))

# ---------- 5. Axes & Styling ----------
ax.set_xlabel('|Final roll angle| at 2.5 m (deg)',   fontsize=13, fontweight='medium')
ax.set_ylabel('|Final roll rate|  at 2.5 m (deg/s)', fontsize=13, fontweight='medium')
ax.set_title('Righting Performance  —  ideal point is (0, 0)',
             fontsize=16, fontweight='bold', pad=25)

# Star at the ideal (0, 0) point
ax.scatter(0, 0, marker='*', s=400, color='gold', edgecolors='#888800',
           linewidths=1, zorder=6, label='Ideal')

abs_x = report['F_roll_2p5m'].abs()
abs_y = report['F_rollrate_2p5m'].abs()
x_margin = abs_x.max() * 0.12
y_margin = abs_y.max() * 0.12
ax.set_xlim(left=-180, right=180)
ax.set_ylim(bottom=-y_margin, top=abs_y.max() + y_margin)

ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- 6. Split Colorbars ----------
sm_roll = plt.cm.ScalarMappable(cmap=cmap_roll, norm=norm_roll)
sm_rate = plt.cm.ScalarMappable(cmap=cmap_rate, norm=norm_rate)

cb_left = plt.colorbar(sm_roll, ax=ax, location='left', pad=0.10, fraction=0.04)
cb_left.set_label('Left Half (Red): |Initial roll angle| (deg)', fontsize=11)
cb_left.outline.set_visible(False)

cb_right = plt.colorbar(sm_rate, ax=ax, location='right', pad=0.04, fraction=0.04)
cb_right.set_label('Right Half (Blue): |Initial roll rate| (deg/s)', fontsize=11)
cb_right.outline.set_visible(False)

# Shape legend
shape_handles = [
    mlines.Line2D([], [], color='#333333', marker=SHAPE_PATHS[t]['edge'], 
                  linestyle='None', markersize=10, markerfacecolor='none', label=t)
    for t in ['r45', 'r90', 'r180']
]
ax.legend(handles=shape_handles, title='Trial Type', fontsize=11,
          title_fontsize=12, loc='upper left', framealpha=0.9, edgecolor='#E0E0E0')

# ---------- 7. Text Anti-Overlap Handling ----------
try:
    from adjustText import adjust_text
    adjust_text(texts, expand_points=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7))
except ImportError:
    for t in texts:
        t.set_position((t.get_position()[0] + 3, t.get_position()[1] + 3))

# ---------- 8. Output ----------
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f"Saved plot successfully to: {OUT_PATH}")
plt.show()