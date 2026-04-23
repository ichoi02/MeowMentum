import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdb import set_trace as st

MODEL_BASED = '' # '_mbc'
REPORT_PATH = f'./data_analysis/report{MODEL_BASED}.csv'
OUT_PATH    = f'./data_analysis/box_plot{MODEL_BASED}.png'

# --- Formatting Constants ---
GLOB_FONTSIZE = 12
GLOB_LABELSIZE = 10
FIGSIZE = (10, 6) # Slightly wider to accommodate grouped plots
FIG_DPI = 300

def wrap_360(angle):
    if angle > 180:
        return angle - 360
    elif angle < -180:
        return angle + 360
    else:
        return angle
    

# Load and process data
report = pd.read_csv(REPORT_PATH)
for col in ['F_roll_2p5m', 'B_roll_2p5m', 'F_pitch_2p5m', 'B_pitch_2p5m']:
    if col in report.columns:
        report[col] = report[col].apply(wrap_360)
        report[col] = report[col].abs()

# # NOTE adding this in as placeholder for the code
# _report = report.copy()
# _report['morphology'] = 'spine only'
# report = pd.concat([report, _report], ignore_index=True)


# --- Plotting in pure Matplotlib ---
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111)

# Get unique morphologies and trials
morphologies = report['morphology'].dropna().unique()
trials = report['trial'].dropna().unique()

# Define the major x-axis groups
major_groups = ['Total'] + list(trials)

# Set up colors for the morphologies
cmap = plt.cm.Set2(np.linspace(0, 1, max(len(morphologies), 8)))
morph_colors = {morph: cmap[i] for i, morph in enumerate(morphologies)}

# Calculate offsets for side-by-side grouping
n_morphs = len(morphologies)
width = 0.6 / n_morphs if n_morphs > 0 else 0.4
offsets = np.linspace(-0.3 + (width/2), 0.3 - (width/2), n_morphs) if n_morphs > 1 else [0]

# Iterate through Major Groups (Total -> 45 -> 90 -> 180)
for group_idx, group_name in enumerate(major_groups):
    
    # Iterate through Morphologies within each group
    for morph_idx, morph in enumerate(morphologies):
        
        # Filter data based on whether we are in 'Total' or a specific trial
        if group_name == 'Total':
            mask = (report['morphology'] == morph)
        else:
            mask = (report['trial'] == group_name) & (report['morphology'] == morph)
            
        y_data = report[mask]['F_roll_2p5m'].dropna().values
        
        if len(y_data) > 0:
            # Calculate actual position on x-axis
            pos = group_idx + offsets[morph_idx]
            
            # 1. Boxplot layer
            bplot = ax.boxplot(
                y_data, 
                positions=[pos], 
                widths=width * 0.8, # Slightly smaller than allocated width for spacing
                patch_artist=True, 
                showfliers=False,
                boxprops=dict(facecolor=morph_colors[morph], color='black', alpha=0.7),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1)
            )
            
            # 2. Stripplot layer (jittered scatter points)
            jitter = np.random.normal(0, width * 0.15, size=len(y_data))
            ax.scatter(
                np.repeat(pos, len(y_data)) + jitter, 
                y_data, 
                color='black', 
                alpha=0.4, 
                s=15, 
                zorder=3
            )

# Add vertical separators between major groups
for i in range(len(major_groups) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Build custom legend for morphologies
handles = [plt.Rectangle((0,0),1,1, facecolor=morph_colors[m], alpha=0.7, edgecolor='black') for m in morphologies]
ax.legend(handles, morphologies, title="Morphology", frameon=False, fontsize=GLOB_LABELSIZE)

# Apply spine modifications
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# Apply tick and label formatting
ax.set_xticks(range(len(major_groups)))
ax.set_xticklabels(major_groups, fontsize=GLOB_LABELSIZE)
ax.tick_params(axis='y', labelsize=GLOB_LABELSIZE)

ax.set_ylabel('Final Roll Angle at 2.5 m (deg)', fontsize=GLOB_FONTSIZE)
ax.set_xlabel('Trial Type', fontsize=GLOB_FONTSIZE)

# Final layout & save
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=FIG_DPI)
plt.show()

st()