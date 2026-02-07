#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import os, re

np.random.seed(42)

# -----------------------------
# Base and derived metrics
# -----------------------------
METRICS = ['max_rad_mm', 'max_bend_mm', 'max_bend_deg', 'max_lin_mm']
METRICS_1 = ['radial_expansion_ratio', 'bending_distance_ratio', 'bending_angle', 'expansion_ratio']
SUBPLOT_LABELS = ['A', 'C', 'D', 'E']
D0 = 50
L0 = 20

def load_and_process_dataset(path, label=None, default_thickness_ratio=None):
    g_path = os.path.join(path, 'g_metrics.csv')
    s_path = os.path.join(path, 'summary_results.csv')

    g_metrics = pd.read_csv(g_path)
    summary = pd.read_csv(s_path)

    # Extract bending number safely - handle both formats
    # Format 1: bending#_test# 
    # Format 2: word#_test# (e.g., bend_collapse2_test1)
    
    # For summary (video column)
    # First try: bending# format
    summary['bending_num'] = summary['video'].str.extract(r'bending(\d+)')[0]
    # If NaN, try: any_word# before _test or just word#
    mask = summary['bending_num'].isna()
    if mask.any():
        # Try pattern with _test
        summary.loc[mask, 'bending_num'] = summary.loc[mask, 'video'].str.extract(r'[a-zA-Z_]+(\d+)_test')[0]
    # If still NaN, try just extracting last number
    mask = summary['bending_num'].isna()
    if mask.any():
        summary.loc[mask, 'bending_num'] = summary.loc[mask, 'video'].str.extract(r'[a-zA-Z_]+(\d+)')[0]
    
    summary = summary.dropna(subset=['bending_num'])
    summary['bending_num'] = summary['bending_num'].astype(int)

    # For g_metrics (Prototype ID column)
    # First try: bending# format
    g_metrics['bending_num'] = g_metrics['Prototype ID'].str.extract(r'bending(\d+)')[0]
    # If NaN, try: any_word# before _test or just word#
    mask = g_metrics['bending_num'].isna()
    if mask.any():
        g_metrics.loc[mask, 'bending_num'] = g_metrics.loc[mask, 'Prototype ID'].str.extract(r'[a-zA-Z_]+(\d+)_test')[0]
    mask = g_metrics['bending_num'].isna()
    if mask.any():
        g_metrics.loc[mask, 'bending_num'] = g_metrics.loc[mask, 'Prototype ID'].str.extract(r'[a-zA-Z_]+(\d+)')[0]
    
    g_metrics = g_metrics.dropna(subset=['bending_num'])
    g_metrics['bending_num'] = g_metrics['bending_num'].astype(int)

    # Add thickness_ratio if not present
    if 'thickness_ratio' not in g_metrics.columns:
        if default_thickness_ratio is not None:
            g_metrics['thickness_ratio'] = default_thickness_ratio
        else:
            g_metrics['thickness_ratio'] = 1.0  # Default to 1

    # Merge
    merged = summary.merge(
        g_metrics[['bending_num', 'arc_length_ratio', 'thickness_ratio']],
        on='bending_num'
    )

    # Derived metrics
    merged['radial_expansion_ratio'] = (merged['max_rad_mm'] + D0) / D0
    merged['bending_distance_ratio'] = merged['max_bend_mm'] / (D0 / 2)
    merged['bending_angle'] = merged['max_bend_deg']
    merged['expansion_ratio'] = merged['max_lin_mm'] / L0

    # Label dataset (for legend)
    merged['dataset_label'] = label if label else os.path.basename(path)

    return merged

# Load dataset 1 with default thickness_ratio = 1
merged1 = load_and_process_dataset('1data/bending1', label='Dataset 1', default_thickness_ratio=1.0)
# Load dataset 2 (should have thickness_ratio in the data)
merged2 = load_and_process_dataset('1data/bendingfinal', label='Dataset 2')

merged_all = pd.concat([merged1, merged2], ignore_index=True, join='outer')

print(f"Merged1 rows: {len(merged1)}, Merged2 rows: {len(merged2)}")
print(f"Dataset 1 thickness_ratio: {merged1['thickness_ratio'].iloc[0] if len(merged1) > 0 else 'N/A'}")
print(f"Dataset 2 thickness_ratio: {merged2['thickness_ratio'].iloc[0] if len(merged2) > 0 else 'N/A'}")

plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

# Define more sophisticated color scheme
# Using colorblind-friendly palette with better contrast
dataset_colors = {}
color_palette = [ '#A23B72',"#021720",]  # Blue and Rose colors
marker_styles = ['o', 's']  # Circle and square markers

legend_entries_added = set()

for i, metric in enumerate(METRICS_1):
    ax = axes[i]
    
    # Store data for determining y-axis limits
    all_values = []
    
    # First pass: collect all data to determine axis limits
    for merged, _ in [(merged1, 'Dataset 1'), (merged2, 'Dataset 2')]:
        if metric in merged.columns:
            all_values.extend(merged[metric].values)
    
    # Calculate y-axis limits with space for legend
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        # Add 25% extra space at the top for legend
        if i ==1 or i == 2:
            y_max_extended = y_max + 0.45 * y_range
        else:
            y_max_extended = y_max + 0.05 * y_range
        y_min_extended = y_min - 0.05 * y_range
        ax.set_ylim(y_min_extended, y_max_extended)

    # Plot data
    for dataset_idx, (merged, dataset_name) in enumerate([(merged1, 'Dataset 1'), (merged2, 'Dataset 2')]):
        # Get thickness ratio for this dataset
        if 'thickness_ratio' in merged.columns and not merged['thickness_ratio'].isna().all():
            th_ratio = merged['thickness_ratio'].iloc[0]
        else:
            th_ratio = 1.0
        
        # Create legend label with better formatting
        legend_label = f"$th_{{\\mathrm{{max}}}}/th_{{\\mathrm{{min}}}}$ = {th_ratio:.2f}"
        
        # Assign color and marker based on dataset
        if th_ratio not in dataset_colors:
            dataset_colors[th_ratio] = color_palette[len(dataset_colors) % len(color_palette)]
        
        color = dataset_colors[th_ratio]
        marker = marker_styles[dataset_idx % len(marker_styles)]
        
        # Check if we should add legend entry
        legend_key = (th_ratio, i)
        
        for arc_ratio_idx, arc_ratio in enumerate(sorted(merged['arc_length_ratio'].unique())):
            vals = merged[merged['arc_length_ratio'] == arc_ratio][metric].values
            vals_sorted = np.sort(vals)
            
            x_jitter = np.random.normal(0, 0.00, len(vals_sorted))
            
            for j, val in enumerate(vals_sorted):
                # Only add legend on first point
                add_label = legend_key not in legend_entries_added and j == 0 and arc_ratio_idx == 0
                if add_label:
                    legend_entries_added.add(legend_key)
                
                ax.scatter(
                    arc_ratio + x_jitter[j],
                    val,
                    alpha=0.7,
                    s=90,
                    color=color,
                    marker=marker,
                    edgecolors='white',
                    linewidth=1.2,
                    zorder=2,
                    label=legend_label if add_label else None
                )

    # Axis formatting
    if metric == 'radial_expansion_ratio':
        y_label = 'Radial Expansion Ratio'
    elif metric == 'bending_distance_ratio':
        y_label = 'Bending Distance Ratio'
    elif metric == 'bending_angle':
        y_label = 'Bending Angle (deg)'
    elif metric == 'expansion_ratio':
        y_label = 'Expansion Ratio'
    else:
        y_label = metric.replace('_', ' ').title()

    ax.set_xlabel('Arc Length Ratio', fontsize=14, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=14, fontweight='medium')
    
    # Enhanced title with better positioning
    ax.set_title(f'{SUBPLOT_LABELS[i]}: {y_label}', 
                 fontsize=15, fontweight='bold', loc='left', pad=10)
    
    # Improved grid styling
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    # Legend always in top right with larger font
    legend = ax.legend(
        fontsize=13,
        loc='upper right',
        framealpha=0.95,
        edgecolor='gray',
        frameon=True,
        shadow=True,
        borderpad=0.8,
        handletextpad=0.5,
        columnspacing=1.0,
        markerscale=1.2
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Enhanced tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, length=5, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Thicker spines for better definition
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

r = 25
arc_ratios = np.array(sorted(merged_all['arc_length_ratio'].unique()))
b = ( arc_ratios**2 * r - 2 * np.sqrt(arc_ratios**2 * r**2 - r**2) ) / (arc_ratios**2 - 2)
print(arc_ratios**2 * r)
print(- 2 * np.sqrt(arc_ratios**2 * r**2 - r**2))
print(arc_ratios**2 - 2)
axes[1].plot(arc_ratios, b, '--', label=' ')

plt.tight_layout(pad=1)

save_path = os.path.join(os.getcwd(), 'bending_dataset.png')
plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
save_path = os.path.join(os.getcwd(), 'bending_dataset.svg')
plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
if os.path.exists(save_path):
    size = os.path.getsize(save_path) / 1024
    print(f"Saved {save_path} ({size:.1f} KB)")
plt.close(fig)