"""
Plot fluid temperature at each interface between CTES modules.

During CHARGING  flow goes:  inlet → M0 → M1 → ... → M13 → outlet
During DISCHARGING flow goes: inlet → M13 → M12 → ... → M0 → outlet

The plot shows T_fluid at z=L of each module (= the physical interface
between module i and module i+1), making the heat exchange along the
series chain easy to understand.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"

# --- Find latest simulation CSV ---
csv_files = glob.glob(os.path.join(project_root, "src", "data", "**",
                                    "simulation_results_*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError("No simulation results found. Run Part_1 first.")
csv_path = max(csv_files, key=os.path.getmtime)
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# --- Check columns ---
TF_COLS = [f"module_{i:02d}_Tf_zL_C" for i in range(14)]
TS_COLS = [f"module_{i:02d}_Ts_outlet_C" for i in range(14)]
missing = [c for c in TF_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing fluid columns: {missing}\nRe-run Part_1 to generate them.")

# --- Time axis ---
t0 = df.index[0]
hours = (pd.to_datetime(df.index) - pd.to_datetime(t0)).total_seconds() / 3600.0
hours = hours.values

# --- Colors ---
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#c0392b', '#2980b9',
    '#27ae60', '#d35400', '#8e44ad', '#16a085',
    '#e67e22', '#34495e',
]

# =====================================================================
# FIGURE 1: Fluid temperature at each module interface over time
# =====================================================================
fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                         height_ratios=[3, 3, 1],
                         sharex=True,
                         gridspec_kw={'hspace': 0.08})

ax_f, ax_s, ax_mode = axes

# Panel 1: Fluid T at z=L for each module
for i, col in enumerate(TF_COLS):
    label = f'M{i}→M{i+1}' if i < 13 else f'M{i} exit'
    ax_f.plot(hours, df[col].values, linewidth=1.2, color=colors[i], label=label)

# Also plot CTES inlet temperature
if 'ctes_inlet_temp_C' in df.columns:
    ax_f.plot(hours, df['ctes_inlet_temp_C'].values, linewidth=1.5,
              color='black', linestyle='--', alpha=0.6, label='CTES inlet (charge)')

ax_f.set_ylabel('Fluid temperature [°C]', fontsize=12)
ax_f.set_title('Fluid temperature at each module interface — full factory simulation',
               fontsize=13, fontweight='bold')
ax_f.legend(loc='upper left', ncol=4, fontsize=7, framealpha=0.9)
ax_f.grid(True, alpha=0.3, linestyle='--')

# Panel 2: Concrete T for comparison
for i, col in enumerate(TS_COLS):
    ax_s.plot(hours, df[col].values, linewidth=1.0, color=colors[i],
              alpha=0.8, label=f'Module {i}')
ax_s.set_ylabel('Concrete outlet T [°C]', fontsize=12)
ax_s.legend(loc='upper left', ncol=4, fontsize=7, framealpha=0.9)
ax_s.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Operation mode strip
if 'operation_mode' in df.columns:
    mode_colors_map = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c',
                       'D': '#3498db', '-': '#bdc3c7'}
    mode_labels = {'A': 'Charge only', 'B': 'Solar→factory',
                   'C': 'Solar+CTES→factory', 'D': 'CTES→factory', '-': 'Standby'}
    for idx_i in range(len(df)):
        mode = df['operation_mode'].iloc[idx_i]
        c = mode_colors_map.get(str(mode), '#bdc3c7')
        ax_mode.axvspan(hours[idx_i] - 0.08, hours[idx_i] + 0.08,
                        color=c, alpha=0.8, linewidth=0)
    patches = [Patch(facecolor=mode_colors_map[k], label=f'{k}: {mode_labels[k]}')
               for k in ['A', 'B', 'C', 'D', '-']
               if k in df['operation_mode'].astype(str).values]
    ax_mode.legend(handles=patches, loc='upper right', ncol=len(patches),
                   fontsize=7, framealpha=0.9)
    ax_mode.set_yticks([])
    ax_mode.set_ylabel('Mode', fontsize=11)

ax_mode.set_xlabel('Time [hours]', fontsize=12)

plt.tight_layout()
out1 = os.path.join(project_root, 'module_fluid_temperatures.png')
plt.savefig(out1, dpi=300, bbox_inches='tight')
print(f"Saved: {out1}")

# =====================================================================
# FIGURE 2: Snapshot "bar chart" of fluid T along the chain at key times
# =====================================================================
snapshot_hours = [0, 12, 24, 36, 48, 50, 60, 72, 96, 120, 144, 168]
snapshot_hours = [h for h in snapshot_hours if h <= hours[-1]]

n_snap = len(snapshot_hours)
fig2, axes2 = plt.subplots(1, n_snap, figsize=(2.5 * n_snap, 5), sharey=True)
if n_snap == 1:
    axes2 = [axes2]

positions = np.arange(14)

for ax_i, target_h in zip(axes2, snapshot_hours):
    idx = np.argmin(np.abs(hours - target_h))
    actual_h = hours[idx]
    mode = df['operation_mode'].iloc[idx] if 'operation_mode' in df.columns else '?'

    tf_vals = [df[TF_COLS[i]].iloc[idx] for i in range(14)]
    ts_vals = [df[TS_COLS[i]].iloc[idx] for i in range(14)]

    ax_i.barh(positions, tf_vals, height=0.4, color='steelblue',
              alpha=0.8, label='T_fluid')
    ax_i.barh(positions + 0.4, ts_vals, height=0.4, color='coral',
              alpha=0.8, label='T_concrete')

    ax_i.set_yticks(positions + 0.2)
    ax_i.set_yticklabels([f'M{i}' for i in range(14)], fontsize=8)
    ax_i.set_title(f't={actual_h:.0f}h\n({mode})', fontsize=9)
    ax_i.set_xlabel('T [°C]', fontsize=9)
    ax_i.grid(True, alpha=0.3, axis='x')

axes2[0].legend(fontsize=8, loc='lower right')
fig2.suptitle('Fluid vs Concrete temperature along the module chain — snapshots',
              fontsize=13, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(project_root, 'module_fluid_snapshots.png')
plt.savefig(out2, dpi=300, bbox_inches='tight')
print(f"Saved: {out2}")

plt.show()
