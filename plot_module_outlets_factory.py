"""
Plot the 14 CTES module outlet temperatures from the actual Part_1 simulation
(full factory logic: charging, discharging, standby).
Reads the latest simulation CSV which already contains the columns.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"

# --- Find latest simulation CSV ---
csv_files = glob.glob(os.path.join(project_root, "src", "data", "**", "simulation_results_*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError("No simulation results found. Run Part_1 first.")
csv_path = max(csv_files, key=os.path.getmtime)
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# --- Module columns ---
MODULE_COLS = [f"module_{i:02d}_Ts_outlet_C" for i in range(14)]
missing = [c for c in MODULE_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing module columns in CSV: {missing}")

# --- Time axis ---
t0 = df.index[0]
hours = (pd.to_datetime(df.index) - pd.to_datetime(t0)).total_seconds() / 3600.0
hours = hours.values  # numpy array

# --- Colors ---
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#c0392b', '#2980b9',
    '#27ae60', '#d35400', '#8e44ad', '#16a085',
    '#e67e22', '#34495e',
]

# --- Figure: 3 panels (module temps, DNI, operation mode) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10),
                                     height_ratios=[4, 1, 0.6],
                                     sharex=True,
                                     gridspec_kw={'hspace': 0.08})

# Panel 1: module outlet temperatures
for i, col in enumerate(MODULE_COLS):
    ax1.plot(hours, df[col].values, linewidth=1.2, color=colors[i],
             label=f'Module {i}')

ax1.set_ylabel('Concrete outlet temperature [°C]', fontsize=12)
ax1.set_title('14 CTES module outlet temperatures — full factory simulation (Part_1)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', ncol=4, fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Panel 2: DNI
if 'dni_W_m2' in df.columns:
    ax2.fill_between(hours, df['dni_W_m2'].values, alpha=0.5, color='orange', label='DNI')
    ax2.set_ylabel('DNI [W/m²]', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: operation mode color strip
if 'operation_mode' in df.columns:
    mode_colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c', 'D': '#3498db', '-': '#bdc3c7'}
    mode_labels = {'A': 'Charge only', 'B': 'Solar→factory', 'C': 'Solar+CTES→factory',
                   'D': 'CTES→factory', '-': 'Standby'}
    for idx_i in range(len(df)):
        mode = df['operation_mode'].iloc[idx_i]
        c = mode_colors.get(mode, '#bdc3c7')
        ax3.axvspan(hours[idx_i] - 0.08, hours[idx_i] + 0.08,
                    color=c, alpha=0.8, linewidth=0)
    # legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=mode_colors[k], label=f'{k}: {mode_labels[k]}')
               for k in ['A', 'B', 'C', 'D', '-'] if k in df['operation_mode'].values]
    ax3.legend(handles=patches, loc='upper right', ncol=len(patches), fontsize=7, framealpha=0.9)
    ax3.set_yticks([])
    ax3.set_ylabel('Mode', fontsize=11)

ax3.set_xlabel('Time [hours]', fontsize=12)

plt.tight_layout()
out_png = os.path.join(project_root, 'module_outlets_factory_simulation.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved: {out_png}")

# --- Summary ---
print("\n" + "=" * 70)
print("FINAL outlet T per module")
print("=" * 70)
for i, col in enumerate(MODULE_COLS):
    print(f"  Module {i:2d}:  {df[col].iloc[-1]:.1f} °C")

plt.show()
