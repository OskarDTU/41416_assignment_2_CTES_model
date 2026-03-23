"""
Plot the 14 average concrete temperatures per CTES module over time.
Uses module_XX_Ts_avg_C columns from the latest Part_1 simulation CSV.
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"

# --- Find latest simulation CSV ---
csv_files = glob.glob(os.path.join(project_root, "src", "data", "**",
                                    "simulation_results_*.csv"), recursive=True)
csv_path = max(csv_files, key=os.path.getmtime)
print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

AVG_COLS = [f"module_{i:02d}_Ts_avg_C" for i in range(14)]
missing = [c for c in AVG_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}\nRe-run Part_1 to generate them.")

t0 = df.index[0]
hours = (pd.to_datetime(df.index) - pd.to_datetime(t0)).total_seconds().values / 3600.0

import matplotlib.cm as cm

# Orange (hot, Module 0) → Dark red (cold, Module 13)  — skip pale yellow
cmap = cm.get_cmap('YlOrRd')
colors = [cmap(0.25 + 0.75 * i / 13) for i in range(14)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10),
                                     height_ratios=[4, 1, 0.6],
                                     sharex=True,
                                     gridspec_kw={'hspace': 0.08})

for i, col in enumerate(AVG_COLS):
    ax1.plot(hours, df[col].values, linewidth=1.3, color=colors[i],
             label=f'Module {i}')

ax1.set_ylabel('Average concrete temperature [°C]', fontsize=12)
ax1.set_title('Average concrete temperature per CTES module — full factory simulation',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', ncol=4, fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# DNI panel
if 'dni_W_m2' in df.columns:
    ax2.fill_between(hours, df['dni_W_m2'].values, alpha=0.5, color='orange', label='DNI')
    ax2.set_ylabel('DNI [W/m²]', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

# Operation mode strip
if 'operation_mode' in df.columns:
    mc = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c', 'D': '#3498db', '-': '#bdc3c7'}
    ml = {'A': 'Charge only', 'B': 'Solar→factory', 'C': 'Solar+CTES→factory',
          'D': 'CTES→factory', '-': 'Standby'}
    for idx_i in range(len(df)):
        mode = str(df['operation_mode'].iloc[idx_i])
        ax3.axvspan(hours[idx_i] - 0.08, hours[idx_i] + 0.08,
                    color=mc.get(mode, '#bdc3c7'), alpha=0.8, linewidth=0)
    patches = [Patch(facecolor=mc[k], label=f'{k}: {ml[k]}')
               for k in ['A', 'B', 'C', 'D', '-'] if k in df['operation_mode'].astype(str).values]
    ax3.legend(handles=patches, loc='upper right', ncol=len(patches), fontsize=7, framealpha=0.9)
    ax3.set_yticks([])
    ax3.set_ylabel('Mode', fontsize=11)

ax3.set_xlabel('Time [hours]', fontsize=12)

plt.tight_layout()
out_png = os.path.join(project_root, 'module_avg_temperatures.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved: {out_png}")

print("\nFinal average T per module:")
for i, col in enumerate(AVG_COLS):
    print(f"  Module {i:2d}: {df[col].iloc[-1]:.1f} °C")

plt.show()
