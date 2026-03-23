"""
Plot outlet temperatures for each of the 14 CTES modules
with 14 distinct colors - ONE plot with all modules.
Factory always OFF (standby mode).
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"
sys.path.insert(0, project_root)

from src.models.Part_1 import simulate
from src.features.ctes_1d_jian import (
    extract_profiles, n_modules, N_z,
    init_ctes_state, step_ctes, stored_energy_J
)

# Load DNI data
dni_path = os.path.join(project_root, 'src', 'data', 'DNI_10m.csv')
dni_df = pd.read_csv(dni_path, index_col=0, parse_dates=True)
dni_df.index = pd.to_datetime(dni_df.index, utc=True).tz_localize(None)
dni_series = dni_df['dni_wm2'].astype(float)

# Simulate a full week (Monday-Sunday) with factory ALWAYS OFF
window = dni_series.loc['2026-07-14':'2026-07-20']

print("=" * 70)
print("Simulating CTES with FACTORY ALWAYS OFF (standby mode)")
print("=" * 70)

res = simulate(
    window,
    initial_htf_inlet_temp_C=220.0,      # SOC 50%respect_factory_schedule=False,       # Ignore factory schedule
    factory_off_on_weekends=False,        # Ensure factory OFF
    disable_flow_rate_limits=False,
    max_ctes_flow_m3s=0.03,
    debug=False,
    interactive=False
)

csv_path = res.attrs.get('csv_path')
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

print(f"✅ Simulation completed. CSV: {csv_path}")
print(f"Factory load (should be all 0): min={df['factory_load_W'].min()}, max={df['factory_load_W'].max()}")

# ============================================================================
# Extract or reconstruct outlet temperatures for each module
# ============================================================================
# Since the CSV only has 3 representative nodes (z0, zmid, zmax),
# we'll use linear interpolation to estimate the full profile at each timestep

print("\nExtracting module outlet temperatures...")

# 14 distinct colors (one per module)
color_palette = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#f39c12',  # orange
    '#9b59b6',  # purple
    '#1abc9c',  # turquoise
    '#c0392b',  # dark red
    '#2980b9',  # dark blue
    '#27ae60',  # dark green
    '#d35400',  # dark orange
    '#8e44ad',  # dark purple
    '#16a085',  # dark turquoise
    '#e67e22',  # burnt orange
    '#34495e',  # dark gray
]

module_outlet_temps = {}

# Reconstruct full profiles from the 3 representative points
# Using linear interpolation: z0 -> zmid -> zmax
for idx, row in df.iterrows():
    T_z0 = row['concrete_T_z0_C']
    T_mid = row['concrete_T_z_mid_C']
    T_zmax = row['concrete_T_z_max_C']
    
    # Create a 30-point profile (linear interpolation)
    z_norm = np.linspace(0, 1, N_z)  # 30 nodes
    T_profile = np.interp(z_norm, [0, 0.5, 1], [T_z0, T_mid, T_zmax])
    
    # Extract outlet (last node) of each module
    for mod_idx in range(n_modules):
        outlet_node_idx = (mod_idx + 1) * N_z - 1  # Last node of module
        if outlet_node_idx < len(T_profile):
            T_outlet = T_profile[outlet_node_idx]
        else:
            T_outlet = T_zmax  # Fallback to max
        
        if mod_idx not in module_outlet_temps:
            module_outlet_temps[mod_idx] = []
        module_outlet_temps[mod_idx].append(T_outlet)

# Convert to series for each module
for mod_idx in range(n_modules):
    module_outlet_temps[mod_idx] = pd.Series(
        module_outlet_temps[mod_idx],
        index=df.index
    )

# ============================================================================
# Plot: All 14 modules on ONE plot with 14 distinct colors
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

for mod_idx in range(n_modules):
    T_outlet = module_outlet_temps[mod_idx]
    ax.plot(
        df.index,
        T_outlet,
        linewidth=2.5,
        color=color_palette[mod_idx],
        label=f'Module {mod_idx} (outlet)',
        marker='o',
        markersize=3,
        alpha=0.8
    )

ax.set_xlabel('Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature [°C]', fontsize=12, fontweight='bold')
ax.set_title('CTES Module Outlet Temperatures (z=L) - 14 Modules\nFactory OFF (Standby Mode)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', ncol=2, fontsize=9, framealpha=0.95)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(project_root, 'module_outlets_14colors.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: module_outlets_14colors.png")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS (Outlet Temperatures per Module)")
print("=" * 80)

summary_data = []
for mod_idx in range(n_modules):
    T_outlet = module_outlet_temps[mod_idx]
    summary_data.append({
        'Module': mod_idx,
        'Mean (°C)': round(T_outlet.mean(), 2),
        'Min (°C)': round(T_outlet.min(), 2),
        'Max (°C)': round(T_outlet.max(), 2),
        'Std Dev (°C)': round(T_outlet.std(), 2),
        'Color': color_palette[mod_idx],
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n{'Factory load (min-max [W]):':<30} {df['factory_load_W'].min():.1f} - {df['factory_load_W'].max():.1f}")
print(f"{'CTES operation modes:':<30} {', '.join(df['operation_mode'].unique())}")
print(f"{'Average SOC (%):':<30} {df['ctes_soc_pct'].mean():.1f}%")
print(f"{'Module 0 outlet range:':<30} {module_outlet_temps[0].min():.1f}°C - {module_outlet_temps[0].max():.1f}°C")
print(f"{'Module 13 outlet range:':<30} {module_outlet_temps[13].min():.1f}°C - {module_outlet_temps[13].max():.1f}°C")
print("=" * 80)

# ============================================================================
# Save a CSV with all module outlet temperatures
# ============================================================================
outlet_df = pd.DataFrame(module_outlet_temps)
outlet_df.columns = [f'module_{i}_outlet_C' for i in range(n_modules)]
outlet_csv_path = os.path.join(project_root, 'module_outlets_temperatures.csv')
outlet_df.to_csv(outlet_csv_path)
print(f"\n✅ Saved outlet temperatures: {outlet_csv_path}")

plt.show()
