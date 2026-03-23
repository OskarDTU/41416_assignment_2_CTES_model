"""
Module outlet temperatures with REAL DNI solar profile.

Charges the CTES only when the sun produces useful power (DNI > 0);
at night the modules are in standby and lose heat through the insulation.
Uses a vectorised Forward-Euler integrator for speed.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"
sys.path.insert(0, project_root)

# Import only the constants/geometry we need — no ODE solver imports
from src.features.ctes_1d_jian import (
    n_modules, N_z, L_module, S_f, S_s, P,
    rho_f as _rho_f, Cp_f, rho_con, Cp_con,
    n_pipes, compute_h_E,
    T_amb, v_wind,
    th_iso, lambda_iso, S_con_ext, S_iso_ext, eps_iso, sigma,
)
from src.data.constants import rho_f
from src.features.solarpower import solar_collector_outlet_temperature

dz = L_module / (N_z - 1)

# Pre-compute heat-loss constants (avoid repeated Python function calls)
_T_sky_K = 0.0552 * (T_amb + 273.15) ** 1.5          # sky temp in K
_h_conv  = 1.53 * v_wind + 1.43
_R_iso   = th_iso / (lambda_iso * S_con_ext)
_loss_denom = L_module * S_s * n_pipes * rho_con * Cp_con


def _vectorised_step(T_f, T_s, T_in, m_dot, dt_sub, h_E, coeff_f, coeff_s, U):
    """
    One Forward-Euler sub-step for ALL 14 modules × 30 nodes.
    Fully numpy-vectorised — no Python loops over nodes.
    """
    # --- advection (charging: flow in +z direction) ---
    T_f_prev = np.empty_like(T_f)
    T_f_prev[:, 1:] = T_f[:, :-1]
    T_f_prev[0, 0] = T_in
    for m in range(1, n_modules):
        T_f_prev[m, 0] = T_f[m - 1, -1]

    advection = -U * (T_f - T_f_prev) / dz

    # --- interphase exchange ---
    delta = T_s - T_f
    dTf = advection + coeff_f * delta

    # --- vectorised heat loss (Buscemi) ---
    T_iso = T_amb + (T_s - T_amb) * 0.1
    T_iso_K = T_iso + 273.15
    h_rad = sigma * eps_iso * (T_iso_K**4 - _T_sky_K**4) / (T_iso - T_amb + 1e-9)
    he = _h_conv + h_rad
    R_ext = 1.0 / (he * S_iso_ext)
    Qloss = np.maximum((T_s - T_amb) / (_R_iso + R_ext), 0.0)
    loss_rate = Qloss / _loss_denom

    dTs = -coeff_s * delta - loss_rate

    return T_f + dt_sub * dTf, T_s + dt_sub * dTs


def _standby_step(T_f, T_s, dt_sub):
    """
    One Forward-Euler sub-step in STANDBY (no flow).
    Only heat loss from concrete; fluid equilibrates with concrete.
    """
    # --- vectorised heat loss (Buscemi) ---
    T_iso = T_amb + (T_s - T_amb) * 0.1
    T_iso_K = T_iso + 273.15
    h_rad = sigma * eps_iso * (T_iso_K**4 - _T_sky_K**4) / (T_iso - T_amb + 1e-9)
    he = _h_conv + h_rad
    R_ext = 1.0 / (he * S_iso_ext)
    Qloss = np.maximum((T_s - T_amb) / (_R_iso + R_ext), 0.0)
    loss_rate = Qloss / _loss_denom

    dTs = -loss_rate
    # In standby the fluid just sits inside → equilibrate towards concrete
    h_E_val_sb, _, _ = compute_h_E(1e-6)  # near-zero flow
    coeff_f_sb = h_E_val_sb * P / (_rho_f * S_f * Cp_f)
    delta = T_s - T_f
    dTf = coeff_f_sb * delta

    return T_f + dt_sub * dTf, T_s + dt_sub * dTs


# ---- Parameters ---------------------------------------------------------
T_INIT   = 130.0         # initial uniform temperature [°C] (SOC = 0%)
Q_VOL    = 0.021         # max collector volumetric flow rate [m³/s]
COLLECTOR_EFF = 0.47     # solar collector efficiency
COLLECTOR_AREA = 6602.0  # collector area [m²]

DT_MACRO = 600.0         # macro-step for recording [s] (= 10 min, matches DNI)

# ---- Load DNI data -------------------------------------------------------
dni_path = os.path.join(project_root, 'src', 'data', 'DNI_10m.csv')
dni_df = pd.read_csv(dni_path, index_col=0, parse_dates=True)
dni_df.index = pd.to_datetime(dni_df.index, utc=True).tz_localize(None)
dni_series = dni_df['dni_wm2'].astype(float)

n_macro = len(dni_series)
MAX_HOURS = n_macro * DT_MACRO / 3600.0

print("=" * 70)
print("Module outlets with REAL DNI solar profile")
print(f"  DNI data: {n_macro} timesteps ({MAX_HOURS:.1f} h)")
print(f"  T_init = {T_INIT} °C   Q_max = {Q_VOL} m³/s")
print(f"  Collector: {COLLECTOR_AREA} m², eff = {COLLECTOR_EFF}")
print("=" * 70)

# ---- Run ----------------------------------------------------------------
T_f = np.full((n_modules, N_z), T_INIT)
T_s = np.full((n_modules, N_z), T_INIT)

hours  = []
outlet = {m: [] for m in range(n_modules)}
modes  = []   # 'charge' or 'standby' for each recorded step

htf_in_temp_C = T_INIT  # initial HTF inlet temperature (evolves)

# Pre-compute standby sub-stepping
dt_sub_sb = DT_MACRO / 10  # 60 s sub-steps for standby (plenty stable)
N_SUB_SB  = 10

RECORD = 1   # record every macro step

for macro in range(n_macro):
    t_h = macro * DT_MACRO / 3600.0
    dni_val = float(dni_series.iloc[macro])

    # Determine if charging or standby
    if dni_val > 5.0:  # meaningful DNI threshold
        # Compute solar power and collector outlet temperature
        solar_power_avail = dni_val * COLLECTOR_EFF * COLLECTOR_AREA
        try:
            col_res = solar_collector_outlet_temperature(
                t_in=htf_in_temp_C,
                m_dot=Q_VOL,
                dni=dni_val,
                efficiency=COLLECTOR_EFF,
                area=COLLECTOR_AREA,
                volumetric=True,
                temp_unit="C",
            )
            T_col_out = col_res.get("t_out_C", np.nan)
            solar_power_W = col_res["power_W"]
        except Exception:
            T_col_out = np.nan
            solar_power_W = 0.0

        if not np.isnan(T_col_out) and solar_power_W > 0 and T_col_out > htf_in_temp_C + 1.0:
            # --- CHARGING ---
            M_DOT = Q_VOL * rho_f
            U_val = (M_DOT / n_pipes) / (_rho_f * S_f)
            dt_cfl = 0.8 * dz / U_val
            N_SUB = max(1, int(np.ceil(DT_MACRO / dt_cfl)))
            dt_sub = DT_MACRO / N_SUB

            h_E_val, _, _ = compute_h_E(M_DOT / n_pipes)
            coeff_f = h_E_val * P / (_rho_f * S_f * Cp_f)
            coeff_s = h_E_val * P / (rho_con * S_s * Cp_con)

            for _ in range(N_SUB):
                T_f, T_s = _vectorised_step(T_f, T_s, T_col_out, M_DOT,
                                            dt_sub, h_E_val, coeff_f, coeff_s, U_val)

            # Update HTF inlet temp: return from last module outlet
            htf_in_temp_C = float(T_f[-1, -1])
            mode = 'charge'
        else:
            # Collector doesn't produce useful power → standby
            for _ in range(N_SUB_SB):
                T_f, T_s = _standby_step(T_f, T_s, dt_sub_sb)
            mode = 'standby'
    else:
        # --- STANDBY (night) ---
        for _ in range(N_SUB_SB):
            T_f, T_s = _standby_step(T_f, T_s, dt_sub_sb)
        mode = 'standby'

    # Record
    if macro % RECORD == 0:
        hours.append(t_h)
        for m in range(n_modules):
            outlet[m].append(float(T_s[m, -1]))
        modes.append(mode)

    # progress every ~24 h
    if macro % 144 == 0:
        temps = [float(T_s[m, -1]) for m in range(n_modules)]
        print(f"  t={t_h:7.1f} h | {mode:7s} | M0={temps[0]:.1f}  M6={temps[6]:.1f}  M13={temps[13]:.1f} °C | DNI={dni_val:.0f}")

hours = np.array(hours)

# ---- Plot ---------------------------------------------------------------
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#c0392b', '#2980b9',
    '#27ae60', '#d35400', '#8e44ad', '#16a085',
    '#e67e22', '#34495e',
]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                               sharex=True, gridspec_kw={'hspace': 0.08})

# Top: module temperatures
for m in range(n_modules):
    ax.plot(hours, outlet[m], linewidth=1.2, color=colors[m],
            label=f'Module {m}')
ax.set_ylabel('Concrete outlet temperature [°C]', fontsize=12)
ax.set_title(
    f'Module outlet temperatures with real DNI profile\n'
    f'SOC(0)=0%,  Q_max={Q_VOL} m³/s,  {COLLECTOR_AREA} m² collector',
    fontsize=13, fontweight='bold')
ax.legend(loc='upper left', ncol=4, fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Bottom: DNI profile
dni_hours = np.arange(len(dni_series)) * DT_MACRO / 3600.0
ax2.fill_between(dni_hours, dni_series.values, alpha=0.5, color='orange', label='DNI')
ax2.set_xlabel('Time [hours]', fontsize=12)
ax2.set_ylabel('DNI [W/m²]', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_png = os.path.join(project_root, 'module_outlets_solar_profile.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_png}")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY  (final outlet T per module)")
print("=" * 70)
for m in range(n_modules):
    print(f"  Module {m:2d}:  {outlet[m][-1]:.1f} °C")

n_charge = sum(1 for m in modes if m == 'charge')
n_standby = sum(1 for m in modes if m == 'standby')
print(f"\n  Charging steps:  {n_charge}  ({n_charge * DT_MACRO / 3600:.1f} h)")
print(f"  Standby steps:   {n_standby}  ({n_standby * DT_MACRO / 3600:.1f} h)")

plt.show()
