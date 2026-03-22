"""
Continuous-charging experiment: charge the CTES at a constant flow rate
and inlet temperature (no factory, no discharge) until all 14 modules
reach the same outlet temperature.  Shows how the thermal front
propagates from Module 0 → Module 13 over time.

Uses a vectorised Forward-Euler integrator for speed (the standard
step_ctes / solve_ivp loop is far too slow for multi-day runs).
"""

import sys
import os
import numpy as np
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

# ---- Parameters ---------------------------------------------------------
T_INIT   = 130.0         # initial uniform temperature [°C] (SOC = 0%)
T_IN     = 310.0         # constant charging inlet temperature [°C]
Q_VOL    = 0.019         # volumetric flow rate [m³/s]  (full Buscemi design)
M_DOT    = Q_VOL * rho_f # mass flow rate [kg/s]
MAX_HOURS = 200          # safety limit [h]
CONVERGE  = 1.0          # all modules within this ΔT → stop [°C]

# CFL-stable sub-step:  dt_sub < dz / U
U_val = (M_DOT / n_pipes) / (_rho_f * S_f)
dt_cfl = 0.8 * dz / U_val          # [s] safe sub-step
DT_MACRO = 600.0                   # macro-step for recording [s]
N_SUB    = max(1, int(np.ceil(DT_MACRO / dt_cfl)))
dt_sub   = DT_MACRO / N_SUB

# Pre-compute h_E and coupling coefficients (constant for fixed m_dot)
h_E_val, _, _ = compute_h_E(M_DOT / n_pipes)
coeff_f = h_E_val * P / (_rho_f * S_f * Cp_f)
coeff_s = h_E_val * P / (rho_con * S_s * Cp_con)

print("=" * 70)
print("Continuous charging experiment")
print(f"  T_in  = {T_IN} °C    Q = {Q_VOL} m³/s  ({M_DOT:.1f} kg/s)")
print(f"  T_init= {T_INIT} °C   max = {MAX_HOURS} h")
print(f"  U = {U_val:.4f} m/s   dt_cfl = {dt_cfl:.1f} s   N_sub = {N_SUB}   h_E = {h_E_val:.2f} W/(m2K)")
print("=" * 70)

# ---- Run ----------------------------------------------------------------
T_f = np.full((n_modules, N_z), T_INIT)
T_s = np.full((n_modules, N_z), T_INIT)

hours  = []
outlet = {m: [] for m in range(n_modules)}

RECORD = 6   # record every RECORD macro-steps (= every hour)
n_macro = int(MAX_HOURS * 3600 / DT_MACRO)
converged = False

for macro in range(n_macro):
    t_h = macro * DT_MACRO / 3600.0

    # Record every RECORD macro steps
    if macro % RECORD == 0:
        hours.append(t_h)
        for m in range(n_modules):
            outlet[m].append(float(T_s[m, -1]))

        # convergence
        if len(hours) > 1:
            spread = max(outlet[m][-1] for m in range(n_modules)) \
                   - min(outlet[m][-1] for m in range(n_modules))
            if spread < CONVERGE:
                print(f"\n  Converged at t = {t_h:.1f} h  (spread = {spread:.2f} °C)")
                converged = True
                break

        # progress every 10 h
        if macro % (RECORD * 10) == 0:
            temps = [outlet[m][-1] for m in range(n_modules)]
            print(f"  t={t_h:7.1f} h | M0={temps[0]:.1f}  M6={temps[6]:.1f}  M13={temps[13]:.1f} °C")

    # sub-step
    for _ in range(N_SUB):
        T_f, T_s = _vectorised_step(T_f, T_s, T_IN, M_DOT, dt_sub, h_E_val, coeff_f, coeff_s, U_val)

if not converged:
    t_h = n_macro * DT_MACRO / 3600.0
    hours.append(t_h)
    for m in range(n_modules):
        outlet[m].append(float(T_s[m, -1]))
    print(f"\n  Did NOT converge within {MAX_HOURS} h")

hours = np.array(hours)

# ---- Plot ---------------------------------------------------------------
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#c0392b', '#2980b9',
    '#27ae60', '#d35400', '#8e44ad', '#16a085',
    '#e67e22', '#34495e',
]

fig, ax = plt.subplots(figsize=(14, 7))
for m in range(n_modules):
    ax.plot(hours, outlet[m], linewidth=1.8, color=colors[m],
            label=f'Module {m}')

ax.set_xlabel('Time [hours]', fontsize=12)
ax.set_ylabel('Concrete outlet temperature [°C]', fontsize=12)
ax.set_title(
    f'Continuous charging  –  T_in = {T_IN} °C,  Q = {Q_VOL} m³/s\n'
    'Solid outlet temperature (z = L) per module',
    fontsize=13, fontweight='bold')
ax.axhline(T_IN, color='grey', ls='--', lw=0.8, label=f'T_in = {T_IN} °C')
ax.legend(loc='lower right', ncol=3, fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_png = os.path.join(project_root, 'module_outlets_continuous_charge.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_png}")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY  (final outlet T per module)")
print("=" * 70)
for m in range(n_modules):
    print(f"  Module {m:2d}:  {outlet[m][-1]:.1f} °C")

plt.show()
