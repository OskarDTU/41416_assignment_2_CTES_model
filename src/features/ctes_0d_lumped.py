"""
CTES 0D Lumped Model
====================
Simplified model: concrete temperature is spatially uniform,
it varies only in time. The fluid is modelled as quasi-steady
(no thermal inertia).

Governing equations:

  Solid (concrete):
    M_s * Cp_con * dT_s/dt = Q_exch - Q_loss

  Fluid (steady-state energy balance):
    m_dot * Cp_f * (T_in - T_out) = Q_exch
    T_f_avg = (T_in + T_out) / 2

  Fluid-solid heat exchange (whole module):
    Q_exch = h_E * A_exch * (T_f_avg - T_s)
    A_exch = P * L_module * n_pipes

  Heat loss to environment (Buscemi et al. eq. 14-19):
    Q_loss = (T_s - T_amb) / (th_iso/(lambda_iso*S_con_ext) + 1/(h_ext*S_iso_ext))

References:
  - Buscemi et al. (2018), Energy Conversion and Management, 166, 719-734
  - Jian et al. (2015), Applied Thermal Engineering, 75, 213-223
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.constants import (
    rho_con, Cp_con, lambda_con,
    D_int, D_ext,
    lambda_iso, th_iso, epsilon_iso,
    V_con, n_modules,
    module_width, module_height, module_length,
)

# =============================================================================
# GEOMETRY — equivalent element approach (Buscemi et al., Fig. 10)
# =============================================================================
D_int_pipe  = D_int
D_ext_pipe  = D_ext
r_pipe_in   = D_int_pipe / 2       # [m] inner radius (fluid side)         = a'
r_pipe_out  = D_ext_pipe / 2       # [m] outer radius of pipe = inner radius of concrete annulus = a
L_module    = float(module_length) # [m]
V_con_total = float(V_con)         # [m3] total concrete volume
eps_iso     = epsilon_iso          # [-]

# n_pipes: used only to derive r_eq and to scale energy to the full module.
# It does NOT appear in the ODE — all equations are written on the equivalent element.
n_pipes = 818

V_eq  = V_con_total / (n_modules * n_pipes)   # [m3] concrete volume per equivalent element
S_s   = V_eq / L_module                        # [m2] concrete annular cross-section
r_eq  = np.sqrt(S_s / np.pi + r_pipe_out**2)  # [m]  outer equivalent radius  (= b in Jian eq. 3.50)

S_f   = np.pi * r_pipe_in**2                  # [m2] fluid cross-section (inner pipe)
P     = np.pi * D_ext_pipe                    # [m]  fluid-solid exchange perimeter

# Mass and surfaces for one full MODULE (= n_pipes equivalent elements)
M_s_module    = rho_con * V_con_total / n_modules  # [kg]  concrete mass per module
A_exch_module = P * L_module * n_pipes              # [m2]  total exchange area per module

# External surfaces for heat loss (per module)
W_con     = module_width  - 2 * th_iso
H_con     = module_height - 2 * th_iso
S_con_ext = 2 * (W_con * L_module + H_con * L_module)
S_iso_ext = 2 * (module_width * L_module + module_height * L_module)

# =============================================================================
# FLUID PROPERTIES — Paratherm-NF thermal oil at ~200 C
# =============================================================================
rho_f = 780.0    # [kg/m3]
Cp_f  = 2200.0   # [J/(kg·K)]
mu_f  = 0.5e-3   # [Pa·s]
k_f   = 0.12     # [W/(m·K)]
Pr_f  = mu_f * Cp_f / k_f

# =============================================================================
# OPERATING CONDITIONS
# =============================================================================
T_min   = 136.0    # [C]  minimum operating temperature (return oil)
T_max   = 280.0    # [C]  maximum operating temperature (from solar field)
T_amb   = 30.0     # [C]  ambient temperature
v_wind  = 3.0      # [m/s]
sigma   = 5.670e-8 # [W/(m2·K4)]

# Total oil flow rate (Buscemi et al.)
Q_vol_total = 0.019                       # [m3/s]
m_dot_total = rho_f * Q_vol_total         # [kg/s]  total mass flow rate
m_dot_eq    = m_dot_total / n_pipes       # [kg/s]  flow rate per equivalent element

print("=" * 55)
print("EQUIVALENT ELEMENT GEOMETRY")
print("=" * 55)
print(f"  V_eq  (concrete)     = {V_eq*1e6:.2f} cm3")
print(f"  a = r_pipe_out       = {r_pipe_out*1e3:.2f} mm  (inner radius of solid)")
print(f"  b = r_eq             = {r_eq*1e3:.2f} mm  (outer equivalent radius)")
print(f"  S_s (solid section)  = {S_s*1e6:.1f} mm2")
print(f"  S_f (fluid section)  = {S_f*1e6:.1f} mm2")
print(f"  M_s per module       = {M_s_module/1000:.1f} ton")
print(f"  A_exch per module    = {A_exch_module:.1f} m2")
print(f"  m_dot per element    = {m_dot_eq*1e3:.3f} g/s")

# =============================================================================
# EFFECTIVE HEAT TRANSFER COEFFICIENT h_E  (Jian et al. eq. 3.50)
#
#   1/h_E = 1/h + (1/k_s) * [4ab^4*ln(b/a) - 3ab^4 + 4a^3*b^2 - a^5] / [4(b^2-a^2)^2]
#
#   h   = convective coefficient fluid -> pipe wall (Gnielinski)
#   k_s = concrete thermal conductivity
#   a   = r_pipe_out  (inner radius of concrete annulus)
#   b   = r_eq        (outer equivalent radius)
# =============================================================================
def compute_h_E(m_dot_per_element):
    v  = m_dot_per_element / (rho_f * S_f)
    Re = rho_f * v * D_int_pipe / mu_f

    # Nusselt number — Gnielinski correlation (ref. [30] in Buscemi et al.)
    if Re < 2300:
        Nu = 3.66                                    # laminar fully-developed
    elif Re < 4000:
        f_t    = (0.790 * np.log(4000) - 1.64)**(-2)
        Nu_lam = 3.66
        Nu_tur = (f_t/8)*(4000-1000)*Pr_f / (1 + 12.7*np.sqrt(f_t/8)*(Pr_f**(2/3)-1))
        alpha  = (Re - 2300) / (4000 - 2300)
        Nu     = (1 - alpha)*Nu_lam + alpha*Nu_tur   # smooth transition
    else:
        f  = (0.790 * np.log(Re) - 1.64)**(-2)
        Nu = (f/8)*(Re-1000)*Pr_f / (1 + 12.7*np.sqrt(f/8)*(Pr_f**(2/3)-1))  # Gnielinski

    h = Nu * k_f / D_int_pipe  # [W/(m2·K)]

    # Radial conduction correction term (Jian et al. eq. 3.50)
    a, b = r_pipe_out, r_eq
    num  = 4*a*b**4*np.log(b/a) - 3*a*b**4 + 4*a**3*b**2 - a**5
    den  = 4 * (b**2 - a**2)**2
    h_E  = 1.0 / (1.0/h + (1.0/lambda_con)*(num/den))

    return h_E, h, Re

h_E_ref, h_ref, Re_ref = compute_h_E(m_dot_eq)
print(f"\n{'='*55}")
print("HEAT TRANSFER COEFFICIENTS  (T_fluid = 200 C)")
print(f"{'='*55}")
print(f"  Re  = {Re_ref:.0f}  ({'turbulent' if Re_ref>4000 else 'transitional/laminar'})")
print(f"  h   = {h_ref:.2f} W/(m2·K)  (convection only)")
print(f"  h_E = {h_E_ref:.2f} W/(m2·K)  (convection + conduction correction)")

# =============================================================================
# HEAT LOSS MODEL  (Buscemi et al. eq. 14-19)
# =============================================================================
def T_sky(T_amb_C):
    """Apparent sky temperature [C]"""
    return 0.0552 * (T_amb_C + 273.15)**1.5 - 273.15

def h_ext_coeff(T_iso_C, T_amb_C, vw):
    """Global external heat transfer coefficient [W/(m2·K)]: convection + radiation"""
    h_conv = 1.53 * vw + 1.43
    T_iso_K = T_iso_C + 273.15
    T_sky_K = T_sky(T_amb_C) + 273.15
    h_rad   = sigma * eps_iso * (T_iso_K**4 - T_sky_K**4) / (T_iso_C - T_amb_C + 1e-9)
    return h_conv + h_rad

def Q_loss_module(T_s, T_amb_in, vw):
    """Total heat loss from one module to the environment [W]"""
    T_iso = T_amb_in + (T_s - T_amb_in) * 0.1   # estimated insulation surface temperature
    he    = h_ext_coeff(T_iso, T_amb_in, vw)
    R_iso = th_iso / (lambda_iso * S_con_ext)
    R_ext = 1.0 / (he * S_iso_ext)
    return max((T_s - T_amb_in) / (R_iso + R_ext), 0.0)

# =============================================================================
# 0D LUMPED ODE
#
# State vector: y = [T_s]   (average concrete temperature in one module)
#
# Fluid outlet temperature is derived algebraically at each time step
# by solving simultaneously:
#   (1) m_dot*Cp_f*(T_in - T_out) = Q_exch
#   (2) Q_exch = h_E*A*(T_f_avg - T_s),  with T_f_avg = (T_in + T_out)/2
# => T_out = (T_s + (NTU/2 - 1)*T_in) / (NTU/2 + 1)
# where NTU = h_E * A_exch / (m_dot * Cp_f)
# =============================================================================
def compute_Q_exch_and_T_out(T_s, T_in, m_dot):
    """
    Solve simultaneously for Q_exch and T_out.

    Equations:
      (1)  Q_exch = h_E * A * (T_f_avg - T_s),   T_f_avg = (T_in + T_out) / 2
      (2)  Q_exch = m_dot * Cp_f * (T_in - T_out)

    Substituting (2) into (1):
      Q_exch = h_E * A * (T_in - T_s) / (1 + NTU/2)
    where NTU = h_E * A / (m_dot * Cp_f)

    Then:  T_out = T_in - Q_exch / (m_dot * Cp_f)
    """
    h_E, _, _ = compute_h_E(m_dot / n_pipes)
    NTU    = h_E * A_exch_module / (m_dot * Cp_f)
    Q_exch = h_E * A_exch_module * (T_in - T_s) / (1.0 + NTU / 2.0)
    T_out  = T_in - Q_exch / (m_dot * Cp_f)
    return Q_exch, T_out

def ctes_0d_rhs(t, y, T_in, m_dot, mode):
    T_s = y[0]
    Ql  = Q_loss_module(T_s, T_amb, v_wind)

    if mode == 'storage':
        # No fluid flow — only heat loss to environment
        dTs = -Ql / (M_s_module * Cp_con)
        return [dTs]

    Q_exch, _ = compute_Q_exch_and_T_out(T_s, T_in, m_dot)
    dTs = (Q_exch - Ql) / (M_s_module * Cp_con)
    return [dTs]

# =============================================================================
# SIMULATION: weekly cycle
# =============================================================================
T_in_charging    = 280.0   # [C] oil from solar field
T_in_discharging = 136.0   # [C] return oil from pasta factory

def get_mode(t_hours):
    """Return (mode, T_in) as a function of hour of the week."""
    day    = int(t_hours // 24)
    hour   = t_hours % 24
    weekend = day < 2
    sun     = 8.0 <= hour < 18.0
    if weekend:
        return ('charging', T_in_charging) if sun else ('storage', None)
    else:
        return ('charging', T_in_charging) if sun else ('discharging', T_in_discharging)

t_total_h = 7 * 24
t_eval_h  = np.arange(0, t_total_h + 0.25, 0.25)

SOC_arr   = np.zeros(len(t_eval_h))
E_arr     = np.zeros(len(t_eval_h))
T_out_arr = np.zeros(len(t_eval_h))

y_curr = [T_min]

print("\nRunning simulation...")
for k in range(len(t_eval_h) - 1):
    t_h          = t_eval_h[k]
    mode, T_in   = get_mode(t_h)
    T_in_val     = T_in if T_in is not None else T_min

    sol = solve_ivp(
        ctes_0d_rhs,
        [t_h*3600, (t_h+0.25)*3600],
        y_curr,
        args=(T_in_val, m_dot_total, mode),
        method='Radau', rtol=1e-6, atol=1e-8,
    )
    y_curr = list(sol.y[:, -1])

    T_s = y_curr[0]
    SOC_arr[k+1]   = np.clip((T_s - T_min)/(T_max - T_min), 0, 1)
    E_arr[k+1]     = rho_con * Cp_con * (V_con_total/n_modules) * (T_s - T_min) / 3.6e9
    if mode != 'storage':
        _, T_out = compute_Q_exch_and_T_out(T_s, T_in_val, m_dot_total)
        T_out_arr[k+1] = T_out
    else:
        T_out_arr[k+1] = T_s

print("Done.")
print(f"\n{'='*55}")
print("RESULTS SUMMARY")
print(f"{'='*55}")
print(f"  Max SOC reached:           {SOC_arr.max()*100:.1f}%")
print(f"  Max energy (1 module):     {E_arr.max():.2f} MWh")
print(f"  Max energy (14 modules):   {E_arr.max()*n_modules:.2f} MWh")

# =============================================================================
# PLOTS
# =============================================================================
day_labels = ['Sat','Sun','Mon','Tue','Wed','Thu','Fri']

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

axes[0].plot(t_eval_h, SOC_arr*100, color='#2ecc71', lw=2)
axes[0].set_ylabel('SOC [%]')
axes[0].set_title('State of Charge')
axes[0].set_ylim(0, 110)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_eval_h, E_arr,             color='#3498db', lw=2,   label='1 module')
axes[1].plot(t_eval_h, E_arr * n_modules, color='#e74c3c', lw=1.5, ls='--',
             label=f'All {n_modules} modules')
axes[1].set_ylabel('Stored energy [MWh]')
axes[1].set_title('Energy stored in concrete')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_eval_h, T_out_arr, color='#e67e22', lw=2)
axes[2].axhline(T_min, color='b', ls='--', alpha=0.6, label=f'T_min = {T_min} C')
axes[2].axhline(T_max, color='r', ls='--', alpha=0.6, label=f'T_max = {T_max} C')
axes[2].set_ylabel('Fluid outlet temperature [C]')
axes[2].set_title('Fluid outlet temperature')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

for ax in axes:
    for d in range(8):
        ax.axvline(d*24, color='gray', ls=':', alpha=0.5)
    ax.set_xticks([d*24 + 12 for d in range(7)])
    ax.set_xticklabels(day_labels)

axes[-1].set_xlabel('Day of the week')
plt.suptitle('CTES 0D Lumped Model — Weekly cycle (summer)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'ctes_0d_lumped_results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {output_path}")
