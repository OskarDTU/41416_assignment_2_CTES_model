"""
CTES 1D Axial Model — Jian et al. (2015)
==========================================
1D axial model with lumped capacitance in the radial direction.

Governing equations (Jian et al. 2015, eq. 3.47-3.50):

  Fluid:
    dT_f/dt + U * dT_f/dz = (h_E * P) / (rho_f * S_f * Cp_f) * (T_s - T_f)

  Solid (concrete):
    dT_s/dt = -(h_E * P) / (rho_s * S_s * Cp_s) * (T_s - T_f) - q_loss

  Effective heat transfer coefficient h_E (eq. 3.50):
    1/h_E = 1/h + (1/k_s) * [4ab^4*ln(b/a) - 3ab^4 + 4a^3*b^2 - a^5] / [4*(b^2-a^2)^2]

    where:
      h   = convective coefficient fluid -> pipe wall (Gnielinski)
      k_s = concrete thermal conductivity
      a   = r_pipe_out  (inner radius of concrete annulus)
      b   = r_eq        (outer equivalent radius)

References:
  - Jian et al. (2015), Applied Thermal Engineering, 75, 213-223
  - Buscemi et al. (2018), Energy Conversion and Management, 166, 719-734
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys, os

# Add project root to path so 'src' package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.constants import (
    rho_con, Cp_con, lambda_con,
    D_int, D_ext,
    lambda_iso, th_iso, epsilon_iso,
    V_con, n_modules,
    module_width, module_height, module_length,
    rho_f, Cp_f, mu_f, k_f,
    T_min, T_max, T_amb, v_wind, sigma,
    Q_vol_total, n_pipes,
)

# =============================================================================
# GEOMETRY — equivalent element approach (Buscemi et al., Fig. 10)
# =============================================================================
D_int_pipe  = D_int
D_ext_pipe  = D_ext
r_pipe_in   = D_int_pipe / 2       # [m] inner radius (fluid side)            = a'
r_pipe_out  = D_ext_pipe / 2       # [m] outer pipe radius = inner solid radius = a
L_module    = float(module_length) # [m]
V_con_total = float(V_con)         # [m3] total concrete volume
eps_iso     = epsilon_iso          # [-]

# n_pipes: used ONLY to derive r_eq and to scale energy to the full module.
# It does NOT appear in the ODE — all equations are written on one equivalent element.
n_pipes = 818

V_eq  = V_con_total / (n_modules * n_pipes)   # [m3] concrete volume per equivalent element
S_s   = V_eq / L_module                        # [m2] concrete annular cross-section
r_eq  = np.sqrt(S_s / np.pi + r_pipe_out**2)  # [m]  outer equivalent radius (= b in Jian eq. 3.50)

S_f   = np.pi * r_pipe_in**2                  # [m2] fluid cross-section (inner pipe)
P     = np.pi * D_ext_pipe                    # [m]  fluid-solid exchange perimeter

# External surfaces for heat loss (per module)
W_con     = module_width  - 2 * th_iso
H_con     = module_height - 2 * th_iso
S_con_ext = 2 * (W_con * L_module + H_con * L_module)
S_iso_ext = 2 * (module_width * L_module + module_height * L_module)

# Derived parameters from constants
Pr_f = mu_f * Cp_f / k_f

# Total oil flow rate and per-element mass flows
m_dot_total = rho_f * Q_vol_total         # [kg/s] total mass flow rate
m_dot_eq    = m_dot_total / n_pipes       # [kg/s] flow rate per equivalent element

if __name__ == "__main__":
    print("=" * 55)
    print("EQUIVALENT ELEMENT GEOMETRY")
    print("=" * 55)
    print(f"  V_eq  (concrete)     = {V_eq*1e6:.2f} cm3")
    print(f"  a = r_pipe_out       = {r_pipe_out*1e3:.2f} mm  (inner radius of solid)")
    print(f"  b = r_eq             = {r_eq*1e3:.2f} mm  (outer equivalent radius)")
    print(f"  S_s (solid section)  = {S_s*1e6:.1f} mm2")
    print(f"  S_f (fluid section)  = {S_f*1e6:.1f} mm2")
    print(f"  m_dot per element    = {m_dot_eq*1e3:.3f} g/s")

# =============================================================================
# EFFECTIVE HEAT TRANSFER COEFFICIENT h_E  (Jian et al. eq. 3.50)
#
#   1/h_E = 1/h + (1/k_s) * [4ab^4*ln(b/a) - 3ab^4 + 4a^3*b^2 - a^5] / [4*(b^2-a^2)^2]
# =============================================================================
def compute_h_E(m_dot_per_element):
    """
    Compute the effective heat transfer coefficient h_E.

    Parameters
    ----------
    m_dot_per_element : float  [kg/s]  mass flow rate in one equivalent element

    Returns
    -------
    h_E [W/(m2·K)], h [W/(m2·K)], Re [-]
    """
    v  = m_dot_per_element / (rho_f * S_f)
    Re = rho_f * v * D_int_pipe / mu_f

    # Gnielinski correlation (ref. [30] in Buscemi et al.)
    if Re < 2300:
        Nu = 3.66                                         # laminar fully-developed
    elif Re < 4000:
        f_t    = (0.790 * np.log(4000) - 1.64)**(-2)
        Nu_lam = 3.66
        Nu_tur = (f_t/8)*(4000-1000)*Pr_f / (1 + 12.7*np.sqrt(f_t/8)*(Pr_f**(2/3)-1))
        alpha  = (Re - 2300) / (4000 - 2300)
        Nu     = (1 - alpha)*Nu_lam + alpha*Nu_tur        # smooth transition
    else:
        f  = (0.790 * np.log(Re) - 1.64)**(-2)
        Nu = (f/8)*(Re-1000)*Pr_f / (1 + 12.7*np.sqrt(f/8)*(Pr_f**(2/3)-1))

    h = Nu * k_f / D_int_pipe  # [W/(m2·K)] convective coefficient

    # Radial conduction correction term (Jian et al. eq. 3.50)
    a, b = r_pipe_out, r_eq
    num  = 4*a*b**4*np.log(b/a) - 3*a*b**4 + 4*a**3*b**2 - a**5
    den  = 4 * (b**2 - a**2)**2
    h_E  = 1.0 / (1.0/h + (1.0/lambda_con)*(num/den))

    return h_E, h, Re

if __name__ == "__main__":
    h_E_ref, h_ref, Re_ref = compute_h_E(m_dot_eq)
    print(f"\n{'='*55}")
    print("HEAT TRANSFER COEFFICIENTS  (T_fluid = 200 C)")
    print(f"{'='*55}")
    print(f"  Re  = {Re_ref:.0f}  ({'turbulent' if Re_ref>4000 else 'transitional/laminar'})")
    print(f"  h   = {h_ref:.2f} W/(m2·K)  (convection only)")
    print(f"  h_E = {h_E_ref:.2f} W/(m2·K)  (convection + conduction correction)")
    print(f"  Reduction = {(1-h_E_ref/h_ref)*100:.1f}%  due to radial conduction resistance")

# =============================================================================
# HEAT LOSS MODEL  (Buscemi et al. eq. 14-19)
# =============================================================================
def T_sky(T_amb_C):
    """Apparent sky temperature [C]"""
    return 0.0552 * (T_amb_C + 273.15)**1.5 - 273.15

def h_ext_coeff(T_iso_C, T_amb_C, vw):
    """Global external heat transfer coefficient [W/(m2·K)]: convection + radiation"""
    h_conv  = 1.53 * vw + 1.43
    T_iso_K = T_iso_C + 273.15
    T_sky_K = T_sky(T_amb_C) + 273.15
    h_rad   = sigma * eps_iso * (T_iso_K**4 - T_sky_K**4) / (T_iso_C - T_amb_C + 1e-9)
    return h_conv + h_rad

def Q_loss_module(T_s_avg, T_amb_in, vw):
    """Total heat loss from one module to the environment [W]"""
    T_iso = T_amb_in + (T_s_avg - T_amb_in) * 0.1   # estimated insulation surface temperature
    he    = h_ext_coeff(T_iso, T_amb_in, vw)
    R_iso = th_iso / (lambda_iso * S_con_ext)
    R_ext = 1.0 / (he * S_iso_ext)
    return max((T_s_avg - T_amb_in) / (R_iso + R_ext), 0.0)

# =============================================================================
# 1D AXIAL ODE SYSTEM
#
# State vector layout (per axial node j):
#   y = [..., T_f(j), T_s(j), ...]
#   index 2*j     -> fluid temperature at node j
#   index 2*j + 1 -> solid temperature at node j
#
# Total size: 2 * N_z
# =============================================================================
N_z    = 30                                  # number of axial nodes
z_nodes = np.linspace(0, L_module, N_z)      # [m] axial positions
dz      = z_nodes[1] - z_nodes[0]            # [m] axial step size

def ctes_1d_rhs(t, y, T_in, m_dot, mode):
    """
    Right-hand side of the 1D ODE system.

    Parameters
    ----------
    T_in  : float  [C]     fluid inlet temperature
    m_dot : float  [kg/s]  total mass flow rate
    mode  : str           'charging', 'discharging', or 'storage'
    """
    dydt = np.zeros_like(y)

    # Storage mode: no fluid flow, only heat loss
    if mode == 'storage':
        for j in range(N_z):
            T_s_j  = y[2*j + 1]
            Ql     = Q_loss_module(T_s_j, T_amb, v_wind)
            ql_vol = Ql / (L_module * S_s * n_pipes)     # [W/m3]
            dydt[2*j]     = 0.0
            dydt[2*j + 1] = -ql_vol / (rho_con * Cp_con)
        return dydt

    # Compute h_E at average fluid temperature
    T_f_avg_all = np.mean([y[2*j] for j in range(N_z)])
    h_E, _, _   = compute_h_E(m_dot / n_pipes)

    # Fluid velocity in one equivalent element
    U = (m_dot / n_pipes) / (rho_f * S_f)

    # Source term coefficients
    coeff_f = h_E * P / (rho_f  * S_f  * Cp_f)    # [1/s] for fluid equation
    coeff_s = h_E * P / (rho_con * S_s * Cp_con)  # [1/s] for solid equation

    for j in range(N_z):
        T_f_j = y[2*j]
        T_s_j = y[2*j + 1]

        # ---- Fluid equation: upwind advection scheme ----
        if mode == 'charging':
            # charging: fluid enters at z=L (cold end), flows in -z direction
            T_f_next  = T_in if j == N_z-1 else y[2*(j+1)]
            advection = U * (T_f_next - T_f_j) / dz
        else:
            # discharging: fluid enters at z=0, flows in +z direction
            T_f_prev  = T_in if j == 0 else y[2*(j-1)]
            advection = -U * (T_f_j - T_f_prev) / dz

        dydt[2*j] = advection + coeff_f * (T_s_j - T_f_j)

        # ---- Solid equation ----
        Ql     = Q_loss_module(T_s_j, T_amb, v_wind)
        ql_vol = Ql / (L_module * S_s * n_pipes)
        dydt[2*j + 1] = -coeff_s * (T_s_j - T_f_j) - ql_vol / (rho_con * Cp_con)

    return dydt

# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================
def extract_profiles(y):
    """Extract T_f and T_s arrays from state vector."""
    T_f = np.array([y[2*j]   for j in range(N_z)])
    T_s = np.array([y[2*j+1] for j in range(N_z)])
    return T_f, T_s

def stored_energy(y, T_ref=T_min):
    """Energy stored in one module [MWh]."""
    _, T_s = extract_profiles(y)
    E = rho_con * Cp_con * S_s * n_pipes * np.trapz(T_s - T_ref, z_nodes)
    return E / 3.6e9

def SOC(y):
    """State of Charge [-]."""
    _, T_s = extract_profiles(y)
    return np.clip((np.mean(T_s) - T_min) / (T_max - T_min), 0.0, 1.0)

def T_outlet(y, mode):
    """Fluid outlet temperature [C]."""
    T_f, _ = extract_profiles(y)
    return T_f[0] if mode == 'charging' else T_f[-1]


def heat_loss_W(y):
    """Instantaneous CTES thermal loss to ambient [W] for all modules.

    This uses the same `Q_loss_module` formulation as the ODE source term and
    aggregates losses over axial nodes and modules.
    """
    _, T_s = extract_profiles(y)
    # Match discretization used in rhs: node-wise loss distributed over module length.
    loss_one_module_W = float(np.sum([Q_loss_module(float(T_s_j), T_amb, v_wind) for T_s_j in T_s]) * dz / L_module)
    return loss_one_module_W * n_modules


# ---------------------------------------------------------------------------
# Convenience helpers for integrating the CTES model from external code
# ---------------------------------------------------------------------------
def init_ctes_state(T_init=None):
    """Return an initial state vector `y` for the CTES model.

    - `T_init` if provided sets all temperatures to that value (C). Otherwise uses `T_min`.
    """
    T0 = T_min if T_init is None else float(T_init)
    return np.full(2 * N_z, T0)


def stored_energy_J(y, T_ref=None):
    """Return stored energy (J) for all modules relative to `T_ref` (defaults to T_min).

    This mirrors `stored_energy()` but returns Joules and includes all `n_modules`.
    """
    if T_ref is None:
        T_ref = T_min
    _, T_s = extract_profiles(y)
    E = rho_con * Cp_con * S_s * n_pipes * np.trapz(T_s - T_ref, z_nodes)
    return float(E * n_modules)


def fluid_energy_J(y, T_ref=None):
    """Return HTF sensible energy (J) inside CTES volume relative to `T_ref`."""
    if T_ref is None:
        T_ref = T_min
    T_f, _ = extract_profiles(y)
    E = rho_f * Cp_f * S_f * n_pipes * np.trapz(T_f - T_ref, z_nodes)
    return float(E * n_modules)


def step_ctes(y_curr, T_in_C, m_dot_total_kg_s, mode, dt_seconds, rtol=1e-5, atol=1e-7):
    """Integrate the CTES 1D model for `dt_seconds`.

    Parameters
    - y_curr: current state vector
    - T_in_C: inlet fluid temperature (C)
    - m_dot_total_kg_s: total mass flow rate (kg/s) through the storage (use 0 for storage mode)
    - mode: 'charging', 'discharging', or 'storage'
    - dt_seconds: timestep seconds

    Returns dict with keys: `y`, `T_out_C`, `energy_J`, `SOC`, plus
    model-side energy-balance diagnostics for debugging.
    """
    if mode not in ('charging', 'discharging', 'storage'):
        raise ValueError("mode must be 'charging', 'discharging', or 'storage'")

    # For charging/discharging, ensure a small positive mass flow is provided.
    if mode in ('charging', 'discharging') and (m_dot_total_kg_s is None or m_dot_total_kg_s <= 0):
        # fall back to storage-only step (no advection)
        mode_use = 'storage'
    else:
        mode_use = mode

    T_in_val = float(T_in_C) if (T_in_C is not None) else float(T_min)
    m_dot_val = float(m_dot_total_kg_s) if (m_dot_total_kg_s is not None) else 0.0
    dt_val = max(1.0, float(dt_seconds))
    E_prev_J = stored_energy_J(y_curr)
    E_fluid_prev_J = fluid_energy_J(y_curr)
    q_loss_prev_W = max(0.0, float(heat_loss_W(y_curr)))
    sol = solve_ivp(
        ctes_1d_rhs,
        [0.0, float(dt_seconds)],
        y_curr,
        args=(T_in_val, m_dot_val, mode_use),
        method='Radau',
        rtol=rtol,
        atol=atol,
    )
    y_new = sol.y[:, -1]

    E_J = stored_energy_J(y_new)
    soc = SOC(y_new)
    T_out = T_outlet(y_new, mode_use)
    q_loss_new_W = max(0.0, float(heat_loss_W(y_new)))
    E_fluid_new_J = fluid_energy_J(y_new)
    # Use solver trajectory to compute timestep-averaged powers, so closure compares
    # integrated storage changes against integrated boundary terms.
    if sol.t.size >= 2:
        t_path = sol.t
        y_path = sol.y.T
        q_loss_path_W = np.array([max(0.0, float(heat_loss_W(y_i))) for y_i in y_path], dtype=float)
        q_loss_avg_W = float(np.trapz(q_loss_path_W, t_path) / dt_val)
        if mode_use in ('charging', 'discharging'):
            q_fluid_boundary_path_W = np.array(
                [m_dot_val * Cp_f * (T_in_val - float(T_outlet(y_i, mode_use))) * float(n_modules) for y_i in y_path],
                dtype=float,
            )
            q_fluid_to_solid_total_W = float(np.trapz(q_fluid_boundary_path_W, t_path) / dt_val)
        else:
            q_fluid_to_solid_total_W = 0.0
    else:
        q_loss_avg_W = 0.5 * (q_loss_prev_W + q_loss_new_W)
        if mode_use in ('charging', 'discharging'):
            q_fluid_to_solid_total_W = m_dot_val * Cp_f * (T_in_val - float(T_out)) * float(n_modules)
        else:
            q_fluid_to_solid_total_W = 0.0
    q_fluid_to_solid_module_W = q_fluid_to_solid_total_W / float(n_modules)
    dE_W_fd = (float(E_J) - float(E_prev_J)) / dt_val
    dE_fluid_W_fd = (float(E_fluid_new_J) - float(E_fluid_prev_J)) / dt_val
    # Build power diagnostics from the ODE RHS along the integration path so
    # closure uses the same semi-discrete balance as the solver.
    if sol.t.size >= 2:
        t_path = sol.t
        y_path = sol.y.T
        dE_solid_rhs_path_W = []
        dE_fluid_rhs_path_W = []
        solid_factor = rho_con * Cp_con * S_s * float(n_pipes) * float(n_modules) * dz
        fluid_factor = rho_f * Cp_f * S_f * float(n_pipes) * float(n_modules) * dz
        for t_i, y_i in zip(t_path, y_path):
            rhs_i = ctes_1d_rhs(float(t_i), y_i, T_in_val, m_dot_val, mode_use)
            dE_fluid_rhs_path_W.append(float(fluid_factor * np.sum(rhs_i[0::2])))
            dE_solid_rhs_path_W.append(float(solid_factor * np.sum(rhs_i[1::2])))
        dE_W_rhs = float(np.trapz(np.array(dE_solid_rhs_path_W, dtype=float), t_path) / dt_val)
        dE_fluid_W_rhs = float(np.trapz(np.array(dE_fluid_rhs_path_W, dtype=float), t_path) / dt_val)
    else:
        dE_W_rhs = dE_W_fd
        dE_fluid_W_rhs = dE_fluid_W_fd

    # Robust net charge into concrete from solid-side first law.
    q_to_solid_corr_W = dE_W_fd + q_loss_avg_W
    # Fluid-vs-solid transfer mismatch diagnostic.
    closure_solid_corr_W = (q_fluid_to_solid_total_W - dE_fluid_W_fd) - q_to_solid_corr_W
    # First-law closure for combined (solid+fluid) CTES control volume.
    closure_combined_W = (dE_W_rhs + dE_fluid_W_rhs) - (q_fluid_to_solid_total_W - q_loss_avg_W)
    return {
        "y": y_new,
        "T_out_C": float(T_out),
        "energy_J": float(E_J),
        "SOC": float(soc),
        "diag_q_fluid_to_solid_module_W": float(q_fluid_to_solid_module_W),
        "diag_q_fluid_to_solid_total_W": float(q_fluid_to_solid_total_W),
        "diag_q_to_solid_corr_W": float(q_to_solid_corr_W),
        "diag_q_loss_avg_W": float(q_loss_avg_W),
        "diag_dE_W": float(dE_W_fd),
        "diag_dE_fluid_W": float(dE_fluid_W_fd),
        "diag_dE_rhs_W": float(dE_W_rhs),
        "diag_dE_fluid_rhs_W": float(dE_fluid_W_rhs),
        "diag_closure_solid_corr_W": float(closure_solid_corr_W),
        "diag_closure_combined_W": float(closure_combined_W),
        "diag_energy_prev_J": float(E_prev_J),
        "diag_energy_new_J": float(E_J),
        "diag_fluid_energy_prev_J": float(E_fluid_prev_J),
        "diag_fluid_energy_new_J": float(E_fluid_new_J),
    }

"""
# =============================================================================
# SIMULATION: weekly cycle
# =============================================================================
T_in_charging    = 280.0   # [C] oil from solar field
T_in_discharging = 136.0   # [C] return oil from pasta factory

def get_mode(t_hours):
    #Return (mode, T_in) as a function of hour of the week.
    day     = int(t_hours // 24)
    hour    = t_hours % 24
    weekend = day < 2
    sun     = 8.0 <= hour < 18.0
    if weekend:
        return ('charging', T_in_charging) if sun else ('storage', None)
    else:
        return ('charging', T_in_charging) if sun else ('discharging', T_in_discharging)

# Initial condition: everything at T_min
y0 = np.full(2 * N_z, T_min)

t_total_h = 7 * 24
t_eval_h  = np.arange(0, t_total_h + 0.25, 0.25)

SOC_arr   = np.zeros(len(t_eval_h))
E_arr     = np.zeros(len(t_eval_h))
T_out_arr = np.zeros(len(t_eval_h))

SOC_arr[0]   = SOC(y0)
E_arr[0]     = stored_energy(y0)
T_out_arr[0] = T_min

y_curr = y0.copy()

print("\nRunning simulation...")
for k in range(len(t_eval_h) - 1):
    t_h          = t_eval_h[k]
    mode, T_in   = get_mode(t_h)
    T_in_val     = T_in if T_in is not None else T_min

    sol = solve_ivp(
        ctes_1d_rhs,
        [t_h*3600, (t_h+0.25)*3600],
        y_curr,
        args=(T_in_val, m_dot_total, mode),
        method='Radau',
        rtol=1e-5, atol=1e-7,
    )
    y_curr = sol.y[:, -1]

    SOC_arr[k+1]   = SOC(y_curr)
    E_arr[k+1]     = stored_energy(y_curr)
    T_out_arr[k+1] = T_outlet(y_curr, mode)

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
day_labels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']

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
plt.suptitle('CTES 1D Axial Model (Jian et al.) — Weekly cycle (summer)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'ctes_1d_jian_results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {output_path}")
"""