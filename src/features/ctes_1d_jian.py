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
            T_f    = fluid temperature [C]
            T_s    = solid (concrete) temperature [C]

            U      = axial fluid velocity in the pipe [m/s]
            P      = fluid-solid exchange perimeter [m]

            rho_f  = HTF density [kg/m3]
            S_f    = HTF flow cross-sectional area [m2]
            Cp_f   = HTF specific heat capacity [J/(kg K)]

            rho_s  = concrete density [kg/m3]
            S_s    = concrete equivalent cross-sectional area [m2]
            Cp_s   = concrete specific heat capacity [J/(kg K)]

            q_loss = volumetric heat-loss source term in the solid equation [K/s equivalent]

            h_E    = effective fluid-solid heat transfer coefficient [W/(m2 K)]
            h      = convective coefficient fluid -> pipe wall (Gnielinski) [W/(m2 K)]
            k_s    = concrete thermal conductivity [W/(m K)]

            a      = r_pipe_out (inner radius of concrete annulus) [m]
            b      = r_eq       (outer equivalent radius) [m]
            ln()   = natural logarithm

State vector layout:
  The ODE state vector `y` stores fluid and solid temperatures for every axial
  node in an interleaved pattern:

    index:  0        1        2        3        4        5    ...
    value: T_f(0)  T_s(0)  T_f(1)  T_s(1)  T_f(2)  T_s(2)  ...

  For node j:
    y[2*j]     = T_f(j)  — fluid temperature at axial node j  [C]
    y[2*j + 1] = T_s(j)  — solid temperature at axial node j  [C]

  A single module has size 2*N_z. The full multi-module state is the N_z blocks
  concatenated in physical flow order (module 0 first).

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
# Pipe inner diameter used by fluid domain equations.
D_int_pipe  = D_int
# Pipe outer diameter used for fluid-solid exchange perimeter.
D_ext_pipe  = D_ext
# Fluid-side pipe radius (cross-section for mass/energy transport).
r_pipe_in   = D_int_pipe / 2       # [m] inner radius (fluid side)            = a'
# Pipe outer radius; this is also the inner radius of the concrete annulus.
r_pipe_out  = D_ext_pipe / 2       # [m] outer pipe radius = inner solid radius = a
# Active module length along the axial (z) direction.
L_module    = float(module_length) # [m]
# Total concrete volume across the whole CTES system.
V_con_total = float(V_con)         # [m3] total concrete volume
# Insulation emissivity for radiation losses.
eps_iso     = epsilon_iso          # [-]

# n_pipes: used ONLY to derive r_eq and to scale energy to the full module.
# It does NOT appear in the ODE — all equations are written on one equivalent element.
n_pipes = 818

# Concrete volume represented by one equivalent pipe-concrete element.
V_eq  = V_con_total / (n_modules * n_pipes)   # [m3] concrete volume per equivalent element
# Equivalent concrete annulus cross-section = volume / length.
S_s   = V_eq / L_module                        # [m2] concrete annular cross-section
# Equivalent annulus outer radius from area relation: pi*(r_eq^2-r_pipe_out^2)=S_s.
r_eq  = np.sqrt(S_s / np.pi + r_pipe_out**2)  # [m]  outer equivalent radius (= b in Jian eq. 3.50)

# Fluid cross-sectional area inside the pipe.
S_f   = np.pi * r_pipe_in**2                  # [m2] fluid cross-section (inner pipe)
# Wetted perimeter for fluid-solid convective exchange.
P     = np.pi * D_ext_pipe                    # [m]  fluid-solid exchange perimeter

# External surfaces for heat loss (per module)
# Concrete core width after removing insulation thickness on both sides.
W_con     = module_width  - 2 * th_iso
# Concrete core height after removing insulation thickness on both sides.
H_con     = module_height - 2 * th_iso
# Concrete external side area (two long side faces + two top/bottom faces) per module.
S_con_ext = 2 * (W_con * L_module + H_con * L_module)
# Insulation outer side area used in external convection/radiation resistance.
S_iso_ext = 2 * (module_width * L_module + module_height * L_module)

# Derived parameters from constants
# Prandtl number of HTF: momentum diffusivity / thermal diffusivity.
Pr_f = mu_f * Cp_f / k_f

# Total oil flow rate and per-element mass flows
# Total HTF mass flow from volumetric flow and density.
m_dot_total = rho_f * Q_vol_total         # [kg/s] total mass flow rate
# Equivalent-element mass flow by splitting total flow across all pipes.
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
    Compute the effective heat transfer coefficient h_E (Jian et al. eq. 3.50).

    Parameters
    ----------
    m_dot_per_element : float
        Mass flow rate through one equivalent pipe element [kg/s].

    Returns
    -------
    h_E : float
        Effective fluid-solid heat transfer coefficient [W/(m2·K)], combining
        convective film resistance and radial conduction resistance in concrete.
    h : float
        Convective film coefficient (Gnielinski) at the pipe inner wall [W/(m2·K)].
    Re : float
        Reynolds number for the flow inside one equivalent pipe [-].

    Notes
    -----
    This function uses the equivalent-element geometry defined at module scope.
    The returned `h_E` is the coefficient used in the 1D energy balances to
    convert a local fluid-solid temperature difference into a heat flux:
    q'' = h_E * (T_s - T_f) [W/m2].
    """
    # Convert mass flow to average velocity in one equivalent pipe.
    v  = m_dot_per_element / (rho_f * S_f)
    # Reynolds number for flow regime and Nusselt correlation.
    Re = rho_f * v * D_int_pipe / mu_f

    # Gnielinski correlation (ref. [30] in Buscemi et al.)
    if Re < 2300:
        Nu = 3.66                                         # laminar fully-developed
    elif Re < 4000:
        # Transitional branch: blend laminar and turbulent estimates smoothly.
        f_t    = (0.790 * np.log(4000) - 1.64)**(-2)
        Nu_lam = 3.66
        Nu_tur = (f_t/8)*(4000-1000)*Pr_f / (1 + 12.7*np.sqrt(f_t/8)*(Pr_f**(2/3)-1))
        alpha  = (Re - 2300) / (4000 - 2300)
        Nu     = (1 - alpha)*Nu_lam + alpha*Nu_tur        # smooth transition
    else:
        f  = (0.790 * np.log(Re) - 1.64)**(-2)
        Nu = (f/8)*(Re-1000)*Pr_f / (1 + 12.7*np.sqrt(f/8)*(Pr_f**(2/3)-1))

    # Convert Nusselt number to convective film coefficient.
    h = Nu * k_f / D_int_pipe  # [W/(m2·K)] convective coefficient

    # Radial conduction correction term (Jian et al. eq. 3.50)
    # Geometry terms from Jian eq. 3.50 for radial conduction correction.
    a, b = r_pipe_out, r_eq
    num  = 4*a*b**4*np.log(b/a) - 3*a*b**4 + 4*a**3*b**2 - a**5
    den  = 4 * (b**2 - a**2)**2
    # Effective coefficient combines convection resistance and radial conduction resistance.
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
    """
    Compute the apparent sky (radiative sink) temperature.

    Parameters
    ----------
    T_amb_C : float
        Ambient air temperature [C].

    Returns
    -------
    float
        Apparent sky temperature [C], computed via the empirical relation
        T_sky = 0.0552 * T_amb_K^1.5 (evaluated in K, returned in C).
    """
    # Empirical sky-temperature relation in C (internally evaluated in K and converted back).
    return 0.0552 * (T_amb_C + 273.15)**1.5 - 273.15

def h_ext_coeff(T_iso_C, T_amb_C, vw):
    """
    Compute the global external heat transfer coefficient at the insulation surface.

    Combines forced/free convection and linearised radiation (Buscemi et al. eq. 14-19).

    Parameters
    ----------
    T_iso_C : float
        Estimated outer surface temperature of the insulation [C].
    T_amb_C : float
        Ambient air temperature [C].
    vw : float
        Wind speed [m/s].

    Returns
    -------
    float
        Combined external heat transfer coefficient h_ext = h_conv + h_rad [W/(m2·K)].
    """
    # Forced/free convection outside insulation.
    h_conv  = 1.53 * vw + 1.43
    T_iso_K = T_iso_C + 273.15
    T_sky_K = T_sky(T_amb_C) + 273.15
    # Linearized radiative coefficient around current temperatures.
    h_rad   = sigma * eps_iso * (T_iso_K**4 - T_sky_K**4) / (T_iso_C - T_amb_C + 1e-9)
    return h_conv + h_rad

def Q_loss_module(T_s_avg, T_amb_in, vw):
    """
    Compute the total steady-state heat loss from one CTES module to the environment.

    Models heat flow through two series resistances: insulation conduction (R_iso)
    and external convection/radiation (R_ext).

    Parameters
    ----------
    T_s_avg : float
        Average solid (concrete) temperature at the axial node being evaluated [C].
    T_amb_in : float
        Ambient air temperature [C].
    vw : float
        Wind speed [m/s].

    Returns
    -------
    float
        Heat loss from one full CTES module [W], clamped to >= 0
        so the model does not admit ambient heat gain.

    Notes
    -----
    Despite the name `T_s_avg`, the caller may pass either a true axial average
    solid temperature or the local solid temperature of one axial node. In the
    ODE, the returned module-level loss is converted to a local volumetric sink
    before being inserted into the solid energy equation.
    """
    # Approximate insulation outer-surface temperature from solid-to-ambient delta.
    T_iso = T_amb_in + (T_s_avg - T_amb_in) * 0.1   # estimated insulation surface temperature
    # External heat transfer coefficient (convection + radiation).
    he    = h_ext_coeff(T_iso, T_amb_in, vw)
    # Insulation conduction thermal resistance.
    R_iso = th_iso / (lambda_iso * S_con_ext)
    # External convection/radiation thermal resistance.
    R_ext = 1.0 / (he * S_iso_ext)
    # Heat loss through series thermal resistances; clamped to non-negative.
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
STATE_SIZE_MODULE = 2 * N_z

def ctes_1d_rhs(t, y, T_in, m_dot, mode):
    """
    Right-hand side of the 1D axial ODE system (Jian et al. eq. 3.47-3.49).

    Implements coupled fluid/solid PDEs discretised on N_z axial nodes using
    a first-order upwind scheme for advection.

    Parameters
    ----------
    t : float
        Current time [s] (required by solve_ivp signature, not used explicitly).
    y : ndarray, shape (2*N_z,)
        State vector for one module, interleaved as
        [T_f(0), T_s(0), T_f(1), T_s(1), ...], with all temperatures in [C].
    T_in : float
        Fluid inlet temperature applied at the upstream boundary of this module [C].
    m_dot : float
        Total HTF mass flow rate through the full CTES string [kg/s].
        The per-pipe equivalent-element flow is computed internally as m_dot / n_pipes.
    mode : str
        Operating mode: 'charging' for flow in +z, 'discharging' for flow in -z,
        or 'storage' for no flow and heat-loss-only evolution.

    Returns
    -------
    dydt : ndarray, shape (2*N_z,)
        Time-derivative vector of the state, in the same interleaved layout as `y`.

        For each axial node j:
        - dydt[2*j]     = dT_f(j)/dt, the fluid temperature rate [C/s]
        - dydt[2*j + 1] = dT_s(j)/dt, the solid temperature rate [C/s]

        Since a temperature difference of 1 C is equal to a temperature
        difference of 1 K, these rates can equivalently be interpreted as [K/s].

    Notes
    -----
    Physical interpretation of the returned derivatives:
    - Fluid entries combine axial advection and fluid-solid heat exchange.
    - Solid entries combine fluid-solid heat exchange and ambient heat loss.
    - In 'storage' mode, all fluid derivatives are zero because advection is off
      and the fluid equation is not advanced.
    """
    # dydt stores local temperature rates, node by node, in the same layout as y.
    dydt = np.zeros_like(y)

    # Storage mode: no fluid flow, only heat loss
    if mode == 'storage':
        for j in range(N_z):
            # Solid temperature at node j.
            T_s_j  = y[2*j + 1]
            # Module-level ambient loss evaluated at local solid temperature.
            Ql     = Q_loss_module(T_s_j, T_amb, v_wind)
            # Convert module heat loss [W] to volumetric sink [W/m3] for PDE form.
            ql_vol = Ql / (L_module * S_s * n_pipes)     # [W/m3]
            dydt[2*j]     = 0.0
            # Convert volumetric power sink to temperature-rate sink [K/s].
            dydt[2*j + 1] = -ql_vol / (rho_con * Cp_con)
        return dydt

    # Compute h_E at average fluid temperature
    # Optional mean-fluid diagnostic (kept for potential property updates).
    T_f_avg_all = np.mean([y[2*j] for j in range(N_z)])
    # h_E evaluated using per-pipe mass flow.
    h_E, _, _   = compute_h_E(m_dot / n_pipes)

    # Fluid velocity in one equivalent element
    # Axial HTF velocity in one equivalent element.
    U = (m_dot / n_pipes) / (rho_f * S_f)

    # Source term coefficients
    # Interphase coupling prefactors that convert delta-T into dT/dt source terms.
    coeff_f = h_E * P / (rho_f  * S_f  * Cp_f)    # [1/s] for fluid equation
    coeff_s = h_E * P / (rho_con * S_s * Cp_con)  # [1/s] for solid equation

    for j in range(N_z):
        T_f_j = y[2*j]
        T_s_j = y[2*j + 1]

        # ---- Fluid equation: upwind advection scheme ----
        if mode == 'charging':
            # charging: fluid enters at z=0, flows in +z direction
            # Upwind gradient for +z flow direction.
            T_f_prev  = T_in if j == 0 else y[2*(j-1)]
            advection = -U * (T_f_j - T_f_prev) / dz
        else:
            # discharging: fluid enters at z=L, flows in -z direction
            # Upwind gradient for -z flow direction.
            T_f_next  = T_in if j == N_z-1 else y[2*(j+1)]
            advection = U * (T_f_next - T_f_j) / dz

        # Fluid time derivative: advection + fluid-solid heat exchange.
        dydt[2*j] = advection + coeff_f * (T_s_j - T_f_j)

        # ---- Solid equation ----
        # Ambient loss from this axial location.
        Ql     = Q_loss_module(T_s_j, T_amb, v_wind)
        # Convert to volumetric sink for local solid energy balance.
        ql_vol = Ql / (L_module * S_s * n_pipes)
        # Solid time derivative: interphase exchange minus ambient loss sink.
        dydt[2*j + 1] = -coeff_s * (T_s_j - T_f_j) - ql_vol / (rho_con * Cp_con)

    return dydt

# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================
def extract_profiles(y):
    """
    Extract fluid and solid axial temperature profiles from the state vector.

    For multi-module states (len(y) = 2*N_z*n_mod), profiles are concatenated
    over modules from z=0 to z=max.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_mod,)
        CTES state vector. Interleaved per module as [T_f(0), T_s(0), ..., T_f(N_z-1), T_s(N_z-1)].
        Temperatures in [C].

    Returns
    -------
    T_f : ndarray, shape (N_z*n_mod,)
        Fluid temperature profile along the full axial length [C], ordered in
        physical flow order module-by-module.
    T_s : ndarray, shape (N_z*n_mod,)
        Solid (concrete) temperature profile along the full axial length [C],
        ordered in physical flow order module-by-module.
    """
    # Normalize input to contiguous numeric array for slicing/indexing.
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size % STATE_SIZE_MODULE != 0:
        raise ValueError("CTES state length must be a multiple of 2*N_z")
    # Infer number of modules represented in state vector.
    n_mod_state = y_arr.size // STATE_SIZE_MODULE
    if n_mod_state == 1:
        T_f = np.array([y_arr[2*j] for j in range(N_z)], dtype=float)
        T_s = np.array([y_arr[2*j+1] for j in range(N_z)], dtype=float)
        return T_f, T_s

    T_f_parts = []
    T_s_parts = []
    for k in range(n_mod_state):
        # Slice state block for one module.
        yk = y_arr[k * STATE_SIZE_MODULE:(k + 1) * STATE_SIZE_MODULE]
        T_f_parts.append(np.array([yk[2*j] for j in range(N_z)], dtype=float))
        T_s_parts.append(np.array([yk[2*j+1] for j in range(N_z)], dtype=float))
    return np.concatenate(T_f_parts), np.concatenate(T_s_parts)


def _split_module_states(y):
    """
    Split a full CTES state vector into individual per-module blocks.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_mod,)
        Full CTES state vector covering all modules.

    Returns
    -------
    list of ndarray, each shape (2*N_z,)
        One state block per module, in physical order (module 0 first).
        Each block preserves the interleaved layout
        [T_f(0), T_s(0), ..., T_f(N_z-1), T_s(N_z-1)] with temperatures in [C].
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size % STATE_SIZE_MODULE != 0:
        raise ValueError("CTES state length must be a multiple of 2*N_z")
    n_mod_state = y_arr.size // STATE_SIZE_MODULE
    # Return independent module blocks so step logic can update each module separately.
    return [y_arr[k * STATE_SIZE_MODULE:(k + 1) * STATE_SIZE_MODULE].copy() for k in range(n_mod_state)]


def _combine_module_states(states):
    """
    Concatenate a list of per-module state blocks into a single full state vector.

    Parameters
    ----------
    states : list of array-like, each shape (2*N_z,)
        Per-module state blocks in physical order.

    Returns
    -------
    ndarray, shape (2*N_z*n_mod,)
        Full concatenated CTES state vector with temperatures in [C].
        Returns an empty array if `states` is empty.
    """
    if not states:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(s, dtype=float) for s in states])


def _module_energy_J(y_mod, T_ref):
    """
    Compute the sensible energy stored in the concrete of one module.

    Parameters
    ----------
    y_mod : array-like, shape (2*N_z,)
        State vector for a single module.
    T_ref : float
        Reference temperature for sensible energy calculation [C].

    Returns
    -------
    float
        Sensible energy in the solid (concrete) of one module relative to T_ref [J].
    """
    _, T_s_mod = extract_profiles(y_mod)
    # Integrate solid sensible energy relative to reference temperature along z.
    return float(rho_con * Cp_con * S_s * n_pipes * np.trapz(T_s_mod - T_ref, z_nodes))


def _module_fluid_energy_J(y_mod, T_ref):
    """
    Compute the sensible energy of the HTF inside one module.

    Parameters
    ----------
    y_mod : array-like, shape (2*N_z,)
        State vector for a single module.
    T_ref : float
        Reference temperature for sensible energy calculation [C].

    Returns
    -------
    float
        Sensible energy of the fluid volume inside one module relative to T_ref [J].
    """
    T_f_mod, _ = extract_profiles(y_mod)
    # Integrate fluid sensible energy relative to reference temperature along z.
    return float(rho_f * Cp_f * S_f * n_pipes * np.trapz(T_f_mod - T_ref, z_nodes))


def _module_heat_loss_W(y_mod):
    """
    Compute the instantaneous heat loss to ambient from one module.

    Averages node-wise Q_loss_module calls along the axial direction.

    Parameters
    ----------
    y_mod : array-like, shape (2*N_z,)
        State vector for a single module.

    Returns
    -------
    float
        Total heat loss from one module to the environment [W].
    """
    _, T_s_mod = extract_profiles(y_mod)
    # Average node-wise module losses along z (Riemann sum scaled by dz/L).
    return float(np.sum([Q_loss_module(float(T_s_j), T_amb, v_wind) for T_s_j in T_s_mod]) * dz / L_module)

def stored_energy(y, T_ref=T_min):
    """
    Return the total sensible energy stored in the CTES concrete.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.
    T_ref : float, optional
        Reference temperature for sensible energy [C]. Defaults to T_min.

    Returns
    -------
    float
        Total sensible energy in the solid phase across all modules [MWh].
        The reference level is the concrete temperature `T_ref`.
    """
    return stored_energy_J(y, T_ref=T_ref) / 3.6e9

def SOC(y):
    """
    Compute the State of Charge of the CTES.

    Defined as the average solid temperature mapped linearly between T_min and T_max.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.

    Returns
    -------
    float
        State of Charge in [0, 1], where 0 = fully discharged (T_min) and
        1 = fully charged (T_max). This is dimensionless.
    """
    _, T_s = extract_profiles(y)
    # SOC from average solid temperature mapped linearly between T_min and T_max.
    return np.clip((np.mean(T_s) - T_min) / (T_max - T_min), 0.0, 1.0)

def T_outlet(y, mode):
    """
    Return the fluid outlet temperature of the CTES string.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.
    mode : str
        'charging'    -> outlet is at z = L (last node of last module).
        'discharging' -> outlet is at z = 0 (first node of first module).

    Returns
    -------
    float
        Fluid temperature at the outlet face of the full module string [C].
    """
    T_f, _ = extract_profiles(y)
    return float(T_f[-1] if mode == 'charging' else T_f[0])


def heat_loss_W(y):
    """
    Compute the instantaneous total heat loss of the CTES to ambient.

    Uses the same Q_loss_module formulation as the ODE source term, aggregated
    over all axial nodes and all modules.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.

    Returns
    -------
    float
        Total heat loss to the environment across all modules [W].
        Positive values correspond to heat leaving the storage.
    """
    mods = _split_module_states(y)
    if len(mods) == 1:
        return _module_heat_loss_W(mods[0]) * n_modules
    return float(sum(_module_heat_loss_W(m) for m in mods))


# ---------------------------------------------------------------------------
# Convenience helpers for integrating the CTES model from external code
# ---------------------------------------------------------------------------
def init_ctes_state(T_init=None):
    """
    Create a uniform initial state vector for the CTES model.

    Parameters
    ----------
    T_init : float or None, optional
        Uniform initial temperature for all fluid and solid nodes [C].
        Defaults to T_min if not provided.

    Returns
    -------
    ndarray, shape (2*N_z*n_modules,)
        Initial CTES state vector with all fluid and solid temperatures set to
        `T_init` [C].
    """
    # Default initialization at lower operating temperature.
    T0 = T_min if T_init is None else float(T_init)
    # One module state with uniform fluid/solid temperature.
    y_mod = np.full(STATE_SIZE_MODULE, T0)
    # Replicate module state across all modules in series.
    return np.tile(y_mod, n_modules)


def stored_energy_J(y, T_ref=None):
    """
    Return the total sensible energy stored in the CTES concrete in Joules.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.
    T_ref : float or None, optional
        Reference temperature [C]. Defaults to T_min.

    Returns
    -------
    float
        Total stored energy in the solid (concrete) across all modules [J],
        relative to the reference temperature `T_ref`.
    """
    if T_ref is None:
        T_ref = T_min
    mods = _split_module_states(y)
    if len(mods) == 1:
        return _module_energy_J(mods[0], T_ref) * n_modules
    return float(sum(_module_energy_J(m, T_ref) for m in mods))


def fluid_energy_J(y, T_ref=None):
    """
    Return the sensible energy of the HTF currently inside the CTES volume.

    Parameters
    ----------
    y : array-like, shape (2*N_z*n_modules,)
        Full CTES state vector.
    T_ref : float or None, optional
        Reference temperature [C]. Defaults to T_min.

    Returns
    -------
    float
        Total fluid sensible energy inside the CTES across all modules [J],
        relative to the reference temperature `T_ref`.
    """
    if T_ref is None:
        T_ref = T_min
    mods = _split_module_states(y)
    if len(mods) == 1:
        return _module_fluid_energy_J(mods[0], T_ref) * n_modules
    return float(sum(_module_fluid_energy_J(m, T_ref) for m in mods))


def step_ctes(y_curr, T_in_C, m_dot_total_kg_s, mode, dt_seconds, rtol=1e-5, atol=1e-7):
    """
    Advance the CTES 1D model by one timestep using scipy solve_ivp (Radau).

    Modules are marched in physical flow order so that the outlet of each module
    feeds the inlet of the next. Falls back to storage mode if m_dot <= 0.

    Parameters
    ----------
    y_curr : array-like, shape (2*N_z*n_modules,)
        Current CTES state vector (all temperatures in [C]).
    T_in_C : float
        Fluid inlet temperature at the entry of the storage string [C].
    m_dot_total_kg_s : float
        Total HTF mass flow rate through the full CTES string [kg/s].
        Pass 0 (or None) for storage mode.
    mode : str
        'charging'    — hot fluid enters at z=0, flows toward z=L.
        'discharging' — cool fluid enters at z=L, flows toward z=0.
        'storage'     — no flow; only ambient heat loss is integrated.
    dt_seconds : float
        Integration timestep [s]. Minimum enforced internally at 1 s.
    rtol : float, optional
        Relative tolerance for solve_ivp. Default 1e-5.
    atol : float, optional
        Absolute tolerance for solve_ivp. Default 1e-7.

    Returns
    -------
    dict with keys:
        y : ndarray, shape (2*N_z*n_modules,)
            Updated CTES state vector after the timestep, with temperatures in [C].
        T_out_C : float
            Fluid outlet temperature at the exit face of the storage string [C].
        energy_J : float
            Total sensible energy stored in concrete after the step [J].
        SOC : float
            State of Charge after the step [-], in [0, 1].
        diag_q_fluid_to_solid_total_W : float
            Sensible power transferred from HTF to solid (m_dot*Cp*(T_in-T_out)) [W].
        diag_q_fluid_to_solid_module_W : float
            Same quantity divided by number of modules [W].
        diag_q_loss_avg_W : float
            Trapezoidal-average ambient heat loss over the step [W].
        diag_q_to_solid_corr_W : float
            Solid charging power inferred from dE_solid/dt + q_loss [W].
        diag_dE_W : float
            Finite-difference solid energy rate (dE_solid/dt) [W].
        diag_dE_fluid_W : float
            Finite-difference fluid energy rate (dE_fluid/dt) [W].
        diag_dE_rhs_W, diag_dE_fluid_rhs_W : float
            Copies of the FD rates (reserved for RHS-based estimates) [W].
        diag_closure_solid_corr_W : float
            Residual of solid-only energy balance [W]; ideally ~0.
        diag_closure_combined_W : float
            Residual of combined fluid+solid energy balance [W]; ideally ~0.
        diag_energy_prev_J, diag_energy_new_J : float
            Solid stored energy before and after the step [J].
        diag_fluid_energy_prev_J, diag_fluid_energy_new_J : float
            Fluid stored energy before and after the step [J].

    Notes
    -----
    This function advances the state one macro-timestep and returns both the
    new temperatures and diagnostic power/energy terms. All `diag_*_W` entries
    are powers [J/s], while `diag_*_J` entries are energies [J].
    """
    if mode not in ('charging', 'discharging', 'storage'):
        raise ValueError("mode must be 'charging', 'discharging', or 'storage'")

    # For charging/discharging, ensure a small positive mass flow is provided.
    if mode in ('charging', 'discharging') and (m_dot_total_kg_s is None or m_dot_total_kg_s <= 0):
        # fall back to storage-only step (no advection)
        mode_use = 'storage'
    else:
        mode_use = mode

    # Sanitize scalar inputs and enforce minimum step size for diagnostics.
    T_in_val = float(T_in_C) if (T_in_C is not None) else float(T_min)
    m_dot_val = float(m_dot_total_kg_s) if (m_dot_total_kg_s is not None) else 0.0
    dt_val = max(1.0, float(dt_seconds))
    # Pre-step energies/loss for finite-difference power balance checks.
    E_prev_J = stored_energy_J(y_curr)
    E_fluid_prev_J = fluid_energy_J(y_curr)
    q_loss_prev_W = max(0.0, float(heat_loss_W(y_curr)))

    # Work with explicit module blocks to march modules in physical flow order.
    mods_prev = _split_module_states(y_curr)
    n_mod_state = len(mods_prev)
    mods_new = [m.copy() for m in mods_prev]

    if mode_use == 'storage' or m_dot_val <= 0.0:
        # Storage mode: advance each module without advection/mass flow.
        for k in range(n_mod_state):
            sol_k = solve_ivp(
                ctes_1d_rhs,
                [0.0, float(dt_seconds)],
                mods_prev[k],
                args=(T_in_val, 0.0, 'storage'),
                method='Radau',
                rtol=rtol,
                atol=atol,
            )
            mods_new[k] = sol_k.y[:, -1]
        # Report outlet from combined state for API consistency.
        T_out = T_outlet(_combine_module_states(mods_new), 'discharging')
        q_fluid_to_solid_total_W = 0.0
    else:
        # Physical ordering: module 0 at z=0 side, module n-1 at z=max side.
        if mode_use == 'charging':
            order = range(0, n_mod_state)
        else:
            order = range(n_mod_state - 1, -1, -1)

        # March inlet/outlet temperature through modules in physical flow direction.
        T_in_chain = T_in_val
        for k in order:
            sol_k = solve_ivp(
                ctes_1d_rhs,
                [0.0, float(dt_seconds)],
                mods_prev[k],
                args=(T_in_chain, m_dot_val, mode_use),
                method='Radau',
                rtol=rtol,
                atol=atol,
            )
            mods_new[k] = sol_k.y[:, -1]
            # Outlet of current module becomes inlet for the next module in chain.
            T_in_chain = T_outlet(mods_new[k], mode_use)
        T_out = float(T_in_chain)
        # Fluid sensible power change across full storage string.
        q_fluid_to_solid_total_W = m_dot_val * Cp_f * (T_in_val - T_out)

    # Post-step aggregate state and diagnostics.
    y_new = _combine_module_states(mods_new)
    E_J = stored_energy_J(y_new)
    E_fluid_new_J = fluid_energy_J(y_new)
    soc = SOC(y_new)
    q_loss_new_W = max(0.0, float(heat_loss_W(y_new)))
    # Trapezoidal average loss over the step for balance accounting.
    q_loss_avg_W = 0.5 * (q_loss_prev_W + q_loss_new_W)
    # Per-module transfer power for easier interpretation.
    q_fluid_to_solid_module_W = q_fluid_to_solid_total_W / max(1, n_mod_state)
    # Finite-difference solid/fluid stored-energy rates.
    dE_W_fd = (float(E_J) - float(E_prev_J)) / dt_val
    dE_fluid_W_fd = (float(E_fluid_new_J) - float(E_fluid_prev_J)) / dt_val
    dE_W_rhs = dE_W_fd
    dE_fluid_W_rhs = dE_fluid_W_fd

    # Net power to solid and diagnostic closures.
    # Corrected solid charging power implied by solid dE plus ambient loss.
    q_to_solid_corr_W = dE_W_fd + q_loss_avg_W
    # Closure of solid-only balance including fluid energy change.
    closure_solid_corr_W = (q_fluid_to_solid_total_W - dE_fluid_W_fd) - q_to_solid_corr_W
    # Combined closure over fluid + solid control volume.
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