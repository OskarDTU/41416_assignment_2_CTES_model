# =============================================================================
# Exergy Analysis of the CTES
# Charging, Discharging, and Standby configurations
#
# Exergy of a heat flow (eq. 3.62):
#   X_Q = Q * (1 - T0/Tb)
#
# Exergy of a mass flow (eq. 3.63):
#   X_flow = m_dot * Cp * [(T - T0) - T0 * ln(T/T0)]
#
# Overall exergy efficiency (eq. 3.64):
#   eta = X_recovered (discharging) / X_supplied (charging)
# =============================================================================

import pandas as pd
import numpy as np
from CoolProp.CoolProp import PropsSI

import glob
import os

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.constants import (
    rho_con, Cp_con,
    V_con,
    rho_f, Cp_f,
    T_amb,
)
# Find the most recent simulation CSV automatically
csv_files = glob.glob("src/data/**/simulation_results_*.csv", recursive=True)
if not csv_files:
    raise FileNotFoundError("No simulation results found. Run Part_1.py first.")

csv_path = max(csv_files, key=os.path.getmtime)  # most recent
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# --- Constants ---
T_o  = 293.15  # K, dead state temperature (20 °C)
dt   = 600.0   # s, simulation timestep (10 min)
fluid = "INCOMP::PNF"

from src.data.constants import (
    rho_con, Cp_con,
    V_con,
    rho_f, Cp_f,
    T_amb,
)

M_con = rho_con * V_con  # kg, total concrete mass


# =============================================================================
# EXERGY FUNCTIONS
# =============================================================================

def exergy_flow(m_dot, T_K, Cp):
    """
    Exergy power of a mass flow (eq. 3.63) [W].
    m_dot : mass flow rate [kg/s]
    T_K   : fluid temperature [K]
    Cp    : specific heat capacity [J/(kg·K)]
    """
    return m_dot * Cp * ((T_K - T_o) - T_o * np.log(T_K / T_o))


def exergy_heat(Q_W, T_K):
    """
    Exergy associated with a heat loss to the surroundings (eq. 3.62) [W].
    Q_W : heat loss rate [W], positive leaving the system
    T_K : surface temperature at which heat is transferred [K]
    """
    if T_K <= T_o:
        return 0.0
    return Q_W * (1.0 - T_o / T_K)


def exergy_stored_profile(T_z0_C, T_mid_C, T_zmax_C):
    """
    Exergy stored in the CTES concrete [J], computed via Simpson's rule
    over three axial nodes (z=0, z=L/2, z=L).
    Works with both scalars and pandas Series.
    """
    T_nodes_K = np.array([T_z0_C, T_mid_C, T_zmax_C]) + 273.15
    ex_nodes  = (M_con / 3.0) * Cp_con * (
        (T_nodes_K - T_o) - T_o * np.log(T_nodes_K / T_o)
    )
    # Simpson's rule: (1/6) * [f(0) + 4*f(L/2) + f(L)] * interval_length (=3)
    return (ex_nodes[0] + 4.0 * ex_nodes[1] + ex_nodes[2]) / 6.0 * 3.0


# =============================================================================
# PRE-COMPUTE COLUMNS COMMON TO ALL CONFIGURATIONS
# =============================================================================

# Average concrete temperature [K] (used for exergy heat loss)
df["T_con_avg_K"] = (
    (df["concrete_T_z0_C"] + df["concrete_T_z_max_C"]) / 2.0 + 273.15
)

# Exergy stored in the concrete [J]
df["ex_stored_J"] = exergy_stored_profile(
    df["concrete_T_z0_C"],
    df["concrete_T_z_mid_C"],
    df["concrete_T_z_max_C"],
)

# Rate of change of stored exergy [W]
df["dex_stored_dt_W"] = df["ex_stored_J"].diff() / dt
df.at[df.index[0], "dex_stored_dt_W"] = 0.0

# Heat loss to surroundings [W]
df["Q_loss_W"] = df["ctes_current_loss_kW"].abs() * 1000.0

# Exergy of heat loss [W]
df["X_loss_W"] = df.apply(
    lambda r: exergy_heat(r["Q_loss_W"], r["T_con_avg_K"]), axis=1
)

# Initialise exergy columns
df["ex_in_W"]   = 0.0
df["ex_out_W"]  = 0.0
df["ex_dest_W"] = 0.0
df["config"]    = "standby"


# =============================================================================
# CHARGING
# Exergy balance:
#   ex_in = dex_stored/dt + ex_out + X_loss + ex_dest
#   => ex_dest = (ex_in - ex_out) - dex_stored/dt - X_loss
# =============================================================================

mask_ch = df["m_ctes_charge_use_m3s"] > 1e-6
ch = df[mask_ch].copy()

T_in_ch  = ch["ctes_inlet_temp_C"] + 273.15    # K, hot fluid entering CTES
T_out_ch = ch["ctes_temp_C"]        + 273.15    # K, fluid leaving CTES
m_dot_ch = ch["m_ctes_charge_use_m3s"] * rho_f # kg/s

ch["ex_in_W"]   = exergy_flow(m_dot_ch, T_in_ch,  Cp_f)
ch["ex_out_W"]  = exergy_flow(m_dot_ch, T_out_ch, Cp_f)
ch["ex_dest_W"] = (
      (ch["ex_in_W"] - ch["ex_out_W"])
    - ch["dex_stored_dt_W"]
    - ch["X_loss_W"]
).clip(lower=0.0)
ch["config"] = "charging"

df.update(ch[["ex_in_W", "ex_out_W", "ex_dest_W", "config"]])


# =============================================================================
# DISCHARGING
# Exergy balance:
#   -dex_stored/dt = (ex_out - ex_in) + X_loss + ex_dest
#   => ex_dest = -dex_stored/dt - (ex_out - ex_in) - X_loss
# =============================================================================

mask_dch = (df["m_ctes_use_m3s"] > 1e-6) & (
    df["operation_mode"].isin(["C", "D", "C*", "D*"])
)
dch = df[mask_dch].copy()

T_in_dch  = dch["oil_out_C"]   + 273.15    # K, cool fluid entering CTES (HEX return)
T_out_dch = dch["ctes_temp_C"] + 273.15    # K, hot fluid leaving CTES (to HEX)
m_dot_dch = dch["m_ctes_use_m3s"] * rho_f  # kg/s

dch["ex_in_W"]   = exergy_flow(m_dot_dch, T_in_dch,  Cp_f)
dch["ex_out_W"]  = exergy_flow(m_dot_dch, T_out_dch, Cp_f)
dch["ex_dest_W"] = (
      (-dch["dex_stored_dt_W"])
    - (dch["ex_out_W"] - dch["ex_in_W"])
    - dch["X_loss_W"]
).clip(lower=0.0)
dch["config"] = "discharging"

df.update(dch[["ex_in_W", "ex_out_W", "ex_dest_W", "config"]])


# =============================================================================
# STANDBY
# Exergy balance:
#   -dex_stored/dt = X_loss + ex_dest
#   => ex_dest = -dex_stored/dt - X_loss
# =============================================================================

mask_sb = ~mask_ch & ~mask_dch
sb = df[mask_sb].copy()

sb["ex_dest_W"] = (
    (-sb["dex_stored_dt_W"]) - sb["X_loss_W"]
).clip(lower=0.0)
sb["config"] = "standby"

df.update(sb[["ex_dest_W", "config"]])


# =============================================================================
# OVERALL EXERGY EFFICIENCY (eq. 3.64)
# eta = X_recovered (discharging) / X_supplied (charging)
# =============================================================================

X_supplied  = ((df.loc[mask_ch,  "ex_in_W"]  - df.loc[mask_ch,  "ex_out_W"]) * dt).sum()
X_recovered = ((df.loc[mask_dch, "ex_out_W"] - df.loc[mask_dch, "ex_in_W"])  * dt).sum()
eta_ex = X_recovered / X_supplied if X_supplied > 0 else np.nan


# =============================================================================
# RESULTS TABLE
# =============================================================================

def summarise(mask, label):
    sub = df[mask]
    n_h        = len(sub) * dt / 3600.0
    X_in_MWh   = (sub["ex_in_W"]          * dt).sum() / 3.6e9
    X_out_MWh  = (sub["ex_out_W"]         * dt).sum() / 3.6e9
    X_loss_MWh = (sub["X_loss_W"]         * dt).sum() / 3.6e9
    X_dest_MWh = (sub["ex_dest_W"]        * dt).sum() / 3.6e9
    dX_MWh     = (sub["dex_stored_dt_W"]  * dt).sum() / 3.6e9
    return {
        "Configuration"   : label,
        "Duration [h]"    : round(n_h,        1),
        "X_in [MWh]"      : round(X_in_MWh,   4),
        "X_out [MWh]"     : round(X_out_MWh,  4),
        "X_loss [MWh]"    : round(X_loss_MWh, 4),
        "X_dest [MWh]"    : round(X_dest_MWh, 4),
        "dX_stored [MWh]" : round(dX_MWh,     4),
    }

rows = [
    summarise(mask_ch,  "Charging"),
    summarise(mask_dch, "Discharging"),
    summarise(mask_sb,  "Standby"),
]

total = {
    "Configuration"   : "TOTAL",
    "Duration [h]"    : round(len(df) * dt / 3600.0, 1),
    "X_in [MWh]"      : round(sum(r["X_in [MWh]"]      for r in rows), 4),
    "X_out [MWh]"     : round(sum(r["X_out [MWh]"]     for r in rows), 4),
    "X_loss [MWh]"    : round(sum(r["X_loss [MWh]"]    for r in rows), 4),
    "X_dest [MWh]"    : round(sum(r["X_dest [MWh]"]    for r in rows), 4),
    "dX_stored [MWh]" : round(sum(r["dX_stored [MWh]"] for r in rows), 4),
}
rows.append(total)

results_table = pd.DataFrame(rows).set_index("Configuration")

print("\n" + "=" * 75)
print("CTES EXERGY ANALYSIS — RESULTS TABLE")
print("=" * 75)
print(results_table.to_string())
print(f"\nOverall exergy efficiency (eq. 3.64):  eta_ex = {eta_ex*100:.1f} %")
print(f"  X_supplied  (charging)    = {X_supplied  / 3.6e9:.4f} MWh")
print(f"  X_recovered (discharging) = {X_recovered / 3.6e9:.4f} MWh")
print("=" * 75)

# Save enriched DataFrame and results table
df.to_csv("simulation_results_exergy.csv")
results_table.to_csv("exergy_results_table.csv")
print("\nFiles saved:")
print("  simulation_results_exergy.csv  — full timestep-by-timestep data")
print("  exergy_results_table.csv       — summary table")