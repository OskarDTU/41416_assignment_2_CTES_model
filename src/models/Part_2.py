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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.constants import (
    rho_con, Cp_con,
    V_con,
    rho_f, Cp_f,
    T_amb,
    n_modules,
)

# --- Constants ---
T_o   = 293.15        # K, dead state temperature (20 °C)
dt    = 600.0         # s, simulation timestep (10 min)
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


# Column names for the 14 module average temperatures saved by Part_1
MODULE_COLS = [f"module_{i:02d}_Ts_avg_C" for i in range(n_modules)]


def exergy_stored_14modules(df_modules_C):
    """
    Exergy stored in the CTES concrete [J], summed over 14 modules.
    Each module has mass M_con / n_modules.

    df_modules_C : DataFrame or 2-D array, shape (n_timesteps, 14) [°C]
    Returns       : Series or 1-D array of total exergy stored [J]
    """
    T_K = np.asarray(df_modules_C, dtype=float) + 273.15  # (n_timesteps, 14)
    M_mod = M_con / n_modules
    ex_per_module = M_mod * Cp_con * (
        (T_K - T_o) - T_o * np.log(T_K / T_o)
    )
    return ex_per_module.sum(axis=1)


# =============================================================================
# MAIN EXERGY ANALYSIS FUNCTION
# =============================================================================

def exergy_analysis(df):
    """
    Perform exergy analysis on CTES simulation results.

    Input  : df — DataFrame returned by simulate() in Part_1.py
    Output : (df_ex, results_table, eta_ex)
               df_ex         — df enriched with exergy columns
               results_table — summary DataFrame (MWh per configuration)
               eta_ex        — overall exergy efficiency [-]
    """
    df = df.copy()

    # --- Fill NaN in module outlet temperatures before any computation ---
    for col in MODULE_COLS:
        df[col] = df[col].interpolate(method="linear").bfill().ffill()

    # --- Average concrete temperature [K] (used for exergy heat loss) ---
    df["T_con_avg_K"] = df[MODULE_COLS].mean(axis=1) + 273.15

    # --- Exergy stored in the concrete [J] (14-module sum) ---
    df["ex_stored_J"] = exergy_stored_14modules(df[MODULE_COLS])

    # --- Rate of change of stored exergy [W] ---
    df["dex_stored_dt_W"] = df["ex_stored_J"].diff() / dt
    df.at[df.index[0], "dex_stored_dt_W"] = 0.0

    # --- Heat loss to surroundings [W] ---
    df["Q_loss_W"] = df["ctes_current_loss_kW"].abs() * 1000.0

    # --- Exergy of heat loss [W] — vectorised, no apply needed ---
    df["X_loss_W"] = (
        df["Q_loss_W"] * (1.0 - T_o / df["T_con_avg_K"])
    ).clip(lower=0.0)

    # --- Initialise exergy columns ---
    df["ex_in_W"]   = 0.0
    df["ex_out_W"]  = 0.0
    df["ex_dest_W"] = 0.0
    df["config"]    = "standby"

    # --- Non-overlapping masks ---
    # Charging: CTES receives flow from collector AND is not discharging
    mask_ch  = (df["m_ctes_charge_use_m3s"] > 1e-6) & (df["m_ctes_use_m3s"] <= 1e-6)
    # Discharging: CTES supplies flow to HEX (includes mode C where both happen)
    mask_dch = df["m_ctes_use_m3s"] > 1e-6
    # Standby: no flow in either direction
    mask_sb  = ~mask_ch & ~mask_dch

    # =========================================================================
    # CHARGING
    # Exergy balance:
    #   ex_in = dex_stored/dt + ex_out + X_loss + ex_dest
    #   => ex_dest = (ex_in - ex_out) - dex_stored/dt - X_loss
    # =========================================================================
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

    # =========================================================================
    # DISCHARGING
    # Exergy balance:
    #   -dex_stored/dt = (ex_out - ex_in) + X_loss + ex_dest
    #   => ex_dest = -dex_stored/dt - (ex_out - ex_in) - X_loss
    # =========================================================================
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

    # =========================================================================
    # STANDBY
    # Exergy balance:
    #   -dex_stored/dt = X_loss + ex_dest
    #   => ex_dest = -dex_stored/dt - X_loss
    # =========================================================================
    sb = df[mask_sb].copy()

    sb["ex_dest_W"] = (
        (-sb["dex_stored_dt_W"]) - sb["X_loss_W"]
    ).clip(lower=0.0)
    sb["config"] = "standby"
    df.update(sb[["ex_dest_W", "config"]])

    # --- Overall exergy efficiency (eq. 3.64) ---
    # clip per timestep to avoid negative contributions distorting the total
    X_supplied  = (((df.loc[mask_ch,  "ex_in_W"] - df.loc[mask_ch,  "ex_out_W"]).clip(lower=0)) * dt).sum()
    X_recovered = (((df.loc[mask_dch, "ex_out_W"] - df.loc[mask_dch, "ex_in_W"]).clip(lower=0)) * dt).sum()
    eta_ex = X_recovered / X_supplied if X_supplied > 0 else np.nan

    # --- Results table ---
    def summarise(mask, label):
        sub = df[mask]
        return {
            "Configuration"   : label,
            "Duration [h]"    : round(len(sub) * dt / 3600.0, 1),
            "X_in [MWh]"      : round((sub["ex_in_W"]         * dt).sum() / 3.6e9, 4),
            "X_out [MWh]"     : round((sub["ex_out_W"]        * dt).sum() / 3.6e9, 4),
            "X_loss [MWh]"    : round((sub["X_loss_W"]        * dt).sum() / 3.6e9, 4),
            "X_dest [MWh]"    : round((sub["ex_dest_W"]       * dt).sum() / 3.6e9, 4),
            "dX_stored [MWh]" : round((sub["dex_stored_dt_W"] * dt).sum() / 3.6e9, 4),
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

    return df, results_table, eta_ex


# =============================================================================
# ENTRY POINT — only executed when running Part_2.py directly
# =============================================================================

if __name__ == "__main__":
    csv_files = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "data", "**", "simulation_results_*.csv"),
        recursive=True
    )
    if not csv_files:
        raise FileNotFoundError("No simulation results found. Run Part_1.py first.")

    csv_path = max(csv_files, key=os.path.getmtime)
    print(f"Loading: {csv_path}")

    df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_ex, table, eta = exergy_analysis(df_raw)

    df_ex.to_csv("src/data/simulation_results_exergy.csv")
    table.to_csv("src/data/exergy_results_table.csv")
    print("\nFiles saved:")
    print("  src/data/simulation_results_exergy.csv")
    print("  src/data/exergy_results_table.csv")
