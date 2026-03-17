import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
import glob
import re
from datetime import datetime

# default data directory
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def _find_latest_sim_csv(data_dir=DATA_DIR):
    # Search recursively because simulation outputs are saved in timestamp folders.
    pattern = os.path.join(data_dir, '**', 'simulation_results_*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_simulation_results(csv_path: Optional[str] = None, out_pdf: Optional[str] = None):
    # auto-detect latest CSV when not provided
    if csv_path is None:
        csv_path = _find_latest_sim_csv()
        if csv_path is None:
            raise FileNotFoundError(f'No simulation CSV found in {DATA_DIR}')

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Energy: Joules -> MWh
    if 'ctes_energy_J' in df.columns:
        energy_MWh = df['ctes_energy_J'].astype(float) / 3.6e9
    else:
        energy_MWh = pd.Series(np.nan, index=df.index)

    # Prefer SOC exported by simulation; fallback to model-capacity-based estimate.
    if 'ctes_soc_pct' in df.columns:
        soc = (df['ctes_soc_pct'].astype(float) / 100.0).clip(0, 1)
    else:
        try:
            try:
                from ..features.ctes_1d_jian import init_ctes_state, stored_energy_J, T_max
            except Exception:
                from features.ctes_1d_jian import init_ctes_state, stored_energy_J, T_max
            cap_J = stored_energy_J(init_ctes_state(T_init=float(T_max)))
            soc = (df['ctes_energy_J'].astype(float) / cap_J).clip(0, 1)
        except Exception:
            # final fallback: normalize to the run's energy span if possible
            if 'ctes_energy_J' in df.columns and df['ctes_energy_J'].astype(float).notna().any():
                e = df['ctes_energy_J'].astype(float)
                e_min = float(e.min(skipna=True))
                e_max = float(e.max(skipna=True))
                if e_max > e_min:
                    soc = ((e - e_min) / (e_max - e_min)).clip(0, 1)
                else:
                    soc = pd.Series(np.nan, index=df.index)
            else:
                soc = pd.Series(np.nan, index=df.index)

    # Outlet temperature if present
    if 'ctes_temp_C' in df.columns:
        t_out = df['ctes_temp_C'].astype(float)
    else:
        t_out = pd.Series(np.nan, index=df.index)

    # Make plots similar to the original CTES script
    day_labels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    # include concrete temperatures if present (z=0, z_mid, z_max)
    has_concrete = 'concrete_T_z0_C' in df.columns
    nplots = 4 if has_concrete else 3
    fig, axes = plt.subplots(nplots, 1, figsize=(13, 11), sharex=True)
    if nplots == 3:
        axes = axes
    else:
        # ensure axes is indexable
        axes = axes

    axes[0].plot(df.index, soc * 100, color='#2ecc71', lw=2)
    axes[0].set_ylabel('SOC [%]')
    axes[0].set_title('State of Charge')
    axes[0].set_ylim(0, 110)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, energy_MWh, color='#3498db', lw=2, label='1 system')
    axes[1].set_ylabel('Stored energy [MWh]')
    axes[1].set_title('Energy stored in concrete')
    axes[1].grid(True, alpha=0.3)

    if has_concrete:
        concrete_z0 = df['concrete_T_z0_C'].astype(float)
        # attempt to find mid and max concrete temps if available
        concrete_zmid = df['concrete_T_z_mid_C'].astype(float) if 'concrete_T_z_mid_C' in df.columns else None
        concrete_zmax = df['concrete_T_z_max_C'].astype(float) if 'concrete_T_z_max_C' in df.columns else None
        axes[2].plot(df.index, concrete_z0, color='#9b59b6', lw=2, label='z=0')
        if concrete_zmid is not None:
            axes[2].plot(df.index, concrete_zmid, color='#8e44ad', lw=1.5, label='z=mid')
        if concrete_zmax is not None:
            axes[2].plot(df.index, concrete_zmax, color='#5e3370', lw=1.5, label='z=max')
        axes[2].set_ylabel('Concrete T [C]')
        axes[2].set_title('Concrete temperature (z positions)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(df.index, t_out, color='#e67e22', lw=2)
        axes[3].set_ylabel('Fluid outlet temperature [C]')
        axes[3].set_title('Fluid outlet temperature')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].plot(df.index, t_out, color='#e67e22', lw=2)
        axes[2].set_ylabel('Fluid outlet temperature [C]')
        axes[2].set_title('Fluid outlet temperature')
        axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_tick_params(rotation=20)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    # derive timestamp from CSV filename when possible
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(csv_path))
    if m:
        ts = m.group(1)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if out_pdf is None:
        out_pdf = os.path.join(os.path.dirname(csv_path), f'simulation_plots_{ts}.pdf')

    # save vector PDF
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Plots saved: {out_pdf}')

    # Combined Collector + Flows figure (collector plots + flows in same figure)
    has_collector = 'solar_power_W' in df.columns or 'collector_t_out_C' in df.columns
    flow_cols = [c for c in ['m_col_vol_flow_m3s','m_col_use_m3s','m_ctes_use_m3s','m_oil_hex_m3s'] if c in df.columns]
    if has_collector or flow_cols:
        nsub = 2 + (1 if ('collector_t_out_C' in df.columns) else 0)
        # We'll create a 3-row figure: collector power, collector temp (if present), flows
        figC, axsC = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # collector power
        if 'solar_power_W' in df.columns:
            axsC[0].plot(df.index, df['solar_power_W'].astype(float) / 1e3, color='#f39c12')
            axsC[0].set_ylabel('Collector power [kW]')
            axsC[0].grid(True, alpha=0.3)
        else:
            axsC[0].text(0.5, 0.5, 'No collector power data', ha='center')

        # collector outlet temp (if present)
        if 'collector_t_out_C' in df.columns:
            axsC[1].plot(df.index, df['collector_t_out_C'].astype(float), color='#1abc9c')
            axsC[1].set_ylabel('Collector outlet T [C]')
            axsC[1].grid(True, alpha=0.3)
        else:
            axsC[1].text(0.5, 0.5, 'No collector temp data', ha='center')

        # flows
        if flow_cols:
            for c in flow_cols:
                axsC[2].plot(df.index, df[c].astype(float), label=c)
            axsC[2].set_ylabel('Flow [m3/s]')
            axsC[2].legend()
            axsC[2].grid(True, alpha=0.3)
        else:
            axsC[2].text(0.5, 0.5, 'No flow data', ha='center')

        axsC[2].set_xlabel('Time')
        combined_pdf = os.path.join(os.path.dirname(csv_path), f'simulation_collector_and_flows_{ts}.pdf')
        figC.tight_layout()
        figC.savefig(combined_pdf, bbox_inches='tight')
        print(f'Collector+Flows plots saved: {combined_pdf}')

if __name__ == '__main__':
    latest = _find_latest_sim_csv()
    if latest is not None:
        plot_simulation_results(csv_path=latest)
    else:
        print('No simulation CSV found in', DATA_DIR)
