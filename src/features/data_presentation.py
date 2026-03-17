import glob
import os
import re
from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))

FONT_SIZE = 10.5
FIG_WIDTH_A4 = 8.27


def _find_latest_sim_csv(data_dir=DATA_DIR):
    pattern = os.path.join(data_dir, '**', 'simulation_results_*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _timestamp_from_path(path: str) -> str:
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(path))
    if m:
        return m.group(1)
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _series(df: pd.DataFrame, col: str, scale: float = 1.0, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float) * scale
    return pd.Series(default, index=df.index, dtype=float)


def _apply_plot_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
    })


def _week_window(idx: pd.Index):
    idx = pd.DatetimeIndex(idx)
    if len(idx) == 0:
        return None, None
    saturday_points = idx[idx.weekday == 5]
    x0 = saturday_points[0] if len(saturday_points) > 0 else idx[0]
    x1 = min(x0 + pd.Timedelta(days=7) - pd.Timedelta(minutes=10), idx[-1])
    if x1 <= x0:
        x0, x1 = idx[0], idx[-1]
    return x0, x1


def _weekday_label(x, _pos):
    return mdates.num2date(x).strftime('%A')


def _format_time_axis(ax, x0, x1):
    ax.set_xlim(x0, x1)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_weekday_label))


def _legend_below(ax, ncol=3):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=ncol, frameon=True)


def _align_soc_axes(ax_soc, ax_energy, e_ctes_MWh: pd.Series):
    """Align SoC (%) and energy (MWh) twin y-axes on the same vertical scale."""
    soc_ticks = np.arange(0.0, 101.0, 20.0)
    ax_soc.set_ylim(0.0, 100.0)
    ax_soc.set_yticks(soc_ticks)

    if e_ctes_MWh.notna().any():
        e_min = float(e_ctes_MWh.min(skipna=True))
        e_max = float(e_ctes_MWh.max(skipna=True))
        if np.isfinite(e_min) and np.isfinite(e_max):
            if abs(e_max - e_min) < 1e-12:
                e_max = e_min + 1.0
            ax_energy.set_ylim(e_min, e_max)
            e_ticks = e_min + (soc_ticks / 100.0) * (e_max - e_min)
            ax_energy.set_yticks(e_ticks)


def plot_simulation_results(csv_path: Optional[str] = None, out_pdf: Optional[str] = None):
    if csv_path is None:
        csv_path = _find_latest_sim_csv()
        if csv_path is None:
            raise FileNotFoundError(f'No simulation CSV found in {DATA_DIR}')

    _apply_plot_style()

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    ts = _timestamp_from_path(csv_path)
    out_dir = os.path.dirname(csv_path)
    x0, x1 = _week_window(df.index)

    # Core solar power channels [kW]
    p_solar_raw = _series(df, 'solar_power_raw_W', 1 / 1000.0)
    p_solar_raw = p_solar_raw.where(p_solar_raw > 1e-6, 0.0)
    p_solar_used = _series(df, 'solar_power_W', 1 / 1000.0)
    p_solar_to_hex = _series(df, 'solar_contribution_W', 1 / 1000.0)
    p_solar_to_ctes = _series(df, 'ctes_charge_input_W', 1 / 1000.0)
    p_solar_curt = _series(df, 'solar_power_curtailed_W', 1 / 1000.0)
    dni = _series(df, 'dni_W_m2', 1.0)
    dni = dni.where(dni > 1e-6, 0.0)

    # CTES thermal powers [kW] with sign convention requested by user
    p_ctes_chg = _series(df, 'ctes_charge_input_W', 1 / 1000.0)
    p_ctes_dis = -_series(df, 'ctes_discharge_output_W', 1 / 1000.0)
    p_ctes_loss = _series(df, 'ctes_current_loss_kW', 1.0)

    soc_pct = _series(df, 'ctes_soc_pct', 1.0)
    e_ctes_MWh = _series(df, 'ctes_energy_MWh', 1.0)
    if e_ctes_MWh.isna().all():
        e_ctes_MWh = _series(df, 'ctes_energy_J', 1 / 3.6e9)

    # Figure 1: three stacked panels
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_WIDTH_A4, 8.8), sharex=True)

    ax1.plot(df.index, p_solar_raw, label='Potential & DNI', lw=1.3, color='black')
    ax1.plot(df.index, p_solar_used, label='Used', lw=1.3)
    ax1.plot(df.index, p_solar_to_hex, label='To HEX', lw=1.3)
    ax1.plot(df.index, p_solar_to_ctes, label='To CTES', lw=1.3)
    ax1.plot(df.index, p_solar_curt, label='Curtailment', lw=1.3, color='red')
    ax1.set_ylabel('Collector power [kW]')
    ax1.grid(True, alpha=0.3)
    ax1r = ax1.twinx()
    ax1r.plot(df.index, dni, color='black', lw=1.0, ls='-', label='_nolegend_')
    ax1r.set_ylabel('DNI [W/m$^2$]')
    valid_ratio = (dni > 1e-9) & np.isfinite(dni) & np.isfinite(p_solar_raw)
    if valid_ratio.any():
        ratio_kw_per_wm2 = float((p_solar_raw[valid_ratio] / dni[valid_ratio]).median())
        if ratio_kw_per_wm2 > 0:
            y0, y1 = ax1.get_ylim()
            ax1r.set_ylim(y0 / ratio_kw_per_wm2, y1 / ratio_kw_per_wm2)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True)

    ax2.plot(df.index, p_ctes_chg, label='CTES charging input', lw=1.3)
    ax2.plot(df.index, p_ctes_dis, label='CTES discharging output', lw=1.3)
    ax2.plot(df.index, p_ctes_loss, label='CTES thermal loss', lw=1.3)
    ax2.set_ylabel('CTES thermal power [kW]')
    ax2.grid(True, alpha=0.3)
    _legend_below(ax2, ncol=2)

    ax3.plot(df.index, soc_pct, lw=1.5)
    ax3.set_ylabel('CTES SoC [%]')
    ax3.grid(True, alpha=0.3)
    ax3r = ax3.twinx()
    _align_soc_axes(ax3, ax3r, e_ctes_MWh)
    ax3r.set_ylabel('CTES energy [MWh]')

    _format_time_axis(ax1, x0, x1)
    _format_time_axis(ax2, x0, x1)
    _format_time_axis(ax3, x0, x1)

    fig1.tight_layout()
    power_pdf = out_pdf if out_pdf is not None else os.path.join(out_dir, f'thermal_power_overview_{ts}.pdf')
    fig1.savefig(power_pdf, bbox_inches='tight')
    print(f'Figure saved: {power_pdf}')

    # Figure 2: stacked-area alternative + SoC panel
    fig2, (ax21, ax22, ax23) = plt.subplots(3, 1, figsize=(FIG_WIDTH_A4, 8.2), sharex=True)

    ax21.stackplot(
        df.index,
        [p_solar_to_hex.clip(lower=0.0), p_solar_to_ctes.clip(lower=0.0), p_solar_curt.clip(lower=0.0)],
        labels=['To HEX', 'To CTES', 'Curtailment'],
        alpha=0.75,
        colors=['tab:green', 'tab:purple', 'red'],
    )
    ax21.plot(df.index, p_solar_raw, color='black', lw=1.0, ls='-', label='Potential/DNI')
    ax21.set_ylabel('Power [kW]')
    ax21.grid(True, alpha=0.3)
    ax21r = ax21.twinx()
    ax21r.plot(df.index, dni, color='black', lw=1.0, ls='-')
    ax21r.set_ylabel('DNI [W/m$^2$]')
    ax21.set_ylim(bottom=0.0)
    ax21r.set_ylim(bottom=0.0)
    _legend_below(ax21, ncol=2)

    p_factory_solar = _series(df, 'solar_to_factory_kW', 1.0)
    p_factory_ctes = _series(df, 'ctes_to_factory_kW', 1.0)
    p_factory_backup = _series(df, 'backup_heater_kW', 1.0)
    p_factory_load = _series(df, 'factory_load_kW', 1.0)
    ax22.stackplot(
        df.index,
        [p_factory_solar.clip(lower=0.0), p_factory_ctes.clip(lower=0.0), p_factory_backup.clip(lower=0.0)],
        labels=['Factory from solar', 'Factory from CTES', 'Factory from backup'],
        alpha=0.70,
        colors=['tab:blue', 'tab:orange', 'red'],
    )
    ax22.plot(df.index, p_factory_load, color='black', lw=1.0, ls='-', label='Factory thermal demand')
    ax22.set_ylabel('Power [kW]')
    ax22.grid(True, alpha=0.3)
    _legend_below(ax22, ncol=2)

    ax23.plot(df.index, soc_pct, lw=1.5)
    ax23.set_ylabel('CTES SoC [%]')
    ax23.grid(True, alpha=0.3)
    ax23r = ax23.twinx()
    _align_soc_axes(ax23, ax23r, e_ctes_MWh)
    ax23r.set_ylabel('CTES energy [MWh]')

    _format_time_axis(ax21, x0, x1)
    _format_time_axis(ax22, x0, x1)
    _format_time_axis(ax23, x0, x1)

    fig2.tight_layout()
    stacked_pdf = os.path.join(out_dir, f'thermal_power_overview_stacked_{ts}.pdf')
    fig2.savefig(stacked_pdf, bbox_inches='tight')
    print(f'Figure saved: {stacked_pdf}')

    # Figure 3: HTF temperatures + concrete profile temperatures
    fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(FIG_WIDTH_A4, 7.1), sharex=True)

    t_return = _series(df, 'htf_in_temp_C', 1.0)
    t_col_out = _series(df, 'collector_t_out_C', 1.0)
    t_hex_in = _series(df, 'hex_inlet_temp_C', 1.0)
    t_hex_out = _series(df, 'oil_out_C', 1.0)
    t_ctes_in = _series(df, 'ctes_inlet_temp_C', 1.0)
    t_ctes_out = _series(df, 'ctes_temp_C', 1.0)

    ax31.plot(df.index, t_return, label='HTF return', lw=1.2, color='tab:blue')
    ax31.plot(df.index, t_col_out, label='Collector outlet', lw=1.2, color='tab:orange')
    ax31.plot(df.index, t_hex_in, label='HEX inlet', lw=1.3, color='tab:green')
    ax31.plot(df.index, t_hex_out, label='HEX outlet', lw=1.2, color='tab:red', ls='--')
    ax31.plot(df.index, t_ctes_in, label='CTES inlet', lw=1.2, color='tab:purple', ls=':')
    ax31.plot(df.index, t_ctes_out, label='CTES outlet', lw=1.2, color='tab:brown', ls='-.')
    ax31.set_ylabel('Temperature [C]')
    ax31.grid(True, alpha=0.3)
    _legend_below(ax31, ncol=3)

    t_z0 = _series(df, 'concrete_T_z0_C', 1.0)
    t_zmid = _series(df, 'concrete_T_z_mid_C', 1.0)
    t_zmax = _series(df, 'concrete_T_z_max_C', 1.0)
    ax32.plot(df.index, t_z0, label='z=0', lw=1.2)
    ax32.plot(df.index, t_zmid, label='z=mid', lw=1.2)
    ax32.plot(df.index, t_zmax, label='z=max', lw=1.2)
    ax32.set_ylabel('$T_{CTES}$ [°C]')
    ax32.grid(True, alpha=0.3)
    _legend_below(ax32, ncol=3)

    _format_time_axis(ax31, x0, x1)
    _format_time_axis(ax32, x0, x1)

    fig3.tight_layout()
    temps_pdf = os.path.join(out_dir, f'htf_and_ctes_temperatures_{ts}.pdf')
    fig3.savefig(temps_pdf, bbox_inches='tight')
    print(f'Figure saved: {temps_pdf}')

    # Figure 4: key HTF flow rates + HEX flow share fractions
    fig4, (ax4, ax5) = plt.subplots(2, 1, figsize=(FIG_WIDTH_A4, 6.0), sharex=True)

    f_col_total = _series(df, 'm_col_vol_flow_m3s', 1.0)
    f_col_hex = _series(df, 'm_col_use_m3s', 1.0)
    f_col_to_ctes = _series(df, 'm_ctes_charge_use_m3s', 1.0)
    f_ctes_to_hex = _series(df, 'm_ctes_use_m3s', 1.0)
    f_hex_to_ctes = f_ctes_to_hex.copy()
    f_recirc = _series(df, 'm_recirc_m3s', 1.0)

    ax4.plot(df.index, f_col_total, label='Collector total', lw=1.2)
    ax4.plot(df.index, f_col_to_ctes, label='Collector to CTES', lw=1.2)
    ax4.plot(df.index, f_ctes_to_hex, label='CTES to HEX', lw=1.2)
    ax4.plot(df.index, f_hex_to_ctes, label='HEX to CTES return', lw=1.0, ls='--')
    ax4.plot(df.index, f_recirc, label='HEX recirculation', lw=1.2)
    ax4.set_ylabel('Flow [m$^3$/s]')
    ax4.grid(True, alpha=0.3)
    _legend_below(ax4, ncol=3)

    flow_total = _series(df, 'm_oil_hex_m3s', 1.0).replace(0.0, np.nan)
    frac_solar = 100.0 * (f_col_hex / flow_total)
    frac_ctes = 100.0 * (f_ctes_to_hex / flow_total)
    frac_recirc = 100.0 * (f_recirc / flow_total)
    ax5.stackplot(
        df.index,
        [frac_solar.fillna(0.0), frac_ctes.fillna(0.0), frac_recirc.fillna(0.0)],
        labels=['Solar', 'CTES', 'Recirculation'],
        alpha=0.75,
        colors=['tab:blue', 'tab:orange', 'tab:green'],
    )
    ax5.set_ylabel('HEX flow share [%]')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    _legend_below(ax5, ncol=3)

    _format_time_axis(ax4, x0, x1)
    _format_time_axis(ax5, x0, x1)

    fig4.tight_layout()
    flow_pdf = os.path.join(out_dir, f'htf_flow_rates_{ts}.pdf')
    fig4.savefig(flow_pdf, bbox_inches='tight')
    print(f'Figure saved: {flow_pdf}')


if __name__ == '__main__':
    latest = _find_latest_sim_csv()
    if latest is not None:
        plot_simulation_results(csv_path=latest)
    else:
        print('No simulation CSV found in', DATA_DIR)
