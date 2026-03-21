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


def _find_summary_csv_for_results(results_csv_path: str) -> Optional[str]:
    out_dir = os.path.dirname(results_csv_path)
    candidates = glob.glob(os.path.join(out_dir, 'simulation_summary_*.csv'))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _read_summary_csv(summary_csv_path: str) -> pd.DataFrame:
    # Written by Part_1 with sep=';' and decimal=','.
    try:
        return pd.read_csv(summary_csv_path, sep=';', decimal=',')
    except Exception:
        return pd.read_csv(summary_csv_path)


def _write_summary_table_from_csv(summary_csv_path: str, out_dir: str) -> Optional[str]:
    try:
        sdf = _read_summary_csv(summary_csv_path)
        if sdf.empty:
            return None
        row = sdf.iloc[0]
    except Exception:
        return None

    def _get(name: str):
        v = row.get(name, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    def _fmt(v: float, pct: bool = False) -> str:
        if not np.isfinite(v):
            return '--'
        if pct:
            return f"{100.0 * float(v):.2f}"
        return f"{float(v):.2f}"

    def _unit_tex(unit: str) -> str:
        return unit.replace('%', r'\%')

    rows_main = [
        ('Factory heat demand', _get('total_factory_consumed_MWh'), '[MWh]', False),
        ('Solar thermal production', _get('total_solar_produced_MWh'), '[MWh]', False),
        ('Solar curtailment', _get('total_solar_curtailed_MWh'), '[MWh]', False),
        ('Solar to factory (direct)', _get('total_solar_supplied_to_factory_MWh'), '[MWh]', False),
        ('CTES to factory', _get('total_ctes_supplied_to_factory_MWh'), '[MWh]', False),
        ('CTES charge input', _get('total_ctes_charge_input_MWh'), '[MWh]', False),
        ('CTES discharge output', _get('total_ctes_discharge_output_MWh'), '[MWh]', False),
        ('Backup heater energy', _get('total_backup_heater_MWh'), '[MWh]', False),
        ('CTES stored energy at start', 0.0 if abs(_get('ctes_energy_start_MWh')) < 0.05 else _get('ctes_energy_start_MWh'), '[MWh]', False),
        ('CTES stored energy at end', _get('ctes_energy_end_MWh'), '[MWh]', False),
        ('CTES thermal loss', _get('total_ctes_loss_abs_MWh'), '[MWh]', False),
    ]
    produced_MWh = _get('total_solar_produced_MWh')
    curtailed_MWh = _get('total_solar_curtailed_MWh')
    denom_capture = produced_MWh + curtailed_MWh
    capture_eff = produced_MWh / denom_capture if np.isfinite(denom_capture) and denom_capture > 0.0 else np.nan

    rows_pct = [
        ('Solar fraction (direct + CTES)', _get('solar_fraction'), '[%]', True),
        ('Available solar captured', capture_eff, '[%]', True),
        ('CTES RTE (SoC-adjusted)', _get('ctes_round_trip_efficiency'), '[%]', True),
    ]

    latex_lines = [
        '\\renewcommand{\\arraystretch}{1.2}',
        '\\begin{table}[h]',
        '\\centering',
        '\\begin{tabular}{l|r c}',
        '\\hline',
        'Parameter & Value & Unit \\\\',
        '\\hline',
    ]
    for name, val, unit, is_pct in rows_main:
        latex_lines.append(f'{name} & {_fmt(val, pct=is_pct)} & {_unit_tex(unit)} \\\\')
    latex_lines.append('\\hline')
    for name, val, unit, is_pct in rows_pct:
        latex_lines.append(f'{name} & {_fmt(val, pct=is_pct)} & {_unit_tex(unit)} \\\\')
    latex_lines.extend([
        '\\hline',
        '\\end{tabular}',
        '\\caption{Summary of CTES simulation energy flows and key performance indicators.}',
        '\\label{tab:ctes-simulation-summary}',
        '\\end{table}',
        ''
    ])

    latex_path = os.path.join(out_dir, 'simulation_summary_table.tex')
    try:
        with open(latex_path, 'w', encoding='utf-8') as f_ltx:
            f_ltx.write('\n'.join(latex_lines))
        print(f'Figure table saved: {latex_path}')
        return latex_path
    except Exception:
        return None


def _align_soc_axes(ax_soc, ax_energy, soc_pct: pd.Series, e_ctes_MWh: pd.Series):
    """Align SoC (%) and energy (MWh) twin y-axes on the same vertical scale."""
    soc_ticks = np.arange(0.0, 101.0, 20.0)
    ax_soc.set_ylim(0.0, 100.0)
    ax_soc.set_yticks(soc_ticks)

    if e_ctes_MWh.notna().any():
        e_full = np.nan
        valid = (soc_pct > 1e-6) & soc_pct.notna() & e_ctes_MWh.notna()
        if valid.any():
            # Infer 100% energy from E/(SOC/100) to keep both axes physically aligned.
            e_full = float((e_ctes_MWh[valid] / (soc_pct[valid] / 100.0)).median())
        if not np.isfinite(e_full) or e_full <= 0.0:
            e_full = float(e_ctes_MWh.max(skipna=True))
        if np.isfinite(e_full) and e_full > 0.0:
            ax_energy.set_ylim(0.0, e_full)
            e_ticks = (soc_ticks / 100.0) * e_full
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

    # Build LaTeX summary table from existing summary CSV in the same output folder.
    summary_csv = _find_summary_csv_for_results(csv_path)
    if summary_csv is not None:
        _write_summary_table_from_csv(summary_csv, out_dir)

    # Core solar power channels [kW]
    p_solar_raw = _series(df, 'solar_power_raw_W', 1 / 1000.0)
    p_solar_raw = p_solar_raw.where(p_solar_raw > 1e-6, 0.0)
    p_solar_used = _series(df, 'solar_power_W', 1 / 1000.0)
    p_solar_to_hex = _series(df, 'solar_contribution_W', 1 / 1000.0)
    # Use direct HTF->CTES charging channel for solar split plots.
    p_solar_to_ctes = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)
    if p_solar_to_ctes.isna().all():
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
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_WIDTH_A4, 7.6), sharex=True)

    ax1.plot(df.index, p_solar_raw, label='Potential & DNI', lw=2.6, color='black')
    ax1.plot(df.index, p_solar_used, label='Used', lw=2.6)
    ax1.plot(df.index, p_solar_to_hex, label='To HEX', lw=2.6)
    ax1.plot(df.index, p_solar_to_ctes, label='To CTES', lw=2.6)
    ax1.plot(df.index, p_solar_curt, label='Curtailment', lw=2.6, color='red')
    ax1.set_ylabel('Collector power [kW]')
    ax1.grid(True, alpha=0.3)
    ax1r = ax1.twinx()
    ax1r.plot(df.index, dni, color='black', lw=2.0, ls='-', label='_nolegend_')
    ax1r.set_ylabel('DNI [W/m$^2$]')
    valid_ratio = (dni > 1e-9) & np.isfinite(dni) & np.isfinite(p_solar_raw)
    if valid_ratio.any():
        ratio_kw_per_wm2 = float((p_solar_raw[valid_ratio] / dni[valid_ratio]).median())
        if ratio_kw_per_wm2 > 0:
            y0, y1 = ax1.get_ylim()
            ax1r.set_ylim(y0 / ratio_kw_per_wm2, y1 / ratio_kw_per_wm2)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True)

    ax2.plot(df.index, p_ctes_chg, label='Charging', lw=2.6)
    ax2.plot(df.index, p_ctes_dis, label='Discharging', lw=2.6)
    ax2.plot(df.index, p_ctes_loss, label='Thermal loss', lw=2.6)
    ax2.set_ylabel('CTES thermal power [kW]')
    ax2.grid(True, alpha=0.3)
    _legend_below(ax2, ncol=2)

    ax3.plot(df.index, soc_pct, lw=3.0)
    ax3.set_ylabel('CTES SoC [%]')
    ax3.grid(True, alpha=0.3)
    ax3r = ax3.twinx()
    _align_soc_axes(ax3, ax3r, soc_pct, e_ctes_MWh)
    ax3r.set_ylabel('CTES energy [MWh]')

    _format_time_axis(ax1, x0, x1)
    _format_time_axis(ax2, x0, x1)
    _format_time_axis(ax3, x0, x1)

    fig1.tight_layout()
    power_pdf = out_pdf if out_pdf is not None else os.path.join(out_dir, 'thermal_power_overview.pdf')
    fig1.savefig(power_pdf, bbox_inches='tight')
    print(f'Figure saved: {power_pdf}')

    # Figure 2: stacked-area alternative with CTES power and SoC panels
    fig2, (ax21, ax22, ax23, ax24) = plt.subplots(4, 1, figsize=(FIG_WIDTH_A4, 11.6), sharex=True)

    ax21.stackplot(
        df.index,
        [p_solar_to_hex.clip(lower=0.0), p_solar_to_ctes.clip(lower=0.0), p_solar_curt.clip(lower=0.0)],
        labels=['To HEX', 'To CTES', 'Curtailment'],
        alpha=0.75,
        colors=['tab:green', 'tab:purple', 'red'],
    )
    ax21.plot(df.index, p_solar_raw, color='black', lw=2.0, ls='-', label='Potential & DNI')
    ax21.set_ylabel('Collector power [kW]')
    ax21.grid(True, alpha=0.3)
    ax21r = ax21.twinx()
    ax21r.plot(df.index, dni, color='black', lw=2.0, ls='-')
    ax21r.set_ylabel('DNI [W/m$^2$]')
    ax21.set_ylim(bottom=0.0)
    ax21r.set_ylim(bottom=0.0)
    _legend_below(ax21, ncol=2)

    ax22.plot(df.index, p_ctes_chg, label='Charging', lw=2.6, color='tab:green')
    ax22.plot(df.index, p_ctes_dis, label='Discharging', lw=2.6, color='tab:orange')
    ax22.plot(df.index, p_ctes_loss, label='Thermal loss', lw=2.4, color='tab:red')
    ax22.axhline(0.0, color='black', lw=1.6, alpha=0.6)
    ax22.set_ylabel('CTES power [kW]')
    ax22.grid(True, alpha=0.3)
    _legend_below(ax22, ncol=3)

    p_factory_solar = _series(df, 'solar_to_factory_kW', 1.0)
    p_factory_ctes = _series(df, 'ctes_to_factory_kW', 1.0)
    p_factory_backup = _series(df, 'backup_heater_kW', 1.0)
    p_factory_load = _series(df, 'factory_load_kW', 1.0)
    ax23.stackplot(
        df.index,
        [p_factory_solar.clip(lower=0.0), p_factory_ctes.clip(lower=0.0), p_factory_backup.clip(lower=0.0)],
        labels=['From collectors', 'From CTES', 'From backup heater'],
        alpha=0.70,
        colors=['tab:blue', 'tab:orange', 'red'],
    )
    ax23.plot(df.index, p_factory_load, color='black', lw=2.0, ls='-', label='Factory thermal demand')
    ax23.set_ylabel('Factory power [kW]')
    ax23.grid(True, alpha=0.3)
    _legend_below(ax23, ncol=2)

    ax24.plot(df.index, soc_pct, lw=3.0)
    ax24.set_ylabel('CTES SoC [%]')
    ax24.grid(True, alpha=0.3)
    ax24r = ax24.twinx()
    _align_soc_axes(ax24, ax24r, soc_pct, e_ctes_MWh)
    ax24r.set_ylabel('CTES energy [MWh]')

    _format_time_axis(ax21, x0, x1)
    _format_time_axis(ax22, x0, x1)
    _format_time_axis(ax23, x0, x1)
    _format_time_axis(ax24, x0, x1)

    fig2.tight_layout()
    fig2.subplots_adjust(hspace=0.38)
    stacked_pdf = os.path.join(out_dir, 'thermal_power_overview_stacked.pdf')
    fig2.savefig(stacked_pdf, bbox_inches='tight')
    print(f'Figure saved: {stacked_pdf}')

    # Figure 3: HTF temperatures + concrete profile temperatures
    fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(FIG_WIDTH_A4, 6.1), sharex=True)

    t_return = _series(df, 'htf_in_temp_C', 1.0)
    t_col_out = _series(df, 'collector_t_out_C', 1.0)
    t_hex_in = _series(df, 'hex_inlet_temp_C', 1.0)
    t_hex_out = _series(df, 'oil_out_C', 1.0)
    t_ctes_in = _series(df, 'ctes_inlet_temp_C', 1.0)
    t_ctes_out = _series(df, 'ctes_temp_C', 1.0)

    ax31.plot(df.index, t_return, label='HTF return', lw=2.4, color='tab:blue')
    ax31.plot(df.index, t_col_out, label='Collector outlet', lw=2.4, color='tab:orange')
    ax31.plot(df.index, t_hex_in, label='HEX inlet', lw=2.6, color='tab:green')
    ax31.plot(df.index, t_hex_out, label='HEX outlet', lw=2.4, color='tab:red', ls='--')
    ax31.plot(df.index, t_ctes_in, label='CTES inlet', lw=2.4, color='tab:purple', ls=':')
    ax31.plot(df.index, t_ctes_out, label='CTES outlet', lw=2.4, color='tab:brown', ls='-.')
    ax31.set_ylabel('Temperature [C]')
    ax31.grid(True, alpha=0.3)
    _legend_below(ax31, ncol=3)

    t_z0 = _series(df, 'concrete_T_z0_C', 1.0)
    t_zmid = _series(df, 'concrete_T_z_mid_C', 1.0)
    t_zmax = _series(df, 'concrete_T_z_max_C', 1.0)
    ax32.plot(df.index, t_z0, label='z=0', lw=2.4)
    ax32.plot(df.index, t_zmid, label='z=mid', lw=2.4)
    ax32.plot(df.index, t_zmax, label='z=max', lw=2.4)
    ax32.set_ylabel('$T_{CTES}$ [°C]')
    ax32.grid(True, alpha=0.3)
    _legend_below(ax32, ncol=3)

    _format_time_axis(ax31, x0, x1)
    _format_time_axis(ax32, x0, x1)

    fig3.tight_layout()
    temps_pdf = os.path.join(out_dir, 'htf_and_ctes_temperatures.pdf')
    fig3.savefig(temps_pdf, bbox_inches='tight')
    print(f'Figure saved: {temps_pdf}')

    # Figure 4: key HTF flow rates + HEX flow share fractions
    fig4, (ax4, ax5) = plt.subplots(2, 1, figsize=(FIG_WIDTH_A4, 6.0), sharex=True)

    f_col_total = _series(df, 'm_col_vol_flow_m3s', 1.0)
    f_col_hex = _series(df, 'm_col_use_m3s', 1.0)
    f_col_to_ctes = _series(df, 'm_ctes_charge_use_m3s', 1.0)
    f_ctes_to_hex = _series(df, 'm_ctes_use_m3s', 1.0)
    f_recirc = _series(df, 'm_recirc_m3s', 1.0)

    ax4.plot(df.index, f_col_total, label='Collector total', lw=2.4)
    ax4.plot(df.index, f_col_hex, label='Collector to HEX', lw=2.4)
    ax4.plot(df.index, f_col_to_ctes, label='Collector to CTES', lw=2.4)
    ax4.plot(df.index, f_ctes_to_hex, label='CTES to HEX', lw=2.4)
    ax4.plot(df.index, f_recirc, label='HEX recirculation', lw=2.4, ls='--')
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
    flow_pdf = os.path.join(out_dir, 'htf_flow_rates.pdf')
    fig4.savefig(flow_pdf, bbox_inches='tight')
    print(f'Figure saved: {flow_pdf}')

    # Figure 5: CTES charging diagnostics (raw vs net transfer and curtailment)
    fig5, (ax51, ax52) = plt.subplots(2, 1, figsize=(FIG_WIDTH_A4, 6.8), sharex=True)

    p_ctes_charge_raw = _series(df, 'ctes_charge_raw_W', 1 / 1000.0)
    # Backward compatibility when older CSVs do not include raw/net split yet.
    if p_ctes_charge_raw.isna().all():
        p_ctes_charge_raw = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)
    p_ctes_charge_net = _series(df, 'ctes_charge_net_W', 1 / 1000.0)
    if p_ctes_charge_net.isna().all():
        p_ctes_charge_net = _series(df, 'ctes_charge_input_W', 1 / 1000.0)
    p_ctes_charge_inf = _series(df, 'ctes_charge_inferred_W', 1 / 1000.0)
    p_ctes_loss = _series(df, 'ctes_current_loss_kW', 1.0)

    ax51.plot(df.index, p_ctes_charge_raw, label='Charge raw (fluid->solid)', lw=2.6, color='tab:green')
    ax51.plot(df.index, p_ctes_charge_net, label='Charge net (to solid)', lw=2.4, color='tab:blue')
    ax51.plot(df.index, p_ctes_charge_inf, label='Charge inferred', lw=2.0, color='tab:purple', ls=':')
    ax51.plot(df.index, p_ctes_loss, label='Thermal loss', lw=2.0, color='tab:red')
    ax51.axhline(0.0, color='black', lw=1.3, alpha=0.6)
    ax51.set_ylabel('CTES power [kW]')
    ax51.grid(True, alpha=0.3)
    _legend_below(ax51, ncol=2)

    p_solar_raw = _series(df, 'solar_power_raw_W', 1 / 1000.0)
    p_solar_eff = _series(df, 'solar_power_effective_W', 1 / 1000.0)
    p_solar_curt = _series(df, 'solar_power_curtailed_W', 1 / 1000.0)
    p_col_to_ctes = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)
    ax52.plot(df.index, p_solar_raw, label='Solar potential', lw=2.4, color='black')
    ax52.plot(df.index, p_solar_eff, label='Solar effective', lw=2.2, color='tab:orange')
    ax52.plot(df.index, p_solar_curt, label='Solar curtailed', lw=2.2, color='tab:red')
    ax52.plot(df.index, p_col_to_ctes, label='To CTES channel', lw=2.2, color='tab:green', ls='--')
    ax52.set_ylabel('Solar / channel [kW]')
    ax52.grid(True, alpha=0.3)
    _legend_below(ax52, ncol=2)

    _format_time_axis(ax51, x0, x1)
    _format_time_axis(ax52, x0, x1)

    fig5.tight_layout()
    diag_pdf = os.path.join(out_dir, 'ctes_charging_diagnostics.pdf')
    fig5.savefig(diag_pdf, bbox_inches='tight')
    print(f'Figure saved: {diag_pdf}')


if __name__ == '__main__':
    latest = _find_latest_sim_csv()
    if latest is not None:
        plot_simulation_results(csv_path=latest)
    else:
        print('No simulation CSV found in', DATA_DIR)
