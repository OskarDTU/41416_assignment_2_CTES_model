import glob
import os
import re
import warnings
from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def _full_window(idx: pd.Index):
    idx = pd.DatetimeIndex(idx)
    if len(idx) == 0:
        return None, None
    return idx[0], idx[-1]


def _time_span_days(idx: pd.Index) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 0.0
    return float((idx[-1] - idx[0]) / pd.Timedelta(days=1))


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


def _module_temp_columns(df: pd.DataFrame):
    cols_avg = []
    cols_out = []
    for col in df.columns:
        m_avg = re.match(r'module_(\d+)_Ts_avg_C$', str(col))
        if m_avg:
            cols_avg.append((int(m_avg.group(1)), col))
            continue
        m_out = re.match(r'module_(\d+)_Ts_outlet_C$', str(col))
        if m_out:
            cols_out.append((int(m_out.group(1)), col))
    cols_avg.sort(key=lambda x: x[0])
    cols_out.sort(key=lambda x: x[0])
    return [c for _, c in cols_avg] if cols_avg else [c for _, c in cols_out]


def plot_simulation_results(
    csv_path: Optional[str] = None,
    out_pdf: Optional[str] = None,
    use_wide_for_long_runs: bool = True,
    long_run_days_threshold: float = 13.0,
    long_run_width_scale: float = 2.0,
    show_full_window_for_long_runs: bool = True,
):
    if csv_path is None:
        csv_path = _find_latest_sim_csv()
        if csv_path is None:
            raise FileNotFoundError(f'No simulation CSV found in {DATA_DIR}')

    _apply_plot_style()

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    ts = _timestamp_from_path(csv_path)
    out_dir = os.path.dirname(csv_path)
    is_long_run = _time_span_days(df.index) >= float(long_run_days_threshold)
    if show_full_window_for_long_runs and is_long_run:
        x0, x1 = _full_window(df.index)
    else:
        x0, x1 = _week_window(df.index)
    width_scale = float(long_run_width_scale) if (use_wide_for_long_runs and is_long_run) else 1.0
    fig_width = FIG_WIDTH_A4 * width_scale

    # Build LaTeX summary table from existing summary CSV in the same output folder.
    summary_csv = _find_summary_csv_for_results(csv_path)
    if summary_csv is not None:
        _write_summary_table_from_csv(summary_csv, out_dir)

    # Core solar power channels [kW]
    p_solar_raw = _series(df, 'solar_power_raw_W', 1 / 1000.0)
    p_solar_raw = p_solar_raw.where(p_solar_raw > 1e-6, 0.0)
    p_solar_to_hex = _series(df, 'solar_contribution_W', 1 / 1000.0)
    # Use direct HTF->CTES charging channel for solar split plots.
    p_solar_to_ctes = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)
    if p_solar_to_ctes.isna().all():
        p_solar_to_ctes = _series(df, 'ctes_charge_input_W', 1 / 1000.0)
    # "Used" collector power is the routed useful channels only.
    p_solar_used = (p_solar_to_hex + p_solar_to_ctes).clip(lower=0.0)
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

    p_factory_solar = _series(df, 'solar_to_factory_kW', 1.0)
    p_factory_ctes = _series(df, 'ctes_to_factory_kW', 1.0)
    p_factory_backup = _series(df, 'backup_heater_kW', 1.0)
    p_factory_load = _series(df, 'factory_load_kW', 1.0)

    f_col_total = _series(df, 'm_col_vol_flow_m3s', 1.0)
    f_col_hex = _series(df, 'm_col_use_m3s', 1.0)
    f_col_to_ctes = _series(df, 'm_ctes_charge_use_m3s', 1.0)
    f_ctes_to_hex = _series(df, 'm_ctes_use_m3s', 1.0)
    f_recirc = _series(df, 'm_recirc_m3s', 1.0)
    flow_total = _series(df, 'm_oil_hex_m3s', 1.0).replace(0.0, np.nan)
    frac_solar = 100.0 * (f_col_hex / flow_total)
    frac_ctes = 100.0 * (f_ctes_to_hex / flow_total)
    frac_recirc = 100.0 * (f_recirc / flow_total)

    t_return = _series(df, 'htf_in_temp_C', 1.0)
    t_col_out = _series(df, 'collector_t_out_C', 1.0)
    t_hex_in = _series(df, 'hex_inlet_temp_C', 1.0)
    t_hex_out = _series(df, 'oil_out_C', 1.0)
    t_ctes_in = _series(df, 'ctes_inlet_temp_C', 1.0)
    t_ctes_out = _series(df, 'ctes_temp_C', 1.0)
    t_mix_target = _series(df, 'mix_target_used_C', 1.0)
    if t_mix_target.isna().all():
        t_mix_target = _series(df, 'mix_target_C', 1.0)

    p_ctes_charge_raw = _series(df, 'ctes_charge_raw_W', 1 / 1000.0)
    if p_ctes_charge_raw.isna().all():
        p_ctes_charge_raw = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)
    p_ctes_charge_net = _series(df, 'ctes_charge_net_W', 1 / 1000.0)
    if p_ctes_charge_net.isna().all():
        p_ctes_charge_net = _series(df, 'ctes_charge_input_W', 1 / 1000.0)
    p_ctes_charge_inf = _series(df, 'ctes_charge_inferred_W', 1 / 1000.0)

    p_solar_eff = _series(df, 'solar_power_effective_W', 1 / 1000.0)
    p_col_to_ctes = _series(df, 'ctes_charge_htf_W', 1 / 1000.0)

    module_cols = _module_temp_columns(df)
    n_modules = len(module_cols)
    module_cmap = cm.get_cmap('hot_r')
    # Use a broad hot_r span for stronger visual separation between adjacent modules.
    module_colors = module_cmap(np.linspace(0.1, 0.98, max(n_modules, 1)))
    module_index = np.arange(1, n_modules + 1)
    module_cmap_discrete = ListedColormap(module_colors)
    module_norm_discrete = BoundaryNorm(np.arange(0.5, n_modules + 1.5, 1.0), module_cmap_discrete.N)

    def _draw_collector_power(ax, _fig=None):
        ax.plot(df.index, p_solar_raw, label='Potential/DNI', lw=2.6, color='black')
        ax.plot(df.index, p_solar_used, label='Used', lw=2.6)
        ax.plot(df.index, p_solar_to_hex, label='To HEX', lw=2.6)
        ax.plot(df.index, p_solar_to_ctes, label='To CTES', lw=2.6)
        ax.plot(df.index, p_solar_curt, label='Curtailment', lw=2.6, color='red')
        ax.set_ylabel('Collector power [kW]')
        ax.grid(True, alpha=0.3)
        axr = ax.twinx()
        axr.plot(df.index, dni, color='black', lw=2.0, ls='-', label='_nolegend_')
        axr.set_ylabel('DNI [W/m$^2$]')
        valid_ratio = (dni > 1e-9) & np.isfinite(dni) & np.isfinite(p_solar_raw)
        if valid_ratio.any():
            ratio_kw_per_wm2 = float((p_solar_raw[valid_ratio] / dni[valid_ratio]).median())
            if ratio_kw_per_wm2 > 0:
                y0, y1 = ax.get_ylim()
                axr.set_ylim(y0 / ratio_kw_per_wm2, y1 / ratio_kw_per_wm2)
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True)

    def _draw_ctes_power(ax, _fig=None):
        ax.plot(df.index, p_ctes_chg, label='Charging', lw=2.6, color='tab:green')
        ax.plot(df.index, p_ctes_dis, label='Discharging', lw=2.6, color='tab:orange')
        ax.plot(df.index, p_ctes_loss, label='Thermal loss', lw=2.4, color='tab:red')
        ax.axhline(0.0, color='black', lw=1.4, alpha=0.6)
        ax.set_ylabel('CTES power [kW]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=3)

    def _draw_soc(ax, _fig=None):
        ax.plot(df.index, soc_pct, lw=3.0, color='tab:blue', label='SoC')
        ax.set_ylabel('CTES SoC [%]')
        ax.grid(True, alpha=0.3)
        axr = ax.twinx()
        _align_soc_axes(ax, axr, soc_pct, e_ctes_MWh)
        axr.set_ylabel('CTES energy [MWh]')

    def _draw_factory_power(ax, _fig=None):
        ax.plot(df.index, p_factory_solar, label='From collectors', lw=2.5, color='tab:blue')
        ax.plot(df.index, p_factory_ctes, label='From CTES', lw=2.5, color='tab:orange')
        ax.plot(df.index, p_factory_backup, label='From backup heater', lw=2.5, color='tab:red')
        ax.plot(df.index, p_factory_load, label='Factory thermal demand', lw=2.2, color='black', ls='--')
        ax.set_ylabel('Factory power [kW]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=2)

    def _draw_htf_temps(ax, _fig=None):
        ax.plot(df.index, t_return, label='HTF return', lw=2.4, color='tab:blue')
        ax.plot(df.index, t_col_out, label='Collector outlet', lw=2.4, color='tab:orange')
        ax.plot(df.index, t_hex_in, label='HEX inlet', lw=2.6, color='tab:green')
        ax.plot(df.index, t_mix_target, label='HEX inlet target', lw=2.2, color='lightgreen', ls='--')
        ax.plot(df.index, t_hex_out, label='HEX outlet', lw=2.4, color='tab:red', ls='--')
        ax.plot(df.index, t_ctes_in, label='CTES inlet', lw=2.4, color='tab:purple', ls=':')
        ax.plot(df.index, t_ctes_out, label='CTES outlet', lw=2.4, color='tab:brown', ls='-.')
        ax.set_ylabel('Temperature [C]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=3)

    def _draw_module_temps(ax, fig):
        if n_modules > 0:
            for idx, col in enumerate(module_cols):
                ls = '--' if (idx % 2 == 1) else '-'
                ax.plot(df.index, _series(df, col, 1.0), lw=2.0, color=module_colors[idx], ls=ls)
            sm = cm.ScalarMappable(cmap=module_cmap_discrete, norm=module_norm_discrete)
            sm.set_array([])
            cax = inset_axes(
                ax,
                width='1.8%',
                height='100%',
                loc='lower left',
                bbox_to_anchor=(1.01, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = fig.colorbar(sm, cax=cax, ticks=module_index)
            cbar.set_label('Module #')
            cbar.set_ticklabels([str(i) for i in module_index])
        else:
            ax.plot(df.index, _series(df, 'concrete_T_z0_C', 1.0), lw=2.2, color='tab:blue', label='z=0')
            ax.plot(df.index, _series(df, 'concrete_T_z_mid_C', 1.0), lw=2.2, color='tab:orange', label='z=mid')
            ax.plot(df.index, _series(df, 'concrete_T_z_max_C', 1.0), lw=2.2, color='tab:red', label='z=max')
            _legend_below(ax, ncol=3)
        ax.set_ylabel('$T_{CTES,mod,avg}$ [°C]')
        ax.grid(True, alpha=0.3)

    def _draw_flow_rates(ax, _fig=None):
        ax.plot(df.index, f_col_total, label='Collector total', lw=2.4)
        ax.plot(df.index, f_col_hex, label='Collector to HEX', lw=2.4)
        ax.plot(df.index, f_col_to_ctes, label='Collector to CTES', lw=2.4)
        ax.plot(df.index, f_ctes_to_hex, label='CTES to HEX', lw=2.4)
        ax.plot(df.index, f_recirc, label='HEX recirculation', lw=2.4, ls='--')
        ax.set_ylabel('Flow [m$^3$/s]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=3)

    def _draw_hex_shares(ax, _fig=None):
        ax.plot(df.index, frac_solar, label='Solar', lw=2.5, color='tab:blue')
        ax.plot(df.index, frac_ctes, label='CTES', lw=2.5, color='tab:orange')
        ax.plot(df.index, frac_recirc, label='Recirculation', lw=2.5, color='tab:green', ls='--')
        ax.set_ylabel('HEX flow share [%]')
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=3)

    def _draw_ctes_charge_diag(ax, _fig=None):
        ax.plot(df.index, p_ctes_charge_raw, label='Charge raw (fluid->solid)', lw=2.6, color='tab:green')
        ax.plot(df.index, p_ctes_charge_net, label='Charge net (to solid)', lw=2.4, color='tab:blue')
        ax.plot(df.index, p_ctes_charge_inf, label='Charge inferred', lw=2.0, color='tab:purple', ls=':')
        ax.plot(df.index, p_ctes_loss, label='Thermal loss', lw=2.0, color='tab:red')
        ax.axhline(0.0, color='black', lw=1.3, alpha=0.6)
        ax.set_ylabel('CTES power [kW]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=2)

    def _draw_solar_diag(ax, _fig=None):
        ax.plot(df.index, p_solar_raw, label='Solar potential', lw=2.4, color='black')
        ax.plot(df.index, p_solar_eff, label='Solar effective', lw=2.2, color='tab:orange')
        ax.plot(df.index, p_solar_to_hex, label='To HEX', lw=2.2, color='tab:blue')
        ax.plot(df.index, p_solar_curt, label='Solar curtailed', lw=2.2, color='tab:red')
        ax.plot(df.index, p_col_to_ctes, label='To CTES', lw=2.2, color='tab:green', ls='--')
        ax.set_ylabel('Collector power [kW]')
        ax.grid(True, alpha=0.3)
        _legend_below(ax, ncol=3)

    panels = [
        ('collector_power_and_dni.pdf', _draw_collector_power, 3.6),
        ('ctes_thermal_power.pdf', _draw_ctes_power, 3.2),
        ('ctes_soc_and_energy.pdf', _draw_soc, 3.2),
        ('factory_power_sources.pdf', _draw_factory_power, 3.2),
        ('htf_temperatures.pdf', _draw_htf_temps, 3.3),
        ('ctes_module_outlet_temperatures.pdf', _draw_module_temps, 3.4),
        ('htf_flow_rates.pdf', _draw_flow_rates, 3.2),
        ('hex_flow_share_percent.pdf', _draw_hex_shares, 3.2),
        ('ctes_charging_balance.pdf', _draw_ctes_charge_diag, 3.3),
        ('collector_power_curtailment_vs_ctes.pdf', _draw_solar_diag, 3.3),
    ]

    for fname, drawer, fig_h in panels:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_h), sharex=True)
        drawer(ax, fig)
        _format_time_axis(ax, x0, x1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout.*')
            fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Figure saved: {out_path}')

    fig_all, axes = plt.subplots(len(panels), 1, figsize=(fig_width, 24.0), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (_fname, drawer, _fig_h) in zip(axes, panels):
        drawer(ax, fig_all)
        _format_time_axis(ax, x0, x1)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout.*')
        fig_all.tight_layout()
    combined_pdf = out_pdf if out_pdf is not None else os.path.join(out_dir, 'all_plots_stacked.pdf')
    fig_all.savefig(combined_pdf, bbox_inches='tight')
    plt.close(fig_all)
    print(f'Figure saved: {combined_pdf}')


if __name__ == '__main__':
    latest = _find_latest_sim_csv()
    if latest is not None:
        plot_simulation_results(csv_path=latest)
    else:
        print('No simulation CSV found in', DATA_DIR)
