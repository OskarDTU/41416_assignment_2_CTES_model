"""
CTES exergy analysis for pre-Monday operating target selection.

Implements a practical exergy workflow using the existing 1D CTES model:
- Dead state fixed at T0 = 25 C and p0 = 1 atm.
- Oil exergy from CoolProp enthalpy/entropy (INCOMP::PNF).
- Concrete exergy from solid sensible exergy (constant Cp).
- Convective and radiative loss exergy tracked separately.
- Candidate target SOC levels are evaluated up to Monday 08:00.
- Monday discharge performance is simulated to estimate useful exergy delivery.

The goal is to recommend the best CTES target before Monday morning discharge starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

try:
    from src.features import ctes_1d_jian as ctes
except Exception:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.features import ctes_1d_jian as ctes


# Fixed dead-state requested by user.
T0_C = 25.0
T0_K = T0_C + 273.15
P0_PA = 101_325.0

# Factory-relevant thresholds from the high-level plant logic.
T_FACTORY_PRIMARY_C = 182.0
T_FACTORY_FALLBACK_C = 160.0

# CTES-side inlet temperatures used for charge/discharge probes.
T_IN_CHARGE_C = 280.0
T_IN_DISCHARGE_C = 136.0

# Candidate targets selected in discussion.
DEFAULT_TARGET_SOCS = (0.60, 0.80, 0.95)

# Analysis time windows.
DEFAULT_PRE_START = pd.Timestamp("2026-03-14 00:00:00")  # Saturday
DEFAULT_PRE_END = pd.Timestamp("2026-03-16 08:00:00")    # Monday 08:00
DEFAULT_DISCHARGE_HOURS = 8.0
DEFAULT_DT_S = 1800

# Oil working fluid used in the project.
OIL_FLUID = "INCOMP::PNF"

# Use a one-module proxy state to keep multi-candidate screening tractable.
# The CTES model already scales inventory terms to full plant where applicable.
USE_SINGLE_MODULE_PROXY = True


@dataclass
class CandidateSummary:
    target_soc: float
    soc_at_monday: float
    exergy_inventory_monday_MWh: float
    pre_monday_loss_ex_conv_MWh: float
    pre_monday_loss_ex_rad_MWh: float
    pre_monday_loss_ex_total_MWh: float
    monday_useful_ex_primary_MWh: float
    monday_useful_ex_fallback_MWh: float
    monday_hot_duration_h: float
    monday_fallback_duration_h: float
    recommendation_score_MWh: float


@dataclass
class SocOptimizationRow:
    target_soc: float
    soc_achieved: float
    reached_target: bool
    charge_hours_to_target: float
    charge_input_energy_MWh: float
    exergy_inventory_start_MWh: float
    monday_useful_ex_primary_MWh: float
    monday_useful_ex_fallback_MWh: float
    monday_hot_duration_h: float
    monday_fallback_duration_h: float
    exergy_eff_primary: float
    exergy_eff_fallback: float


class OilExergyCache:
    """Small cache to avoid repeated CoolProp calls for near-identical node temperatures."""

    def __init__(self, fluid: str, p_ref_pa: float, t0_k: float, p0_pa: float):
        self.fluid = fluid
        self.p_ref_pa = p_ref_pa
        self.t0_k = t0_k
        self.p0_pa = p0_pa
        self._b_cache: Dict[float, float] = {}
        self._h0 = PropsSI("H", "T", self.t0_k, "P", self.p0_pa, self.fluid)
        self._s0 = PropsSI("S", "T", self.t0_k, "P", self.p0_pa, self.fluid)

    def specific_exergy_J_per_kg(self, temp_c: float) -> float:
        key = float(np.round(temp_c, 3))
        if key in self._b_cache:
            return self._b_cache[key]

        t_k = temp_c + 273.15
        h = PropsSI("H", "T", t_k, "P", self.p_ref_pa, self.fluid)
        s = PropsSI("S", "T", t_k, "P", self.p_ref_pa, self.fluid)
        b = (h - self._h0) - self.t0_k * (s - self._s0)
        self._b_cache[key] = float(b)
        return float(b)


def _solid_specific_exergy_J_per_kg(temp_c: np.ndarray, t0_k: float) -> np.ndarray:
    """Specific exergy of an incompressible solid with constant Cp."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    cp = ctes.Cp_con
    return cp * ((t_k - t0_k) - t0_k * np.log(np.maximum(t_k, 1e-9) / t0_k))


def _loss_components_from_node_temperature(temp_s_c: float) -> Tuple[float, float, float, float]:
    """
    Return local equivalent loss components for one node temperature.

    Returns
    -------
    q_total_W, q_conv_W, q_rad_W, t_iso_k
    """
    t_amb = ctes.T_amb
    t_iso_c = t_amb + 0.1 * (temp_s_c - t_amb)

    h_conv = 1.53 * ctes.v_wind + 1.43
    t_iso_k = t_iso_c + 273.15
    t_sky_k = ctes.T_sky(t_amb) + 273.15
    h_rad = ctes.sigma * ctes.eps_iso * (t_iso_k**4 - t_sky_k**4) / (t_iso_c - t_amb + 1e-9)

    r_iso = ctes.th_iso / (ctes.lambda_iso * ctes.S_con_ext)
    r_ext = 1.0 / ((h_conv + h_rad) * ctes.S_iso_ext)

    q_total = max((temp_s_c - t_amb) / (r_iso + r_ext), 0.0)
    if (h_conv + h_rad) <= 0.0 or q_total <= 0.0:
        return 0.0, 0.0, 0.0, t_iso_k

    frac_conv = h_conv / (h_conv + h_rad)
    q_conv = q_total * frac_conv
    q_rad = q_total - q_conv
    return float(q_total), float(q_conv), float(q_rad), float(t_iso_k)


def _inventory_exergy_J(y: np.ndarray, oil_exergy: OilExergyCache) -> Tuple[float, float, float]:
    """Return (total, concrete, fluid) exergy inventory in J for full CTES state."""
    modules = ctes._split_module_states(y)

    total_solid = 0.0
    total_fluid = 0.0
    for y_mod in modules:
        t_f, t_s = ctes.extract_profiles(y_mod)

        b_s = _solid_specific_exergy_J_per_kg(t_s, T0_K)
        ex_s = ctes.rho_con * ctes.S_s * ctes.n_pipes * np.trapezoid(b_s, ctes.z_nodes)
        total_solid += float(ex_s)

        b_f = np.array([oil_exergy.specific_exergy_J_per_kg(float(tc)) for tc in t_f], dtype=float)
        ex_f = ctes.rho_f * ctes.S_f * ctes.n_pipes * np.trapezoid(b_f, ctes.z_nodes)
        total_fluid += float(ex_f)

    return total_solid + total_fluid, total_solid, total_fluid


def _loss_exergy_rates_W(y: np.ndarray) -> Tuple[float, float, float]:
    """Return convective, radiative, and total loss exergy rates [W]."""
    modules = ctes._split_module_states(y)
    ex_conv_total = 0.0
    ex_rad_total = 0.0

    for y_mod in modules:
        _, t_s = ctes.extract_profiles(y_mod)
        ex_conv_sum = 0.0
        ex_rad_sum = 0.0

        for temp_s in t_s:
            _, q_conv, q_rad, t_iso_k = _loss_components_from_node_temperature(float(temp_s))
            carnot = max(1.0 - T0_K / max(t_iso_k, 1e-9), 0.0)
            ex_conv_sum += q_conv * carnot
            ex_rad_sum += q_rad * carnot

        ex_conv_mod = ex_conv_sum * ctes.dz / ctes.L_module
        ex_rad_mod = ex_rad_sum * ctes.dz / ctes.L_module
        ex_conv_total += ex_conv_mod
        ex_rad_total += ex_rad_mod

    return float(ex_conv_total), float(ex_rad_total), float(ex_conv_total + ex_rad_total)


def _sun_is_on(ts: pd.Timestamp) -> bool:
    h = ts.hour + ts.minute / 60.0
    return 8.0 <= h < 18.0


def _simulate_pre_monday(
    target_soc: float,
    dt_s: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    oil_exergy: OilExergyCache,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Weekend charge/storage simulation until Monday 08:00."""
    if USE_SINGLE_MODULE_PROXY:
        y = np.full(ctes.STATE_SIZE_MODULE, ctes.T_min, dtype=float)
    else:
        y = ctes.init_ctes_state(T_init=ctes.T_min)

    t_range = pd.date_range(start=start, end=end, freq=f"{dt_s}s", inclusive="left")
    rows: List[Dict[str, float]] = []

    for ts in t_range:
        soc = float(ctes.SOC(y))
        if _sun_is_on(ts) and soc < target_soc:
            mode = "charging"
            t_in = T_IN_CHARGE_C
            m_dot = ctes.m_dot_total
        else:
            mode = "storage"
            t_in = None
            m_dot = 0.0

        step = ctes.step_ctes(y, t_in, m_dot, mode, dt_s)
        y = step["y"]

        ex_total, ex_solid, ex_fluid = _inventory_exergy_J(y, oil_exergy)
        ex_conv_w, ex_rad_w, ex_total_loss_w = _loss_exergy_rates_W(y)

        rows.append(
            {
                "timestamp": ts,
                "target_soc": target_soc,
                "mode": mode,
                "soc": float(ctes.SOC(y)),
                "stored_energy_MWh": float(ctes.stored_energy_J(y) / 3.6e9),
                "exergy_total_MWh": ex_total / 3.6e9,
                "exergy_solid_MWh": ex_solid / 3.6e9,
                "exergy_fluid_MWh": ex_fluid / 3.6e9,
                "loss_ex_conv_kW": ex_conv_w / 1e3,
                "loss_ex_rad_kW": ex_rad_w / 1e3,
                "loss_ex_total_kW": ex_total_loss_w / 1e3,
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["dt_s"] = dt_s
    return y, out


def _simulate_monday_discharge(
    y_init: np.ndarray,
    dt_s: int,
    discharge_hours: float,
    oil_exergy: OilExergyCache,
) -> pd.DataFrame:
    """Discharge probe after Monday 08:00 to quantify useful exergy delivery."""
    y = np.asarray(y_init, dtype=float).copy()
    n_steps = int(np.ceil(discharge_hours * 3600.0 / dt_s))
    t_in_b = oil_exergy.specific_exergy_J_per_kg(T_IN_DISCHARGE_C)

    rows: List[Dict[str, float]] = []
    for k in range(n_steps):
        step = ctes.step_ctes(y, T_IN_DISCHARGE_C, ctes.m_dot_total, "discharging", dt_s)
        y = step["y"]

        t_out = float(step["T_out_C"])
        b_out = oil_exergy.specific_exergy_J_per_kg(t_out)
        ex_rate = max(ctes.m_dot_total * (b_out - t_in_b), 0.0)

        useful_primary = ex_rate if t_out >= T_FACTORY_PRIMARY_C else 0.0
        useful_fallback = ex_rate if t_out >= T_FACTORY_FALLBACK_C else 0.0

        rows.append(
            {
                "hour_from_monday": (k + 1) * dt_s / 3600.0,
                "outlet_temp_C": t_out,
                "flow_exergy_to_factory_kW": ex_rate / 1e3,
                "useful_exergy_primary_kW": useful_primary / 1e3,
                "useful_exergy_fallback_kW": useful_fallback / 1e3,
                "soc": float(ctes.SOC(y)),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["dt_s"] = dt_s
    return out


def _integrate_power_series_MWh(power_kW: Iterable[float], dt_s: int) -> float:
    p = np.asarray(list(power_kW), dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.sum(p) * dt_s / 3600.0 / 1000.0)


def _init_state() -> np.ndarray:
    """Initialize either the single-module proxy state or full CTES state."""
    if USE_SINGLE_MODULE_PROXY:
        return np.full(ctes.STATE_SIZE_MODULE, ctes.T_min, dtype=float)
    return ctes.init_ctes_state(T_init=ctes.T_min)


def _charge_to_target_soc(
    target_soc: float,
    dt_s: int,
    max_charge_hours: float,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Charge CTES from Tmin until reaching target SOC or hitting max hours.

    Returns
    -------
    y_state, elapsed_h, charge_input_energy_J, reached_target
    """
    y = _init_state()
    elapsed_h = 0.0
    charge_in_j = 0.0
    n_steps = int(np.ceil(max_charge_hours * 3600.0 / dt_s))

    reached = False
    for _ in range(n_steps):
        soc_now = float(ctes.SOC(y))
        if soc_now >= target_soc - 1e-4:
            reached = True
            break

        step = ctes.step_ctes(y, T_IN_CHARGE_C, ctes.m_dot_total, "charging", dt_s)
        y = step["y"]
        q_in_w = max(0.0, float(step.get("diag_q_fluid_to_solid_total_W", 0.0)))
        charge_in_j += q_in_w * dt_s
        elapsed_h += dt_s / 3600.0

    return y, elapsed_h, charge_in_j, reached


def _evaluate_soc_operating_point(
    target_soc: float,
    dt_s: int,
    monday_discharge_hours: float,
    oil_exergy: OilExergyCache,
    max_charge_hours: float,
) -> Tuple[SocOptimizationRow, pd.DataFrame]:
    """Build state at target SOC, then evaluate Monday-discharge exergy performance."""
    y_target, t_charge_h, q_in_j, reached = _charge_to_target_soc(target_soc, dt_s, max_charge_hours)
    soc_ach = float(ctes.SOC(y_target))

    ex_start_j, _, _ = _inventory_exergy_J(y_target, oil_exergy)
    dis_df = _simulate_monday_discharge(y_target, dt_s, monday_discharge_hours, oil_exergy)

    useful_primary = _integrate_power_series_MWh(dis_df["useful_exergy_primary_kW"], dt_s)
    useful_fallback = _integrate_power_series_MWh(dis_df["useful_exergy_fallback_kW"], dt_s)
    hot_duration = float((dis_df["outlet_temp_C"] >= T_FACTORY_PRIMARY_C).sum() * dt_s / 3600.0)
    fallback_duration = float((dis_df["outlet_temp_C"] >= T_FACTORY_FALLBACK_C).sum() * dt_s / 3600.0)

    ex_start_mwh = float(ex_start_j / 3.6e9)
    eff_primary = useful_primary / max(ex_start_mwh, 1e-12)
    eff_fallback = useful_fallback / max(ex_start_mwh, 1e-12)

    row = SocOptimizationRow(
        target_soc=float(target_soc),
        soc_achieved=soc_ach,
        reached_target=bool(reached),
        charge_hours_to_target=float(t_charge_h),
        charge_input_energy_MWh=float(q_in_j / 3.6e9),
        exergy_inventory_start_MWh=ex_start_mwh,
        monday_useful_ex_primary_MWh=float(useful_primary),
        monday_useful_ex_fallback_MWh=float(useful_fallback),
        monday_hot_duration_h=hot_duration,
        monday_fallback_duration_h=fallback_duration,
        exergy_eff_primary=float(eff_primary),
        exergy_eff_fallback=float(eff_fallback),
    )
    return row, dis_df


def _load_dni_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "dni_wm2" not in df.columns:
        raise ValueError("DNI CSV must include 'time' and 'dni_wm2' columns")
    t = pd.to_datetime(df["time"], utc=True)
    s = pd.Series(pd.to_numeric(df["dni_wm2"], errors="coerce").fillna(0.0).values, index=t)
    return s.sort_index()


def _available_solar_energy_per_m2_J(
    dni_csv_path: str,
    collector_efficiency: float,
    horizon_hours: float,
) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Integrate available thermal energy per m2 over the first horizon window in DNI data."""
    s = _load_dni_series(dni_csv_path)
    if s.empty:
        raise ValueError("DNI series is empty")

    start = s.index[0]
    end = start + pd.Timedelta(hours=horizon_hours)
    sw = s[(s.index >= start) & (s.index < end)]
    if sw.empty:
        raise ValueError("No DNI samples found in requested horizon")

    if len(sw.index) >= 2:
        dt_s = float(np.median(np.diff(sw.index.view("i8")) / 1e9))
    else:
        dt_s = 600.0

    e_per_m2_j = float(np.sum(sw.values * collector_efficiency * dt_s))
    return e_per_m2_j, start.tz_convert(None), end.tz_convert(None)


def _plot_soc_optimization(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = summary_df["target_soc"] * 100.0
    ax1.plot(x, summary_df["exergy_eff_primary"], marker="o", lw=2, label="Primary exergy efficiency")
    ax1.set_xlabel("Target SOC [%]")
    ax1.set_ylabel("Exergy efficiency [-]")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, summary_df["monday_useful_ex_primary_MWh"], marker="s", lw=2, color="tab:orange", label="Useful exergy >=182 C")
    ax2.set_ylabel("Useful exergy [MWh]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_solar_area(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = summary_df["target_soc"] * 100.0
    colors = ["tab:green" if r else "tab:blue" for r in summary_df["recommended"]]
    ax.bar(x, summary_df["required_collector_area_m2"], color=colors, width=3.5)
    ax.set_xlabel("Target SOC [%]")
    ax.set_ylabel("Required collector area [m2]")
    ax.set_title("Solar farm size required to reach SOC by Monday start")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_optimal_soc_then_size_solar(
    soc_grid: Iterable[float] = tuple(np.round(np.arange(0.30, 0.91, 0.05), 2)),
    dt_s: int = 1800,
    monday_discharge_hours: float = 8.0,
    dni_csv_path: str = "src/data/DNI_10m.csv",
    collector_efficiency: float = 0.47,
    weekend_horizon_hours: float = 56.0,
    out_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    New workflow:
    1) Find optimal SOC from CTES exergy efficiency.
    2) Size collector area needed to reach that SOC by Monday start.
    """
    if out_dir is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "figures" / f"ctes_soc_opt_then_solar_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_t_amb = ctes.T_amb
    ctes.T_amb = float(T0_C)

    oil_exergy = OilExergyCache(OIL_FLUID, P0_PA, T0_K, P0_PA)
    results: List[SocOptimizationRow] = []
    discharge_curves: List[pd.DataFrame] = []

    # Stage 1: fast screening on proxy.
    old_proxy = USE_SINGLE_MODULE_PROXY
    try:
        globals()["USE_SINGLE_MODULE_PROXY"] = True
        screen_rows = []
        for soc_t in soc_grid:
            row, _ = _evaluate_soc_operating_point(float(soc_t), dt_s, monday_discharge_hours, oil_exergy, max_charge_hours=72.0)
            screen_rows.append(row)
        screen_df = pd.DataFrame([r.__dict__ for r in screen_rows])
        screen_df = screen_df.sort_values("exergy_eff_primary", ascending=False).reset_index(drop=True)

        top_targets = sorted(set(float(v) for v in screen_df["target_soc"].head(3).tolist()))

        # Stage 2: full 14-module confirmation on top candidates.
        globals()["USE_SINGLE_MODULE_PROXY"] = False
        for soc_t in top_targets:
            row, ddf = _evaluate_soc_operating_point(soc_t, dt_s, monday_discharge_hours, oil_exergy, max_charge_hours=72.0)
            results.append(row)
            ddf = ddf.copy()
            ddf["target_soc"] = soc_t
            discharge_curves.append(ddf)
    finally:
        globals()["USE_SINGLE_MODULE_PROXY"] = old_proxy
        ctes.T_amb = old_t_amb

    summary_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("target_soc").reset_index(drop=True)
    if summary_df.empty:
        raise RuntimeError("SOC optimization produced no results")

    # Compute required collector area from required charge input and available DNI thermal energy.
    e_avail_per_m2_j, start_ts, end_ts = _available_solar_energy_per_m2_J(
        dni_csv_path=dni_csv_path,
        collector_efficiency=collector_efficiency,
        horizon_hours=weekend_horizon_hours,
    )
    summary_df["solar_window_start"] = str(start_ts)
    summary_df["solar_window_end"] = str(end_ts)
    summary_df["available_thermal_per_m2_MWh"] = e_avail_per_m2_j / 3.6e9
    summary_df["required_collector_area_m2"] = summary_df["charge_input_energy_MWh"] / np.maximum(summary_df["available_thermal_per_m2_MWh"], 1e-12)

    # Select best SOC by exergy efficiency (primary threshold criterion).
    summary_df["recommended"] = False
    best_idx = summary_df["exergy_eff_primary"].idxmax()
    summary_df.loc[best_idx, "recommended"] = True

    # Save artifacts.
    summary_df.to_csv(out_dir / "soc_optimization_summary.csv", index=False)
    pd.DataFrame([r.__dict__ for r in screen_rows]).to_csv(out_dir / "soc_screening_proxy.csv", index=False)
    pd.concat(discharge_curves, ignore_index=True).to_csv(out_dir / "soc_optimization_discharge_curves.csv", index=False)

    _plot_soc_optimization(summary_df, out_dir / "soc_optimization_efficiency.png")
    _plot_solar_area(summary_df, out_dir / "soc_optimization_required_solar_area.png")

    return summary_df, out_dir


def run_ctes_exergy_study(
    target_socs: Iterable[float] = DEFAULT_TARGET_SOCS,
    dt_s: int = DEFAULT_DT_S,
    pre_start: pd.Timestamp = DEFAULT_PRE_START,
    pre_end: pd.Timestamp = DEFAULT_PRE_END,
    monday_discharge_hours: float = DEFAULT_DISCHARGE_HOURS,
    out_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame], Dict[float, pd.DataFrame], Path]:
    """Run exergy study and save outputs (CSV + figures)."""
    if out_dir is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "figures" / f"ctes_exergy_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enforce ambient reference requested for analysis and loss model.
    old_t_amb = ctes.T_amb
    ctes.T_amb = float(T0_C)

    oil_exergy = OilExergyCache(OIL_FLUID, P0_PA, T0_K, P0_PA)

    pre_data: Dict[float, pd.DataFrame] = {}
    discharge_data: Dict[float, pd.DataFrame] = {}
    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    summaries: List[CandidateSummary] = []

    try:
        for target in target_socs:
            y_monday, pre_df = _simulate_pre_monday(target, dt_s, pre_start, pre_end, oil_exergy)
            dis_df = _simulate_monday_discharge(y_monday, dt_s, monday_discharge_hours, oil_exergy)

            pre_data[target] = pre_df
            discharge_data[target] = dis_df
            profiles[target] = ctes.extract_profiles(y_monday)

            soc_at_monday = float(ctes.SOC(y_monday))
            ex_total_j, _, _ = _inventory_exergy_J(y_monday, oil_exergy)

            loss_conv_mwh = _integrate_power_series_MWh(pre_df["loss_ex_conv_kW"], dt_s)
            loss_rad_mwh = _integrate_power_series_MWh(pre_df["loss_ex_rad_kW"], dt_s)
            loss_total_mwh = loss_conv_mwh + loss_rad_mwh

            useful_primary_mwh = _integrate_power_series_MWh(dis_df["useful_exergy_primary_kW"], dt_s)
            useful_fallback_mwh = _integrate_power_series_MWh(dis_df["useful_exergy_fallback_kW"], dt_s)

            hot_duration_h = float((dis_df["outlet_temp_C"] >= T_FACTORY_PRIMARY_C).sum() * dt_s / 3600.0)
            fallback_duration_h = float((dis_df["outlet_temp_C"] >= T_FACTORY_FALLBACK_C).sum() * dt_s / 3600.0)

            # Net useful exergy score before Monday discharge objective.
            score_mwh = useful_primary_mwh - loss_total_mwh

            summaries.append(
                CandidateSummary(
                    target_soc=float(target),
                    soc_at_monday=soc_at_monday,
                    exergy_inventory_monday_MWh=float(ex_total_j / 3.6e9),
                    pre_monday_loss_ex_conv_MWh=loss_conv_mwh,
                    pre_monday_loss_ex_rad_MWh=loss_rad_mwh,
                    pre_monday_loss_ex_total_MWh=loss_total_mwh,
                    monday_useful_ex_primary_MWh=useful_primary_mwh,
                    monday_useful_ex_fallback_MWh=useful_fallback_mwh,
                    monday_hot_duration_h=hot_duration_h,
                    monday_fallback_duration_h=fallback_duration_h,
                    recommendation_score_MWh=score_mwh,
                )
            )
    finally:
        ctes.T_amb = old_t_amb

    summary_df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values("target_soc").reset_index(drop=True)
    summary_df["recommended"] = False
    if not summary_df.empty:
        best_idx = summary_df["recommendation_score_MWh"].idxmax()
        summary_df.loc[best_idx, "recommended"] = True

    summary_path = out_dir / "ctes_exergy_candidate_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pre_all = pd.concat(pre_data.values(), ignore_index=True)
    pre_all_path = out_dir / "ctes_exergy_pre_monday_timeseries.csv"
    pre_all.to_csv(pre_all_path, index=False)

    dis_all = []
    for target, dis_df in discharge_data.items():
        tmp = dis_df.copy()
        tmp["target_soc"] = target
        dis_all.append(tmp)
    dis_all_df = pd.concat(dis_all, ignore_index=True)
    dis_all_path = out_dir / "ctes_exergy_monday_discharge_timeseries.csv"
    dis_all_df.to_csv(dis_all_path, index=False)

    _plot_pre_monday(pre_data, out_dir / "ctes_exergy_pre_monday.png")
    _plot_monday_discharge(discharge_data, out_dir / "ctes_exergy_monday_discharge.png")
    _plot_monday_profiles(profiles, out_dir / "ctes_exergy_monday_profiles.png")
    _plot_candidate_bars(summary_df, out_dir / "ctes_exergy_candidate_bars.png")

    return summary_df, pre_data, discharge_data, out_dir


def _plot_pre_monday(pre_data: Dict[float, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for target, df in pre_data.items():
        x = pd.to_datetime(df["timestamp"])
        label = f"Target SOC {int(round(target * 100))}%"
        axes[0].plot(x, df["soc"] * 100.0, lw=2, label=label)
        axes[1].plot(x, df["exergy_total_MWh"], lw=2, label=label)

        dt_local = float(df.attrs.get("dt_s", DEFAULT_DT_S))
        loss_cum = np.cumsum(df["loss_ex_total_kW"].to_numpy()) * dt_local / 3600.0 / 1000.0
        axes[2].plot(x, loss_cum, lw=2, label=label)

    axes[0].set_ylabel("SOC [%]")
    axes[0].set_title("Pre-Monday SOC evolution")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Inventory exergy [MWh]")
    axes[1].set_title("Pre-Monday CTES exergy inventory")
    axes[1].grid(alpha=0.3)

    axes[2].set_ylabel("Cumulative loss exergy [MWh]")
    axes[2].set_title("Pre-Monday convective+radiative exergy loss")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.3)

    axes[0].legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_monday_discharge(discharge_data: Dict[float, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for target, df in discharge_data.items():
        x = df["hour_from_monday"]
        label = f"Target SOC {int(round(target * 100))}%"
        axes[0].plot(x, df["outlet_temp_C"], lw=2, label=label)

        dt_local = float(df.attrs.get("dt_s", DEFAULT_DT_S))
        useful_cum = np.cumsum(df["useful_exergy_primary_kW"].to_numpy()) * dt_local / 3600.0 / 1000.0
        axes[1].plot(x, useful_cum, lw=2, label=label)

    axes[0].axhline(T_FACTORY_PRIMARY_C, color="k", ls="--", lw=1.2, label="Primary threshold 182 C")
    axes[0].axhline(T_FACTORY_FALLBACK_C, color="gray", ls=":", lw=1.2, label="Fallback threshold 160 C")
    axes[0].set_ylabel("Outlet temperature [C]")
    axes[0].set_title("Monday discharge outlet temperature")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Cumulative useful exergy [MWh]")
    axes[1].set_xlabel("Hours from Monday 08:00")
    axes[1].set_title("Useful exergy delivered above 182 C")
    axes[1].grid(alpha=0.3)

    axes[0].legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_monday_profiles(profiles: Dict[float, Tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    z_per_module = ctes.z_nodes

    for target, (t_f, t_s) in profiles.items():
        n_mod = max(1, int(len(t_f) // len(z_per_module)))
        z_full = np.concatenate([z_per_module + k * ctes.L_module for k in range(n_mod)])
        label = f"Target SOC {int(round(target * 100))}%"
        axes[0].plot(z_full, t_f, lw=2, label=label)
        axes[1].plot(z_full, t_s, lw=2, label=label)

    axes[0].set_title("Monday 08:00 fluid profile")
    axes[0].set_xlabel("Axial position across CTES string [m]")
    axes[0].set_ylabel("Fluid temperature [C]")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Monday 08:00 concrete profile")
    axes[1].set_xlabel("Axial position across CTES string [m]")
    axes[1].set_ylabel("Concrete temperature [C]")
    axes[1].grid(alpha=0.3)

    axes[0].legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return

    x = np.arange(len(summary_df))
    labels = [f"{int(round(v * 100))}%" for v in summary_df["target_soc"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x - 0.18, summary_df["monday_useful_ex_primary_MWh"], width=0.35, label="Useful exergy (>=182 C)")
    axes[0].bar(x + 0.18, summary_df["pre_monday_loss_ex_total_MWh"], width=0.35, label="Pre-Monday loss exergy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("MWh")
    axes[0].set_title("Useful exergy vs loss exergy")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(loc="best")

    colors = ["tab:green" if r else "tab:blue" for r in summary_df["recommended"]]
    axes[1].bar(x, summary_df["recommendation_score_MWh"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Score [MWh]")
    axes[1].set_title("Recommendation score = useful_primary - loss_total")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _print_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        print("No candidate results.")
        return

    best_row = summary_df.loc[summary_df["recommendation_score_MWh"].idxmax()]

    print("\nCTES Exergy Candidate Summary")
    print("=" * 56)
    cols = [
        "target_soc",
        "soc_at_monday",
        "exergy_inventory_monday_MWh",
        "pre_monday_loss_ex_total_MWh",
        "monday_useful_ex_primary_MWh",
        "monday_hot_duration_h",
        "recommendation_score_MWh",
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    print("\nRecommended target before Monday morning:")
    print(
        f"  SOC target {best_row['target_soc']*100:.0f}% "
        f"(score={best_row['recommendation_score_MWh']:.4f} MWh, "
        f"useful_primary={best_row['monday_useful_ex_primary_MWh']:.4f} MWh, "
        f"loss={best_row['pre_monday_loss_ex_total_MWh']:.4f} MWh)"
    )

    print("\nOutputs written to:")
    print(f"  {out_dir}")


def main() -> None:
    summary_df, out_dir = run_optimal_soc_then_size_solar()
    print("\nSOC-first optimization summary")
    print("=" * 56)
    cols = [
        "target_soc",
        "soc_achieved",
        "exergy_eff_primary",
        "monday_useful_ex_primary_MWh",
        "charge_input_energy_MWh",
        "required_collector_area_m2",
        "recommended",
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    best = summary_df.loc[summary_df["recommended"]].iloc[0]
    print("\nRecommended operating point")
    print(
        f"  SOC={best['target_soc']*100:.0f}% "
        f"(ex_eff={best['exergy_eff_primary']:.4f}, "
        f"useful_ex={best['monday_useful_ex_primary_MWh']:.4f} MWh, "
        f"required_area={best['required_collector_area_m2']:.1f} m2)"
    )
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
