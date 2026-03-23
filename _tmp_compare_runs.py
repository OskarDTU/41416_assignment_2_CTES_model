import pandas as pd
import numpy as np

FILES = [
    r"e:\\OneDrive\\Documenten\\GitHub\\41416_assignment_2_CTES_model\\src\\data\\20260322_124442\\simulation_results_20260322_124442.csv",
    r"e:\\OneDrive\\Documenten\\GitHub\\41416_assignment_2_CTES_model\\src\\data\\20260322_104859\\simulation_results_20260322_104859.csv",
]

out_lines = []
for fp in FILES:
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    init_htf = float(df["htf_in_temp_C"].iloc[0])
    init_soc = float(df["ctes_soc_pct"].iloc[0])

    late = df[df.index.weekday >= 2]  # Wed..Fri
    dis = late["ctes_discharge_output_W"].fillna(0.0)
    active = dis[dis > 1e4]
    ddis = active.diff().dropna()

    mean_abs_step = float(ddis.abs().mean()) if len(ddis) else np.nan
    std_step = float(ddis.std()) if len(ddis) else np.nan
    frac_big = float((dis.diff().abs() > 3e5).mean()) if len(dis) > 1 else np.nan

    # During active discharge, evaluate outlet temp and contribution consistency.
    mask_dis = df["ctes_discharge_output_W"].fillna(0.0) > 1e4
    temp_series = pd.to_numeric(df.loc[mask_dis, "ctes_temp_C"], errors="coerce").dropna()
    tmin = float(temp_series.min()) if len(temp_series) else np.nan
    tmax = float(temp_series.max()) if len(temp_series) else np.nan

    req = pd.to_numeric(df.loc[mask_dis, "ctes_contribution_W"], errors="coerce").fillna(0.0)
    out = pd.to_numeric(df.loc[mask_dis, "ctes_discharge_output_W"], errors="coerce").fillna(0.0)
    gap = (out - req)
    gap_p95 = float(gap.abs().quantile(0.95)) if len(gap) else np.nan

    out_lines.append(f"FILE={fp}")
    out_lines.append(f"  init_htf_in_temp_C={init_htf:.1f} init_soc_pct={init_soc:.3f}")
    out_lines.append(f"  late_active_points={len(active)} mean_abs_step_W={mean_abs_step:.1f} std_step_W={std_step:.1f} frac_big_steps={frac_big:.4f}")
    out_lines.append(f"  ctes_temp_during_discharge_C=[{tmin:.2f},{tmax:.2f}]")
    out_lines.append(f"  discharge_vs_contribution_gap_p95_W={gap_p95:.1f}")

with open(r"e:\\OneDrive\\Documenten\\GitHub\\41416_assignment_2_CTES_model\\_tmp_compare_runs_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
