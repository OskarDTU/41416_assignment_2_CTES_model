import pandas as pd
import numpy as np
from src.models.Part_1 import simulate

D = pd.read_csv('src/data/DNI_10m.csv')
D['time'] = pd.to_datetime(D['time'])
S = D.set_index('time')['dni_wm2']
W = S.loc['2026-07-11 00:00:00+00:00':'2026-07-18 00:00:00+00:00']

for t0 in (120.0, 130.0):
    r = simulate(
        W,
        initial_htf_inlet_temp_C=t0,respect_factory_schedule=True,
        factory_off_on_weekends=False,
        disable_flow_rate_limits=False,
        max_ctes_flow_m3s=0.03,
        debug=False,
        interactive=False,
    )
    dis = r['ctes_discharge_output_W'].fillna(0.0)
    late = r.loc[r.index.weekday >= 2]
    a = late['ctes_discharge_output_W'].fillna(0.0)
    active = a > 1e4
    dstd = float(a[active].diff().dropna().std()) if active.any() else float('nan')
    frac_big = float((a.diff().abs() > 3e5).mean())
    min_t_dis = float(r.loc[dis > 1e4, 'ctes_temp_C'].dropna().min()) if (dis > 1e4).any() else float('nan')
    max_t_dis = float(r.loc[dis > 1e4, 'ctes_temp_C'].dropna().max()) if (dis > 1e4).any() else float('nan')
    print(f'RUN init={t0:.0f} path={r.attrs.get("csv_path","")}')
    print(f'  late_active_pts={int(active.sum())} diff_std_W={dstd:.1f} frac_big_steps={frac_big:.4f}')
    print(f'  ctes_temp_during_discharge[min,max]=[{min_t_dis:.2f},{max_t_dis:.2f}]')
