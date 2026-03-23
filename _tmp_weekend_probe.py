import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-11 00:00:00+00:00':'2026-07-12 23:50:00+00:00']
res = simulate(window, respect_factory_schedule=True, factory_off_on_weekends=True, disable_flow_rate_limits=False, max_ctes_flow_m3s=0.03, debug=False, interactive=False)

cap = 0.021
sel = res[(res['factory_load_W'] < 1e-6) & (res['solar_power_curtailed_W'] > 1.0)]
sel2 = sel[(res['collector_t_out_C'] >= 309.9) & (res['m_col_vol_flow_m3s'] < 0.020)]
print('rows no-load + curtailed:', len(sel))
print('rows also at T_out max and flow<0.02:', len(sel2))
if len(sel2) > 0:
    cols = ['dni_wm2','collector_t_in_C','collector_t_out_C','solar_power_available_W','solar_power_W','collector_curtailment_W','solar_power_curtailed_W','m_col_vol_flow_m3s','m_col_use_m3s','m_ctes_charge_use_m3s','ctes_charge_htf_W','ctes_soc_pct','ctes_flow_limit_reason','op_mode']
    print(sel2[cols].head(20).to_string())
