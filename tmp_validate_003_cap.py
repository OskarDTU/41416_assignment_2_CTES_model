import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-14 06:00:00+00:00':'2026-07-14 14:00:00+00:00']

res = simulate(window, initial_no_load_hours=0.0, respect_factory_schedule=True, factory_off_on_weekends=False, disable_flow_rate_limits=False, max_ctes_flow_m3s=0.03, debug=False, interactive=False)
print('curtail_MWh', round(float(res['solar_power_curtailed_W'].sum()*600/3.6e9),4))
print('ctes_charge_MWh', round(float(res['ctes_charge_htf_W'].clip(lower=0).sum()*600/3.6e9),4))
print('max_col_flow', round(float(res['m_col_vol_flow_m3s'].max()),5))
