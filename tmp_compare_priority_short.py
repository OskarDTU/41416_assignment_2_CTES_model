import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-14 06:00:00+00:00':'2026-07-14 10:00:00+00:00']

base = simulate(window, initial_no_load_hours=0.0, respect_factory_schedule=True, factory_off_on_weekends=False, disable_flow_rate_limits=False, max_ctes_flow_m3s=0.021, early_sun_charge_priority=False, debug=False, interactive=False)
prio = simulate(window, initial_no_load_hours=0.0, respect_factory_schedule=True, factory_off_on_weekends=False, disable_flow_rate_limits=False, max_ctes_flow_m3s=0.021, early_sun_charge_priority=True, debug=False, interactive=False)

for name, df in [('base', base), ('prio', prio)]:
    curt = float(df['solar_power_curtailed_W'].sum()*600/3.6e9)
    qin = float(df['ctes_charge_htf_W'].clip(lower=0).sum()*600/3.6e9)
    print(name, 'curt_MWh=', round(curt,4), 'ctes_charge_MWh=', round(qin,4))
