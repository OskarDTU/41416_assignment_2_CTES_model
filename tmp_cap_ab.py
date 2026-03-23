import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-14 06:00:00+00:00':'2026-07-14 08:00:00+00:00']

for cap in [0.021, 0.03]:
    res = simulate(window,respect_factory_schedule=True, factory_off_on_weekends=False, disable_flow_rate_limits=False, max_ctes_flow_m3s=cap, debug=False, interactive=False)
    curt = float(res['solar_power_curtailed_W'].sum()*600/3.6e9)
    qin = float(res['ctes_charge_htf_W'].clip(lower=0).sum()*600/3.6e9)
    print('cap', cap, 'curtail_MWh', round(curt,4), 'ctes_charge_MWh', round(qin,4))
