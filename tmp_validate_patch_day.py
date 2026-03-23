import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-14 00:00:00+00:00':'2026-07-14 23:50:00+00:00']
res = simulate(window,respect_factory_schedule=True, factory_off_on_weekends=False, disable_flow_rate_limits=False, max_ctes_flow_m3s=0.021, debug=False, interactive=False)
night = res['solar_power_W'] < 1e-6
print('night steps:', int(night.sum()))
print('night ctes_charge_htf_W>0:', int((res.loc[night, 'ctes_charge_htf_W']>1.0).sum()))
print('night ctes_charge_input_W>0:', int((res.loc[night, 'ctes_charge_input_W']>1.0).sum()))
