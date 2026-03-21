import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-11 00:00:00+00:00':'2026-07-18 00:00:00+00:00']

res = simulate(
    window,
    initial_htf_inlet_temp_C=220.0,  # <-- Qui scegli il SOC iniziale (50% in questo caso)
    initial_no_load_hours=0.0,
    respect_factory_schedule=True,
    factory_off_on_weekends=False,
    disable_flow_rate_limits=False,
    max_ctes_flow_m3s=0.03,
    debug=False,
    interactive=False
)

curt = float(res['solar_power_curtailed_W'].sum()*600/3.6e9)
qin = float(res['ctes_charge_htf_W'].clip(lower=0).sum()*600/3.6e9)
qout = float(res['ctes_discharge_output_W'].clip(lower=0).sum()*600/3.6e9)
print('curtail_MWh', round(curt,4))
print('ctes_charge_MWh', round(qin,4))
print('ctes_discharge_MWh', round(qout,4))
print('max collector flow m3/s', float(res['m_col_vol_flow_m3s'].max()))
