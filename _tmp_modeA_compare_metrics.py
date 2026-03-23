import json
import pandas as pd
from src.models.Part_1 import simulate

d = pd.read_csv('src/data/DNI_10m_3days.csv')
d['time'] = pd.to_datetime(d['time'])
s = d.set_index('time')['dni_wm2']
s = s[s > 0].iloc[:36]
kwargs = dict(
    respect_factory_schedule=True,
    factory_off_on_weekends=True,
    debug=False,
    interactive=False,
    enable_flow_optimizer=True,
    flow_optimizer_grid_points=7,
)
a = simulate(s, mode_a_temperature_priority=False, **kwargs)
b = simulate(s, mode_a_temperature_priority=True, **kwargs)

dt = 600.0

def m(df):
    arows = df[df['operation_mode'].astype(str).str.startswith('A')]
    return {
        'curt_MWh': float(df['solar_power_curtailed_W'].sum() * dt / 3.6e9),
        'charge_MWh': float(df['ctes_charge_input_W'].sum() * dt / 3.6e9),
        'eff_MWh': float(df['solar_power_effective_W'].sum() * dt / 3.6e9),
        'mcol_A_med': float(arows['m_col_vol_flow_m3s'].median()) if len(arows) else float('nan'),
        'tout_A_med': float(arows['collector_t_out_C'].median()) if len(arows) else float('nan'),
    }

ma = m(a)
mb = m(b)
out = {
    'baseline': ma,
    'temp_priority': mb,
    'delta_temp_minus_base': {k: mb[k] - ma[k] for k in ma}
}
with open('_tmp_modeA_compare_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('done')
