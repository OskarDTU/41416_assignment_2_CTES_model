import io
import contextlib
import pandas as pd
from src.models.Part_1 import simulate

dni = pd.read_csv('src/data/DNI_10m.csv')
dni['time'] = pd.to_datetime(dni['time'])
dni = dni.set_index('time')['dni_wm2']
window = dni.loc['2026-07-11 04:00:00+00:00':'2026-07-11 12:00:00+00:00']

kwargs = dict(
    respect_factory_schedule=True,
    factory_off_on_weekends=True,
    disable_flow_rate_limits=False,
    max_ctes_flow_m3s=0.03,
    debug=False,
    interactive=False,
)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    base = simulate(window, **kwargs)
with contextlib.redirect_stdout(buf):
    opt = simulate(window, enable_flow_optimizer=True, flow_optimizer_grid_points=7, **kwargs)

for name, df in [('legacy', base), ('optimized', opt)]:
    no_load = df[df['factory_load_W'] < 1e-6]
    curt_MWh = float(no_load['solar_power_curtailed_W'].sum() * 600.0 / 3.6e9)
    chg_MWh = float(no_load['ctes_charge_htf_W'].clip(lower=0).sum() * 600.0 / 3.6e9)
    mcol_med = float(no_load['m_col_vol_flow_m3s'].median())
    mchg_med = float(no_load['m_ctes_charge_use_m3s'].median())
    print(name, 'curt_MWh=', round(curt_MWh,4), 'chg_MWh=', round(chg_MWh,4), 'mcol_med=', round(mcol_med,5), 'mchg_med=', round(mchg_med,5))

print('optimizer mode counts:')
print(opt['flow_optimizer_mode'].value_counts().to_string())
print('optimizer evals median:', float(opt['flow_optimizer_evals'].median()))
