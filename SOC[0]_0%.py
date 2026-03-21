import sys
import os
import pandas as pd

project_root = r"C:\DTU\Spring26\Energy system design and optimization\41416_assignment_2_CTES_model"
sys.path.insert(0, project_root)

from src.models.Part_1 import simulate
from src.models.Part_2 import exergy_analysis
from src.features.data_presentation import plot_simulation_results

dni_path = os.path.join(project_root, 'src', 'data', 'DNI_10m.csv')
dni_df = pd.read_csv(dni_path, index_col=0, parse_dates=True)
dni_df.index = pd.to_datetime(dni_df.index, utc=True).tz_localize(None)
dni_series = dni_df['dni_wm2'].astype(float)
window = dni_series.loc['2026-07-11':'2026-07-18']

SOC_target = 0  

T_min = 130.0
T_max = 310.0
T_init = T_min + SOC_target * (T_max - T_min)

res = simulate(
    window,
    initial_htf_inlet_temp_C=T_init,
)

print('Simulation completed.')
csv_path = res.attrs.get('csv_path')
print('CSV saved in:', csv_path)

# --- Energy results ---
curt = float(res['solar_power_curtailed_W'].sum() * 600 / 3.6e9)
qin  = float(res['ctes_charge_htf_W'].clip(lower=0).sum() * 600 / 3.6e9)
qout = float(res['ctes_discharge_output_W'].clip(lower=0).sum() * 600 / 3.6e9)
print('curtail_MWh',          round(curt, 4))
print('ctes_charge_MWh',      round(qin,  4))
print('ctes_discharge_MWh',   round(qout, 4))
print('max collector flow m3/s', float(res['m_col_vol_flow_m3s'].max()))

# --- Exergy analysis ---
res_ex, table, eta = exergy_analysis(res)
print('overall exergy efficiency', round(eta, 4))

# --- Plots ---
plot_simulation_results(csv_path=csv_path)