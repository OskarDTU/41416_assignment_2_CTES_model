#In part 1 of the project, you are supposed to do a project on dynamic 
# modelling of the thermodynamic process of an energy storage plant and simulate
#the operation of the plant for a relevant period of operation time.

#First, we shall model the thermodynamic process of an energy storage plant
#We shall look at the charging, storage, and discharging of the storage plant

# %% Import necessary libraries

from typing import Optional, Union
import pandas as pd
import numpy as np
from datetime import timedelta

from CoolProp.CoolProp import PropsSI
import os
import sys
# try both relative and absolute import styles so this module is usable
try:
	from ..features.solarpower import solar_collector_outlet_temperature
except Exception:
	try:
		from features.solarpower import solar_collector_outlet_temperature
	except Exception:
		solar_collector_outlet_temperature = None

try:
    from ..features.ctes_1d_jian import init_ctes_state, step_ctes, stored_energy_J, T_outlet, extract_profiles, heat_loss_W, N_z as _CTES_N_z
except Exception:
	try:
		from features.ctes_1d_jian import init_ctes_state, step_ctes, stored_energy_J, T_outlet, extract_profiles, heat_loss_W, N_z as _CTES_N_z
	except Exception:
		init_ctes_state = step_ctes = stored_energy_J = T_outlet = extract_profiles = heat_loss_W = None
		_CTES_N_z = 30

try:
	from ..data.constants import n_modules as ctes_series_modules
except Exception:
	try:
		from data.constants import n_modules as ctes_series_modules
	except Exception:
		ctes_series_modules = 14

# %% Helper functions for the simulation loop
# Load DNI input from various formats (scalar, Series, or CSV path)
def _load_dni_input(dni_input: Union[str, float, int, pd.Series]) -> Union[float, pd.Series]:
	"""Return DNI as float or pandas Series depending on input type."""
	if isinstance(dni_input, (float, int)):
		return float(dni_input)
	if isinstance(dni_input, pd.Series):
		return dni_input.astype(float)
	if isinstance(dni_input, str):
		df = pd.read_csv(dni_input, header=0)
		try:
			df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
			df = df.set_index(df.columns[0])
		except Exception:
			# leave as-is if parsing fails
			pass
		# prefer column named like the example
		if "dni_wm2" in df.columns:
			return df["dni_wm2"].astype(float)
		# otherwise take the first numeric column
		for col in df.columns:
			try:
				s = pd.to_numeric(df[col], errors="coerce")
				if s.notna().any():
					s.index = df.index
					return s
			except Exception:
				continue
		raise ValueError("Could not find a numeric DNI column in CSV")
	raise TypeError("dni_input must be a path, float, int, or pandas Series")

#Oil/water heat exchanger model for supplying hot water to the pasta factory.
def oil_water_hex(
	oil_in_C: float,
	oil_flow_m3s: float,
	desired_water_power_W: float,
	water_vol_flow_m3s: Optional[float] = None,
	water_in_C: float = 220.0,
	water_target_C: Optional[float] = None,
	oil_fluid: str = "INCOMP::PNF",
	water_fluid: str = "Water",
	pinch_delta: float = 5.0,
	):
	"""Compute oil outlet temperature and achievable water heating power using constant-cp.

	Returns (oil_out_C, provided_water_power_W, water_out_C).

	- Uses CoolProp to obtain density and cp at inlet temperatures.
	- Assumes constant cp (no integral over T).
	- If `water_vol_flow_m3s` is provided and `water_target_C` is provided, the
	  function will compute the power required to reach `water_target_C` and then
	  report how much of that power can be provided by the oil given its flow.
	- If flows are zero or insufficient, provided power will be limited accordingly.
	"""
	# Guard values
	if oil_flow_m3s <= 0 or desired_water_power_W <= 0:
		return float(oil_in_C), 0.0, float(water_in_C)

	#Convert inlet temperatures to K for CoolProp
	T_oil_in_K = oil_in_C + 273.15
	T_water_in_K = water_in_C + 273.15

	# Get oil properties at inlet conditions, with fallbacks if CoolProp fails (e.g. if not installed or fluid not found)
	try:
		rho_oil = PropsSI("D", "T", T_oil_in_K, "Q", 0, oil_fluid)
		cp_oil = PropsSI("C", "T", T_oil_in_K, "Q", 0, oil_fluid)
	except Exception:
		rho_oil = 870.0
		cp_oil = 2200.0

	# compute mass flow of oil from volumetric flow and density
	m_dot_oil = oil_flow_m3s * rho_oil
	if m_dot_oil <= 0:
		return float(oil_in_C), 0.0, float(water_in_C)

	# If water flow and target provided, compute requested water density and cp, then required power to reach target temperature
	if water_vol_flow_m3s is not None and water_target_C is not None:
		try:
			rho_water = PropsSI("D", "T", T_water_in_K, "P", 8e5, water_fluid)
			cp_water = PropsSI("C", "T", T_water_in_K, "P", 8e5, water_fluid)
		except Exception:
			rho_water = 1000.0
			cp_water = 4180.0
			
		m_dot_water = water_vol_flow_m3s * rho_water
		required_W = m_dot_water * cp_water * (water_target_C - water_in_C)
		desired_water_power_W = min(desired_water_power_W, required_W)

	# maximum power oil can provide cooling to the water given its flow and temperature difference, ensuring a pinch point for driving force
	min_oil_out_C = water_in_C + pinch_delta  # ensure oil outlet is at least the pinch delta above water inlet for some driving force
	max_available_W = m_dot_oil * cp_oil * max(oil_in_C - min_oil_out_C, 0.0)
	provided_W = min(max_available_W, desired_water_power_W)

	# compute oil outlet temperature from provided power
	oil_out_C = oil_in_C - provided_W / (m_dot_oil * cp_oil)
	
	# compute water outlet temperature if a water flow was given
	if water_vol_flow_m3s is not None:
		try:
			rho_water = PropsSI("D", "T", T_water_in_K, "Q", 0, water_fluid)
			cp_water = PropsSI("C", "T", T_water_in_K, "Q", 0, water_fluid)
		except Exception:
			rho_water = 1000.0
			cp_water = 4180.0
		m_dot_water = water_vol_flow_m3s * rho_water
		delta_T_water = provided_W / (m_dot_water * cp_water)
		water_out_C = water_in_C + delta_T_water
	else:
		water_out_C = water_in_C

	return float(oil_out_C), float(provided_W), float(water_out_C)

# %% Main simulation loop for the solar+CTES+pasta-factory system.
def simulate(
	dni_input: Union[str, float, int, pd.Series],
 	collector_efficiency: float = 0.47,
 	collector_area_m2: float = 6602.0,
	initial_htf_inlet_temp_C: float = 135.0,
 	max_htf_temp_C: float = 310.0,
 	max_collector_flow_m3s: Optional[float] = None,
 	max_pump_flow_m3s: Optional[float] = None,
 	max_ctes_flow_m3s: Optional[float] = None,
	ctes_discharge_cutoff_C: float = 120.0,
	ctes_discharge_hysteresis_C: float = 5,
 	timestep_seconds: int = 600,
	respect_factory_schedule: bool = True,
	factory_off_on_weekends: bool = True,
 	fluid: str = "INCOMP::PNF",
	disable_flow_rate_limits: bool = True,
	early_sun_charge_priority: bool = True,
	early_sun_start_hour: int = 6,
	early_sun_end_hour: int = 11,
	early_sun_collector_overdrive_factor: float = 1.0,
	enable_inner_coupled_solver: bool = True,
	inner_solver_max_iter: int = 6,
	inner_solver_relaxation: float = 0.35,
	enable_flow_optimizer: bool = True,
	flow_optimizer_grid_points: int = 9,
	flow_optimizer_use_full_ctes_eval: bool = False,
 	debug: bool = False,
 	interactive: bool = False,
):
	"""High-level simulation loop for the solar+CTES+pasta-factory system.

	This implements the main structure and control logic without a detailed CTES model.
	It calls `solar_collector_outlet_temperature` from `src/features/solarpower.py` when available.
	"""
	if solar_collector_outlet_temperature is None:
		raise RuntimeError("solar_collector_outlet_temperature is unavailable. Ensure features.solarpower is importable.")

	dni = _load_dni_input(dni_input)
	# Ensure a datetime index for iteration
	if not isinstance(dni, pd.Series):
		raise TypeError("dni_input must resolve to a pandas Series for simulation runs")
	if dni.index.inferred_type not in ("datetime64",):
		try:
			dni.index = pd.to_datetime(dni.index)
		except Exception:
			raise ValueError("DNI series must have a datetime index")

	start = dni.index[0]

	# ensure a fallback `ctes` dict exists for legacy stub calls
	ctes = {
		"energy_J": 0.0,
		"capacity_J": 5e9,
		"min_temp_C": 120.0,
		"max_temp_C": 310.0,
		"temp_C": initial_htf_inlet_temp_C,
		"max_discharge_W": 5e6,
	}
	# initialize CTES state using the 1D model if available
	if init_ctes_state is not None and stored_energy_J is not None:
		ctes_y = init_ctes_state(T_init=initial_htf_inlet_temp_C)
		ctes_energy_J = stored_energy_J(ctes_y)
		# compute capacity as energy between T_min and T_max using the 1D model helper
		try:
			ctes_capacity_J = stored_energy_J(init_ctes_state(T_init=max_htf_temp_C))
		except Exception:
			ctes_capacity_J = 5e9
		# reflect values in the legacy dict for any stub code paths
		ctes["energy_J"] = ctes_energy_J
		ctes["capacity_J"] = ctes_capacity_J
		ctes["temp_C"] = initial_htf_inlet_temp_C
	else:
		# fall back to the simple placeholder if CTES model unavailable
		ctes_y = None
		ctes_energy_J = 0.0
		ctes_capacity_J = 5e9
		ctes = {
			"energy_J": ctes_energy_J,
			"capacity_J": ctes_capacity_J,
			"min_temp_C": 120.0,
			"max_temp_C": 300.0,
			"temp_C": initial_htf_inlet_temp_C,
			"max_discharge_W": 5e6,
		}

	# record initial CTES energy to allow reporting deltas (J)
	initial_ctes_energy_J = float(ctes_energy_J)

	records = []
	# cumulative energy trackers (J)
	cum_factory_consumed_J = 0.0
	cum_solar_produced_J = 0.0
	cum_solar_curtailed_J = 0.0
	cum_solar_supplied_to_factory_J = 0.0
	cum_ctes_supplied_to_factory_J = 0.0
	cum_ctes_charge_input_J = 0.0
	cum_ctes_discharge_output_J = 0.0
	cum_backup_heater_J = 0.0
	cum_tolerated_deficit_J = 0.0
	# CTES loss tracker (J)
	cum_ctes_loss_J = 0.0
	# water-side parameters for the factory HEX (given in the project)
	factory_water_flow_m3s = 0.023  # superheated water mass flow (m3/s)
	factory_water_in_C = 120.0
	factory_water_target_C = 140.0
	# Nominal factory thermal load from specified water side (P=8 bar,
	# 120 C -> 140 C at 0.023 m3/s), computed with CoolProp.
	try:
		rho_w_nom = PropsSI("D", "T", factory_water_in_C + 273.15, "P", 8e5, "Water")
		cp_w_nom = PropsSI("C", "T", factory_water_in_C + 273.15, "P", 8e5, "Water")
		factory_nominal_load_W = float(factory_water_flow_m3s * rho_w_nom * cp_w_nom * (factory_water_target_C - factory_water_in_C))
	except Exception:
		factory_nominal_load_W = 2.5e6
	# target HTF conditions into the O/W HEX to reach water target (from your note)
	target_hex_htf_temp_C = 182.0
	target_hex_htf_temp_fallback_C = 160.0
	factory_water_fallback_C = 133.0
	# Keep default design flow at 0.021 m3/s (well below 0.21 m3/s).
	target_hex_htf_flow_m3s = 0.021
	# Collector split budget (HEX + CTES charge) per timestep.
	collector_split_budget_m3s = target_hex_htf_flow_m3s
	# Hard physical cap for collector-side volumetric flow.
	collector_flow_hard_cap_m3s = 0.021
	ctes_min_discharge_temp_C = float(ctes_discharge_cutoff_C)
	ctes_discharge_enabled = bool(initial_htf_inlet_temp_C >= (ctes_min_discharge_temp_C + float(ctes_discharge_hysteresis_C)))
	pinch_delta_C = 5  # minimal approach temperature between oil and water

	# initial HTF inlet temperature (can be updated by recirculation / mixing logic)
	htf_in_temp_C = initial_htf_inlet_temp_C

	# Main loop iterating over DNI time series
	last_prompt_hour = -1
	last_progress_hour = -1
	total_ctes_discharge_iterations = 0
	ctes_discharge_solve_events = 0
	max_ctes_discharge_iterations = 0
	hourly_ctes_discharge_iterations = 0
	hourly_ctes_discharge_solve_events = 0
	hourly_max_ctes_discharge_iterations = 0
	hourly_bucket_hour = None
	hourly_mode_counts = {}
	prev_ctes_discharge_flow_m3s = 0.0
	first_debug_print = True
	debug_table_header_printed = False
	# Initialize CTES step debug variables to avoid undefined local references
	last_ctes_mdot = 0.0
	last_ctes_T_in = float('nan')
	last_ctes_T_out = float('nan')
	# display variable for charging (may be capped for reporting)
	actual_charge_display_W = 0.0
	for ts, dni_value in dni.items():
		elapsed_h = (ts - start) / pd.Timedelta(hours=1)
		curr_hour = int(np.floor(elapsed_h))
		ctes_flow_dir = 'none'
		op_mode = '-'
		ctes_to_factory_W = 0.0
		ctes_state_advanced = False
		requested_ctes_to_factory_W = 0.0
		m_ctes_charge_use_m3s = 0.0
		actual_charge_W = 0.0
		actual_charge_display_W = 0.0
		ctes_model_diag = None
		res = None  # compatibility for static analyzers with stale flow inference
		res_step = None
		# compute minute-of-hour for gating debug prints/prompts
		if isinstance(ts, pd.Timestamp):
			minute_of_hour = int(ts.minute)
			hour_of_day = int(ts.hour)
		else:
			minute_of_hour = int(((elapsed_h - curr_hour) * 60) % 60)
			hour_of_day = int(curr_hour % 24)
		# interactive per-hour pause when requested (only during first 10 minutes)
		if interactive:
			if curr_hour > last_prompt_hour and minute_of_hour < 10:
				last_prompt_hour = curr_hour
				print(f"--- Simulated time: {ts} (hour {curr_hour}) ---")
				ok = input('Enter any character to abort: ').strip().lower()
				if ok != '':
					raise SystemExit('Simulation aborted by user')
		# lightweight progress output for non-debug runs
		if (not debug) and (not interactive) and (curr_hour > last_progress_hour) and (minute_of_hour < 10):
			last_progress_hour = curr_hour
			if hourly_bucket_hour is None:
				hourly_bucket_hour = curr_hour
			elif curr_hour != hourly_bucket_hour:
				hourly_avg_iters = (hourly_ctes_discharge_iterations / hourly_ctes_discharge_solve_events) if hourly_ctes_discharge_solve_events > 0 else 0.0
				mode_counts_txt = ','.join([f"{k}:{v}" for k, v in sorted(hourly_mode_counts.items())]) if len(hourly_mode_counts) > 0 else '-'
				print(
					f"Progress: simulated {ts} (hour {curr_hour}) | "
					f"mode={mode_counts_txt} | "
					f"events={hourly_ctes_discharge_solve_events}, "
					f"iters={hourly_ctes_discharge_iterations}, "
					f"avg={hourly_avg_iters:.2f}, "
					f"max={hourly_max_ctes_discharge_iterations}"
				)
				hourly_bucket_hour = curr_hour
				hourly_ctes_discharge_iterations = 0
				hourly_ctes_discharge_solve_events = 0
				hourly_max_ctes_discharge_iterations = 0
				hourly_mode_counts = {}
		# factory schedule can be disabled to isolate night-time supply behavior.
		factory_active = True
		if respect_factory_schedule:
			if factory_off_on_weekends and isinstance(ts, pd.Timestamp) and ts.weekday() >= 5:
				factory_active = False

		factory_load_W = factory_nominal_load_W if factory_active else 0.0

		# snapshot CTES energy at timestep start for loss attribution
		prev_ctes_energy_J = float(ctes_energy_J)
		# per-timestep diagnostics defaults
		actual_charge_W = 0.0
		actual_charge_display_W = 0.0
		provided_from_ctes_W = 0.0
		solar_curtailed_W = 0.0
		solar_power_effective_W = 0.0
		collector_curtailment_W = 0.0
		# CTES flow-limiter diagnostics (for debug table visibility).
		m_ctes_needed_mix = np.nan
		m_ctes_needed_power = np.nan
		m_ctes_use_pre_clip = 0.0
		m_ctes_use_post_clip = 0.0
		m_ctes_trim_reduction = 0.0
		m_ctes_overflow_reduction = 0.0
		m_ctes_discharge_fit_reduction = 0.0
		ctes_flow_limit_reason = 'none'
		ctes_discharge_solver_mode = 'none'
		ctes_discharge_iterations = 0
		ctes_charge_iterations = 0
		flow_optimizer_mode = 'legacy'
		flow_optimizer_evals = 0
		flow_optimizer_best_objective_W = np.nan
		flow_optimizer_best_charge_flow_m3s = 0.0
		flow_optimizer_factory_shortfall_W = np.nan

		# compute solar collector output (uses volumetric flow for Paratherm NF)
		# First compute available solar power (independent of flow)
		solar_power_available_W = (float(dni_value) if not pd.isna(dni_value) else 0.0) * collector_efficiency * collector_area_m2

		# Morning overdrive can relax collector hard cap to absorb more solar into CTES.
		collector_flow_hard_cap_step_m3s = collector_flow_hard_cap_m3s
		if early_sun_charge_priority and (early_sun_start_hour <= hour_of_day < early_sun_end_hour):
			collector_flow_hard_cap_step_m3s = collector_flow_hard_cap_m3s * max(1.0, float(early_sun_collector_overdrive_factor))

		def _cap_collector_flow(flow_m3s: float) -> float:
			flow = min(float(max(0.0, flow_m3s)), float(collector_flow_hard_cap_step_m3s))
			if not disable_flow_rate_limits:
				if max_collector_flow_m3s is not None:
					flow = min(flow, float(max_collector_flow_m3s))
				if max_pump_flow_m3s is not None:
					flow = min(flow, float(max_pump_flow_m3s))
			return float(max(0.0, flow))

		def _htf_props_at(T_C: float) -> tuple[float, float]:
			try:
				rho = PropsSI("D", "T", float(T_C) + 273.15, "Q", 0, fluid)
				cp = PropsSI("C", "T", float(T_C) + 273.15, "Q", 0, fluid)
			except Exception:
				rho, cp = 870.0, 2200.0
			return float(max(rho, 1e-9)), float(max(cp, 1e-9))

		collector_t_in_C = float(htf_in_temp_C)
		m_col_vol_flow_m3s = 0.0
		inner_solver_iters = 0
		inner_solver_dTin_C = np.nan
		inner_solver_dm_col_m3s = np.nan
		inner_solver_converged = False

		# Optional flow optimizer: choose collector flow that maximizes expected
		# CTES charging while prioritizing factory supply when active.
		if enable_flow_optimizer and (solar_power_available_W > 0.0) and (step_ctes is not None) and (ctes_y is not None):
			collector_t_in_C = float(htf_in_temp_C)
			rho_opt, cp_opt = _htf_props_at(collector_t_in_C)
			grid_n = max(3, int(flow_optimizer_grid_points))
			candidate_flows = np.linspace(0.0, float(collector_flow_hard_cap_step_m3s), grid_n)

			best_key = None
			best_flow = 0.0
			best_objective = 0.0
			best_charge_flow = 0.0
			best_shortfall = 0.0

			for flow_guess in candidate_flows:
				flow_cand = _cap_collector_flow(float(flow_guess))
				flow_optimizer_evals += 1
				if flow_cand <= 0.0:
					continue

				try:
					col_try = solar_collector_outlet_temperature(
						t_in=collector_t_in_C,
						m_dot=flow_cand,
						dni=float(dni_value) if not pd.isna(dni_value) else 0.0,
						efficiency=collector_efficiency,
						area=collector_area_m2,
						volumetric=True,
						temp_unit="C",
					)
					P_col = float(max(0.0, col_try.get("power_W", 0.0)))
					T_out_cand = float(col_try.get("t_out_C", np.nan))
				except Exception:
					continue

				if not np.isfinite(T_out_cand):
					continue
				if T_out_cand > max_htf_temp_C:
					T_out_cand = float(max_htf_temp_C)
					try:
						cp_eff = PropsSI("C", "T", ((collector_t_in_C + T_out_cand) / 2.0) + 273.15, "Q", 0, fluid)
					except Exception:
						cp_eff = cp_opt
					P_col = max(0.0, flow_cand * rho_opt * max(float(cp_eff), 1e-9) * max(0.0, (T_out_cand - collector_t_in_C)))

				m_charge_cand = flow_cand
				factory_shortfall = 0.0
				if factory_active:
					T_out_min_req_C = factory_water_target_C + pinch_delta_C
					dT_src = max(1e-6, T_out_cand - T_out_min_req_C)
					m_dot_need = factory_load_W / max(cp_opt * dT_src, 1e-9)
					m_hex_need = m_dot_need / max(rho_opt, 1e-9)
					m_hex_use = min(flow_cand, max(0.0, m_hex_need))
					m_charge_cand = max(0.0, flow_cand - m_hex_use)
					factory_supply_est = min(P_col, m_hex_use * rho_opt * cp_opt * dT_src)
					factory_shortfall = max(0.0, factory_load_W - factory_supply_est)

				objective = 0.0
				if m_charge_cand > 0.0:
					if flow_optimizer_use_full_ctes_eval:
						try:
							res_try = step_ctes(ctes_y, T_out_cand, m_charge_cand * rho_opt, 'charging', timestep_seconds)
							objective = max(0.0, float(res_try.get('diag_q_to_solid_corr_W', res_try.get('diag_q_fluid_to_solid_total_W', 0.0))))
						except Exception:
							objective = 0.0
					else:
						# Fast proxy: charging potential from current CTES charging outlet
						# (state-aware), bounded by collector thermal power.
						try:
							T_ctes_chg_out = float(T_outlet(ctes_y, 'charging')) if T_outlet is not None else htf_in_temp_C
						except Exception:
							T_ctes_chg_out = htf_in_temp_C
						dT_charge = max(0.0, T_out_cand - T_ctes_chg_out)
						q_proxy = m_charge_cand * rho_opt * cp_opt * dT_charge
						objective = max(0.0, min(P_col, q_proxy))

				if factory_active:
					key = (-float(factory_shortfall), float(objective), float(m_charge_cand), float(flow_cand))
				else:
					key = (float(objective), float(m_charge_cand), float(flow_cand))

				if (best_key is None) or (key > best_key):
					best_key = key
					best_flow = float(flow_cand)
					best_objective = float(objective)
					best_charge_flow = float(m_charge_cand)
					best_shortfall = float(factory_shortfall)

			if best_key is not None:
				m_col_vol_flow_m3s = best_flow
				flow_optimizer_mode = 'optimized'
				flow_optimizer_best_objective_W = best_objective
				flow_optimizer_best_charge_flow_m3s = best_charge_flow
				flow_optimizer_factory_shortfall_W = best_shortfall
				inner_solver_converged = True
			else:
				flow_optimizer_mode = 'optimizer_failed'

		if (flow_optimizer_mode == 'legacy') and enable_inner_coupled_solver and factory_active and (solar_power_available_W > 0.0):
			T_in_guess = float(htf_in_temp_C)
			m_prev = 0.0
			relax = float(np.clip(inner_solver_relaxation, 0.05, 0.95))
			for it_idx in range(max(1, int(inner_solver_max_iter))):
				inner_solver_iters = int(it_idx + 1)
				rho_g, cp_g = _htf_props_at(T_in_guess)
				dT_hex = max(1e-6, target_hex_htf_temp_C - T_in_guess)
				dT_limit = max(1e-6, max_htf_temp_C - T_in_guess)
				m_dot_need_factory = float(factory_load_W) / max(cp_g * dT_hex, 1e-9)
				m_dot_solar_cap = float(solar_power_available_W) / max(cp_g * dT_limit, 1e-9)
				m_try = _cap_collector_flow(min(m_dot_need_factory, m_dot_solar_cap) / rho_g)

				# Evaluate collector response with guessed inlet and flow.
				try:
					col_try = solar_collector_outlet_temperature(
						t_in=T_in_guess,
						m_dot=m_try,
						dni=float(dni_value) if not pd.isna(dni_value) else 0.0,
						efficiency=collector_efficiency,
						area=collector_area_m2,
						volumetric=True,
						temp_unit="C",
					)
					T_out_try = float(col_try.get("t_out_C", np.nan))
				except Exception:
					T_out_try = np.nan

				if not np.isfinite(T_out_try):
					T_out_try = T_in_guess
				T_out_try = min(T_out_try, float(max_htf_temp_C))

				# Approximate return-side coupling for this timestep and relax to avoid chatter.
				T_return_est = max(factory_water_target_C + pinch_delta_C, min(T_out_try, target_hex_htf_temp_C))
				T_in_new = (1.0 - relax) * T_in_guess + relax * T_return_est
				inner_solver_dm_col_m3s = float(abs(m_try - m_prev))
				inner_solver_dTin_C = float(abs(T_in_new - T_in_guess))
				if (inner_solver_dm_col_m3s < 1e-5) and (inner_solver_dTin_C < 0.05):
					m_prev = m_try
					T_in_guess = T_in_new
					inner_solver_converged = True
					break
				m_prev = m_try
				T_in_guess = T_in_new

			collector_t_in_C = float(T_in_guess)
			m_col_vol_flow_m3s = float(m_prev)
		elif flow_optimizer_mode == 'legacy':
			# Fallback: solar-driven flow sizing constrained by max collector outlet temperature.
			rho_htf, cp_htf = _htf_props_at(htf_in_temp_C)
			deltaT_limit = max_htf_temp_C - htf_in_temp_C
			if deltaT_limit <= 0:
				m_col_vol_flow_m3s = 0.0
			else:
				try:
					cp_ctrl = PropsSI("C", "T", ((htf_in_temp_C + max_htf_temp_C) / 2.0) + 273.15, "Q", 0, fluid)
				except Exception:
					cp_ctrl = cp_htf
				cp_ctrl = max(float(cp_ctrl), 1e-9)
				m_dot_mass_needed = solar_power_available_W / (cp_ctrl * max(deltaT_limit, 1e-9))
				m_col_vol_flow_m3s = _cap_collector_flow(m_dot_mass_needed / rho_htf)
			collector_t_in_C = float(htf_in_temp_C)

		# Use the control-solved collector inlet for field power/outlet evaluation.
		rho_htf, cp_htf = _htf_props_at(collector_t_in_C)

		# compute collector outlet temperature and actual absorbed power with the given flow
		try:
			col_res = solar_collector_outlet_temperature(
				t_in=collector_t_in_C,
				m_dot=m_col_vol_flow_m3s,
				dni=float(dni_value) if not pd.isna(dni_value) else 0.0,
				efficiency=collector_efficiency,
				area=collector_area_m2,
				volumetric=True,
				temp_unit="C",
			)
			solar_power_W = col_res["power_W"]
			collector_t_out_C = col_res.get("t_out_C", np.nan)
			# if no solar power is produced, report collector outlet as NaN
			if float(solar_power_W) <= 0.0:
				collector_t_out_C = np.nan
			# Enforce HTF temperature limit by curtailing effective absorbed power if needed.
			if not np.isnan(collector_t_out_C) and collector_t_out_C > max_htf_temp_C:
				collector_t_out_C = float(max_htf_temp_C)
				m_dot_mass_col = m_col_vol_flow_m3s * rho_htf
				try:
					cp_eff = PropsSI("C", "T", ((collector_t_in_C + collector_t_out_C) / 2.0) + 273.15, "Q", 0, fluid)
				except Exception:
					cp_eff = cp_htf
				solar_power_W = max(0.0, m_dot_mass_col * cp_eff * (collector_t_out_C - collector_t_in_C))
		except Exception:
			solar_power_W = 0.0
			collector_t_out_C = np.nan
		collector_curtailment_W = max(0.0, solar_power_available_W - max(0.0, solar_power_W))
		# debug printing per-timestep is emitted later after flows/allocation are known

		# --- Determine HTF supply to the O/W HEX (mixing of collector + CTES + optional recirc) ---
		# available flows (use computed collector volumetric flow as available)
		m_col_avail = m_col_vol_flow_m3s
		# If collector outlet temperature is unavailable, collector stream cannot be mixed into HEX.
		if np.isnan(collector_t_out_C):
			m_col_avail = 0.0
		m_col_avail = min(m_col_avail, collector_flow_hard_cap_m3s)
		if not disable_flow_rate_limits:
			if max_collector_flow_m3s is not None:
				m_col_avail = min(m_col_avail, max_collector_flow_m3s)
			if max_pump_flow_m3s is not None:
				m_col_avail = min(m_col_avail, max_pump_flow_m3s)
		# By default CTES flow availability is independent of collector flow,
		# so night-time discharge remains possible.
		if max_ctes_flow_m3s is not None and not disable_flow_rate_limits:
			m_ctes_discharge_avail = float(max_ctes_flow_m3s)
		else:
			m_ctes_discharge_avail = float(target_hex_htf_flow_m3s)
		if not disable_flow_rate_limits and max_pump_flow_m3s is not None:
			m_ctes_discharge_avail = min(m_ctes_discharge_avail, float(max_pump_flow_m3s))
		m_ctes_discharge_avail = max(0.0, m_ctes_discharge_avail)
		T_col = collector_t_out_C
		T_ctes = (T_outlet(ctes_y, 'discharging') if (ctes_y is not None and T_outlet is not None) else htf_in_temp_C)
		if np.isfinite(T_ctes):
			if ctes_discharge_enabled and (T_ctes <= ctes_min_discharge_temp_C):
				ctes_discharge_enabled = False
			elif (not ctes_discharge_enabled) and (T_ctes >= (ctes_min_discharge_temp_C + float(ctes_discharge_hysteresis_C))):
				ctes_discharge_enabled = True
		if (not ctes_discharge_enabled) or (not np.isnan(T_ctes) and T_ctes <= ctes_min_discharge_temp_C):
			m_ctes_discharge_avail = 0.0

		def _safe_mix_temp(m1, T1, m2, T2, default_T):
			num = 0.0
			den = 0.0
			if m1 > 0 and not np.isnan(T1):
				num += m1 * T1
				den += m1
			if m2 > 0 and not np.isnan(T2):
				num += m2 * T2
				den += m2
			return (num / den) if den > 0 else default_T

		# target feed to HEX
		m_req = target_hex_htf_flow_m3s if factory_active else 0.0
		# provisional target for mixing control.
		# In CTES-only mode (no collector stream), keep primary target (182 C).
		if not factory_active:
			T_req = target_hex_htf_temp_C
		elif (m_col_avail <= 1e-9) or np.isnan(T_col):
			T_req = target_hex_htf_temp_C
		elif T_col >= target_hex_htf_temp_C:
			T_req = target_hex_htf_temp_C
		else:
			T_req = target_hex_htf_temp_fallback_C

		# start by using as much collector flow as reasonable up to m_req
		m_col_use = min(m_col_avail, m_req)

		# compute CTES flow needed only when collector stream is thermally insufficient.
		m_ctes_needed = 0.0
		if m_col_use > 0 and (not np.isnan(T_col)) and (T_col < T_req - 1e-9) and (T_ctes - T_req) != 0:
			# solve m_col*T_col + m_ctes*T_ctes = m_req*T_req  => m_ctes = (m_req*T_req - m_col*T_col)/(T_ctes - T_req)
			m_ctes_needed = (m_req * T_req - m_col_use * T_col) / (T_ctes - T_req)
			m_ctes_needed_mix = m_ctes_needed
		elif m_col_use == 0 and (T_ctes - T_req) != 0:
			# Do not force full 0.021 m3/s in CTES-only mode; let power-based sizing
			# determine the required external flow and leave spare HEX capacity for recirculation.
			m_ctes_needed = 0.0

		# Also request CTES support based on power shortfall (not only temperature shortfall),
		# so sunrise periods do not fall into collector-only backup-heavy behavior.
		if factory_active and m_ctes_discharge_avail > 0 and T_ctes > ctes_min_discharge_temp_C:
			power_shortfall_W = max(0.0, factory_load_W - max(0.0, solar_power_W))
			if power_shortfall_W > 0.0:
				T_sink_ref_C = max(factory_water_target_C + pinch_delta_C, htf_in_temp_C)
				deltaT_support = max(1e-6, T_ctes - T_sink_ref_C)
				m_ctes_needed_power = power_shortfall_W / max(rho_htf * cp_htf * deltaT_support, 1e-9)
				m_ctes_needed = max(m_ctes_needed, m_ctes_needed_power)

		m_ctes_use_pre_clip = max(0.0, float(m_ctes_needed))
		m_ctes_use = float(np.clip(m_ctes_needed, 0.0, m_ctes_discharge_avail))
		m_ctes_use_post_clip = m_ctes_use

		# In single-source operation, trim external flow to the minimum thermal demand
		# (with a small margin). This preserves spare HEX capacity for recirculation.
		if factory_active and m_col_use > 1e-12 and m_ctes_use <= 1e-12 and (not np.isnan(T_col)):
			T_out_min_req_C = factory_water_target_C + pinch_delta_C
			dT_src = max(1e-6, T_col - T_out_min_req_C)
			m_dot_need = factory_load_W / max(cp_htf * dT_src, 1e-9)
			m_vol_need = m_dot_need / max(rho_htf, 1e-9)
			m_col_use = min(m_col_use, 1.05 * max(0.0, m_vol_need))
		if factory_active and m_ctes_use > 1e-12 and m_col_use <= 1e-12 and (not np.isnan(T_ctes)):
			m_ctes_before_trim = m_ctes_use
			T_out_min_req_C = factory_water_target_C + pinch_delta_C
			dT_src = max(1e-6, T_ctes - T_out_min_req_C)
			m_dot_need = factory_load_W / max(cp_htf * dT_src, 1e-9)
			m_vol_need = m_dot_need / max(rho_htf, 1e-9)
			m_ctes_use = min(m_ctes_use, 1.05 * max(0.0, m_vol_need))
			m_ctes_trim_reduction += max(0.0, m_ctes_before_trim - m_ctes_use)

		# if CTES unavailable or insufficient, try to supplement with collector more (if more available)
		if m_col_use + m_ctes_use < m_req:
			extra_from_col = min(m_col_avail - m_col_use, m_req - (m_col_use + m_ctes_use))
			m_col_use += extra_from_col

		# If collector power exceeds factory demand in collector-only operation,
		# free a small share of collector flow for simultaneous CTES charging.
		# This avoids B-mode timesteps with curtailment but zero charging flow.
		if (
			factory_active
			and m_ctes_use <= 1e-9
			and m_col_use > 1e-9
			and solar_power_W > max(0.0, factory_load_W)
		):
			surplus_frac = 1.0 - (factory_load_W / max(solar_power_W, 1e-9))
			# Keep the reduction conservative to avoid upsetting HEX delivery.
			reduction_frac = float(np.clip(surplus_frac, 0.0, 0.10))
			m_col_use *= (1.0 - reduction_frac)

		# Enforce maximum HEX throughput: external (collector+CTES) plus recirculation
		# may not exceed m_req (= 0.021 m3/s when factory is active).
		denom_ext = m_col_use + m_ctes_use
		if factory_active and m_req > 0 and denom_ext > m_req:
			overflow = denom_ext - m_req
			# Keep CTES support when needed; reduce collector first.
			reduce_col = min(m_col_use, overflow)
			m_col_use -= reduce_col
			overflow -= reduce_col
			if overflow > 0:
				m_ctes_before_overflow = m_ctes_use
				m_ctes_use = max(0.0, m_ctes_use - overflow)
				m_ctes_overflow_reduction += max(0.0, m_ctes_before_overflow - m_ctes_use)
		denom_ext = m_col_use + m_ctes_use

		# External-source mixed inlet to HEX before recirculation.
		T_mix_ext = _safe_mix_temp(m_col_use, T_col, m_ctes_use, T_ctes, htf_in_temp_C)
		T_mix = T_mix_ext
		m_recirc = 0.0
		# Total HEX flow includes recirculation and is capped by m_req.
		m_oil_hex = (denom_ext if not factory_active else min(m_req, denom_ext))

		# Control mode for this timestep:
		# primary: target HEX inlet 182 C and water target 140 C
		# fallback: if 182 C cannot be reached, use 160 C / 133 C target mode.
		fallback_mode_active = bool(factory_active and (T_mix + 1e-9 < target_hex_htf_temp_C))
		hex_in_target_used_C = target_hex_htf_temp_fallback_C if fallback_mode_active else target_hex_htf_temp_C
		water_target_used_C = factory_water_fallback_C if fallback_mode_active else factory_water_target_C
		mix_target_used_C = hex_in_target_used_C

		def _compute_hex_required_inlet(T_in_C: float, oil_flow_m3s: float) -> tuple[float, bool]:
			req_oil_in_C = np.nan
			hex_ok = False
			if factory_active and oil_flow_m3s > 0:
				# water-side required power to reach factory target
				try:
					rho_w_loc = PropsSI("D", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
					cp_w_loc = PropsSI("C", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
				except Exception:
					rho_w_loc = 1000.0
					cp_w_loc = 4180.0
				m_w_loc = factory_water_flow_m3s * rho_w_loc
				desired_W_full_loc = m_w_loc * cp_w_loc * (water_target_used_C - factory_water_in_C)
				# oil properties at inlet temperature guess
				try:
					rho_oil_loc = PropsSI("D", "T", T_in_C + 273.15, "Q", 0, "INCOMP::PNF")
					cp_oil_loc = PropsSI("C", "T", T_in_C + 273.15, "Q", 0, "INCOMP::PNF")
				except Exception:
					rho_oil_loc = 870.0
					cp_oil_loc = 2200.0
				m_dot_oil_loc = oil_flow_m3s * rho_oil_loc
				oil_out_min_C_loc = water_target_used_C + pinch_delta_C
				if m_dot_oil_loc > 0 and cp_oil_loc > 0:
					req_oil_in_C = oil_out_min_C_loc + desired_W_full_loc / (m_dot_oil_loc * cp_oil_loc)
					hex_ok = T_in_C >= req_oil_in_C
			return req_oil_in_C, hex_ok

		# compute required oil inlet temperature to meet the factory water target given m_oil_hex
		required_oil_in_C, hex_temp_sufficient = _compute_hex_required_inlet(T_mix, m_oil_hex)

		# compute water-side desired power for factory if active
		desired_water_power_W = 0.0
		if factory_active:
			# ask to heat factory water from inlet to target
			try:
				rho_w = PropsSI("D", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
				cp_w = PropsSI("C", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
			except Exception:
				rho_w = 1000.0; cp_w = 4180.0
			m_w = factory_water_flow_m3s * rho_w
			desired_water_power_W = m_w * cp_w * (water_target_used_C - factory_water_in_C)

		# call the detailed oil-water HEX model to see how much power can be delivered
		oil_out_C, provided_W, water_out_C = oil_water_hex(
			oil_in_C=T_mix,
			oil_flow_m3s=m_oil_hex,
			desired_water_power_W=desired_water_power_W if factory_active else 0.0,
			water_vol_flow_m3s=(factory_water_flow_m3s if factory_active else None),
			water_in_C=factory_water_in_C,
			water_target_C=(water_target_used_C if factory_active else None),
				pinch_delta=pinch_delta_C,
		)

		# Recirculation iteration: use spare HEX flow capacity (m_req - external flow)
		# and maximize recirculation toward the active HEX inlet target.
		if factory_active and m_req > 0 and denom_ext > 1e-12:
			m_recirc_cap = max(0.0, m_req - denom_ext)
			if m_recirc_cap > 1e-12:
				for _ in range(4):
					if np.isnan(oil_out_C):
						break
					T_recirc_ref = float(oil_out_C)
					# Recirculation helps whenever return is cooler than external supply.
					if T_mix <= mix_target_used_C + 1e-9:
						break
					if T_recirc_ref >= T_mix_ext - 1e-9:
						break
					if T_recirc_ref < mix_target_used_C - 1e-9:
						# enough cooling potential exists to possibly hit target
						m_need = denom_ext * (T_mix_ext - mix_target_used_C) / max(1e-12, (mix_target_used_C - T_recirc_ref))
						m_recirc_new = float(np.clip(m_need, 0.0, m_recirc_cap))
					else:
						# return is still above target: use all available spare flow for best-effort cooling
						m_recirc_new = float(m_recirc_cap)
					if abs(m_recirc_new - m_recirc) < 1e-10:
						break
					m_recirc = m_recirc_new
					m_oil_hex = denom_ext + m_recirc
					T_mix = ((denom_ext * T_mix_ext) + (m_recirc * T_recirc_ref)) / max(1e-12, m_oil_hex)
					required_oil_in_C, hex_temp_sufficient = _compute_hex_required_inlet(T_mix, m_oil_hex)
					oil_out_C, provided_W, water_out_C = oil_water_hex(
						oil_in_C=T_mix,
						oil_flow_m3s=m_oil_hex,
						desired_water_power_W=desired_water_power_W if factory_active else 0.0,
						water_vol_flow_m3s=(factory_water_flow_m3s if factory_active else None),
						water_in_C=factory_water_in_C,
						water_target_C=(water_target_used_C if factory_active else None),
						pinch_delta=pinch_delta_C,
					)

		# If HEX couldn't meet the full demand and a backup heater would be needed,
		# allow a relaxed pinch (10 C) and re-evaluate to see if that reduces backup heater use.
		if factory_active and (not fallback_mode_active):
			remaining_deficit = max(0.0, desired_water_power_W - provided_W)
			if remaining_deficit > 0 and pinch_delta_C > 10:
				# re-run HEX with relaxed pinch
				oil_out_C_relaxed, provided_W_relaxed, water_out_C_relaxed = oil_water_hex(
					oil_in_C=T_mix,
					oil_flow_m3s=m_oil_hex,
					desired_water_power_W=desired_water_power_W,
					water_vol_flow_m3s=factory_water_flow_m3s,
					water_in_C=factory_water_in_C,
					water_target_C=water_target_used_C,
					pinch_delta=10.0,
				)
				# adopt relaxed result if it supplies more
				if provided_W_relaxed > provided_W:
					provided_W = provided_W_relaxed
					oil_out_C = oil_out_C_relaxed
					water_out_C = water_out_C_relaxed

		# Report HEX sufficiency based on the final achieved duty, not only the
		# earlier inlet-temperature heuristic. This keeps STATUS consistent when a
		# relaxed pinch solve still meets factory duty.
		if factory_active:
			hex_power_tol_W = max(1e3, 1e-3 * factory_load_W)
			hex_temp_sufficient = bool(provided_W >= max(0.0, desired_water_power_W - hex_power_tol_W))

		# account for provided_W as energy used from HTF (charging/discharging logic follows)
		# Split source contribution by external source flows only; recirculation is internal.
		solar_fraction = (m_col_use / max(denom_ext, 1e-12)) if denom_ext > 0 else 0.0
		solar_contribution_W = provided_W * solar_fraction
		ctes_contribution_W = provided_W * (1.0 - solar_fraction)
		# Request CTES against unmet factory demand (after solar contribution), not only
		# against preliminary HEX split. This avoids small artificial backup slices.
		requested_ctes_to_factory_W = max(0.0, factory_load_W - solar_contribution_W)
		ctes_loss_allowance_W = 0.0
		if heat_loss_W is not None and ctes_y is not None:
			try:
				ctes_loss_allowance_W = max(0.0, -float(heat_loss_W(ctes_y)))
			except Exception:
				ctes_loss_allowance_W = 0.0

		# update CTES energy when CTES provided some of the heat (discharge)
		if ctes_contribution_W > 0 and m_ctes_use > 0 and step_ctes is not None:
			m_ctes_before_discharge_fit = m_ctes_use
			# The fluid entering CTES during discharge is the HEX return oil, not
			# the fixed loop inlet reference temperature.
			T_return_to_ctes_C = oil_out_C if not np.isnan(oil_out_C) else htf_in_temp_C
			# Available hydraulic discharge mass flow based on allocated volume flow.
			m_ctes_mass_hydraulic = m_ctes_use * rho_htf if m_ctes_use > 0 else 0.0
			m_ctes_mass_hydraulic_max = m_ctes_discharge_avail * rho_htf if m_ctes_discharge_avail > 0 else 0.0
			# Let the CTES model determine extractable power; size mass flow against
			# target energy directly, bounded only by hydraulic availability.
			m_ctes_mass = m_ctes_mass_hydraulic
			m_ctes_use = m_ctes_mass / max(rho_htf, 1e-9)

			if m_ctes_mass > 0:
				prev_E = ctes_energy_J
				y_before_discharge = ctes_y
				discharge_step_final = None
				provided_from_ctes_instant_W = 0.0
				allowed_drop_J = max(0.0, (requested_ctes_to_factory_W + ctes_loss_allowance_W) * float(timestep_seconds))
				if allowed_drop_J <= 0.0:
					ctes_discharge_solver_mode = 'zero'
					m_ctes_mass = 0.0
					m_ctes_use = 0.0
					ctes_contribution_W = 0.0
					ctes_to_factory_W = 0.0
					provided_from_ctes_instant_W = 0.0
				else:
					target_drop_J = max(0.0, requested_ctes_to_factory_W * float(timestep_seconds))
					# If the initial control request is insufficient, allow discharge sizing
					# to increase flow up to available CTES hydraulic capacity.
					m_upper = max(m_ctes_mass, m_ctes_mass_hydraulic_max)
					ctes_y = y_before_discharge
					discharge_step_upper = step_ctes(ctes_y, T_return_to_ctes_C, m_upper, 'discharging', timestep_seconds)
					drop_upper_J = max(0.0, prev_E - float(discharge_step_upper['energy_J']))
					selected_discharge_step = None

					# If even max hydraulic flow cannot reach target, use max available.
					if drop_upper_J <= target_drop_J:
						ctes_discharge_solver_mode = 'max'
						m_ctes_mass = m_upper
						m_ctes_use = m_ctes_mass / max(rho_htf, 1e-9)
						selected_discharge_step = discharge_step_upper
					else:
						# Two-stage directional stepping: coarse (0.001) then fine (0.0001)
						# to reduce iteration count while still refining near the target.
						ctes_discharge_solver_mode = 'step'
						step_flow_sequence_m3s = (0.001, 0.0001)
						m_guess_mass = float(np.clip(prev_ctes_discharge_flow_m3s, 0.0, m_upper / max(rho_htf, 1e-9))) * rho_htf
						m_try = float(np.clip(m_guess_mass, 0.0, m_upper))
						discharge_step_try = None

						for step_flow_m3s in step_flow_sequence_m3s:
							step_mass = step_flow_m3s * rho_htf
							max_iter = max(2, int(np.ceil(m_upper / max(step_mass, 1e-9))) + 2)
							prev_dir = 0

							for _ in range(max_iter):
								ctes_discharge_iterations += 1
								ctes_y = y_before_discharge
								discharge_step_try = step_ctes(ctes_y, T_return_to_ctes_C, m_try, 'discharging', timestep_seconds)
								drop_try_J = max(0.0, prev_E - float(discharge_step_try['energy_J']))

								if drop_try_J < target_drop_J:
									dir_sign = 1
								elif drop_try_J > target_drop_J:
									dir_sign = -1
								else:
									dir_sign = 0

								if prev_dir != 0 and dir_sign != 0 and dir_sign != prev_dir:
									ctes_discharge_solver_mode = 'step_turn'
									break

								if dir_sign == 0:
									break

								m_next = float(np.clip(m_try + (dir_sign * step_mass), 0.0, m_upper))
								if abs(m_next - m_try) < 1e-12:
									break

								m_try = m_next
								prev_dir = dir_sign

						if discharge_step_try is None:
							ctes_y = y_before_discharge
							discharge_step_try = step_ctes(ctes_y, T_return_to_ctes_C, m_try, 'discharging', timestep_seconds)

						m_ctes_mass = m_try
						m_ctes_use = m_ctes_mass / max(rho_htf, 1e-9)
						selected_discharge_step = discharge_step_try

					ctes_discharge_solve_events += 1
					total_ctes_discharge_iterations += ctes_discharge_iterations
					max_ctes_discharge_iterations = max(max_ctes_discharge_iterations, ctes_discharge_iterations)
					hourly_ctes_discharge_solve_events += 1
					hourly_ctes_discharge_iterations += ctes_discharge_iterations
					hourly_max_ctes_discharge_iterations = max(hourly_max_ctes_discharge_iterations, ctes_discharge_iterations)

					if selected_discharge_step is not None:
						ctes_state_advanced = True
						ctes_y = selected_discharge_step['y']
						ctes_energy_J = selected_discharge_step['energy_J']
						discharge_step_final = selected_discharge_step
						ctes_model_diag = selected_discharge_step
					provided_from_ctes_instant_W = max(0.0, (prev_E - ctes_energy_J) / max(1.0, timestep_seconds))
				# record the actual contribution (may differ from requested)
				ctes_contribution_W = min(provided_from_ctes_instant_W, requested_ctes_to_factory_W, factory_load_W)
				ctes_to_factory_W = ctes_contribution_W
				if m_ctes_mass > 0 and ctes_contribution_W > 0:
					ctes_flow_dir = 'to_hex'
					last_ctes_mdot = m_ctes_mass
					last_ctes_T_in = T_return_to_ctes_C
					last_ctes_T_out = (discharge_step_final.get('T_out_C', np.nan) if discharge_step_final is not None else np.nan)
				else:
					last_ctes_mdot = 0.0
					last_ctes_T_in = np.nan
					last_ctes_T_out = np.nan
			else:
				ctes_contribution_W = 0.0
				ctes_to_factory_W = 0.0
			m_ctes_discharge_fit_reduction += max(0.0, m_ctes_before_discharge_fit - m_ctes_use)
			if m_ctes_use > 1e-12:
				prev_ctes_discharge_flow_m3s = float(m_ctes_use)
		else:
			# fallback to stub behaviour if detailed model is not available
			if ctes_contribution_W > 0:
				ctes_to_factory_W = ctes_contribution_W
				ctes_flow_dir = 'to_hex'

		# Reconcile HEX mixing/recirculation after possible CTES discharge-flow correction.
		# During discharge sizing, m_ctes_use can change; refresh recirc with final external flows.
		if factory_active and m_req > 0:
			denom_ext_post = max(0.0, m_col_use + m_ctes_use)
			if denom_ext_post > m_req:
				overflow_post = denom_ext_post - m_req
				reduce_col_post = min(m_col_use, overflow_post)
				m_col_use -= reduce_col_post
				overflow_post -= reduce_col_post
				if overflow_post > 0:
					m_ctes_before_overflow_post = m_ctes_use
					m_ctes_use = max(0.0, m_ctes_use - overflow_post)
					m_ctes_overflow_reduction += max(0.0, m_ctes_before_overflow_post - m_ctes_use)
				denom_ext_post = max(0.0, m_col_use + m_ctes_use)

			T_ctes_post = (T_outlet(ctes_y, 'discharging') if (ctes_y is not None and T_outlet is not None) else T_ctes)
			if np.isnan(T_ctes_post):
				T_ctes_post = T_ctes
			T_mix_ext_post = _safe_mix_temp(m_col_use, T_col, m_ctes_use, T_ctes_post, htf_in_temp_C)
			m_recirc_cap_post = max(0.0, m_req - denom_ext_post)
			m_recirc_post = 0.0
			T_mix_post = T_mix_ext_post
			if m_recirc_cap_post > 1e-12 and not np.isnan(oil_out_C):
				T_recirc_post = float(oil_out_C)
				if (T_mix_ext_post > mix_target_used_C + 1e-9) and (T_recirc_post < T_mix_ext_post - 1e-9):
					if T_recirc_post < mix_target_used_C - 1e-9:
						m_need_post = denom_ext_post * (T_mix_ext_post - mix_target_used_C) / max(1e-12, (mix_target_used_C - T_recirc_post))
						m_recirc_post = float(np.clip(m_need_post, 0.0, m_recirc_cap_post))
					else:
						m_recirc_post = float(m_recirc_cap_post)
					m_oil_hex_post = denom_ext_post + m_recirc_post
					T_mix_post = ((denom_ext_post * T_mix_ext_post) + (m_recirc_post * T_recirc_post)) / max(1e-12, m_oil_hex_post)
				else:
					m_oil_hex_post = denom_ext_post
			else:
				m_oil_hex_post = denom_ext_post

			if (abs(m_recirc_post - m_recirc) > 1e-10) or (abs(T_mix_post - T_mix) > 1e-9) or (abs(m_oil_hex_post - m_oil_hex) > 1e-10):
				m_recirc = m_recirc_post
				m_oil_hex = m_oil_hex_post
				T_mix = T_mix_post
				required_oil_in_C, hex_temp_sufficient = _compute_hex_required_inlet(T_mix, m_oil_hex)
				oil_out_C, provided_W, water_out_C = oil_water_hex(
					oil_in_C=T_mix,
					oil_flow_m3s=m_oil_hex,
					desired_water_power_W=desired_water_power_W if factory_active else 0.0,
					water_vol_flow_m3s=(factory_water_flow_m3s if factory_active else None),
					water_in_C=factory_water_in_C,
					water_target_C=(water_target_used_C if factory_active else None),
					pinch_delta=pinch_delta_C,
				)

		# remaining solar energy after HEX provisioning is available for charging CTES or exported
		remaining_solar_W = max(0.0, solar_power_W - solar_contribution_W)

		# Ensure delivered factory heat is consistent with corrected CTES contribution.
		# This resolves cases where HEX pre-calculation assumed more CTES contribution than
		# physically extracted after discharge correction.
		provided_W = max(0.0, solar_contribution_W + ctes_to_factory_W)
		if factory_active and factory_water_flow_m3s > 0:
			try:
				rho_w_dbg = PropsSI("D", "T", factory_water_in_C + 273.15, "P", 8e5, "Water")
				cp_w_dbg = PropsSI("C", "T", factory_water_in_C + 273.15, "P", 8e5, "Water")
			except Exception:
				rho_w_dbg = 1000.0
				cp_w_dbg = 4180.0
			m_dot_w_dbg = factory_water_flow_m3s * rho_w_dbg
			if m_dot_w_dbg > 0 and cp_w_dbg > 0:
				water_out_C = factory_water_in_C + provided_W / (m_dot_w_dbg * cp_w_dbg)

		# factory deficit after HEX is met by backup heater (CTES discharge was already handled in HEX split)
		# Apply a small numerical tolerance so solver residue does not trigger false D*.
		raw_factory_deficit_W = max(0.0, factory_load_W - provided_W)
		if factory_active:
			deficit_tolerance_W = max(1e3, 1e-3 * factory_load_W)  # 1 kW or 0.1%
			factory_deficit_W = 0.0 if raw_factory_deficit_W <= deficit_tolerance_W else raw_factory_deficit_W
			# Final HEX sufficiency flag should match delivered factory duty after
			# all control corrections and deficit tolerance handling.
			hex_temp_sufficient = bool(factory_deficit_W <= deficit_tolerance_W)
		else:
			deficit_tolerance_W = 1e3
			factory_deficit_W = 0.0
			hex_temp_sufficient = True
		tolerated_deficit_W = max(0.0, raw_factory_deficit_W - factory_deficit_W)
		backup_heater_W = factory_deficit_W
		provided_from_ctes_W = ctes_to_factory_W

		# Summarize the dominant CTES flow limiter for debug visibility.
		eps_flow = 1e-9
		if not factory_active:
			ctes_flow_limit_reason = 'factory_off'
		elif m_ctes_discharge_avail <= eps_flow:
			if (not ctes_discharge_enabled) or (not np.isnan(T_ctes) and T_ctes <= ctes_min_discharge_temp_C):
				ctes_flow_limit_reason = 'temp_gate'
			else:
				ctes_flow_limit_reason = 'no_avail'
		elif m_ctes_discharge_fit_reduction > eps_flow:
			ctes_flow_limit_reason = 'ctes_fit'
		elif m_ctes_overflow_reduction > eps_flow:
			ctes_flow_limit_reason = 'hex_cap'
		elif m_ctes_trim_reduction > eps_flow:
			ctes_flow_limit_reason = 'trim_min'
		elif m_ctes_use_pre_clip > (m_ctes_use_post_clip + eps_flow):
			ctes_flow_limit_reason = 'avail_cap'
		elif m_ctes_use_pre_clip <= eps_flow:
			ctes_flow_limit_reason = 'no_request'

		# charge CTES from any remaining solar after serving the factory through HEX
		if remaining_solar_W > 0:
			if step_ctes is not None:
				# charging flow is explicitly tracked for debug/reporting
				# Use only collector flow not already allocated to HEX, and enforce
				# the global collector split budget (hex + charge <= target_hex_htf_flow_m3s).
				collector_left_for_charge = max(0.0, m_col_avail - m_col_use)
				split_budget_left = max(0.0, collector_split_budget_m3s - m_col_use)
				served_factory_now = (factory_deficit_W <= deficit_tolerance_W + 1e-9)
				in_early_sun_window = (early_sun_start_hour <= hour_of_day < early_sun_end_hour)
				if early_sun_charge_priority and in_early_sun_window and served_factory_now:
					# Morning priority: use any collector flow not needed by HEX to charge CTES.
					allowed_charge_flow = collector_left_for_charge
				else:
					allowed_charge_flow = split_budget_left
				m_ctes_charge_avail = min(collector_left_for_charge, allowed_charge_flow)
				if not disable_flow_rate_limits and max_ctes_flow_m3s is not None:
					m_ctes_charge_avail = min(m_ctes_charge_avail, float(max_ctes_flow_m3s))
				m_ctes_charge_use_m3s = float(m_ctes_charge_avail)
				m_ctes_charge_mass = m_ctes_charge_use_m3s * rho_htf if m_ctes_charge_use_m3s > 0 else 0.0
				prev_E = ctes_energy_J
				if m_ctes_charge_mass > 0:
					T_in_for_ctes = collector_t_out_C if not np.isnan(collector_t_out_C) else htf_in_temp_C
					# Charging is flow/energy-limited by upstream allocation; use one direct CTES step.
					res_step = step_ctes(ctes_y, T_in_for_ctes, m_ctes_charge_mass, 'charging', timestep_seconds)
					ctes_state_advanced = True
					ctes_model_diag = res_step
				else:
					T_in_for_ctes = None
					res_step = step_ctes(ctes_y, None, 0.0, 'storage', timestep_seconds)
					ctes_state_advanced = True
					ctes_model_diag = res_step
				ctes_y = res_step['y']
				ctes_energy_J = res_step['energy_J']
				# heat exchanged with CTES from the HTF (W)
				last_ctes_mdot = m_ctes_charge_mass
				last_ctes_T_in = T_in_for_ctes if T_in_for_ctes is not None else np.nan
				last_ctes_T_out = res_step.get('T_out_C', np.nan)
				dt_local = max(1.0, float(timestep_seconds))
				# Charging input should represent heat transferred from HTF to CTES,
				# not net storage rise (which can be negative when losses dominate).
				actual_charge_W = 0.0
				if isinstance(res_step, dict) and ('diag_q_to_solid_corr_W' in res_step):
					# Net thermal power to solid storage; subtracts transient fluid inventory effects.
					actual_charge_W = max(0.0, float(res_step.get('diag_q_to_solid_corr_W', 0.0)))
				elif isinstance(res_step, dict) and ('diag_q_fluid_to_solid_total_W' in res_step):
					actual_charge_W = max(0.0, float(res_step.get('diag_q_fluid_to_solid_total_W', 0.0)))
				elif m_ctes_charge_mass > 0 and (T_in_for_ctes is not None) and (not np.isnan(last_ctes_T_out)):
					try:
						cp_charge = PropsSI("C", "T", ((T_in_for_ctes + last_ctes_T_out) / 2.0) + 273.15, "Q", 0, fluid)
					except Exception:
						cp_charge = cp_htf
					cp_charge = max(float(cp_charge), 1e-9)
					actual_charge_W = max(0.0, m_ctes_charge_mass * cp_charge * max(0.0, (T_in_for_ctes - last_ctes_T_out)))
				actual_charge_display_W = min(max(0.0, remaining_solar_W), actual_charge_W)
				if actual_charge_W > 0 and ctes_flow_dir == 'none':
					ctes_flow_dir = 'to_coll'
			else:
				actual_charge_W = remaining_solar_W
				actual_charge_display_W = remaining_solar_W
				if ctes_flow_dir == 'none':
					ctes_flow_dir = 'to_coll'

		# If useful sinks are saturated (factory + CTES), curtail excess solar.
		try:
			ctes_soc_for_curtail = float(stored_energy_J(ctes_y) / max(1.0, ctes_capacity_J)) if (ctes_y is not None and stored_energy_J is not None) else np.nan
		except Exception:
			ctes_soc_for_curtail = np.nan
		ctes_near_full = bool((ctes_soc_for_curtail >= 0.995) if not np.isnan(ctes_soc_for_curtail) else False)
		loss_sink_allowance_W = ctes_loss_allowance_W if ctes_near_full else 0.0
		downstream_curtail_W = max(0.0, remaining_solar_W - max(0.0, actual_charge_display_W))
		solar_loss_offset_W = min(downstream_curtail_W, max(0.0, loss_sink_allowance_W))
		downstream_curtail_W = max(0.0, downstream_curtail_W - solar_loss_offset_W)
		solar_power_effective_W = max(0.0, solar_power_W - downstream_curtail_W)
		# total curtailment includes collector-stage and downstream curtailment.
		solar_curtailed_W = max(0.0, solar_power_available_W - solar_power_effective_W)
		# No external export path is modeled; curtailed solar is dropped.
		remaining_solar_W = 0.0

		# Determine operating mode letter for this timestep (after all allocations).
		# A: collectors->CTES (factory off)
		# B: collectors serve factory (optionally while charging CTES)
		# C: collectors + CTES discharge serve factory
		# D: CTES-only/night discharge serves factory
		if not factory_active:
			if actual_charge_display_W > 1e-6:
				op_mode = 'A'
			else:
				op_mode = '-'
		else:
			if (m_col_use > 1e-9) and (m_ctes_use > 1e-9):
				op_mode = 'C'
			elif (m_col_use > 1e-9):
				op_mode = 'B'
			elif (m_ctes_use > 1e-9):
				op_mode = 'D'
			else:
				op_mode = '-'
		if backup_heater_W > 1e-6 and op_mode in ('A', 'B', 'C', 'D'):
			op_mode = f"{op_mode}*"
		hourly_mode_counts[op_mode] = int(hourly_mode_counts.get(op_mode, 0)) + 1

		# Always advance CTES in storage mode when no charge/discharge step occurred,
		# so ambient losses are present from the beginning.
		if step_ctes is not None and not ctes_state_advanced:
			res_step = step_ctes(ctes_y, None, 0.0, 'storage', timestep_seconds)
			ctes_y = res_step['y']
			ctes_energy_J = res_step['energy_J']
			ctes_model_diag = res_step
			last_ctes_mdot = 0.0
			last_ctes_T_in = np.nan
			last_ctes_T_out = np.nan
			ctes_state_advanced = True

		# Update HTF return temperature for next timestep from the active loop return.
		# This prevents collector inlet from remaining fixed at the initial value.
		htf_in_next_C = htf_in_temp_C
		ret_num = 0.0
		ret_den = 0.0
		if m_oil_hex > 1e-9 and not np.isnan(oil_out_C):
			ret_num += float(m_oil_hex) * float(oil_out_C)
			ret_den += float(m_oil_hex)
		if m_ctes_charge_use_m3s > 1e-9 and not np.isnan(last_ctes_T_out):
			ret_num += float(m_ctes_charge_use_m3s) * float(last_ctes_T_out)
			ret_den += float(m_ctes_charge_use_m3s)
		if ret_den > 0.0:
			htf_in_next_C = ret_num / ret_den
		# only commit finite values
		if np.isfinite(htf_in_next_C):
			htf_in_temp_C = htf_in_next_C

		net_power_W = remaining_solar_W - factory_deficit_W


		# compute CTES energy balance and losses for diagnostics
		dt = float(timestep_seconds)
		# energy change in storage over this timestep
		ctes_energy_change_J = float(ctes_energy_J) - float(prev_ctes_energy_J)
		dE_dt_W = ctes_energy_change_J / max(1.0, dt)
		# Report CTES power channels directly from modeled charge/discharge exchange.
		Q_out_from_ctes_W = max(0.0, ctes_to_factory_W)
		Q_in_htf_W = max(0.0, actual_charge_W)
		# CTES loss reporting is based on direct thermal loss to ambient from the model state.
		if heat_loss_W is not None and ctes_y is not None:
			current_loss_power_W = min(0.0, -float(heat_loss_W(ctes_y)))
		else:
			# fallback when detailed model helper is unavailable
			loss_J = ctes_energy_change_J - (Q_in_htf_W - Q_out_from_ctes_W) * dt
			current_loss_power_W = (loss_J / dt) if loss_J < 0 else 0.0
		Q_in_inferred_W = max(0.0, dE_dt_W + Q_out_from_ctes_W - current_loss_power_W)
		Q_in_to_ctes_W = Q_in_htf_W
		cum_ctes_loss_J += float(current_loss_power_W) * dt
		ctes_delta_MWh_step = ctes_energy_change_J / 3.6e9
		ctes_discharge_MWh_step = (Q_out_from_ctes_W * dt) / 3.6e9
		ctes_charge_MWh_step = (Q_in_to_ctes_W * dt) / 3.6e9
		ctes_loss_MWh_step = (-current_loss_power_W * dt) / 3.6e9
		# state residual: closure using only model state delta vs explicit Qin/Qout
		ctes_balance_residual_state_W = (ctes_energy_change_J / dt) - (Q_in_to_ctes_W - Q_out_from_ctes_W)
		# loss-inclusive residual: closure when adding separately reported ambient-loss term
		ctes_balance_residual_with_loss_W = ctes_balance_residual_state_W - current_loss_power_W
		hex_flow_balance_m3s = m_oil_hex - (m_col_use + m_ctes_use + m_recirc)

		# accumulate energies for summary (use timestep duration)
		cum_factory_consumed_J += float(factory_load_W) * dt
		cum_solar_produced_J += float(solar_power_effective_W) * dt
		cum_solar_curtailed_J += float(solar_curtailed_W) * dt
		cum_solar_supplied_to_factory_J += float(solar_contribution_W) * dt
		cum_ctes_supplied_to_factory_J += float(ctes_contribution_W) * dt
		cum_ctes_charge_input_J += float(Q_in_to_ctes_W) * dt
		cum_ctes_discharge_output_J += float(Q_out_from_ctes_W) * dt
		cum_backup_heater_J += float(backup_heater_W) * dt
		cum_tolerated_deficit_J += float(tolerated_deficit_W) * dt

		# debug printing per-timestep: show key values after allocation
		if debug:
			# print legend/note only once at the very beginning
			if first_debug_print:
				print('\nDEBUG NOTE: m_col_vol_flow_m3s = available collector volumetric flow; m_col_use_m3s = portion used to feed HEX')
				print(f'DEBUG NOTE: factory schedule active={respect_factory_schedule}, weekend_shutdown={factory_off_on_weekends}')
				print('DEBUG NOTE: recirc is internal mixing flow (m3/s) from HEX return to inlet used to trim T_mix to target; it is not extra external flow and is not included in m_oil_hex')
				print('DEBUG NOTE: ctes_flow reports active CTES direction this timestep: to_hex, to_coll, or none')
				# explain CTES energy sign convention once
				print('DEBUG NOTE: CTES stored energy is reported relative to T_min; negative means average concrete temperature < T_min')
				print(f'DEBUG NOTE: factory_nominal_load is computed from water side with CoolProp: {factory_nominal_load_W/1e6:.3f} MW at 120->140 C, 0.023 m3/s, 8 bar')
				print(f'DEBUG NOTE: collector flow hard cap enforced at {collector_flow_hard_cap_m3s:.3f} m3/s; excess solar is curtailed and outlet temperature is capped')
				print(f'DEBUG NOTE: collector split budget enforced: collector_hex + collector_chg <= {collector_split_budget_m3s:.3f} m3/s')
				first_debug_print = False
			# emit per-timestep debug lines every 10 minutes
			if (minute_of_hour % 10) == 0:
				def _cell_num(label, val, decimals=2, label_w=15, value_w=9):
					if pd.isna(val):
						sval = 'nan'
					else:
						sval = f"{float(val):.{decimals}f}"
					return f"{label:<{label_w}s} {sval:>{value_w}s}"
				def _cell_txt(label, txt, label_w=15, value_w=9):
					s = str(txt)
					if len(s) > value_w:
						s = s[:value_w]
					return f"{label:<{label_w}s} {s:>{value_w}s}"
				def _print_row(ts_label, row_tag, cells):
					print(f"  {ts_label:16s} | {row_tag:<9s} | " + " | ".join(cells))
				# ensure local diagnostic variables exist
				actual_charge_W = locals().get('actual_charge_W', 0.0)
				provided_from_ctes_W = locals().get('provided_from_ctes_W', 0.0)
				# CTES temperature and SOC if available
				try:
					# only report CTES outlet temperature when there was flow through CTES
					if 'last_ctes_mdot' in locals() and last_ctes_mdot > 0:
						if 'last_ctes_T_out' in locals() and not np.isnan(last_ctes_T_out):
							ctes_temp_now = float(last_ctes_T_out)
						else:
							ctes_temp_now = float(T_outlet(ctes_y, 'charging')) if (ctes_y is not None and T_outlet is not None) else np.nan
					else:
						ctes_temp_now = np.nan
				except Exception:
					ctes_temp_now = np.nan
				try:
					ctes_soc_now = float(stored_energy_J(ctes_y) / max(1.0, (ctes_capacity_J if 'ctes_capacity_J' in locals() else 1.0))) if (ctes_y is not None and stored_energy_J is not None) else np.nan
				except Exception:
					ctes_soc_now = np.nan
				# kW formatting
				solar_raw_kW = solar_power_available_W / 1000.0
				solar_abs_kW = solar_power_W / 1000.0
				provided_kW = provided_W / 1000.0
				ctes_kW = provided_from_ctes_W / 1000.0
				backup_kW = backup_heater_W / 1000.0
				factory_load_kW = factory_load_W / 1000.0
				# percentages of factory nominal load
				if factory_active and factory_nominal_load_W > 0:
					solar_pct = solar_contribution_W / factory_nominal_load_W * 100.0
					ctes_pct = ctes_to_factory_W / factory_nominal_load_W * 100.0
					backup_pct = backup_heater_W / factory_nominal_load_W * 100.0
				else:
					solar_pct = ctes_pct = backup_pct = float('nan')
				# factory power % (NaN when factory is off)
				if factory_active and factory_nominal_load_W > 0:
					factory_pct = factory_load_W / factory_nominal_load_W * 100.0
				else:
					factory_pct = float('nan')
				# print labeled rows with fixed-width cells for terminal alignment
				if not debug_table_header_printed:
					print("  ----------------------------------------------------------------------------------------------------------------------------------------------------------------")
					print("  time             | row       | each cell is 'label value' with fixed width")
					print("  ----------------------------------------------------------------------------------------------------------------------------------------------------------------")
					print("  TEMP_C labels: return_htf, collector_out, hex_inlet, oil_outlet, ctes_inlet, ctes_outlet, water_inlet, water_outlet")
					print("  FLOW_M3S labels: collector_av, collector_hex, collector_chg, ctes_to_hex, ctes_avail, recirculation, oil_hex")
					print("  CTES_LIM labels: need_mix, need_pow, pre_clip, avail, post_clip, trim_red, ovf_red, fit_red, solver, op_mode, limiter")
					print("  CTES_ITR labels: mode, iter")
					print("  COL_CTRL labels: collector_in, avail_eff, solver_iter, dTin, dMcol, converged")
					print("  POWER_KW labels: factory_load, solar_raw, solar_eff, solar_curt, hex_to_factory, ctes_to_factory, backup_heat, ctes_loss")
					debug_table_header_printed = True
				ts_table = ts.strftime('%Y-%m-%d %H:%M') if isinstance(ts, pd.Timestamp) else str(ts)
				# show how m_ctes_avail was derived for troubleshooting
				if disable_flow_rate_limits:
					ctes_avail_note = 'limits_off'
				else:
					ctes_avail_note = 'ctes_cap' if ("max_ctes_flow_m3s" in locals() and max_ctes_flow_m3s is not None) else 'target_hex'
				has_return_flow = bool((m_oil_hex > 1e-9) or (m_ctes_charge_use_m3s > 1e-9))
				return_htf_display_C = htf_in_temp_C if has_return_flow else np.nan
				ctes_inlet_display_C = last_ctes_T_in if ('last_ctes_mdot' in locals() and last_ctes_mdot > 0) else np.nan
				_print_row(ts_table, 'TEMP_C', [
					_cell_num('return_htf', return_htf_display_C, decimals=2),
					_cell_num('collector_out', collector_t_out_C, decimals=2),
					_cell_num('hex_inlet', T_mix, decimals=2),
					_cell_num('oil_outlet', oil_out_C, decimals=2),
					_cell_num('ctes_inlet', ctes_inlet_display_C, decimals=2),
					_cell_num('ctes_outlet', ctes_temp_now, decimals=2),
					_cell_num('water_inlet', factory_water_in_C, decimals=2),
					_cell_num('water_outlet', water_out_C, decimals=2),
				])
				_print_row(ts_table, 'FLOW_M3S', [
					_cell_num('collector_av', m_col_vol_flow_m3s, decimals=4),
					_cell_num('collector_hex', m_col_use, decimals=4),
					_cell_num('collector_chg', m_ctes_charge_use_m3s, decimals=4),
					_cell_num('ctes_to_hex', m_ctes_use, decimals=4),
					_cell_num('ctes_avail', m_ctes_discharge_avail, decimals=4),
					_cell_num('recirculation', m_recirc, decimals=4),
					_cell_num('oil_hex', m_oil_hex, decimals=4),
				])
				_print_row(ts_table, 'COL_CTRL', [
					_cell_num('collector_in', collector_t_in_C, decimals=2),
					_cell_num('avail_eff', m_col_avail, decimals=4),
					_cell_num('solver_iter', inner_solver_iters, decimals=0),
					_cell_num('dTin', inner_solver_dTin_C, decimals=3),
					_cell_num('dMcol', inner_solver_dm_col_m3s, decimals=6),
					_cell_txt('converged', int(bool(inner_solver_converged))),
					_cell_txt('---', ''),
				])
				# power row includes the most relevant sink/source terms in kW
				charging_kw_display = (actual_charge_display_W/1000.0) if 'actual_charge_display_W' in locals() else (actual_charge_W/1000.0)
				_print_row(ts_table, 'POWER_KW', [
					_cell_num('factory_load', factory_load_kW, decimals=2),
					_cell_num('solar_raw', solar_raw_kW, decimals=2),
					_cell_num('solar_eff', solar_power_effective_W/1000.0, decimals=2),
					_cell_num('solar_curt', solar_curtailed_W/1000.0, decimals=2),
					_cell_num('hex_to_fact', provided_kW, decimals=2),
					_cell_num('ctes_to_fac', ctes_kW, decimals=2),
					_cell_num('backup_heat', backup_kW, decimals=2),
					_cell_num('ctes_loss', current_loss_power_W/1000.0, decimals=3),
				])
				_print_row(ts_table, 'CURTAIL', [
					_cell_num('solar_abs', solar_abs_kW, decimals=2),
					_cell_num('coll_curt', collector_curtailment_W/1000.0, decimals=2),
					_cell_num('tot_curt', solar_curtailed_W/1000.0, decimals=2),
					_cell_num('mix_tgt', mix_target_used_C, decimals=1),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
				])
				flow_code = ctes_flow_dir
				_print_row(ts_table, 'STATUS', [
					_cell_num('dni_wm2', float(dni_value) if not pd.isna(dni_value) else 0.0, decimals=1),
					_cell_txt('ctes_flow', flow_code),
					_cell_txt('op_mode', op_mode),
					_cell_num('charge_kw', charging_kw_display, decimals=2),
					_cell_num('factory_gap', factory_deficit_W/1000.0, decimals=2),
					_cell_num('req_hex_in', required_oil_in_C, decimals=2),
					_cell_num('hex_target', hex_in_target_used_C, decimals=1),
					_cell_txt('mode', ('fallback' if fallback_mode_active else 'primary')),
					_cell_txt('hex_ok', int(bool(hex_temp_sufficient))),
					_cell_txt('ctes_limit', ctes_avail_note),
				])
				_print_row(ts_table, 'SHARE_PCT', [
					_cell_num('solar_pct', solar_pct, decimals=1),
					_cell_num('ctes_pct', ctes_pct, decimals=1),
					_cell_num('backup_pct', backup_pct, decimals=1),
					_cell_num('factory_pct', factory_pct, decimals=1),
					_cell_txt('spare', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
				])
				_print_row(ts_table, 'CTES_LIM', [
					_cell_num('need_mix', m_ctes_needed_mix, decimals=4),
					_cell_num('need_pow', m_ctes_needed_power, decimals=4),
					_cell_num('pre_clip', m_ctes_use_pre_clip, decimals=4),
					_cell_num('avail', m_ctes_discharge_avail, decimals=4),
					_cell_num('post_clip', m_ctes_use_post_clip, decimals=4),
					_cell_num('trim_red', m_ctes_trim_reduction, decimals=4),
					_cell_num('ovf_red', m_ctes_overflow_reduction, decimals=4),
					_cell_num('fit_red', m_ctes_discharge_fit_reduction, decimals=4),
					_cell_txt('solver', ctes_discharge_solver_mode),
					_cell_txt('op_mode', op_mode),
					_cell_txt('limiter', ctes_flow_limit_reason),
				])
				iter_count_mode = int(ctes_charge_iterations) if str(op_mode).startswith('A') else int(ctes_discharge_iterations)
				_print_row(ts_table, 'CTES_ITR', [
					_cell_txt('mode', op_mode),
					_cell_num('iter', iter_count_mode, decimals=0),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
				])
				# CTES diagnostics: energy, SOC, cumulative loss (presented negative), and current loss power (negative when loss)
				ctes_energy_MWh = ctes_energy_J / 3.6e9
				ctes_full_now = bool((ctes_soc_now >= 0.999) if not np.isnan(ctes_soc_now) else False)
				_print_row(ts_table, 'CTES', [
					_cell_num('energy_mwh', ctes_energy_MWh, decimals=3),
					_cell_num('soc_pct', ctes_soc_now*100.0, decimals=1),
					_cell_txt('is_full', int(ctes_full_now)),
					_cell_num('loss_cum', cum_ctes_loss_J/3.6e9, decimals=6),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
					_cell_txt('---', ''),
				])
				_print_row(ts_table, 'CTES_BAL', [
					_cell_num('dE_step', ctes_delta_MWh_step, decimals=4),
					_cell_num('Qin_step', ctes_charge_MWh_step, decimals=4),
					_cell_num('Qout_step', ctes_discharge_MWh_step, decimals=4),
					_cell_num('Qloss_step', ctes_loss_MWh_step, decimals=4),
					_cell_num('bal_state_kW', ctes_balance_residual_state_W/1000.0, decimals=3),
					_cell_num('bal_loss_kW', ctes_balance_residual_with_loss_W/1000.0, decimals=3),
					_cell_txt('---', ''),
				])
				q_in_gap_W = float(Q_in_inferred_W - Q_in_htf_W)
				q_in_gap_pct = (100.0 * q_in_gap_W / max(1.0, float(Q_in_to_ctes_W)))
				closure_with_loss_kW = (-ctes_balance_residual_with_loss_W) / 1000.0
				close_kW_display = closure_with_loss_kW
				storage_delta_run_MWh = (float(ctes_energy_J) - float(initial_ctes_energy_J)) / 3.6e9
				gap_flag = 'OK'
				if abs(q_in_gap_pct) > 50.0:
					gap_flag = 'QIN_GAP!'
				elif abs(q_in_gap_pct) > 20.0:
					gap_flag = 'QIN_GAP'
				bal_flag = 'OK'
				bal_metric_W = abs(ctes_balance_residual_with_loss_W)
				bal_scale_W = max(1e3, float(abs(Q_in_to_ctes_W) + abs(Q_out_from_ctes_W) + abs(current_loss_power_W)))
				if isinstance(ctes_model_diag, dict) and ('diag_closure_combined_W' in ctes_model_diag):
					bal_metric_W = abs(float(ctes_model_diag.get('diag_closure_combined_W', 0.0)))
					close_kW_display = (-float(ctes_model_diag.get('diag_closure_combined_W', 0.0))) / 1000.0
				bal_rel = bal_metric_W / bal_scale_W
				if (bal_metric_W > 5e4) or (bal_rel > 0.20):
					bal_flag = 'BAL!'
				elif (bal_metric_W > 1e4) or (bal_rel > 0.05):
					bal_flag = 'BAL'
				_print_row(ts_table, 'CTES_PWR', [
					_cell_num('Qin_htf_kW', Q_in_htf_W/1000.0, decimals=3),
					_cell_num('Qin_inf_kW', Q_in_inferred_W/1000.0, decimals=3),
					_cell_num('Qin_use_kW', Q_in_to_ctes_W/1000.0, decimals=3),
					_cell_num('Qout_kW', Q_out_from_ctes_W/1000.0, decimals=3),
					_cell_num('Qloss_kW', current_loss_power_W/1000.0, decimals=3),
					_cell_num('dE_kW', dE_dt_W/1000.0, decimals=3),
					_cell_num('Qin_gap_kW', q_in_gap_W/1000.0, decimals=3),
				])
				_print_row(ts_table, 'CTES_CHK', [
					_cell_num('Qin_gap_pct', q_in_gap_pct, decimals=1),
					_cell_num('bal_rel_pct', bal_rel*100.0, decimals=1),
					_cell_num('close_kW', close_kW_display, decimals=3),
					_cell_num('hex_flow_bal', hex_flow_balance_m3s, decimals=5),
					_cell_num('run_dE_MWh', storage_delta_run_MWh, decimals=3),
					_cell_txt('gap_flag', gap_flag),
					_cell_txt('bal_flag', bal_flag),
					_cell_txt('---', ''),
				])
				if isinstance(ctes_model_diag, dict):
					qf2s_mod_kW = float(ctes_model_diag.get('diag_q_fluid_to_solid_module_W', np.nan)) / 1000.0
					qf2s_tot_kW = float(ctes_model_diag.get('diag_q_fluid_to_solid_total_W', np.nan)) / 1000.0
					q_to_solid_corr_kW = float(ctes_model_diag.get('diag_q_to_solid_corr_W', np.nan)) / 1000.0
					qloss_model_kW = float(ctes_model_diag.get('diag_q_loss_avg_W', np.nan)) / 1000.0
					dE_model_kW = float(ctes_model_diag.get('diag_dE_W', np.nan)) / 1000.0
					dE_fluid_model_kW = float(ctes_model_diag.get('diag_dE_fluid_W', np.nan)) / 1000.0
					close_model_kW = float(ctes_model_diag.get('diag_closure_combined_W', np.nan)) / 1000.0
					qf_gap_kW = q_to_solid_corr_kW - (Q_in_htf_W / 1000.0)
					_print_row(ts_table, 'CTES_MOD', [
						_cell_num('qf2s_mod', qf2s_mod_kW, decimals=3),
						_cell_num('qf2s_tot', qf2s_tot_kW, decimals=3),
						_cell_num('q2solid', q_to_solid_corr_kW, decimals=3),
						_cell_num('dEfl_kW', dE_fluid_model_kW, decimals=3),
						_cell_num('dE_kW', dE_model_kW, decimals=3),
						_cell_num('qloss_kW', qloss_model_kW, decimals=3),
						_cell_num('close_kW', close_model_kW, decimals=3),
						_cell_num('qf_gap_kW', qf_gap_kW, decimals=3),
						_cell_txt('mode', ctes_flow_dir),
						_cell_txt('---', ''),
					])
				_print_row(ts_table, 'CUM_MWH', [
					_cell_num('factory', cum_factory_consumed_J/3.6e9, decimals=3),
					_cell_num('solar_eff', cum_solar_produced_J/3.6e9, decimals=3),
					_cell_num('solar_curt', cum_solar_curtailed_J/3.6e9, decimals=3),
					_cell_num('solar_sup', cum_solar_supplied_to_factory_J/3.6e9, decimals=3),
					_cell_num('ctes_sup', cum_ctes_supplied_to_factory_J/3.6e9, decimals=3),
					_cell_num('tol_err', cum_tolerated_deficit_J/3.6e9, decimals=6),
					_cell_txt('---', ''),
				])
				_print_row(ts_table, 'CUM_CTES', [
					_cell_num('Qin_MWh', cum_ctes_charge_input_J/3.6e9, decimals=3),
					_cell_num('Qout_MWh', cum_ctes_discharge_output_J/3.6e9, decimals=3),
					_cell_num('Qloss_MWh', (-cum_ctes_loss_J)/3.6e9, decimals=3),
					_cell_num('dE_run_MWh', storage_delta_run_MWh, decimals=3),
					_cell_num('R_dch_chg', (cum_ctes_discharge_output_J / max(1.0, cum_ctes_charge_input_J)), decimals=3),
					_cell_txt('flags', f"{gap_flag}/{bal_flag}"),
					_cell_txt('---', ''),
				])
				print("  ----------------------------------------------------------------------------------------------------------------------------------------------------------------")
		# prepare CTES temperature for CSV (NaN when no CTES flow)
		try:
			if ('last_ctes_mdot' in locals() and last_ctes_mdot > 0):
				if ('last_ctes_T_out' in locals() and not np.isnan(last_ctes_T_out)):
					ctes_temp_csv = float(last_ctes_T_out)
				else:
					ctes_temp_csv = float(T_outlet(ctes_y, 'charging')) if (ctes_y is not None and T_outlet is not None) else np.nan
			else:
				ctes_temp_csv = np.nan
		except Exception:
			ctes_temp_csv = np.nan

		# --- Per-module concrete temperatures (average and outlet) ---
		if ctes_y is not None and extract_profiles is not None:
			_, _T_s_all = extract_profiles(ctes_y)
			_mod_avg = {
				f"module_{i:02d}_Ts_avg_C": float(np.mean(_T_s_all[i * _CTES_N_z:(i + 1) * _CTES_N_z]))
				for i in range(ctes_series_modules)
			}
			_mod_outlets = {f"module_{i:02d}_Ts_outlet_C": float(_T_s_all[(i + 1) * _CTES_N_z - 1]) for i in range(ctes_series_modules)}
		else:
			_mod_avg = {f"module_{i:02d}_Ts_avg_C": np.nan for i in range(ctes_series_modules)}
			_mod_outlets = {f"module_{i:02d}_Ts_outlet_C": np.nan for i in range(ctes_series_modules)}

		records.append(
			{
				"timestamp": ts,
				"dni_W_m2": float(dni_value) if not pd.isna(dni_value) else 0.0,
				"solar_power_W": solar_power_W,
				"solar_power_raw_W": solar_power_available_W,
				"solar_power_effective_W": solar_power_effective_W,
				"solar_power_curtailed_W": solar_curtailed_W,
				"collector_curtailment_W": collector_curtailment_W,
				"downstream_curtailment_W": downstream_curtail_W if 'downstream_curtail_W' in locals() else np.nan,
				"solar_power_kW": solar_power_available_W / 1000.0,
				"solar_power_absorbed_kW": solar_power_W / 1000.0,
				"solar_power_available_W": solar_power_available_W,
				"collector_t_in_C": collector_t_in_C,
				"collector_solver_iters": inner_solver_iters,
				"collector_solver_dTin_C": inner_solver_dTin_C,
				"collector_solver_dMcol_m3s": inner_solver_dm_col_m3s,
				"collector_solver_converged": bool(inner_solver_converged),
				"flow_optimizer_mode": flow_optimizer_mode,
				"flow_optimizer_evals": int(flow_optimizer_evals),
				"flow_optimizer_best_objective_W": flow_optimizer_best_objective_W,
				"flow_optimizer_best_charge_flow_m3s": flow_optimizer_best_charge_flow_m3s,
				"flow_optimizer_factory_shortfall_W": flow_optimizer_factory_shortfall_W,
				"ctes_discharge_iterations": int(ctes_discharge_iterations),
				"ctes_discharge_solver_mode": ctes_discharge_solver_mode,
				"m_col_avail_eff_m3s": m_col_avail,
				"m_col_vol_flow_m3s": m_col_vol_flow_m3s,
				"m_col_use_m3s": m_col_use,
				"m_ctes_use_m3s": m_ctes_use,
				"m_ctes_charge_use_m3s": m_ctes_charge_use_m3s,
				"m_ctes_discharge_avail_m3s": m_ctes_discharge_avail,
				"m_recirc_m3s": m_recirc,
				"m_oil_hex_m3s": m_oil_hex,
				"htf_in_temp_C": htf_in_temp_C,
				"hex_inlet_temp_C": T_mix,
				"mix_target_C": mix_target_used_C,
				"mix_target_used_C": mix_target_used_C,
				"collector_t_out_C": collector_t_out_C,
				"oil_out_C": oil_out_C,
				"water_out_C": water_out_C,
				"provided_W": provided_W,
				"provided_kW": provided_W / 1000.0,
				"solar_contribution_W": solar_contribution_W,
				"solar_to_factory_kW": solar_contribution_W / 1000.0,
				"ctes_contribution_W": ctes_contribution_W,
				"ctes_to_factory_kW": ctes_contribution_W / 1000.0,
				"remaining_solar_W": remaining_solar_W,
				"required_oil_in_C": required_oil_in_C,
				"hex_temp_sufficient": bool(hex_temp_sufficient),
				"factory_load_W": factory_load_W,
				"factory_load_kW": factory_load_W / 1000.0,
				"operation_mode": op_mode,
				"net_power_W": net_power_W,
				"ctes_energy_J": ctes_energy_J,
				"ctes_energy_MWh": ctes_energy_J / 3.6e9,
				"ctes_soc_pct": (100.0 * (stored_energy_J(ctes_y) / max(1.0, ctes_capacity_J)) if (ctes_y is not None and stored_energy_J is not None) else np.nan),
				"ctes_inlet_temp_C": (float(last_ctes_T_in) if ('last_ctes_T_in' in locals() and not np.isnan(last_ctes_T_in)) else np.nan),
				"ctes_temp_C": ctes_temp_csv,
					"concrete_T_z0_C": (extract_profiles(ctes_y)[1][0] if (ctes_y is not None and extract_profiles is not None) else np.nan),
					"concrete_T_z_mid_C": (lambda y: (extract_profiles(y)[1][len(extract_profiles(y)[1])//2]) if (y is not None and extract_profiles is not None) else np.nan)(ctes_y),
					"concrete_T_z_max_C": (lambda y: (extract_profiles(y)[1][-1]) if (y is not None and extract_profiles is not None) else np.nan)(ctes_y),
				"provided_from_ctes_W": provided_from_ctes_W,
				"provided_from_ctes_kW": provided_from_ctes_W/1000.0,
				"backup_heater_W": backup_heater_W,
				"backup_heater_kW": backup_heater_W/1000.0,
				"tolerated_deficit_W": tolerated_deficit_W,
				"tolerated_deficit_cumulative_MWh": cum_tolerated_deficit_J/3.6e9,
				"ctes_loss_cumulative_MWh": cum_ctes_loss_J/3.6e9,
				"ctes_current_loss_kW": current_loss_power_W/1000.0,
				"ctes_charge_htf_W": Q_in_htf_W,
				"ctes_charge_inferred_W": Q_in_inferred_W,
				"ctes_charge_input_W": Q_in_to_ctes_W,
				"ctes_discharge_output_W": Q_out_from_ctes_W,
				"ctes_balance_residual_state_W": ctes_balance_residual_state_W,
				"ctes_balance_residual_with_loss_W": ctes_balance_residual_with_loss_W,
				"hex_flow_balance_m3s": hex_flow_balance_m3s,
				**_mod_avg,
				**_mod_outlets,
			}
		)

	results_df = pd.DataFrame.from_records(records).set_index("timestamp")
	if (not debug) and (not interactive) and (hourly_bucket_hour is not None):
		hourly_avg_iters = (hourly_ctes_discharge_iterations / hourly_ctes_discharge_solve_events) if hourly_ctes_discharge_solve_events > 0 else 0.0
		mode_counts_txt = ','.join([f"{k}:{v}" for k, v in sorted(hourly_mode_counts.items())]) if len(hourly_mode_counts) > 0 else '-'
		print(
			"CTES hourly iteration stats (final): "
			f"mode={mode_counts_txt}, "
			f"events={hourly_ctes_discharge_solve_events}, "
			f"iters={hourly_ctes_discharge_iterations}, "
			f"avg={hourly_avg_iters:.2f}, "
			f"max={hourly_max_ctes_discharge_iterations}"
		)
	ctes_energy_start_J = float(results_df['ctes_energy_J'].iloc[0]) if ('ctes_energy_J' in results_df.columns and len(results_df) > 0) else np.nan
	ctes_energy_end_J = float(results_df['ctes_energy_J'].iloc[-1]) if ('ctes_energy_J' in results_df.columns and len(results_df) > 0) else np.nan

	# Save results to CSV for later plotting by data_presentation
	try:
		ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
		# create a timestamped output folder for this run
		out_dir = os.path.join('src', 'data', ts)
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"simulation_results_{ts}.csv")
		results_df.to_csv(out_path)
		# attach path and folder to the DataFrame for callers
		results_df.attrs['csv_path'] = out_path
		results_df.attrs['out_dir'] = out_dir
		print(f"Simulation CSV written: {out_path}")

		# Write a one-line summary CSV with totals over the entire period (energies in MWh)
		summary: dict[str, object] = {
			'total_factory_consumed_MWh': cum_factory_consumed_J / 3.6e9,
			'total_solar_produced_MWh': cum_solar_produced_J / 3.6e9,
			'total_solar_curtailed_MWh': cum_solar_curtailed_J / 3.6e9,
			'total_solar_supplied_to_factory_MWh': cum_solar_supplied_to_factory_J / 3.6e9,
			'total_ctes_supplied_to_factory_MWh': cum_ctes_supplied_to_factory_J / 3.6e9,
			'total_backup_heater_MWh': cum_backup_heater_J / 3.6e9,
			'total_tolerated_deficit_MWh': cum_tolerated_deficit_J / 3.6e9,
			'total_ctes_loss_MWh': cum_ctes_loss_J / 3.6e9,
			'total_ctes_charge_input_MWh': cum_ctes_charge_input_J / 3.6e9,
			'total_ctes_discharge_output_MWh': cum_ctes_discharge_output_J / 3.6e9,
			'ctes_discharge_solve_events': int(ctes_discharge_solve_events),
			'ctes_discharge_iterations_total': int(total_ctes_discharge_iterations),
			'ctes_discharge_iterations_avg': (float(total_ctes_discharge_iterations) / float(ctes_discharge_solve_events)) if ctes_discharge_solve_events > 0 else 0.0,
			'ctes_discharge_iterations_max': int(max_ctes_discharge_iterations),
			'ctes_energy_start_MWh': (ctes_energy_start_J / 3.6e9) if np.isfinite(ctes_energy_start_J) else np.nan,
			'ctes_energy_end_MWh': (ctes_energy_end_J / 3.6e9) if np.isfinite(ctes_energy_end_J) else np.nan,
			'ctes_storage_delta_MWh': ((ctes_energy_end_J - ctes_energy_start_J) / 3.6e9) if (np.isfinite(ctes_energy_start_J) and np.isfinite(ctes_energy_end_J)) else np.nan,
		}
		summary['ctes_discharge_to_charge_ratio_window'] = (cum_ctes_discharge_output_J / cum_ctes_charge_input_J) if cum_ctes_charge_input_J > 0 else np.nan
		# Window efficiency can exceed 1.0 if storage starts with high inventory.
		# Include inventory draw from storage delta to avoid misleading "free energy" RTE.
		delta_storage_J = (ctes_energy_end_J - ctes_energy_start_J) if (np.isfinite(ctes_energy_start_J) and np.isfinite(ctes_energy_end_J)) else np.nan
		inventory_draw_J = max(0.0, -float(delta_storage_J)) if np.isfinite(delta_storage_J) else np.nan
		rte_input_soc_adjusted_J = (cum_ctes_charge_input_J + inventory_draw_J) if np.isfinite(inventory_draw_J) else np.nan
		rte_ratio = (cum_ctes_discharge_output_J / rte_input_soc_adjusted_J) if (np.isfinite(rte_input_soc_adjusted_J) and rte_input_soc_adjusted_J > 0) else np.nan
		summary['ctes_round_trip_efficiency'] = (max(0.0, float(rte_ratio)) if np.isfinite(rte_ratio) else np.nan)
		summary['ctes_inventory_draw_MWh'] = (inventory_draw_J / 3.6e9) if np.isfinite(inventory_draw_J) else np.nan
		# CTES first-law closure diagnostic over the simulated window:
		# state-only: dE = Qin - Qout
		# loss-inclusive: dE = Qin - Qout + Qloss_signed, where Qloss_signed is negative for losses.
		if np.isfinite(ctes_energy_start_J) and np.isfinite(ctes_energy_end_J):
			dE_window_J = float(ctes_energy_end_J) - float(ctes_energy_start_J)
			balance_rhs_state_J = float(cum_ctes_charge_input_J) - float(cum_ctes_discharge_output_J)
			balance_rhs_with_loss_J = balance_rhs_state_J + float(cum_ctes_loss_J)
			summary['ctes_energy_balance_residual_state_MWh'] = (dE_window_J - balance_rhs_state_J) / 3.6e9
			summary['ctes_energy_balance_residual_with_loss_MWh'] = (dE_window_J - balance_rhs_with_loss_J) / 3.6e9
		else:
			summary['ctes_energy_balance_residual_state_MWh'] = np.nan
			summary['ctes_energy_balance_residual_with_loss_MWh'] = np.nan
		summary['total_ctes_loss_abs_MWh'] = (-float(cum_ctes_loss_J)) / 3.6e9
		# Solar fraction includes direct solar + CTES-delivered energy (solar-derived).
		total_supplied = cum_solar_supplied_to_factory_J + cum_ctes_supplied_to_factory_J + cum_backup_heater_J
		summary['solar_fraction'] = ((cum_solar_supplied_to_factory_J + cum_ctes_supplied_to_factory_J) / total_supplied) if total_supplied > 0 else np.nan
		summary_df = pd.DataFrame([summary])
		summary_path = os.path.join(out_dir, f'simulation_summary_{ts}.csv')
		summary_df.to_csv(summary_path, index=False, sep=';', decimal=',')
		print(f"Simulation summary saved: {summary_path}")
		avg_iters = (total_ctes_discharge_iterations / ctes_discharge_solve_events) if ctes_discharge_solve_events > 0 else 0.0
		print(
			"CTES discharge iteration stats: "
			f"events={ctes_discharge_solve_events}, "
			f"total={total_ctes_discharge_iterations}, "
			f"avg={avg_iters:.2f}, "
			f"max={max_ctes_discharge_iterations}"
		)
	except Exception:
		# best-effort: ignore if cannot write
		out_path = None

	return results_df

# %% Main entry point for the simulation
if __name__ == "__main__":
	print("This module provides `simulate()` for running the high-level CTES simulation loop.")
	print("Call `simulate(dni_series_or_path, ...)` from your scripts or notebooks.")
	# Quick smoke-run when invoked directly: use the provided DNI file and defaults
	try:
		csv_in = 'src/data/DNI_10m.csv'
		# if running in an interactive terminal, enable interactive prompts and debug
		if sys.stdin is not None and sys.stdin.isatty():
			results = simulate(csv_in, debug=True, interactive=True)
		else:
			results = simulate(csv_in)
		print('Simulation finished.')
		csv_path = results.attrs.get('csv_path') if hasattr(results, 'attrs') else None
		if csv_path:
			print('Results saved to:', csv_path)
	except Exception as e:
		print('Simulation failed:', e)
		csv_path = None

	# Try to plot results using data_presentation if available
	try:
		from ..features.data_presentation import plot_simulation_results
	except Exception:
		try:
			from features.data_presentation import plot_simulation_results
		except Exception:
			plot_simulation_results = None

	if plot_simulation_results is not None:
		try:
			plot_simulation_results(csv_path=csv_path)
		except Exception as e:
			print('Plotting failed:', e)
	else:
		print('Plotting function not available; CSV at', csv_path)


