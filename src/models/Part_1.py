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
# try both relative and absolute import styles so this module is usable
try:
	from ..features.solarpower import solar_collector_outlet_temperature
except Exception:
	try:
		from features.solarpower import solar_collector_outlet_temperature
	except Exception:
		solar_collector_outlet_temperature = None

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

# To Oskar and Federico: This is what you want to replace with your fancy CTES model. Believe this one is completely useless.
def _ctes_stub_step(state: dict, power_W: float, dt_s: float):
	"""Very small placeholder CTES model.

	Inputs:
	- state: dict with keys `energy_J`, `capacity_J`, `min_temp_C`, `max_temp_C`.
	- power_W: positive => charging, negative => discharging (W)
	- dt_s: timestep seconds

	Returns updated state and `provided_power_W` (power the CTES could supply to meet a discharge request).
	This is intentionally simple: stores/withdraws energy and updates an effective temperature.
	"""
	dE = power_W * dt_s
	state["energy_J"] = max(0.0, min(state["capacity_J"], state.get("energy_J", 0.0) + dE))
	# update a naive temperature proxy from stored energy fraction
	frac = state["energy_J"] / state["capacity_J"] if state["capacity_J"] > 0 else 0.0
	state["temp_C"] = state["min_temp_C"] + frac * (state["max_temp_C"] - state["min_temp_C"])
	# available instantaneous discharge power is limited by a max discharge rate
	max_discharge_W = state.get("max_discharge_W", 5e6)
	available_power = min(max_discharge_W, state["energy_J"] / max(dt_s, 1.0))
	return state, available_power


#Oil/water heat exchanger model for supplying hot water to the pasta factory.
def oil_water_hex(
	oil_in_C: float,
	oil_flow_m3s: float,
	desired_water_power_W: float,
	water_vol_flow_m3s: Optional[float] = None,
	water_in_C: float = 120.0,
	water_target_C: Optional[float] = None,
	oil_fluid: str = "INCOMP::PNF",
	water_fluid: str = "Water",
	pinch_delta: float = 16.0,
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

#Useless recirculation function that nobody cares about
def _compute_recirc_fraction(collector_t_out_C: float, htf_in_C: float, max_htf_temp_C: float) -> float:
	"""Compute a conservative recirculation fraction to limit HTF maximum temperature.

	This is a heuristic placeholder: it returns a fraction between 0 and 1 of the
	collector output that should be recirculated back to the inlet to avoid exceeding
	`max_htf_temp_C`. Replace with your detailed mixing/flow control later.
	"""
	if np.isnan(collector_t_out_C) or collector_t_out_C <= max_htf_temp_C:
		return 0.0
	# aim to mix collector out with htf_in to hit max_htf_temp_C: f*collector + (1-f)*htf_in = max
	num = collector_t_out_C - max_htf_temp_C
	den = collector_t_out_C - htf_in_C + 1e-9
	frac = num / den
	return float(np.clip(frac if (frac := num/den) is not None else 0.0, 0.0, 1.0))

# %% Main simulation loop for the solar+CTES+pasta-factory system.
def simulate(
	dni_input: Union[str, float, int, pd.Series],
 	collector_efficiency: float = 0.47,
 	collector_area_m2: float = 6000.0,
 	initial_htf_inlet_temp_C: float = 125.0,
 	max_htf_temp_C: float = 315.0,
 	max_collector_flow_m3s: Optional[float] = None,
 	max_pump_flow_m3s: Optional[float] = None,
 	max_ctes_flow_m3s: Optional[float] = None,
 	timestep_seconds: int = 600,
 	initial_no_load_hours: float = 48.0,
	fluid: str = "INCOMP::PNF",
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
	# initialize simple CTES state
	ctes = {
		"energy_J": 0.0,
		"capacity_J": 5e9,  # placeholder capacity (J)
		"min_temp_C": 120.0,
		"max_temp_C": 300.0,
		"temp_C": initial_htf_inlet_temp_C,
		"max_discharge_W": 5e6,
	}

	records = []
	factory_nominal_load_W = 2.5e6
	# water-side parameters for the factory HEX (given in the project)
	factory_water_flow_m3s = 0.023  # superheated water mass flow (m3/s)
	factory_water_in_C = 120.0
	factory_water_target_C = 140.0
	# target HTF conditions into the O/W HEX to reach water target (from your note)
	target_hex_htf_temp_C = 182.0
	target_hex_htf_flow_m3s = 0.021 #cannot be exceeded
	pinch_delta_C = 16.0  # minimal approach temperature between oil and water

	# initial HTF inlet temperature (can be updated by recirculation / mixing logic)
	htf_in_temp_C = initial_htf_inlet_temp_C

	# Main loop iterating over DNI time series
	for ts, dni_value in dni.items():
		elapsed_h = (ts - start) / pd.Timedelta(hours=1)
		# factory off during first `initial_no_load_hours` OR on weekends
		factory_active = True
		if elapsed_h < initial_no_load_hours:
			factory_active = False
		if isinstance(ts, pd.Timestamp) and ts.weekday() >= 5:
			factory_active = False

		factory_load_W = factory_nominal_load_W if factory_active else 0.0

		# compute solar collector output (uses volumetric flow for Paratherm NF)
		# First compute available solar power (independent of flow)
		solar_power_available_W = (float(dni_value) if not pd.isna(dni_value) else 0.0) * collector_efficiency * collector_area_m2

		# compute a sensible collector volumetric flow to avoid exceeding max HTF temperature
		# use HTF properties at inlet temperature
		try:
			rho_htf = PropsSI("D", "T", htf_in_temp_C + 273.15, "Q", 0, fluid)
			cp_htf = PropsSI("C", "T", htf_in_temp_C + 273.15, "Q", 0, fluid)
		except Exception:
			rho_htf = 870.0
			cp_htf = 2200.0

		# avoid division by zero or negative approach
		deltaT_limit = max_htf_temp_C - htf_in_temp_C
		if deltaT_limit <= 0 or cp_htf <= 0 or rho_htf <= 0:
			m_col_vol_flow_m3s = 0.0
		else:
			m_dot_mass_needed = solar_power_available_W / (cp_htf * deltaT_limit)
			m_col_vol_flow_m3s = m_dot_mass_needed / rho_htf

		# enforce pump/collector upper bounds
		if max_collector_flow_m3s is not None:
			m_col_vol_flow_m3s = min(m_col_vol_flow_m3s, max_collector_flow_m3s)
		if max_pump_flow_m3s is not None:
			m_col_vol_flow_m3s = min(m_col_vol_flow_m3s, max_pump_flow_m3s)

		# ensure non-negative
		m_col_vol_flow_m3s = float(max(0.0, m_col_vol_flow_m3s))

		# compute collector outlet temperature and actual absorbed power with the given flow
		try:
			col_res = solar_collector_outlet_temperature(
				t_in=htf_in_temp_C,
				m_dot=m_col_vol_flow_m3s,
				dni=float(dni_value) if not pd.isna(dni_value) else 0.0,
				efficiency=collector_efficiency,
				area=collector_area_m2,
				volumetric=True,
				temp_unit="C",
			)
			solar_power_W = col_res["power_W"]
			collector_t_out_C = col_res.get("t_out_C", np.nan)
		except Exception:
			solar_power_W = 0.0
			collector_t_out_C = np.nan

		# --- Determine HTF supply to the O/W HEX (mixing of collector + CTES + optional recirc) ---
		# available flows (use computed collector volumetric flow as available)
		m_col_avail = m_col_vol_flow_m3s
		if max_collector_flow_m3s is not None:
			m_col_avail = min(m_col_avail, max_collector_flow_m3s)
		if max_pump_flow_m3s is not None:
			m_col_avail = min(m_col_avail, max_pump_flow_m3s)
		m_ctes_avail = 0.0 if max_ctes_flow_m3s is None else max_ctes_flow_m3s
		T_col = collector_t_out_C
		T_ctes = ctes["temp_C"]

		# target feed to HEX
		m_req = target_hex_htf_flow_m3s
		T_req = target_hex_htf_temp_C

		# start by using as much collector flow as reasonable up to m_req
		m_col_use = min(m_col_avail, m_req)

		# compute CTES flow needed to reach T_req when mixed with m_col_use
		m_ctes_needed = 0.0
		if m_col_use > 0 and (T_ctes - T_req) != 0:
			# solve m_col*T_col + m_ctes*T_ctes = m_req*T_req  => m_ctes = (m_req*T_req - m_col*T_col)/(T_ctes - T_req)
			m_ctes_needed = (m_req * T_req - m_col_use * T_col) / (T_ctes - T_req)
		elif m_col_use == 0 and (T_ctes - T_req) != 0:
			m_ctes_needed = m_req

		m_ctes_use = float(np.clip(m_ctes_needed, 0.0, m_ctes_avail))

		# if CTES unavailable or insufficient, try to supplement with collector more (if more available)
		if m_col_use + m_ctes_use < m_req:
			extra_from_col = min(m_col_avail - m_col_use, m_req - (m_col_use + m_ctes_use))
			m_col_use += extra_from_col

		# compute the mixed inlet temperature to HEX from currently allocated flows
		denom = m_col_use + m_ctes_use
		if denom > 0:
			T_mix = (m_col_use * T_col + m_ctes_use * T_ctes) / denom
		else:
			T_mix = htf_in_temp_C

		# compute a conservative recirculation to lower inlet if T_mix > T_req
		m_recirc = 0.0
		if T_mix > T_req:
			T_recirc = htf_in_temp_C
			if T_recirc < T_mix and (T_req - T_recirc) > 0:
				m_recirc = denom * (T_mix - T_req) / (T_req - T_recirc)
				m_recirc = float(np.clip(m_recirc, 0.0, max(0.0, m_col_avail)))
				denom += m_recirc
				T_mix = ((denom - m_recirc) * T_mix + m_recirc * T_recirc) / denom

		# final oil flow supplied to HEX
		m_oil_hex = denom

		# --- Automatic reallocation: try to meet required oil inlet temperature by shifting flows ---
		# compute required oil inlet to fully meet water demand for current allocation
		reallocated = False
		if factory_active and m_oil_hex > 0:
			try:
				rho_w = PropsSI("D", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
				cp_w = PropsSI("C", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
			except Exception:
				rho_w = 1000.0; cp_w = 4180.0
			m_w = factory_water_flow_m3s * rho_w
			desired_W_full = m_w * cp_w * (factory_water_target_C - factory_water_in_C)

			# function to check if current T_mix can satisfy requirement
			def _check_hex_sufficiency(Tmix, mol_hex):
				try:
					rho_o = PropsSI("D", "T", Tmix + 273.15, "Q", 0, fluid)
					cp_o = PropsSI("C", "T", Tmix + 273.15, "Q", 0, fluid)
				except Exception:
					rho_o = 870.0; cp_o = 2200.0
				m_dot_o = mol_hex * rho_o
				oil_out_min = factory_water_target_C + pinch_delta_C
				if m_dot_o <= 0 or cp_o <= 0:
					return False, np.nan
				req_in = oil_out_min + desired_W_full / (m_dot_o * cp_o)
				return Tmix >= req_in, req_in

			# 1) try increasing CTES flow to its available limit (helps if CTES is hotter than collector)
			if m_ctes_use < m_ctes_avail:
				m_ctes_try = m_ctes_avail
				den = m_col_use + m_ctes_try
				Tmix_try = (m_col_use * T_col + m_ctes_try * T_ctes) / den if den > 0 else htf_in_temp_C
				ok, req = _check_hex_sufficiency(Tmix_try, den)
				if ok:
					m_ctes_use = m_ctes_try
					denom = den; T_mix = Tmix_try; m_oil_hex = denom; required_oil_in_C = req; reallocated = True

			# 2) if still not sufficient, try increasing collector use up to available
			if not reallocated and m_col_use < m_col_avail:
				m_col_try = m_col_avail
				den = m_col_try + m_ctes_use
				Tmix_try = (m_col_try * T_col + m_ctes_use * T_ctes) / den if den > 0 else htf_in_temp_C
				ok, req = _check_hex_sufficiency(Tmix_try, den)
				if ok:
					m_col_use = m_col_try
					denom = den; T_mix = Tmix_try; m_oil_hex = denom; required_oil_in_C = req; reallocated = True

			# 3) if still not, reduce recirculation to raise T_mix (set m_recirc=0)
			if not reallocated and m_recirc > 0:
				den = (m_col_use + m_ctes_use)
				Tmix_try = (m_col_use * T_col + m_ctes_use * T_ctes) / den if den > 0 else htf_in_temp_C
				ok, req = _check_hex_sufficiency(Tmix_try, den)
				if ok:
					m_recirc = 0.0
					denom = den; T_mix = Tmix_try; m_oil_hex = denom; required_oil_in_C = req; reallocated = True

			# compute required_oil_in_C for final allocation if not set
			if not reallocated:
				ok, req = _check_hex_sufficiency(T_mix, m_oil_hex)
				required_oil_in_C = req

		# compute required oil inlet temperature to meet the factory water target given m_oil_hex
		required_oil_in_C = np.nan
		hex_temp_sufficient = False
		if factory_active and m_oil_hex > 0:
			# water-side required power to reach factory target
			try:
				rho_w = PropsSI("D", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
				cp_w = PropsSI("C", "T", factory_water_in_C + 273.15, "Q", 0, "Water")
			except Exception:
				rho_w = 1000.0; cp_w = 4180.0
			m_w = factory_water_flow_m3s * rho_w
			desired_W_full = m_w * cp_w * (factory_water_target_C - factory_water_in_C)

			# oil properties at inlet temperature guess (use T_mix for cp/rho)
			try:
				rho_oil = PropsSI("D", "T", T_mix + 273.15, "Q", 0, "INCOMP::PNF")
				cp_oil = PropsSI("C", "T", T_mix + 273.15, "Q", 0, "INCOMP::PNF")
			except Exception:
				rho_oil = 870.0; cp_oil = 2200.0
			m_dot_oil = m_oil_hex * rho_oil
			# assume minimum oil outlet temperature equals water target plus pinch
			oil_out_min_C = factory_water_target_C + pinch_delta_C
			if m_dot_oil > 0 and cp_oil > 0:
				required_oil_in_C = oil_out_min_C + desired_W_full / (m_dot_oil * cp_oil)
				# check if current mixed HTF inlet temperature is sufficient to reach required inlet
				hex_temp_sufficient = T_mix >= required_oil_in_C

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
			desired_water_power_W = m_w * cp_w * (factory_water_target_C - factory_water_in_C)

		# call the detailed oil-water HEX model to see how much power can be delivered
		oil_out_C, provided_W, water_out_C = oil_water_hex(
			oil_in_C=T_mix,
			oil_flow_m3s=m_oil_hex,
			desired_water_power_W=desired_water_power_W if factory_active else 0.0,
			water_vol_flow_m3s=(factory_water_flow_m3s if factory_active else None),
			water_in_C=factory_water_in_C,
			water_target_C=(factory_water_target_C if factory_active else None),
				pinch_delta=pinch_delta_C,
		)

		# If HEX couldn't meet the full demand and a backup heater would be needed,
		# allow a relaxed pinch (10 C) and re-evaluate to see if that reduces backup heater use.
		if factory_active:
			remaining_deficit = max(0.0, desired_water_power_W - provided_W)
			if remaining_deficit > 0 and pinch_delta_C > 10:
				# re-run HEX with relaxed pinch
				oil_out_C_relaxed, provided_W_relaxed, water_out_C_relaxed = oil_water_hex(
					oil_in_C=T_mix,
					oil_flow_m3s=m_oil_hex,
					desired_water_power_W=desired_water_power_W,
					water_vol_flow_m3s=factory_water_flow_m3s,
					water_in_C=factory_water_in_C,
					water_target_C=factory_water_target_C,
					pinch_delta=10.0,
				)
				# adopt relaxed result if it supplies more
				if provided_W_relaxed > provided_W:
					provided_W = provided_W_relaxed
					oil_out_C = oil_out_C_relaxed
					water_out_C = water_out_C_relaxed

		# account for provided_W as energy used from HTF (charging/discharging logic follows)
		# solar contribution to provided_W is proportional to fraction of oil coming from collector
		solar_fraction = (m_col_use / max(m_oil_hex, 1e-12)) if m_oil_hex > 0 else 0.0
		solar_contribution_W = provided_W * solar_fraction
		ctes_contribution_W = provided_W * (1.0 - solar_fraction)

		# update CTES energy when CTES provided some of the heat (discharge)
		if ctes_contribution_W > 0:
			# CTES supplies heat -> withdraw energy
			ctes, _ = _ctes_stub_step(ctes, -ctes_contribution_W, timestep_seconds)

		# remaining solar energy after HEX provisioning is available for charging CTES or exported
		remaining_solar_W = max(0.0, solar_power_W - solar_contribution_W)

		net_power_W = remaining_solar_W - (factory_load_W - provided_W)

		# Decide HTF mass-flow control in a simple way: if surplus -> charge, if deficit -> discharge
		backup_heater_W = 0.0
		provided_from_ctes_W = 0.0

		if net_power_W >= 0:
			# charge CTES with as much surplus as available after recirculation
			charge_power_W = net_power_W
			ctes, _ = _ctes_stub_step(ctes, charge_power_W, timestep_seconds)
		else:
			# need to discharge to meet factory load
			need_W = -net_power_W
			# try to get available power from CTES
			ctes, available_W = _ctes_stub_step(ctes, -min(need_W, ctes.get("max_discharge_W", 0.0)), timestep_seconds)
			provided_from_ctes_W = min(available_W, need_W)
			remaining_deficit = need_W - provided_from_ctes_W
			if remaining_deficit > 0:
				backup_heater_W = remaining_deficit


		records.append(
			{
				"timestamp": ts,
				"dni_W_m2": float(dni_value) if not pd.isna(dni_value) else 0.0,
				"solar_power_W": solar_power_W,
				"solar_power_available_W": solar_power_available_W,
				"m_col_vol_flow_m3s": m_col_vol_flow_m3s,
				"m_col_use_m3s": m_col_use,
				"m_ctes_use_m3s": m_ctes_use,
				"m_oil_hex_m3s": m_oil_hex,
				"htf_in_temp_C": htf_in_temp_C,
				"collector_t_out_C": collector_t_out_C,
				"oil_out_C": oil_out_C,
				"water_out_C": water_out_C,
				"provided_W": provided_W,
				"solar_contribution_W": solar_contribution_W,
				"ctes_contribution_W": ctes_contribution_W,
				"remaining_solar_W": remaining_solar_W,
				"required_oil_in_C": required_oil_in_C,
				"hex_temp_sufficient": bool(hex_temp_sufficient),
				"factory_load_W": factory_load_W,
				"net_power_W": net_power_W,
				"ctes_energy_J": ctes["energy_J"],
				"ctes_temp_C": ctes["temp_C"],
				"provided_from_ctes_W": provided_from_ctes_W,
				"backup_heater_W": backup_heater_W,
			}
		)

	results_df = pd.DataFrame.from_records(records).set_index("timestamp")
	return results_df

# %% Main entry point for the simulation
if __name__ == "__main__":
	print("This module provides `simulate()` for running the high-level CTES simulation loop.")
	print("Call `simulate(dni_series_or_path, ...)` from your scripts or notebooks.")


