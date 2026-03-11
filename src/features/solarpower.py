"""Solar collector outlet temperature calculation with temperature-dependent cp.

This module provides `solar_collector_outlet_temperature` which computes
the outlet temperature of an HTF stream after receiving solar power from
collectors. The function accepts either a scalar DNI (for one timestep),
a pandas Series of DNI values, or a path to a CSV file (first column
datetime, second column DNI in W/m2). It uses CoolProp to obtain the
temperature-dependent specific heat capacity of Paratherm-NF and solves
the energy balance

	P = m_dot * \int_{T_in}^{T_out} cp(T) dT

for T_out using a robust bisection + numerical integration method.

If CoolProp is unavailable or you prefer a constant cp, pass `cp_override`
in J/kg/K.
"""

from __future__ import annotations

from typing import Optional, Union

import math
import pandas as pd

from logging import getLogger

logger = getLogger(__name__)


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


def solar_collector_outlet_temperature(
	t_in: float,
	m_dot: float,
	dni_input: Union[str, float, int, pd.Series],
	efficiency: float,
	area: float,
	cp_override: Optional[float] = None,
	temp_unit: str = "C",
	pressure: float = 101325.0,
	volumetric: bool = False,
	density_override: Optional[float] = None,
	max_iter: int = 60,
	tol: float = 1e-6,
) -> Union[float, pd.Series]:
	"""Compute outlet temperature(s) after solar collectors.

	Parameters
	- `t_in`: inlet temperature (°C by default, or K if `temp_unit` == 'K').
	- `m_dot`: mass flow rate in kg/s (must be > 0).
	- `dni_input`: path to DNI CSV, a scalar DNI (W/m2), or a pandas Series.
	- `efficiency`: collector efficiency (0-1).
	- `area`: collector area in m^2.
	- `cp_override`: optional constant cp in J/kg/K. If None, CoolProp
	  will be used to evaluate cp(T) for Paratherm-NF.
	- `temp_unit`: 'C' or 'K' indicating units of `t_in`.
	- `pressure`: pressure (Pa) used for CoolProp queries.
	- `max_iter`, `tol`: numerical solver controls.

	Returns a scalar float (°C) if `dni_input` is scalar, or a pandas Series
	of outlet temperatures with matching index when a Series or CSV is used.
	"""

	if m_dot <= 0:
		raise ValueError("m_dot must be positive (kg/s or m^3/s when volumetric=True)")

	dni = _load_dni_input(dni_input)

	# Note: `m_dot` may be a mass flow (kg/s) by default. If the caller
	# wants to provide volumetric flow (m^3/s) instead, they should convert
	# to mass flow before calling. If you want this function to accept
	# volumetric flow directly, set `volumetric=True` and optionally pass
	# `density_override` in kg/m3. (We keep the core function expecting
	# mass flow to avoid ambiguity.)

	# Quick return for zero or negative power
	def compute_for_power(P_w: float) -> float:
		if P_w <= 0:
			# no heating
			return float(t_in if temp_unit.upper() == "C" else (t_in - 273.15))
		# determine mass flow (kg/s). If `volumetric` is True, `m_dot`
		# is treated as volumetric flow (m^3/s) and converted to mass flow
		# using density from CoolProp unless `density_override` is provided.
		if volumetric:
			if density_override is not None:
				rho = float(density_override)
			else:
				try:
					from CoolProp.CoolProp import PropsSI
				except Exception as exc:
					raise RuntimeError(
						"CoolProp unavailable; pass density_override when using volumetric flow"
					) from exc

				# try candidate names for Paratherm-NF
				names = [
					"INCOMP::PNF",
					"INCOMP::ParathermNF",
					"INCOMP::Paratherm_NF",
					"Paratherm-NF",
					"ParathermNF",
				]
				rho = None
				t_k_in = t_in + 273.15
				last_err = None
				for name in names:
					try:
						val = PropsSI("Dmass", "T", t_k_in, "P", pressure, name)
						if val is not None and val > 0:
							rho = float(val)
							break
					except Exception as e:
						last_err = e
						continue
				if rho is None:
					raise RuntimeError("Could not determine density for Paratherm-NF via CoolProp") from last_err

			mass_flow = float(m_dot) * float(rho)
		else:
			mass_flow = float(m_dot)

		# cp function: returns J/kg/K at temperature in degC
		if cp_override is not None:
			def cp_T(_t_c: float) -> float:
				return float(cp_override)
			# no CoolProp needed; set a high valid temperature
			max_valid_t_c = t_in + 1000.0
		else:
			try:
				from CoolProp.CoolProp import PropsSI
			except Exception as exc:
				raise RuntimeError(
					"CoolProp unavailable; pass cp_override to use a constant cp"
				) from exc

			# detect a working fluid string once at the inlet temperature
			names = [
				"INCOMP::PNF",
				"INCOMP::ParathermNF",
				"INCOMP::Paratherm_NF",
				"Paratherm-NF",
				"ParathermNF",
			]
			working_name = None
			t_k_in = t_in + 273.15
			last_err = None
			for name in names:
				try:
					test_val = PropsSI("Cpmass", "T", t_k_in, "P", pressure, name)
					if test_val is not None and test_val > 0:
						working_name = name
						break
				except Exception as e:
					last_err = e
					continue
			if working_name is None:
				raise RuntimeError("Could not determine a working Paratherm-NF fluid name via CoolProp") from last_err

			def cp_T(t_c: float) -> float:
				t_k = t_c + 273.15
				try:
					val = PropsSI("Cpmass", "T", t_k, "P", pressure, working_name)
					return float(val)
				except Exception as e:
					raise RuntimeError(f"CoolProp failed for {working_name} at T={t_c} C") from e

			# determine maximum valid temperature for this fluid (°C)
			max_valid_t_c = None
			step = 20.0
			t_try = t_in + step
			last_ok = t_in
			for _ in range(200):
				try:
					PropsSI("Cpmass", "T", t_try + 273.15, "P", pressure, working_name)
					last_ok = t_try
					t_try += step
				except Exception:
					break
			max_valid_t_c = last_ok

		# energy function: F(T_out) = m_dot * integral_{T_in}^{T_out} cp(T) dT - P_w
		t_in_c = float(t_in if temp_unit.upper() == "C" else (t_in - 273.15))

		def integral_cp(t1: float, t2: float, n: int = 40) -> float:
			# simple trapezoidal integration of cp(T) over T in °C
			if t2 == t1:
				return 0.0
			xs = [t1 + (t2 - t1) * i / n for i in range(n + 1)]
			cps = [cp_T(x) for x in xs]
			s = 0.0
			for i in range(n):
				s += 0.5 * (cps[i] + cps[i + 1]) * (xs[i + 1] - xs[i])
			return s

		def F(t_out_c: float) -> float:
			return mass_flow * integral_cp(t_in_c, t_out_c) - P_w

		# bracket for bisection: start with guess based on constant cp at t_in
		try:
			cp0 = cp_T(t_in_c)
		except Exception:
			cp0 = cp_override if cp_override is not None else 2500.0

		# initial naive delta T
		delta_guess = P_w / (mass_flow * max(cp0, 1e-6))
		t_low = t_in_c
		# clamp initial high to detected maximum valid temperature
		t_high = min(t_in_c + max(10.0, delta_guess * 1.5), max_valid_t_c)

		# expand upper bound until F(t_high) >= 0 (integral exceeds P)
		iter_expand = 0
		while F(t_high) < 0 and iter_expand < 50 and t_high < max_valid_t_c:
			new_high = t_high + max(10.0, (t_high - t_low) * 2.0, delta_guess)
			t_high = min(new_high, max_valid_t_c)
			iter_expand += 1

		if F(t_high) < 0:
			# try the absolute maximum valid temperature
			if F(max_valid_t_c) < 0:
				raise RuntimeError(
					"Unable to bracket root for T_out within CoolProp valid temperature range; power too large"
				)
			t_high = max_valid_t_c

		# bisection
		a, b = t_low, t_high
		fa, fb = F(a), F(b)
		if fa == 0.0:
			return a
		for _ in range(max_iter):
			c = 0.5 * (a + b)
			fc = F(c)
			if abs(fc) < tol:
				return c
			# narrow bracket
			if fa * fc <= 0:
				b, fb = c, fc
			else:
				a, fa = c, fc
		# last resort return midpoint
		return 0.5 * (a + b)

	# Compute power (W)
	if isinstance(dni, (float, int)):
		P = float(dni) * float(efficiency) * float(area)
		return compute_for_power(P)
	else:
		# dni is a pandas Series
		powers = dni.astype(float) * float(efficiency) * float(area)
		# compute for each timestamp; vectorize with apply
		results = powers.apply(lambda p: compute_for_power(float(p)))
		results.name = "t_out_C"
		return results


