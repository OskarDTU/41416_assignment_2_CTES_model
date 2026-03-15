# Load libraries
from CoolProp.CoolProp import PropsSI

# function for cp as a function of T
def cp(T, fluid):
    return PropsSI("C", "T", T, "Q", 0, fluid)

def solar_collector_outlet_temperature(
    t_in, # inlet temperature (°C by default, or K if `temp_unit` == 'K')
    m_dot, # mass flow rate in kg/s (must be > 0)
    dni, # direct normal irradiance (W/m2) as a scalar
    efficiency, # collector efficiency (0-1)
    area, # collector area in m^2
    volumetric=False, # if True, `m_dot` is treated as volumetric flow (m^3/s) and converted to mass flow using `density_override` or CoolProp
    fluid="INCOMP::PNF", # CoolProp fluid name for Paratherm-NF
    density_override=None, # optional fluid density (kg/m^3) for volumetric flow
    temp_unit="C", # 'C' or 'K' indicating units of `t_in`
    ):
    # Calculate potential absorbed power from solar field.
    power = max(0.0, dni * efficiency * area)  # W
    if temp_unit == "C":
        t_in_k = t_in + 273.15                          # Convert °C to K for CoolProp
    else:
        t_in_k = t_in                                  # Already in K
    #Calculate output temperature based on absorbed power and mass flow rate as well as specific heat capacity from CoolProp
    if volumetric:
        # Convert volumetric flow to mass flow using density at inlet conditions
        rho = PropsSI("D", "T", t_in_k, "Q", 0, fluid)  # Density at inlet conditions (kg/m^3)
        m_dot_mass = m_dot * rho                        # Convert volumetric flow (m^3/s) to mass flow (kg/s)
    else:
        m_dot_mass = m_dot

    # No heating possible if no flow or no irradiance.
    if m_dot_mass <= 0 or power <= 0:
        return {"power_W": 0.0, "t_out_C": t_in_k - 273.15}

    # Bound temperature solve to CoolProp fluid validity range.
    try:
        t_min = PropsSI("Tmin", "", 0, "", 0, fluid)
        t_max = PropsSI("Tmax", "", 0, "", 0, fluid)
    except Exception:
        t_min = 250.0
        t_max = 650.0

    t_in_k = float(min(max(t_in_k, t_min + 1e-6), t_max - 1e-6))

    def _cp_safe(Tk):
        Tk_b = min(max(float(Tk), t_min + 1e-6), t_max - 1e-6)
        return cp(Tk_b, fluid)

    def _power_to_fluid(T_out_k_local):
        cp_avg = (_cp_safe(t_in_k) + _cp_safe(T_out_k_local)) / 2.0
        return m_dot_mass * cp_avg * max(T_out_k_local - t_in_k, 0.0)

    # Maximum thermal power the fluid can carry before reaching fluid Tmax.
    t_high = t_max - 1e-6
    power_cap = _power_to_fluid(t_high)

    # Curtail power if irradiance would require temperatures beyond fluid validity range.
    power_target = min(power, power_cap)

    # Solve for outlet temperature using bisection on [t_in, t_high].
    t_low = t_in_k
    t_out_k = t_low
    if power_target > 0:
        for _ in range(80):
            t_mid = 0.5 * (t_low + t_high)
            p_mid = _power_to_fluid(t_mid)
            if abs(p_mid - power_target) <= max(1e-3, 1e-6 * power_target):
                t_out_k = t_mid
                break
            if p_mid < power_target:
                t_low = t_mid
            else:
                t_high = t_mid
            t_out_k = 0.5 * (t_low + t_high)

    actual_power = _power_to_fluid(t_out_k)
    return {"power_W": float(actual_power), "t_out_C": float(t_out_k - 273.15)}
"""
# Example usage
if __name__ == "__main__":
    results = solar_collector_outlet_temperature(
        t_in=220.0,
        m_dot=0.019,
        dni=1000,
        efficiency=0.47,
        area=6000.0,
        volumetric=True,
        temp_unit="C",
    )
    print(results)
"""