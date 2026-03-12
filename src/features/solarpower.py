# Load libraries
from CoolProp.CoolProp import PropsSI

# function for cp as a function of T
def cp(T):
    return PropsSI("C", "T", T, "Q", 0, "INCOMP::PNF")

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
    # Calculate absorbed power
    power = dni * efficiency * area  # W
    t_in_k = t_in + 273.15                          # Convert °C to K for CoolProp
    #Calculate output temperature based on absorbed power and mass flow rate as well as specific heat capacity from CoolProp
    if volumetric:
        # Convert volumetric flow to mass flow using density at inlet conditions
        rho = PropsSI("D", "T", t_in_k, "Q", 0, fluid)  # Density at inlet conditions (kg/m^3)
        m_dot_mass = m_dot * rho                        # Convert volumetric flow (m^3/s) to mass flow (kg/s)
    else:
        m_dot_mass = m_dot
    #Calculate outlet temperature taking changing specific heat into account using an integral approach
    # We can use a numerical approach to solve for t_out since cp is a function of temperature.
    # We can use a simple iterative method to find t_out such that power = m_dot
    # * integral of cp(T) dT from t_in to t_out.
    t_out_k = t_in_k  # Start with inlet temperature as initial guess
    for _ in range(100):  # Limit iterations to prevent infinite loop
        cp_avg = (cp(t_in_k) + cp(t_out_k)) / 2  # Average cp between inlet and current outlet guess
        power_guess = m_dot_mass * cp_avg * (t_out_k - t_in_k)  # Power based on current outlet guess
        if abs(power_guess - power) < 1e-3:  # Check if close enough
            break
        t_out_k += (power - power_guess) / (m_dot_mass * cp_avg)  # Adjust outlet temperature guess
        print(t_out_k-273.15)  # Debug: print current outlet temperature guess in °C
    
    if t_out_k < t_in_k:
        raise ValueError(f"Computed outlet temperature {t_out_k-273.15:.2f} °C is less than inlet temperature {t_in_k-273.15:.2f} °C, which is non-physical. Check inputs.")
    return {"power_W": power, "t_out_C": t_out_k-273.15}

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