import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.features.solarpower import solar_collector_outlet_temperature


def test_solar_outlet_with_volumetric_flow():
    try:
        from CoolProp.CoolProp import PropsSI  # noqa: F401
    except Exception:
        pytest.skip("CoolProp not installed; skipping Paratherm-NF integration test")

    # Given
    t_in = 220.0  # °C
    m_dot_vol = 0.019  # m^3/s
    dni = 1100.0  # W/m2
    efficiency = 0.47
    area = 6000.0

    # When
    t_out = solar_collector_outlet_temperature(
        t_in=t_in,
        m_dot=m_dot_vol,
        dni_input=dni,
        efficiency=efficiency,
        area=area,
        volumetric=True,
        temp_unit="C",
    )
    print(f"Outlet temperature with volumetric flow: {t_out:.2f} °C")
    # Then


test_solar_outlet_with_volumetric_flow()