import numpy as np
from pyfluids import Fluid, FluidsList, Input
from scipy.optimize import root_scalar

pressurant_fill_tank_initial_temp = 25 # deg C
pressurant_fill_tank_initial_p = 300e5 # Pa
pressurant_tank_initial_p = 250e5

tank_p = 40e5

pressurant = Fluid(FluidsList.Nitrogen).with_state(Input.temperature(pressurant_fill_tank_initial_temp), Input.pressure(pressurant_fill_tank_initial_p))
pressurant = pressurant.isentropic_expansion_to_pressure(pressurant_tank_initial_p)
pressurant = pressurant.isenthalpic_expansion_to_pressure(tank_p)
print(f"Pressurant at tank: T: {pressurant.temperature:.2f} deg C, P: {pressurant.pressure/1e5:.2f} Bar)")

pressurant_mdot = 0.1 # kg/s

def colebrook(f: float, Re: float, rel_roughness: float) -> float:
    """
    Colebrook-White equation for turbulent friction factor
    Returns the residual that should equal zero when f is correct
    """
    return (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))

def Re_calc(fluid: Fluid, mdot: float, diameter: float):
    area = np.pi * (diameter/2)**2
    u = mdot / (fluid.density * area)
    rho = fluid.density
    return rho * u * diameter / fluid.dynamic_viscosity

def friction_factor(Re: float, rel_roughness: float) -> float:
    """Calculate Darcy friction factor using Colebrook equation approximation"""
    is_laminar = Re < 4000

    if is_laminar:
        return 64 / Re
    else:  # Turbulent flow - colebrook equation
        f = root_scalar(colebrook, args=(Re, rel_roughness), bracket=[0.00001, 1]).root # Colebrook solver
        # f = 1.325 / (np.log((self.rel_roughness / 3.7) + (5.74 / Re ** 0.9))) ** 2  # Swamee-Jain
        # f = (-1.8 * np.log10((self.rel_roughness/3.7)**1.11 + (6.9 / Re))) ** -2 # Haaland
        return f

def dp(fluid: Fluid, mdot: float, id: float, L: float, abs_roughness: float) -> float:
    rel_roughness = abs_roughness / id
    Re = Re_calc(fluid, mdot, id)
    f = friction_factor(Re, rel_roughness)
    rho = fluid.density
    u = mdot / (rho * (np.pi * (id/2)**2))
    a = np.sqrt(1.4 * 287 * (fluid.temperature + 273.15))
    print(f"Re: {Re:.2e}, f: {f:.4f}, u: {u:.2f} m/s, a: {a:.2f} m/s, M: {u/a:.2f}")
    return f * (L / id) * (rho * u ** 2) * 0.5

abs_roughness = 0.005e-3 # m

pipe_tw = 25.4e-3 * 0.036

pipe_1_4_id = 0.25 * 25.4e-3 - 2 * pipe_tw
pipe_3_8_id = 0.375 * 25.4e-3 - 2 * pipe_tw

pipe_1_4_dp = dp(pressurant, pressurant_mdot, pipe_1_4_id, 1.0, abs_roughness)
pipe_3_8_dp = dp(pressurant, pressurant_mdot, pipe_3_8_id, 1.0, abs_roughness)

print(f"1/4\" Pipe Dp: {pipe_1_4_dp/1e5:.2f} Bar")
print(f"3/8\" Pipe Dp: {pipe_3_8_dp/1e5:.2f} Bar")