import numpy as np
from pyfluids import Fluid, FluidsList, Input
from flow_models import *
from os import system
system('cls')

loading_tube_length = 708e-3
vent_tube_length = 718e-3

tank_id = 177.9e-3
tank_area = 0.25 * np.pi * tank_id**2

ox_tank_vol = 19.25e-3 # m^3

ox_vol = loading_tube_length * tank_area
gas_vol = ox_tank_vol - ox_vol

tank_p = 50e5
ox_vp = 28e5

ox_fluid = FluidsList.NitrousOxide
# ox_fluid = FluidsList.CarbonDioxide

n2 = Fluid(FluidsList.Nitrogen).with_state(Input.temperature(300), Input.pressure(tank_p))
ox = Fluid(ox_fluid).with_state(Input.pressure(ox_vp), Input.quality(0))
ox_vp_rho = ox.density
ox_vp_temp = ox.temperature
ox = ox.isentropic_compression_to_pressure(tank_p)

dP = 1
n2_expanded = n2.isentropic_expansion_to_pressure(n2.pressure - dP)

n2_mass = n2.density * gas_vol
n2_expanded_vol = n2_mass / n2_expanded.density

dV = n2_expanded_vol - gas_vol

dP_dV_gas = dP / dV

ox_rho = ox.density
ox_temp = ox.temperature

ox_mass = ox_rho * ox_vol
n2_mass = n2.density * gas_vol

mdot = 2.33
Vdot = mdot / ox_rho

dP_dt_liq_vent = dP_dV_gas * Vdot

vent_id = 4.5e-3
vent_area = 0.25 * np.pi * vent_id**2
vent_Cd = 0.7
vent_CdA = vent_Cd * vent_area

vent_mdot = spc_mdot(vent_CdA, n2, n2.temperature, tank_p, 101325)[0]
vent_Vdot = vent_mdot / n2.density
dP_dt_gas_vent = dP_dV_gas * vent_Vdot

print(f"dP/dt (Liquid Venting): {dP_dt_liq_vent/1e5:.2f} Bar/s")
print(f"dP/dt (Gas Venting): {dP_dt_gas_vent/1e5:.2f} Bar/s\n")

print(f"Ox Tank Volume: {ox_tank_vol*1e3:.2f} L,"
      f" Ox Volume: {ox_vol*1e3:.2f} L,"
      f" Gas Volume: {gas_vol*1e3:.2f} L")

print(f"Ox Density: {ox_rho:.2f} kg/m³, Ox Temp: {ox_temp:.2f} °C")
print(f"Ox VP Density: {ox_vp_rho:.2f} kg/m³, Ox VP Temp: {ox_vp_temp:.2f} °C")

print(f"Ox Mass: {ox_mass:.3f} kg")
print(f"N2 Mass: {n2_mass:.3f} kg")