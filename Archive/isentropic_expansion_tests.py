from pyfluids import Fluid, FluidsList, Input
import numpy as np


press_tank_vol = 6.8e-3
prop_tank_vol = 6.4e-3

total_vol = press_tank_vol + prop_tank_vol

n2_temp = 15 # degC
n2_pressure = 12e5

n2 = Fluid(FluidsList.Air).with_state(Input.temperature(n2_temp), Input.pressure(n2_pressure))
n2_rho_init = n2.density
n2_mass = n2_rho_init * press_tank_vol


n2_final_density = n2_mass / total_vol
# n2.update(Input.density(n2_final_density), Input.entropy(n2.entropy))
n2.update(Input.density(n2_final_density), Input.enthalpy(n2.enthalpy))
final_p = n2.pressure

print(f"N2 Initial Density: {n2_rho_init:.2f} kg/m³, Initial Mass: {n2_mass:.2f} kg")
print(f"N2 Final Pressure: {final_p/1e5:.2f} Bar, Final Temperature: {n2.temperature:.2f} °C")
