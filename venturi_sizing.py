from string import printable
import numpy as np
from pyfluids import Fluid, FluidsList, Input
from thermo import Chemical
from os import system
system('cls')

p_up = 40e5
vp_up = 30e5

# Hopper engine: 0.15 kg/s - 0.25 kg/s

# mdot = 0.1
# # mdot = 0.25

# # dP: 0.47 - 4.03 bar
# d_inlet = 25.4e-3 * 0.19 # 1/4" swagelok id typ
# d__throat = 3.4e-3 # 3.4 mm nominal, +- 0.1 mm

# # Large engines: 1 lg/s - 2.5 kg/s

# mdot = 1
mdot = 2.5

# dP: 0.48 - 3.72 bar
d_inlet = 25.4e-3 * 0.41 # 1/2" swagelok id typ
d__throat = 9.2e-3 # 9.2 mm nominal, +- 0.1 mm


Cd = 0.95
A_up = np.pi * (d_inlet / 2) ** 2
A__throat = np.pi * (d__throat / 2) ** 2

nitrous_up = Fluid(FluidsList.NitrousOxide)
nitrous_throat = Fluid(FluidsList.NitrousOxide)
n2o = Chemical("10024-97-2")

nitrous_up.update(Input.pressure(vp_up), Input.quality(0))
nitrous_up = nitrous_up.isentropic_compression_to_pressure(p_up)
n2o.calculate((nitrous_up.temperature+273.15), nitrous_up.pressure)

rho_up = nitrous_up.density
mu_up = n2o.mu
v_up = mdot / (rho_up * A_up)
re_up = (rho_up * v_up * d_inlet) / mu_up

# mdot = Cd * A * sqrt(2 * rho * dP / (1 - (A__throat / A_up) ** 2))
# -> dP = mdot * ((1 - (A__throat / A_up) ** 2) / (Cd * A)) ** 2) * 2 * rho

dp = ((1 - (A__throat / A_up) ** 2) / (2 * rho_up)) * (mdot / (Cd * A__throat))**2
p_throat = p_up - dp

nitrous_throat.update(Input.pressure(p_throat), Input.enthalpy(nitrous_up.enthalpy))
n2o.calculate((nitrous_throat.temperature+273.15), nitrous_throat.pressure)
downstream_state = nitrous_throat.phase

rho_throat = nitrous_throat.density
mu_throat = n2o.mu
v_throat = mdot / (rho_throat * A__throat)
re_throat = (rho_throat * v_throat * d__throat) / mu_throat

print(f"Venturi  : dP = {dp / 1e5:6.4f} Bar, mdot = {mdot:5.3f} kg/s, d_up = {d_inlet * 1e3:4.3f} mm, d_throat = {d__throat * 1e3:4.3f} mm")
print(f"Upstream : Re = {re_up:8.3e}, T = {nitrous_up.temperature:6.2f} K, P = {nitrous_up.pressure/1e5:6.2f} Bar, Rho = {rho_up:8.4f} kg/m^3")
print(f"Throat   : Re = {re_throat:8.3e}, T = {nitrous_throat.temperature:6.2f} K, P = {nitrous_throat.pressure/1e5:6.2f} Bar, Rho = {rho_throat:8.4f} kg/m^3")
print(f"Throat   : {downstream_state}")