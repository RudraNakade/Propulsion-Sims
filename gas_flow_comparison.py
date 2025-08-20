import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input
from flow_models import *
from unit_converter import *
from scipy.constants import gas_constant
from os import system

system('cls')


t_0 = degC_to_K(24)
p_0 = 55e5

fluid = Fluid(FluidsList.NitrousOxide)
fluid.update(Input.temperature(K_to_degC(t_0)), Input.pressure(p_0))

sp_gas_const = gas_constant / fluid.molar_mass
gamma = fluid.specific_heat / (fluid.specific_heat - sp_gas_const)
k = (2/(gamma+1))**((gamma+1)/(gamma-1))

Cd = 0.65
annulus_id = 0.5e-3
annulus_od = 0.9e-3
CdA = Cd * np.pi * 0.25 * (annulus_od**2 - annulus_id**2)

p_down_arr = np.linspace(1e5, p_0, 1000)
mdot_spc = np.zeros(len(p_down_arr))
mdot_mixed = np.zeros(len(p_down_arr))

for i, p_down in enumerate(p_down_arr):
    choked = CdA * np.sqrt(gamma * fluid.density * p_0 * k)
    spi = CdA * np.sqrt(2 * fluid.density * (p_0 - p_down))

    mdot_mixed[i] = spi if spi < choked else choked

    mdot_spc[i] = spc_mdot(CdA, fluid, t_0, p_0, p_down)[0]


print(f"Choked mass flow rates:")
print(f"Mixed Model: {mdot_mixed[0]*1e3:.4f} g/s")
print(f"SPC Model: {mdot_spc[0]*1e3:.4f} g/s")

plt.plot(p_down_arr/1e5, mdot_mixed*1e3, color='b', label='SPI / Choked Mixed Model')
plt.plot(p_down_arr/1e5, mdot_spc*1e3, color='g', label='SPC Model')
plt.legend()
plt.xlabel('Downstream Pressure (bar)')
plt.ylabel('Mass Flow Rate (g/s)')
plt.title('Mass Flow Rate vs Pressure')
plt.grid()
plt.show()
