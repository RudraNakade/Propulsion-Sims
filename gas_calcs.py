from pyfluids import Fluid, FluidsList, Input
import numpy as np
from flow_models import spc_modt as spc
import matplotlib.pyplot as plt
import unit_converter as uc
from os import system

system('cls')

fluid = Fluid(FluidsList.Nitrogen)
# fluid = Fluid(FluidsList.Air)
# fluid = Fluid(FluidsList.Helium)
# fluid = Fluid(FluidsList.NitrousOxide)
# fluid = Fluid(FluidsList.Oxygen)

CdA = 20e-6

t_0 = 20

p_up_max = 300e5
p_down = 1e5

p_up_arr = np.linspace(p_down, p_up_max, 500)
mdot = np.zeros_like(p_up_arr)

t_down = np.zeros_like(p_up_arr)
rho_down = np.zeros_like(p_up_arr)

for i, p_up in enumerate(p_up_arr):
    fluid.update(Input.temperature(t_0), Input.pressure(p_up))
    
    gas_const = 8.31447 / fluid.molar_mass
    gamma = (fluid.specific_heat)/(fluid.specific_heat-gas_const)

    mdot[i] = spc(CdA, uc.degC_to_K(fluid.temperature), p_up, p_down, gas_const, gamma)

    fluid.update(Input.pressure(p_down), Input.enthalpy(fluid.enthalpy))
    t_down[i] = fluid.temperature
    rho_down[i] = fluid.density

print(f"Choked mass flow rate: {max(mdot):.3f} kg/s")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# Mass flow rate plot
ax1.plot(p_up_arr/1e5, mdot)
ax1.set_xlabel("Upstream Pressure (bar)")
ax1.set_ylabel("Mass Flow Rate (kg/s)")
ax1.set_xlim(p_down/1e5-0.2, max(p_up_arr)/1e5)
ax1.set_ylim(0, None)
ax1.set_title("Mass Flow Rate vs Upstream Pressure")
ax1.grid()

# Temperature downstream plot
ax2.plot(p_up_arr/1e5, t_down, label='Downstream Temperature')
ax2.axhline(t_0, color='r', linestyle='--', label='Upstream Temperature')
ax2.set_xlabel("Upstream Pressure (bar)")
ax2.set_ylabel("Downstream Temperature (°C)")
ax2.set_xlim(p_down/1e5-0.2, max(p_up_arr)/1e5)
ax2.set_title("Downstream Temperature vs Upstream Pressure")
ax2.grid()

# Density downstream plot
ax3.plot(p_up_arr/1e5, rho_down)
ax3.set_xlabel("Upstream Pressure (bar)")
ax3.set_ylabel("Downstream Density (kg/m³)")
ax3.set_xlim(p_down/1e5-0.2, max(p_up_arr)/1e5)
ax3.set_title("Downstream Density vs Upstream Pressure")
ax3.grid()

plt.tight_layout()
plt.show()
