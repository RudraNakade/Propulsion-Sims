import numpy as np
from pyfluids import Input, Fluid, FluidsList
from unit_converter import *
from flow_models import *

# tank = Fluid(FluidsList.Oxygen)
tank = Fluid(FluidsList.Nitrogen)
tank = Fluid(FluidsList.NitrousOxide)

# mdot = 30e-3
# Cd = 0.65
# id = 3.2e-3

# mdot = 0.5
# mdot = 0.65
# mdot = 1.15
# Cd = 0.65
# id = 6e-3

# CdA = Cd * np.pi * (id / 2)**2

mdot = 4e-3
Cd = 0.65
id = 0.5e-3
od = 0.9e-3

CdA = Cd * np.pi * ((od / 2)**2 - (id / 2)**2)

tank_temp = 298
tank_pressure = 55e5
p_down = 12.32e5

# p_down = psi_to_pa(250)
# p_down = psi_to_pa(200)

tank.update(Input.pressure(tank_pressure), Input.temperature(K_to_degC(tank_temp)))
downstream = tank.isenthalpic_expansion_to_pressure(p_down)
downstream_temp = degC_to_K(downstream.temperature)

tank_density = tank.density
downstream_density = downstream.density

p_0, choking_p, choked = spc_p_0(CdA, mdot, Fluid(FluidsList.Oxygen), downstream_temp, p_down)

dP = p_0 - p_down

print(f"Tank temperature:          {K_to_degC(tank_temp):.2f} °C")
print(f"Downstream temperature:    {K_to_degC(downstream_temp):.2f} °C")
print(f"Tank density:              {tank_density:.2f} kg/m³")
print(f"Downstream density:        {downstream_density:.2f} kg/m³")
print(f"Calculated feed pressure:  {p_0 / 1e5:.2f} bar / {pa_to_psi(p_0):.2f} psi")
print(f"Downstream pressure:       {p_down / 1e5:.2f} bar / {pa_to_psi(p_down):.2f} psi")
print(f"Choking pressure:          {choking_p / 1e5:.2f} bar / {pa_to_psi(choking_p):.2f} psi, choked: {choked}")
print(f"Pressure drop:             {dP / 1e5:.2f} bar / {pa_to_psi(dP):.2f} psi")