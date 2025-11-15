from pyfluids import Fluid, FluidsList, Input
import numpy as np

upstream_gas = Fluid(FluidsList.Nitrogen)

pressurant_tank_temps = np.linspace(-100, 20, 100)
pressurant_tank_p = 52.53e5
prop_tank_p = 33.91e5

upstream_temps = np.zeros_like(pressurant_tank_temps)
upstream_enthalpies = np.zeros_like(pressurant_tank_temps)
downstream_temps = np.zeros_like(pressurant_tank_temps)

for i, temp in enumerate(pressurant_tank_temps):
    upstream_gas.update(Input.temperature(temp), Input.pressure(pressurant_tank_p))

    # Isenthalpic expansion to propellant tank pressure
    downstream_gas = upstream_gas.clone()
    downstream_gas.update(Input.pressure(prop_tank_p), Input.enthalpy(upstream_gas.enthalpy))
    
    upstream_h = upstream_gas.enthalpy

    upstream_temps[i] = upstream_gas.temperature
    upstream_enthalpies[i] = upstream_h
    downstream_temps[i] = downstream_gas.temperature

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(upstream_temps, downstream_temps, 'b-', linewidth=2)
plt.xlabel('Upstream Temperature (degC)')
plt.ylabel('Downstream Temperature (degC)')
plt.title('Isenthalpic Expansion: Upstream vs Downstream Temperature')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(upstream_temps, upstream_enthalpies/1e3, 'r-', linewidth=2)
plt.xlabel('Upstream Temperature (degC)')
plt.ylabel('Upstream Enthalpy (kJ/kg)')
plt.title('Isenthalpic Expansion: Upstream Temperature vs Enthalpy')
plt.grid(True, alpha=0.3)
plt.show()

