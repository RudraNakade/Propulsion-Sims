import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input

T = 10 # deg C

nitrous = Fluid(FluidsList.NitrousOxide)
R = 8.31447/nitrous.molar_mass

Cd = 0.65
D = 1 # mm
A = np.pi * 0.25 * 1e-6 * D**2

p = np.linspace(15,40,200) # bar
mdot = np.zeros(len(p))
mdot_choked = np.zeros(len(p))
mdot_spi = np.zeros(len(p))

pe = 15 # bar

for i in range(len(p)):
    nitrous.update(Input.temperature(T), Input.pressure(p[i]*1e5))
    gamma = (nitrous.specific_heat)/(nitrous.specific_heat-R)
    choking_ratio = ((gamma + 1)/2)**(gamma/(gamma-1))
    k = (2/(gamma+1))**((gamma+1)/(gamma-1))
    # min_choked_p = 2 * pe / (2-gamma*k)
    min_choked_p = pe*choking_ratio
    if p[i] >= min_choked_p:
        mdot[i] = Cd*A*np.sqrt(gamma*nitrous.density*p[i]*1e5*k)
    else:
        mdot[i] = Cd*A*np.sqrt(2*nitrous.density*(p[i]-pe)*1e5)
    mdot_choked[i] = Cd*A*np.sqrt(gamma*nitrous.density*p[i]*1e5*k)
    mdot_spi[i] = Cd*A*np.sqrt(2*nitrous.density*(p[i]-pe)*1e5)

plt.plot(p,mdot_choked*1e3, color='r', label='Choked')
plt.plot(p,mdot_spi*1e3, color='b', label='SPI')
plt.plot(p,mdot*1e3, color='g', label='Combined')
plt.legend()
plt.xlabel('Upstream Pressure (bar)')
plt.ylabel('Mass Flow Rate (g/s)')
plt.title('Mass Flow Rate vs Pressure')
plt.grid()
plt.show()
