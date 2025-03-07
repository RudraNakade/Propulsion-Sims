import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input

T = 10 # deg C

nitrous = Fluid(FluidsList.NitrousOxide)
nitrous.R = 8.31447/nitrous.molar_mass

Cd = 0.65
D = 1 # mm
A = np.pi * 0.25 * 1e-6 * D**2


p = np.linspace(6.45,40,100) # bar
mdot = np.zeros(len(p))
mdot_choked = np.zeros(len(p))
mdot_spi = np.zeros(len(p))

pe = 6.45 # bar

for i in range(len(p)):
    nitrous.update(Input.temperature(T), Input.pressure(p[i]*1e5))
    nitrous.gamma = (nitrous.specific_heat)/(nitrous.specific_heat-nitrous.R)
    choking_ratio = ((nitrous.gamma + 1)/2)**(nitrous.gamma/(nitrous.gamma-1))
    k = (2/(nitrous.gamma+1))**((nitrous.gamma+1)/(nitrous.gamma-1))
    min_choked_p = 2 * pe / (2-nitrous.gamma*k) #pe*choking_ratio
    if p[i] >= min_choked_p:
        mdot[i] = Cd*A*np.sqrt(nitrous.gamma*nitrous.density*p[i]*1e5*k)
    else:
        mdot[i] = Cd*A*np.sqrt(2*nitrous.density*(p[i]-pe)*1e5)
    mdot_choked[i] = Cd*A*np.sqrt(nitrous.gamma*nitrous.density*p[i]*1e5*k)
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