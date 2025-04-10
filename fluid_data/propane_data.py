from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
from os import system

system('cls')

propane = Fluid(FluidsList.nPropane)

T = np.linspace(propane.min_temperature, propane.critical_temperature, 1000)
P = np.zeros(len(T))
D_g = np.zeros(len(T))
D_l = np.zeros(len(T))

for i, temp in enumerate(T):
    propane.update(Input.temperature(temp), Input.quality(0))
    P[i] = propane.pressure/1e5
    D_l[i] = propane.density
    propane.update(Input.temperature(temp), Input.quality(100))
    D_g[i] = propane.density


fig, ax1 = plt.subplots()

ax1.plot(T, P, 'red', label='Vapor Pressure')
ax1.grid()
ax1.grid(which="minor")
ax1.minorticks_on()
ax1.set_xlabel('Temperature (deg C)')
ax1.set_ylabel('Pressure (bar)', color='red')
ax1.set_title('Propane Properties vs Temperature')
ax1.set_xlim(left=T[0],right=T[-1])
ax2 = ax1.twinx()
ax2.plot(T, D_l,'b',label='Liquid Density')
ax2.plot(T, D_g,'m',label='Gas Density')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Density (kg/m^3)' , color='b')
ax2.minorticks_on()
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
fig.tight_layout()

t = 15

propane.update(Input.temperature(t), Input.quality(0))
sat_lqd_rho = propane.density
propane.update(Input.temperature(t), Input.quality(100))
sat_vap_rho = propane.density

print(f"Propane @ {t:.1f} deg C -  saturation pressure: {propane.pressure/1e5:.2f} bar\nliquid density: {sat_lqd_rho:.2f} kg/m^3\nvapor density: {sat_vap_rho:.2f} kg/m^3")

t = -30 # deg C
p = 4 # bar

propane.update(Input.temperature(t), Input.pressure(p*1e5))
lqd_rho = propane.density

print(f'Propane @ {t:.1f} deg C and {p:.1f} bar - liquid density: {lqd_rho:.2f} kg/m^3')

plt.show()