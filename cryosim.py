from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
import enginesim

ox = Fluid(FluidsList.Oxygen)
n2 = Fluid(FluidsList.Nitrogen)

T = np.linspace(65, 150, 100) - 273.15
P = np.concatenate([np.array([1]), np.linspace(10, 100, 10)]) * 1e5
ox_rho = np.zeros((len(P),len(T)))
n2_rho = np.zeros((len(P),len(T)))

ox.update(Input.pressure(101325), Input.enthalpy(-1))
n2.update(Input.pressure(101325), Input.enthalpy(-1))
print(f'Oxygen   - Critial Point: T: {(ox.critical_temperature+273.15):.2f} K / {ox.critical_temperature:.2f} deg C, P: {(ox.critical_pressure/1e5):.2f} Bar')
print(f'Nitrogen - Critial Point: T: {(n2.critical_temperature+273.15):.2f} K / {n2.critical_temperature:.2f} deg C, P: {(n2.critical_pressure/1e5):.2f} Bar\n')
print(f'Storage at 1 Atm:\nOxygen:   {(ox.temperature+273.15):.2f} K / {(ox.temperature):.2f} deg C, {ox.density:.2f} kg/m^3\nNitrogen: {(n2.temperature+273.15):.2f} K / {(n2.temperature):.2f} deg C, {n2.density:.2f} kg/m^3')

for j, pressure in enumerate(P):
    for i, temp in enumerate(T):
        ox.update(Input.temperature(temp), Input.pressure(pressure))
        n2.update(Input.temperature(temp), Input.pressure(pressure))
        ox_rho[j,i] = ox.density
        n2_rho[j,i] = n2.density

oxfig = plt.figure()
n2fig = plt.figure()
diff_fig = plt.figure()

for i in range(len(P)):
    plt.figure(oxfig)
    plt.plot(T+273.15, ox_rho[i,:], label=f'{P[i]/1e5} bar')
    plt.figure(n2fig)
    plt.plot(T+273.15, n2_rho[i,:], label=f'{P[i]/1e5} bar')
    plt.figure(diff_fig)
    plt.plot(T+273.15, ox_rho[i,:]-n2_rho[i,:], label=f'{P[i]/1e5} bar')

plt.figure(oxfig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Oxygen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.ylim(0, 1300)
plt.grid()

plt.figure(n2fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Nitrogen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.ylim(0, 1300)
plt.grid()

plt.figure(diff_fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density Difference (kg/m^3)')
plt.title('Oxygen - Nitrogen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.grid()

plt.show()