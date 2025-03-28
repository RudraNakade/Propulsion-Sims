from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
from os import system

system('cls')

ox = Fluid(FluidsList.Oxygen)
n2 = Fluid(FluidsList.Nitrogen)
propane = Fluid(FluidsList.nPropane)
methane = Fluid(FluidsList.Methane)

T = np.linspace(64, 273, 500) - 273.15
# Create propane-specific temperature array with valid range (T > 85.54)
T_propane = T[T > -187]  # 85.54 K = -187.61°C
# Create methane-specific temperature array with valid range (T > 91K)
T_methane = T[T > -181]  # 91 K = -182.15°C
P = np.concatenate([np.array([1, 5]), np.linspace(10, 100, 10)]) * 1e5
ox_rho = np.zeros((len(P),len(T)))
n2_rho = np.zeros((len(P),len(T)))
# Initialize methane and propane density arrays with appropriate dimensions
propane_rho = np.zeros((len(P),len(T_propane)))
methane_rho = np.zeros((len(P),len(T_methane)))

ox.update(Input.pressure(101325), Input.quality(0))
n2.update(Input.pressure(101325), Input.quality(0))
propane.update(Input.pressure(101325), Input.quality(0))
methane.update(Input.pressure(101325), Input.quality(0))
print(f'Oxygen   - Critial Point: T: {(ox.critical_temperature+273.15):.2f} K / {ox.critical_temperature:.2f} deg C, P: {(ox.critical_pressure/1e5):.2f} Bar')
print(f'Nitrogen - Critial Point: T: {(n2.critical_temperature+273.15):.2f} K / {n2.critical_temperature:.2f} deg C, P: {(n2.critical_pressure/1e5):.2f} Bar')
print(f'Propane  - Critial Point: T: {(propane.critical_temperature+273.15):.2f} K / {propane.critical_temperature:.2f} deg C, P: {(propane.critical_pressure/1e5):.2f} Bar')
print(f'Methane  - Critial Point: T: {(methane.critical_temperature+273.15):.2f} K / {methane.critical_temperature:.2f} deg C, P: {(methane.critical_pressure/1e5):.2f} Bar\n')
print(f'Storage at 1 Atm:\nOxygen:   {(ox.temperature+273.15):.2f} K / {(ox.temperature):.2f} deg C, {ox.density:.2f} kg/m^3\nNitrogen: {(n2.temperature+273.15):.2f} K / {(n2.temperature):.2f} deg C, {n2.density:.2f} kg/m^3\nPropane:  {(propane.temperature+273.15):.2f} K / {(propane.temperature):.2f} deg C, {propane.density:.2f} kg/m^3\nMethane:  {(methane.temperature+273.15):.2f} K / {(methane.temperature):.2f} deg C, {methane.density:.2f} kg/m^3')

for j, pressure in enumerate(P):
    for i, temp in enumerate(T):
        ox.update(Input.temperature(temp), Input.pressure(pressure))
        n2.update(Input.temperature(temp), Input.pressure(pressure))
        ox_rho[j,i] = ox.density
        n2_rho[j,i] = n2.density
    
    # Separate loop for propane with valid temperature range
    for i, temp in enumerate(T_propane):
        propane.update(Input.temperature(temp), Input.pressure(pressure))
        propane_rho[j,i] = propane.density
        
    # Separate loop for methane with valid temperature range
    for i, temp in enumerate(T_methane):
        methane.update(Input.temperature(temp), Input.pressure(pressure))
        methane_rho[j,i] = methane.density

oxfig = plt.figure()
n2fig = plt.figure()
diff_fig = plt.figure()
propane_fig = plt.figure()
methane_fig = plt.figure()

for i in range(len(P)):
    plt.figure(oxfig)
    plt.plot(T+273.15, ox_rho[i,:], label=f'{P[i]/1e5} bar')
    plt.figure(n2fig)
    plt.plot(T+273.15, n2_rho[i,:], label=f'{P[i]/1e5} bar')
    plt.figure(diff_fig)
    plt.plot(T+273.15, ox_rho[i,:]-n2_rho[i,:], label=f'{P[i]/1e5} bar')
    # Use methane-specific temperature array for methane plots
    plt.figure(methane_fig)
    plt.plot(T_methane+273.15, methane_rho[i,:], label=f'{P[i]/1e5} bar')
    # Use propane-specific temperature array for propane plots
    plt.figure(propane_fig)
    plt.plot(T_propane+273.15, propane_rho[i,:], label=f'{P[i]/1e5} bar')

plt.figure(oxfig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Oxygen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.ylim(bottom=0)
plt.grid()

plt.figure(n2fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Nitrogen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.ylim(bottom=0)
plt.grid()

plt.figure(diff_fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density Difference (kg/m^3)')
plt.title('Oxygen - Nitrogen Density vs Temperature')
plt.legend()
plt.xlim(min(T+273.15), max(T+273.15))
plt.grid()

plt.figure(propane_fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Propane Density vs Temperature')
plt.legend()
plt.xlim(min(T_propane+273.15), max(T_propane+273.15))
plt.ylim(bottom=0)
plt.grid()

plt.figure(methane_fig)
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Methane Density vs Temperature')
plt.legend()
plt.xlim(min(T_methane+273.15), max(T_methane+273.15))
plt.ylim(bottom=0)
plt.grid()

plt.show()