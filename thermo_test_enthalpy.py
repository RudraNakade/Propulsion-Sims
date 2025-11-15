from thermo import Chemical
import numpy as np

n2o = Chemical('N2O')

T_init = 19 + 273.15
P_init = 50e5

n2o.calculate(T=T_init, P=P_init)

H_init = n2o.H

dH_arr = np.linspace(0, 50e3, 1000)
T_arr = np.zeros_like(dH_arr)
rho_arr = np.zeros_like(dH_arr)

for i, dH in enumerate(dH_arr):
    try:
        n2o.calculate_PH(H=(dH+H_init), P=P_init)
        T_arr[i] = n2o.T
        rho_arr[i] = n2o.rho
    except:
        T_arr[i] = np.nan
        rho_arr[i] = np.nan

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
ax.plot(dH_arr / 1e3, T_arr, 'g-')
ax2.plot(dH_arr / 1e3, rho_arr, 'b-')
ax.set_xlabel('Enthalpy (kW/(kg/s))')
ax.set_ylabel('Temperature (K)', color='g')
ax2.set_ylabel('Density (kg/m³)', color='b')
ax.grid()
plt.show()