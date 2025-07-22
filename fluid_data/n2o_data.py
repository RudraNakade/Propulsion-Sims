import numpy as np
from thermo.chemical import Chemical
# from thermo.phase_change import EnthalpyVaporization
import matplotlib.pyplot as plt

n2o = Chemical("nitrous oxide")
mw = n2o.MW

pressure = np.array([1, 10, 20, 30, 40, 50, 60, 70]) * 1e5
temperature = np.linspace(200, 300, 100)

rho = np.zeros((len(pressure), len(temperature)))
cp = np.zeros((len(pressure), len(temperature)))
k = np.zeros((len(pressure), len(temperature)))
visc = np.zeros((len(pressure), len(temperature)))
hvap = np.zeros(len(temperature))
hf = np.zeros((len(temperature)))

for j, t in enumerate(temperature):
    for i, p in enumerate(pressure):
        n2o.calculate(t, p)
        rho[i, j] = n2o.rho
        cp[i, j] = n2o.Cp
        k[i, j] = n2o.k
        visc[i, j] = n2o.mu

    hvap[j] = n2o.Hvap if n2o.Hvap is not None else None
    hf[j] = n2o.Hf if n2o.Hf is not None else None

fig, axes = plt.subplots(2, 3, figsize=(12, 10))

# Density plot
for i, p in enumerate(pressure):
    axes[0,0].plot(temperature, rho[i, :], label=f'{p/1e5:.1f} bar', linestyle='-')
axes[0,0].set_xlabel('Temperature (K)')
axes[0,0].set_ylabel('Density (kg/m³)')
axes[0,0].set_title('Density vs Temperature')
axes[0,0].legend()
axes[0,0].grid(True)

# Heat capacity plot
for i, p in enumerate(pressure):
    axes[0,1].plot(temperature, cp[i, :], label=f'{p/1e5:.1f} bar', linestyle='-')
axes[0,1].set_xlabel('Temperature (K)')
axes[0,1].set_ylabel('Heat Capacity (J/kg·K)')
axes[0,1].set_title('Heat Capacity vs Temperature')
axes[0,1].legend()
axes[0,1].grid(True)

# Thermal conductivity plot
for i, p in enumerate(pressure):
    axes[0,2].plot(temperature, k[i, :], label=f'{p/1e5:.1f} bar', linestyle='-')
axes[0,2].set_xlabel('Temperature (K)')
axes[0,2].set_ylabel('Thermal Conductivity (W/m·K)')
axes[0,2].set_title('Thermal Conductivity vs Temperature')
axes[0,2].legend()
axes[0,2].grid(True)

# Viscosity plot
for i, p in enumerate(pressure):
    axes[1,0].plot(temperature, visc[i, :], label=f'{p/1e5:.1f} bar', linestyle='-')
axes[1,0].set_xlabel('Temperature (K)')
axes[1,0].set_ylabel('Viscosity (Pa·s)')
axes[1,0].set_title('Viscosity vs Temperature')
axes[1,0].legend()
axes[1,0].grid(True)

# Heat of vaporization plot
for i,p in enumerate(pressure):
    axes[1,1].plot(temperature, hvap*1e-3)
axes[1,1].set_xlabel('Temperature (K)')
axes[1,1].set_ylabel('Heat of Vaporization (kJ/kg)')
axes[1,1].set_title('Heat of Vaporization vs Temperature')
axes[1,1].grid(True)

# Enthalpy of formation plot
for i, p in enumerate(pressure):
    axes[1,2].plot(temperature, hf*1e-3)
axes[1,2].set_xlabel('Temperature (K)')
axes[1,2].set_ylabel('Enthalpy of Formation (kJ/kg)')
axes[1,2].grid(True)

plt.tight_layout()
plt.show()