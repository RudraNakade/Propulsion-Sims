import numpy as np
from thermo.chemical import Chemical
# from thermo.phase_change import EnthalpyVaporization
import matplotlib.pyplot as plt

ipa = Chemical("isopropanol")
mw = ipa.MW

pressure = np.array([1, 10, 20, 30, 40, 50, 60]) * 1e5
temperature = np.linspace(270, 500, 100)

rho = np.zeros((len(pressure), len(temperature)))
cp = np.zeros((len(pressure), len(temperature)))
k = np.zeros((len(pressure), len(temperature)))
visc = np.zeros((len(pressure), len(temperature)))
qvap = np.zeros(len(temperature))

for j, t in enumerate(temperature):
    for i, p in enumerate(pressure):
        ipa.calculate(t, p)
        rho[i, j] = ipa.rho
        cp[i, j] = ipa.Cp
        k[i, j] = ipa.k
        visc[i, j] = ipa.mu
    qvap[j] = ipa.EnthalpyVaporization(t) / mw if ipa.EnthalpyVaporization(t) is not None else None

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

for i,p in enumerate(pressure):
    axes[1,1].plot(temperature, qvap)
axes[1,1].set_xlabel('Temperature (K)')
axes[1,1].set_ylabel('Heat of Vaporization (kJ/kg)')
axes[1,1].set_title('Heat of Vaporization vs Temperature')
axes[1,1].grid(True)

plt.tight_layout()
plt.show()