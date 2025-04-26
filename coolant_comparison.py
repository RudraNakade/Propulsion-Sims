from pyfluids import Fluid, FluidsList, Input
import numpy as np
import matplotlib.pyplot as plt
from thermo import VaporPressure
from thermo.chemical import Chemical

temperatures = np.linspace(-30,90,100)
temps_K = temperatures + 273.15  # K

ipa = Chemical("isopropanol")
ipa_rho = np.zeros_like(temps_K)
ipa_cp = np.zeros_like(temps_K)
ipa_mu = np.zeros_like(temps_K)
ipa_k = np.zeros_like(temps_K)
ipa_vp = np.zeros_like(temps_K)
for i, T in enumerate(temps_K):
    ipa.T = T
    ipa_rho[i] = ipa.rho
    ipa_cp[i] = ipa.Cp
    ipa_k[i] = ipa.k
    ipa_vp[i] = VaporPressure(CASRN="67-63-0").calculate(T=T, method='WAGNER_MCGARRY')
    ipa_mu[i] = ipa.ViscosityLiquid(P=ipa_vp[i], T=T)

methanol = Fluid(FluidsList.Methanol)
ethanol = Fluid(FluidsList.Ethanol)
propane = Fluid(FluidsList.nPropane)

D_h = 2e-3 # mm
v = 8 # m/s

props = {
    "Methanol": {"fluid": methanol},
    "Ethanol": {"fluid": ethanol},
    "Propane": {"fluid": propane},
    "Isopropanol": {}
}
for name in ["Methanol", "Ethanol", "Propane"]:
    props[name]["rho"] = np.zeros_like(temperatures)
    props[name]["cp"] = np.zeros_like(temperatures)
    props[name]["visc"] = np.zeros_like(temperatures)
    props[name]["k"] = np.zeros_like(temperatures)

for i, T in enumerate(temperatures):
    for name in ["Methanol", "Ethanol", "Propane"]:
        f = props[name]["fluid"]
        f.update(Input.quality(0), Input.temperature(T))
        props[name]["rho"][i] = f.density
        props[name]["cp"][i] = f.specific_heat
        props[name]["visc"][i] = f.dynamic_viscosity
        props[name]["k"][i] = f.conductivity

props["Isopropanol"]["rho"] = ipa_rho
props["Isopropanol"]["cp"] = ipa_cp
props["Isopropanol"]["visc"] = ipa_mu
props["Isopropanol"]["k"] = ipa_k
props["Isopropanol"]["T"] = temps_K
props["Methanol"]["T"] = temperatures + 273.15
props["Ethanol"]["T"] = temperatures + 273.15
props["Propane"]["T"] = temperatures + 273.15

for name in ["Methanol", "Ethanol", "Isopropanol", "Propane"]:
    rho = props[name]["rho"]
    cp = props[name]["cp"]
    visc = props[name]["visc"]
    k = props[name]["k"]
    Re = rho * v * D_h / visc
    Pr = cp * visc / k
    Nu = 0.023 * Re**0.8 * Pr**0.4
    h = Nu * k / D_h
    props[name]["h"] = h


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
labels = ["Methanol", "Ethanol", "Isopropanol", "Propane"]
ylabels = [
    ('Density (kg/m^3)', 'rho'),
    ('Specific Heat (J/kg-K)', 'cp'),
    ('Viscosity (Pa.s)', 'visc'),
    ('Thermal Conductivity (W/m-K)', 'k')
]
titles = ['Density', 'Specific Heat', 'Dynamic Viscosity', 'Thermal Conductivity']

for idx, (ylabel, key) in enumerate(ylabels):
    ax = axs[idx//2, idx%2]
    for color, name in zip(colors, labels):
        ax.plot(props[name]["T"], props[name][key], label=name, color=color)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(ylabel)
    ax.set_title(titles[idx])
    ax.legend()
    ax.grid()
    ax.grid(which='minor', alpha=0.5)
    ax.set_ylim(0, None)

plt.tight_layout()

plt.figure(figsize=(8, 6))
for color, name in zip(colors, labels):
    plt.plot(props[name]["T"], props[name]["h"]*1e-3, label=name, color=color)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Transfer Coefficient (kW/m²-K)')
plt.title(f'Convective Heat Transfer Coefficient vs Temperature (D_h = {D_h*1e3:.2f}mm, v = {v:.2f}m/s)')
plt.legend()
plt.grid(True)
plt.grid(which='minor', alpha=0.5)
plt.ylim(0, None)
plt.tight_layout()
plt.show()