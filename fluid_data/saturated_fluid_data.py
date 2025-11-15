from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
from os import system

system('cls')

def plot_saturation_properties(fluid):
    fluid_class = Fluid(fluid)

    name = fluid_class.name
    T = np.linspace(fluid_class.min_temperature, fluid_class.critical_temperature, 1000)
    P = np.zeros(len(T))
    D_g = np.zeros(len(T))
    D_l = np.zeros(len(T))

    for i, temp in enumerate(T):
        fluid_class.update(Input.temperature(temp), Input.quality(0))
        P[i] = fluid_class.pressure/1e5
        D_l[i] = fluid_class.density
        fluid_class.update(Input.temperature(temp), Input.quality(100))
        D_g[i] = fluid_class.density

    fig, ax1 = plt.subplots()

    ax1.plot(T, P, 'red', label='Vapor Pressure')
    ax1.grid()
    ax1.grid(which="minor")
    ax1.minorticks_on()
    ax1.set_xlabel('Temperature (deg C)')
    ax1.set_ylabel('Pressure (bar)', color='red')
    ax1.set_title(f'{name} Properties vs Temperature')
    ax1.set_xlim(left=T[0],right=T[-1])
    ax2 = ax1.twinx()
    ax2.plot(T, D_l,'b',label='Liquid Density')
    # ax2.plot(T, D_g,'m',label='Gas Density')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Density (kg/m^3)' , color='b')
    ax2.minorticks_on()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.tight_layout()

plot_saturation_properties(FluidsList.NitrousOxide)
# plot_saturation_properties(FluidsList.CarbonDioxide)
# plot_saturation_properties(FluidsList.nPropane)
# plot_saturation_properties(FluidsList.nButane)
# plot_saturation_properties(FluidsList.Ethane)

plt.show()