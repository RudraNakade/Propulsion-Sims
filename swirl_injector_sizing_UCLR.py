# From UCLR Swirl Doc

import numpy as np
import scipy
from pyfluids import Fluid, FluidsList, Input
from os import system

system('cls')

rho_f = 790
dp_f = 30*0.3*1e5

mdot_o = 2.14567
mdot_f = 0.67052
n_elements = 8

fuel = Fluid(FluidsList.Ethanol)
fuel.update(Input.temperature(90), Input.pressure(40e5))
dyn_visc_f = fuel.dynamic_viscosity

print(dyn_visc_f)

element_mdot_o = mdot_o / n_elements
element_mdot_f = mdot_f / n_elements

## Inner Element Sizing
spray_angle_desired = 120
spray_half_angle_desired = spray_angle_desired / 2
n_inlets = 4

def filling_eff_func(eff, spray_half_angle):
    return np.sin(np.pi * spray_half_angle / 180) - (2*np.sqrt(2) * (1 - eff) / ((1 + np.sqrt(1 - eff) * np.sqrt(2 - eff))))

filling_eff = scipy.optimize.fsolve(filling_eff_func, 0.5, args=(spray_half_angle_desired))[0]
