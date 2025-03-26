import numpy as np
import scipy
from pyfluids import Fluid, FluidsList, Input
from os import system
import enginesim as es

system('cls')

rho_f = 790
dp_f = 30*0.15*1e5

mdot_o = 2.14567
mdot_f = 0.67052
n_elements = 8

inlet_cd_f = 0.65

## Inner Element Sizing
spray_angle_desired = 90
spray_half_angle_desired = spray_angle_desired / 2
n_inlets = 4
orifice_dp = 30*0.3*1e5

mdot_f_element = mdot_f / n_elements

mdot_f_orifice = mdot_f_element / n_inlets

# A_inlet_orifice = mdot_f_orifice / (inlet_cd_f * np.sqrt(2 * orifice_dp * rho_f))

def open_area_ratio_func(x, half_ang):
    return np.sin(np.pi*half_ang/180) - (x * np.sqrt(8) / ((1 + np.sqrt(x)) * np.sqrt(1 + x)))

open_area_ratio = scipy.optimize.fsolve(open_area_ratio_func, 0.5, args=(spray_half_angle_desired))[0]

outlet_cd = np.sqrt((1-open_area_ratio)**3 / (1 + open_area_ratio))

A_outlet = mdot_f_element / (outlet_cd * np.sqrt(2 * dp_f * rho_f))

d_outlet = np.sqrt(4 * A_outlet / np.pi)

inlet_cd = np.sqrt(open_area_ratio**3 / (2 - open_area_ratio))

A_inlet_orifice = mdot_f_orifice / (inlet_cd * np.sqrt(2 * dp_f * rho_f))

d_inlet_orifice = np.sqrt(4 * A_inlet_orifice / np.pi)

print(f'Inlet Cd: {inlet_cd:.3f}')
print(f'Outlet Cd: {outlet_cd:.3f}')
print(f'Open Area Ratio: {open_area_ratio:.3f}')
print(f'Outlet Diameter: {d_outlet*1e3:.2f} mm')
print(f'Inlet Orifice Diameter: {d_inlet_orifice*1e3:.2f} mm')

test = es.injector()
test.size_fuel_holes(0.65, 1, 4 * 15)
test_mdot = test.spi_fuel_mdot(dp_f/1e5, rho_f)
print(f'SPI fuel inlet orifice mdot: {test_mdot:.3f} kg/s')

A_inlet = A_inlet_orifice * n_inlets
dp_inlet = (mdot_f_element / (A_inlet * 0.65))**2 / (rho_f * 2)

dp_outlet = (mdot_f_element / (A_outlet * outlet_cd))**2 / (rho_f * 2)

print(f'Inlet Pressure Drop: {dp_inlet/1e5:.2f} bar')
print(f'Outlet Pressure Drop: {dp_outlet/1e5:.2f} bar')
print(f'Total Pressure Drop: {(dp_inlet + dp_outlet)/1e5:.2f} bar')