import numpy as np
import scipy
from pyfluids import Fluid, FluidsList, Input
from os import system
import enginesim as es

system('cls')

rho_f = 790
dp_f = 30*0.3*1e5

mdot_o = 2.14567
mdot_f = 0.67052
n_elements = 8

fuel = Fluid(FluidsList.Ethanol)
fuel.update(Input.temperature(15), Input.pressure(40e5))
dyn_visc_f = fuel.dynamic_viscosity

print(dyn_visc_f)

element_mdot_o = mdot_o / n_elements
element_mdot_f = mdot_f / n_elements

## Inner Element Sizing
spray_angle_desired = 60
n_inlets = 4
nozzle_opening_coeff = 3.5

k = 1 # initial guess

def eps_func(eps, k):
    return k * (eps**1.5) - ((1 - eps) * np.sqrt(2))

def s_func(s, mu, k):
    return np.sqrt(1 - (mu*k)**2) - s*np.sqrt((s**2) - (mu * k)**2) - ((mu * k)**2)*np.log((1 + np.sqrt(1 - (mu*k)**2)) / (s + np.sqrt((s**2) - (mu*k)**2)))

rel_diff = 1

while rel_diff > 1e-2:
    filling_eff = scipy.optimize.fsolve(eps_func, 0.5, args=(k))[0]

    outlet_cd = filling_eff * np.sqrt(filling_eff / (2 - filling_eff))

    d_outlet = np.sqrt(4 * element_mdot_f / (np.pi * outlet_cd * np.sqrt(2 * rho_f * dp_f)))

    r_inlet_axis = nozzle_opening_coeff * d_outlet / 2

    d_inlet_orifice = np.sqrt(2 * nozzle_opening_coeff * d_outlet / (n_inlets * k))

    Re = 4 * element_mdot_f / (np.pi * dyn_visc_f * np.sqrt(n_elements) * d_inlet_orifice)

    inlet_friction_coeff = np.exp((25.8 / ((np.log(Re))**2.58)) - 2)

    B = r_inlet_axis / (0.5 * d_inlet_orifice)
    k = k / (1 + (0.5 * inlet_friction_coeff * (B**2) / (n_inlets - k)))
    # print(k)
    # k = r_inlet_axis * 0.5 * d_outlet / ((n_inlets * 0.25 * d_inlet_orifice**2) + 0.5 * inlet_friction_coeff * r_inlet_axis * (r_inlet_axis - (0.5 * d_outlet)))
    # print(k)

    filling_eff = scipy.optimize.fsolve(eps_func, 0.5, args=(k))[0]

    outlet_cd = filling_eff * np.sqrt(filling_eff / (2 - filling_eff))
    # outlet_cd = outlet_cd / np.sqrt(1 + (outlet_cd * k / r_inlet_axis)**2)

    d_outlet = np.sqrt(4 * element_mdot_f / (np.pi * outlet_cd * np.sqrt(2 * rho_f * dp_f)))

    k = 2 * r_inlet_axis * d_outlet / (n_inlets * d_inlet_orifice**2)

    r_inlet_axis = nozzle_opening_coeff * d_outlet / 2

    # s - gas core diameter to swirl outlet diameter ratio

    s = scipy.optimize.fsolve(s_func, 0.5, args=(filling_eff, k))[0]
    print(f's: {s}')

    spray_angle = 2 * np.arctan(2 * outlet_cd * k / np.sqrt(1 + (s**2) - 4 * ((outlet_cd*k)**2))) * 180 / np.pi

    print(f'k old: {k}')

    k = k * spray_angle_desired / spray_angle

    rel_diff = abs(spray_angle - spray_angle_desired) / spray_angle_desired

    print(f'\nrel_diff: {rel_diff}')
    print(f'D_outlet: {d_outlet*1e3:.2f} mm')
    print(f'D_inlet: {d_inlet_orifice*1e3:.2f} mm')
    print(f'Re: {Re:.2f}')
    print(f'K: {k:.2f}')
    print(f'Outlet Cd: {outlet_cd:.2f}')
    print(f'Filling Efficiency: {filling_eff:.2f}')
    print(f'Spray Angle: {spray_angle:.2f} degrees')
    print(f'Inlet Friction Coefficient: {inlet_friction_coeff:.2f}')
    print(f'Inlet Axis Radius: {r_inlet_axis*1e3:.2f} mm')
    print(f'Gas Core Diameter to Swirl Outlet Diameter Ratio: {s:.2f}\n')

# testinj = es.injector()
# testinj.size_fuel_holes(0.65, d_inlet_orifice*1e3, 4)
# fuel_mass_flow_rate = testinj.spi_fuel_mdot(30*0.3, 790)
# print(f'Total Fuel Mass Flow Rate: {fuel_mass_flow_rate * n_elements:.2f} kg/s')

