from matplotlib.pylab import f
import numpy as np
import scipy
from pyfluids import Fluid, FluidsList, Input
from os import system

system("cls")

rho = 790
dp = 30e5 * 0.3

mdot_total = 0.71
n_elements = 16

fuel = Fluid(FluidsList.Ethanol)
fuel.update(Input.temperature(20), Input.pressure(35e5))
kinematic_visc = fuel.kinematic_viscosity

print(kinematic_visc)

mdot = mdot_total / n_elements # mdot per injector element
print(f"mdot: {mdot:.4f} kg/s")

## Inner Element Sizing
spray_angle_desired = 60
n_inlets = 4
nozzle_opening_coeff = 2.5

k = 1 # initial guess

def eps_func(eps, k):
    return k * (eps**1.5) - ((1 - eps) * np.sqrt(2))

def s_func(s, mu, k):
    return (
        np.sqrt(1 - (mu * k) ** 2)
        - s * np.sqrt((s ** 2) - (mu * k) ** 2)
        - ((mu * k) ** 2)
        * np.log(
            (1 + np.sqrt(1 - (mu * k) ** 2))
            / (s + np.sqrt((s ** 2) - (mu * k) ** 2))
        )
    )

rel_diff = 1

while rel_diff > 1e-4:
    filling_eff = scipy.optimize.fsolve(eps_func, 0.5, args=(k))[0]
    print(f"Initial Filling Efficiency: {filling_eff}")

    outlet_cd = filling_eff * np.sqrt(filling_eff / (2 - filling_eff))
    print(f"Initial Outlet Cd: {outlet_cd}")

    d_outlet = np.sqrt(4 * mdot / (np.pi * outlet_cd * np.sqrt(2 * rho * dp)))
    print(f"Initial D_outlet: {d_outlet*1e3:.2f} mm")

    r_inlet_axis = nozzle_opening_coeff * d_outlet / 2
    print(f"Initial r_inlet_axis: {r_inlet_axis*1e3:.2f} mm")

    d_inlet_orifice = np.sqrt(2 * r_inlet_axis * d_outlet / (n_inlets * k))
    print(f"Initial D_inlet_orifice: {d_inlet_orifice*1e3:.2f} mm")

    est_inlet_dp = (mdot / (0.65 * np.pi * d_inlet_orifice**2))**2 / (2e5 * rho)
    print(f"Est Inlet DP: {est_inlet_dp:.2f} bar")

    Re = 4 * mdot / (np.pi * rho * kinematic_visc * np.sqrt(n_elements) * d_inlet_orifice)

    inlet_friction_coeff = np.exp((25.8 / ((np.log(Re))**2.58)) - 2)

    # B = r_inlet_axis / (0.5 * d_inlet_orifice)
    # k = k / (1 + (0.5 * inlet_friction_coeff * (B**2) / (n_inlets - k)))
    # print(k)
    k_lambda = r_inlet_axis * 0.5 * d_outlet / ((n_inlets * 0.25 * d_inlet_orifice**2) + 0.5 * inlet_friction_coeff * r_inlet_axis * (r_inlet_axis - (0.5 * d_outlet)))
    # print(k)

    filling_eff = scipy.optimize.fsolve(eps_func, 0.5, args=(k_lambda))[0]
    print(f"New Filling Efficiency: {filling_eff}")

    outlet_cd = filling_eff * np.sqrt(filling_eff / (2 - filling_eff))
    print(f"New 1 Outlet Cd: {outlet_cd}")
    outlet_cd = outlet_cd / np.sqrt(1 + (outlet_cd * k_lambda / nozzle_opening_coeff)**2)
    print(f"New 2 Outlet Cd: {outlet_cd}")
    
    d_outlet = np.sqrt(4 * mdot / (np.pi * outlet_cd * np.sqrt(2 * rho * dp)))

    k = 2 * r_inlet_axis * d_outlet / (n_inlets * d_inlet_orifice**2)

    r_inlet_axis = nozzle_opening_coeff * d_outlet / 2

    # s - gas core diameter to swirl outlet diameter ratio

    s = scipy.optimize.root_scalar(s_func, [0, 1], args=(filling_eff, k), method = 'brentq').root
    # s = 0.01

    spray_angle = 2 * np.arctan(2 * outlet_cd * k_lambda / np.sqrt(1 + (s**2) - 4 * ((outlet_cd*k)**2))) * 180 / np.pi
    print(f'spray_angle: {spray_angle}')

    print(f'k_lambda: {k_lambda}')

    print(f'k old: {k}')

    k = k * spray_angle_desired / spray_angle

    print(f'k new: {k}')

    rel_diff = abs(spray_angle - spray_angle_desired) / spray_angle_desired

    print(f'\nrel_diff: {rel_diff}')
    print(f'D_outlet: {d_outlet*1e3:.2f} mm')
    print(f'D_inlet: {d_inlet_orifice*1e3:.2f} mm')
    print(f'Re: {Re:.2f}')
    print(f'Outlet Cd: {outlet_cd:.2f}')
    print(f'Filling Efficiency: {filling_eff:.2f}')
    print(f'Spray Angle: {spray_angle:.2f} degrees')
    print(f'Inlet Friction Coefficient: {inlet_friction_coeff:.2f}')
    print(f'Inlet Axis Radius: {r_inlet_axis*1e3:.2f} mm')
    print(f'S: {s:.2f}\n')

    input('Press Enter to continue...')

# testinj = es.injector()
# testinj.size_fuel_holes(0.65, d_inlet_orifice*1e3, 4)
# fuel_mass_flow_rate = testinj.spi_fuel_mdot(30*0.3, 790)
# print(f'Total Fuel Mass Flow Rate: {fuel_mass_flow_rate * n_elements:.2f} kg/s')

