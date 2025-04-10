import scipy.optimize
from pyfluids import Fluid, FluidsList, Input
import numpy as np
from os import system

system("cls")

### Design parameters
pc = 30e5 # chamber pressure (pa)
target_dp = 30e5 * 0.25 # injector dp (pa)

rho = 790 # kg/m^3
T = 20 # deg C

mdot_total = 0.71 # kg/S

alpha = 60 # spray half angle (deg)

n_elements = 12 # number of injector elements
n_inlet = 3 # number of tangential inlets per element

nozzle_opening_coeff = 1.2

Cd_inlet = 0.7

### Import data
ubar_an_data = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\ubar_an.csv", delimiter=",", names=True)
phi_data = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\phi.csv", delimiter=",", names=True)
mu_data= np.genfromtxt("coax_swirl_sizing\\bazarov_data\\mu.csv", delimiter=",", names=True)
alpha_data = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\alpha.csv", delimiter=",", names=True)

### Sizer

def phi_eq_func(phi, A_eq):
    return A_eq - np.sqrt(2) * (1 - phi) / (phi ** 1.5)

## Calculate initial values
mdot = mdot_total / n_elements

dp_nozzle = target_dp * 0.5

fluid = Fluid(FluidsList.Ethanol)
fluid.update(Input.temperature(T), Input.pressure(pc))

alpha_diff = 1
dp_diff = 1

n = 0

## Step 1 - calculate geometric characteristic parameter (A) and nozzle flow coefficient (mu)
A = np.interp(alpha, alpha_data["y"], alpha_data["x"]) # (geometric characteristic parameter)

while (alpha_diff > 1e-4) and (dp_diff > 1e-4):
    mu = np.interp(A, mu_data["x"], mu_data["y"]) # (nozzle flow coefficient)
    phi = np.interp(A, phi_data["x"], phi_data["y"]) # (filling efficiency)

    ## Step 2 - calculate nozzle radius
    R_n = np.sqrt(mdot / (np.pi * mu * np.sqrt(2 * rho * (dp_nozzle)))) # (R_n) - eq 103
    R_n_old = R_n # save old value of R_n for convergence check

    ## Step 3 - calculate inlet orifice radius
    R_in = R_n * nozzle_opening_coeff # (inlet arm radius) - distance from centreline of inlet orifice to the axis of the swirler
    r_in = np.sqrt(R_in * R_n / (n_inlet * A)) # (inlet orifice radius) - eq 104

    ## Step 4 - calculate following parameters:
    # l_in - inlet orifice length ------------ Bazarov: usually l_in = (3 – 6) x r_in 
    # l_n - nozzle length -------------------- Bazarov: l_n = (0.5 - 2) x R_n
    # l_s - vortex chamber length ------------ Bazarov: l_s > 2 x R_n
    # R_s - vortex chamber radius ------------ Bazarov: R_s = R_in + r_in

    # l_in = 1e-3 # assuming 1mm for now (depends on wall thickess of the element)
    l_in = 3 * r_in
    l_n = 1.25 * R_n
    l_s = 3 * R_n
    R_s = R_in + r_in

    ## Step 5 - calculate Reynolds number and friction coefficient in inlet passage 
    Re_in = (1/np.pi) * mdot / (np.sqrt(n_inlet) * r_in * rho * fluid.kinematic_viscosity)
    friction_coeff = 0.3164 * Re_in**(-0.25)

    ## Step 6 - calculate A_eq (eq 100), find mu_eq and alpha_eq using this
    A_eq = R_in * R_n / ((n_inlet * (r_in**2)) + 0.5 * friction_coeff * R_in * (R_in - R_n)) # eq 100
    mu_eq = np.interp(A_eq, mu_data["x"], mu_data["y"])
    alpha_eq = np.interp(A_eq, alpha_data["x"], alpha_data["y"])

    ## Step 7 - calculate hydraulic-loss coefficient in inlet passages
    passage_tilting_angle = 90 - (np.arctan(R_s / l_in) * 180 / np.pi) # eq 105
    # passage_tilting_angle = 90

    # fig 25 - y1 = 0.9, y2 = 0.5, x1 = 30, x2 = 90
    hydraulic_loss_coeff_inlet = np.interp(passage_tilting_angle, [30, 90], [0.9, 0.5]) # using fig 25
    hydraulic_loss_coeff = hydraulic_loss_coeff_inlet + friction_coeff * l_in / (2 * r_in)

    ## Step 8 - calculate actual flow coefficient (mu_i) using eq 99
    phi_eq = scipy.optimize.root_scalar(phi_eq_func, bracket=[0.01, 1], args=(A_eq), method='brentq').root
    # phi_eq = scipy.optimize.fsolve(phi_eq_func, 0.5, (A_eq))[0]
    mu_eq = (phi_eq ** 1.5) / np.sqrt(2 - phi_eq)
    mu_i = mu_eq / np.sqrt(1 + hydraulic_loss_coeff * ((mu_eq * A_eq * R_n / R_in) ** 2)) # eq 99

    ## Step 9 - calculate new nozzle radius using mu_i
    R_n = np.sqrt(mdot / (np.pi * mu_i * np.sqrt(2 * rho * (dp_nozzle)))) # (R_n) - eq 103

    ## Step 10 - calculate new A value using new R_n
    # A_new = R_in * R_n / (n_inlet * (r_in ** 2))

    A_new = A / (alpha_eq / alpha)

    alpha_diff = abs((alpha - alpha_eq) / alpha)

    # print(f"Relative difference: {rel_diff*1e2:.3f} %")

    A = A_new # update A for next iteration

    dp_inlet = (mdot / (n_inlet * np.pi * Cd_inlet * r_in**2))**2 / (2 * rho) # bar
    total_dp = dp_nozzle + dp_inlet
    stiffness = total_dp / pc

    dp_diff = abs((target_dp - total_dp) / target_dp)

    dp_nozzle *= target_dp / total_dp

    n += 1

print(f"Sizing converged in {n} iterations")

# Calculate gas core radius

r_gas_core = np.sqrt(1 - phi_eq) * R_n
nozzle_film_thickness = R_n - r_gas_core

print("="*60)
print(f"{'Element Parameters':^60}")
print("="*60)
print(f"{'Geometric characteristic parameter (A)':<44}: {A:>8.3f}")
print(f"{'Equivalent geometric parameter (A_eq)':<44}: {A_eq:>8.3f}")
print(f"{'Nozzle flow coefficient (mu)':<44}: {mu:>8.3f}")
print(f"{'Equivalent nozzle flow coefficient (mu_eq)':<44}: {mu_eq:>8.3f}")
print(f"{'Actual flow coefficient (mu_i)':<44}: {mu_i:>8.3f}")
print(f"{'Filling efficiency (phi)':<44}: {phi:>8.3f}")
print(f"{'Equivalent filling efficiency (phi_eq)':<44}: {phi_eq:>8.3f}")
print(f"{'Gas core radius (r_gas_core)':<44}: {r_gas_core*1e3:>8.3f} mm")
print(f"{'Nozzle film thickness':<44}: {nozzle_film_thickness*1e3:>8.3f} mm")
print(f"{'Target spray half angle (alpha)':<44}: {alpha:>8.2f} deg")
print(f"{'Actual spray half angle (alpha_eq)':<44}: {alpha_eq:>8.2f} deg")
print(f"{'Passage tilting angle':<44}: {passage_tilting_angle:>8.2f} deg")
print(f"{'Reynolds number in inlet passage':<44}: {Re_in:>8.0f}")
print(f"{'Friction coefficient':<44}: {friction_coeff:>8.3f}")
print(f"{'Hydraulic-loss coefficient (inlet)':<44}: {hydraulic_loss_coeff_inlet:>8.3f}")
print(f"{'Total hydraulic loss coefficient':<44}: {hydraulic_loss_coeff:>8.3f}")

print("\n" + "="*60)
print(f"{'Dimensions':^60}")
print("="*60)
print(f"{'Nozzle radius (R_n)':<44}: {R_n*1e3:>8.3f} mm")
print(f"{'Inlet orifice radius (r_in)':<44}: {r_in*1e3:>8.3f} mm")
print(f"{'Inlet orifice length (l_in)':<44}: {l_in*1e3:>8.3f} mm")
print(f"{'Inlet arm radius (R_in)':<44}: {R_in*1e3:>8.3f} mm")
print(f"{'Nozzle length (l_n)':<44}: {l_n*1e3:>8.3f} mm")
print(f"{'Vortex chamber length (l_s)':<44}: {l_s*1e3:>8.3f} mm")
print(f"{'Vortex chamber radius (R_s)':<44}: {R_s*1e3:>8.3f} mm")

print("\n" + "="*60)
print(f"{'Pressure Drops':^60}")
print("="*60)
print(f"{'Inlet pressure drop (dp_inlet)':<44}: {dp_inlet/1e5:>8.3f} bar")
print(f"{'Nozzle pressure drop (dp_nozzle)':<44}: {dp_nozzle/1e5:>8.3f} bar")
print(f"{'Total pressure drop (total_dp)':<44}: {total_dp/1e5:>8.3f} bar")
print(f"{'Total Injector stiffness':<44}: {1e2*stiffness:>8.2f} %")
print(f"{'Inlet / Total dP ratio':<44}: {dp_inlet/total_dp:>8.3f}")

print(f"Total orifice area: {(n_inlet * np.pi * r_in**2 * 1e6):.3f} mm^2")