from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

cea = CEA_Obj(
    oxName='N2O',
    fuelName='Isopropanol',
    isp_units='sec',
    cstar_units='m/s',
    pressure_units='Pa',
    temperature_units='K',
    make_debug_prints=False
)

actual_pc = 30e5

# a = -51.5878
# b = 428.9313
# c = 685.0469

a = -46.5673
b = 405.6106
c = 707.9877

def cubic_equation(OF, a, b, c, Pc, mdot_f_ov_At):
    return a * OF**3 + (a + b) * OF**2 + (b + c) * OF + (c - (Pc / mdot_f_ov_At))

def solve_cubic(a, b, c, d):
    """
    Solve cubic equation: a*x^3 + b*x^2 + c*x + d = 0
    Returns the real root(s) of the equation.
    Uses Cardano's formula for solving cubic equations.
    """
    # Normalize coefficients
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for cubic equation")
    
    b = b / a
    c = c / a
    d = d / a
    
    # Calculate intermediate values
    p = c - (b**2) / 3
    q = 2 * (b**3) / 27 - (b * c) / 3 + d
    
    # Calculate discriminant
    discriminant = (q**2) / 4 + (p**3) / 27
    
    if discriminant > 0:
        # One real root
        u = np.cbrt(-q/2 + np.sqrt(discriminant))
        v = np.cbrt(-q/2 - np.sqrt(discriminant))
        x = u + v - b/3
        return x
    elif discriminant == 0:
        # Multiple roots (return the distinct real root)
        u = np.cbrt(-q/2)
        x1 = 2*u - b/3
        x2 = -u - b/3
        # Return the positive root in the expected range
        roots = [x1, x2]
        valid_roots = [r for r in roots if 1 <= r <= 6]
        if valid_roots:
            return valid_roots[0]
        return x1
    else:
        # Three real roots
        r = np.sqrt(-(p**3) / 27)
        theta = np.arccos(-q / (2 * r))
        r = 2 * np.cbrt(r)
        
        x1 = r * np.cos(theta / 3) - b/3
        x2 = r * np.cos((theta + 2*np.pi) / 3) - b/3
        x3 = r * np.cos((theta + 4*np.pi) / 3) - b/3
        
        # Return the root in the valid range (1 to 6)
        roots = [x1, x2, x3]
        valid_roots = [r for r in roots if 1 <= r <= 6]
        if valid_roots:
            return valid_roots[0]
        return x1

def solve_OF(actual_OF: float, actual_pc: float, a: float, b: float, c: float) -> float:
    """
    Solve for O/F ratio using the cubic equation.
    
    Parameters
    ----------
    actual_OF : float
        Actual O/F ratio (used to calculate mdot_f/At)
    actual_pc : float
        Actual chamber pressure in Pa
    a, b, c : float
        Cubic equation coefficients
        
    Returns
    -------
    float
        Solved O/F ratio
    """
    # Calculate actual mdot_f/At from actual conditions
    cstar = cea.get_Cstar(Pc=actual_pc, MR=actual_OF)
    actual_mdot_ov_At = actual_pc / cstar
    actual_mdot_o_ov_At = (actual_OF / (1 + actual_OF)) * actual_mdot_ov_At
    actual_mdot_f_ov_At = actual_mdot_ov_At / (1 + actual_OF)
    
    # Solve cubic equation directly using Cardano's formula
    # Equation: a*OF^3 + (a+b)*OF^2 + (b+c)*OF + (c - Pc/mdot_f) = 0
    coeff_a = a
    coeff_b = a + b
    coeff_c = b + c
    coeff_d = c - (actual_pc / actual_mdot_f_ov_At)
    
    solved_OF = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d)
    
    calc_mdot_o_ov_At = actual_mdot_f_ov_At * solved_OF

    target_OF = 3

    demand_mdot_f_ov_At = calc_mdot_o_ov_At / target_OF
    required_mdot_f_ov_At = actual_mdot_o_ov_At / target_OF

    return solved_OF, calc_mdot_o_ov_At, demand_mdot_f_ov_At, required_mdot_f_ov_At, actual_mdot_f_ov_At

# Test multiple O/F values
test_OFs = np.linspace(1, 5, 30)
solved_OFs = []
failed_OFs = []
percent_errors = []
calc_mdot_o_ov_At_list = []
demand_mdot_f_ov_At_list = []
required_mdot_f_ov_At_list = []
actual_mdot_f_ov_At_list = []

print("\n" + "="*60)
print("Testing Multiple O/F Values")
print("="*60)

for test_OF in test_OFs:
    try:
        solved, calc_mdot_o_ov_At, demand_mdot_f_ov_At, required_mdot_f_ov_At, actual_mdot_f_ov_At = solve_OF(test_OF, actual_pc, a, b, c)
        solved_OFs.append(solved)
        calc_mdot_o_ov_At_list.append(calc_mdot_o_ov_At)
        demand_mdot_f_ov_At_list.append(demand_mdot_f_ov_At)
        required_mdot_f_ov_At_list.append(required_mdot_f_ov_At)
        actual_mdot_f_ov_At_list.append(actual_mdot_f_ov_At)
        error = abs((solved - test_OF) / test_OF) * 100
        percent_errors.append(error)
        print(f"Test O/F: {test_OF:.4f} -> Solved: {solved:.4f} (Error: {error:.2f}%)")
    except Exception as e:
        print(f"Test O/F: {test_OF:.4f} -> FAILED: {e}")
        failed_OFs.append(test_OF)
        solved_OFs.append(np.nan)
        percent_errors.append(np.nan)

print("="*60)
print(f"Successful: {len(test_OFs) - len(failed_OFs)}/{len(test_OFs)}")
print(f"Failed: {len(failed_OFs)}/{len(test_OFs)}")
print("="*60)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'OF Ratio Solver Test - c* = a*OF^2 + b*OF + c')

# Plot 1: Solved vs Test O/F
# Add tinted background for expected operating range
ax1.axvspan(2.5, 3.5, alpha=0.2, color='green', label='Expected Operating Range')

# Filter out NaN values for plotting
valid_mask = ~np.isnan(solved_OFs)
test_OFs_valid = test_OFs[valid_mask]
solved_OFs_valid = np.array(solved_OFs)[valid_mask]

# Plot solved vs actual
ax1.plot(test_OFs_valid, solved_OFs_valid, 'bo-', label='Solved O/F')

# Plot ideal line (y=x)
of_range = np.linspace(test_OFs.min(), test_OFs.max(), 100)
ax1.plot(of_range, of_range, 'r--', label='y = x')

# Mark failed points
if len(failed_OFs) > 0:
    ax1.plot(failed_OFs, [test_OFs.min()] * len(failed_OFs), 'rx', 
            label='Failed Solutions')

ax1.set_xlabel('Test OF')
ax1.set_ylabel('Solved OF')
ax1.set_title(f'OF Solver Test\nPc = {actual_pc/1e5:.0f} bar')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.grid(which='minor', alpha=0.15)
ax1.minorticks_on()

# Plot 2: Percentage Error
ax2.axvspan(2.5, 3.5, alpha=0.2, color='green', label='Expected Operating Range')

# Filter out NaN values for error plot
percent_errors_valid = np.array(percent_errors)[valid_mask]

ax2.plot(test_OFs_valid, percent_errors_valid, 'ro-', label='% Error')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Mark failed points
if len(failed_OFs) > 0:
    ax2.plot(failed_OFs, [0] * len(failed_OFs), 'rx', 
            label='Failed Solutions')

ax2.set_xlabel('Test OF')
ax2.set_ylabel('Percentage Error (%)')
ax2.set_title(f'OF Solver Error\nPc = {actual_pc/1e5:.0f} bar')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.grid(which='minor', alpha=0.15)
ax2.minorticks_on()

# Plot 3: Demand and Required mdot_f/At
ax3.axvspan(2.5, 3.5, alpha=0.2, color='green', label='Expected Operating Range')

# Filter out NaN values
demand_mdot_f_ov_At_valid = np.array(demand_mdot_f_ov_At_list)[valid_mask]
required_mdot_f_ov_At_valid = np.array(required_mdot_f_ov_At_list)[valid_mask]

ax3.plot(test_OFs_valid, demand_mdot_f_ov_At_valid, 'b-o', label='Demand mdot_f/At')
ax3.plot(test_OFs_valid, required_mdot_f_ov_At_valid, 'r-o', label='Required mdot_f/At')
ax3.plot(test_OFs_valid, actual_mdot_f_ov_At_list, 'g-o', label='Input mdot_f/At')

# Mark failed points
if len(failed_OFs) > 0:
    ax3.plot(failed_OFs, [0] * len(failed_OFs), 'rx', 
            label='Failed Solutions')

ax3.set_xlabel('Test OF')
ax3.set_ylabel('mdot_f / At (kg/s per m²)')
ax3.set_title(f'Fuel mdot / unit throat area\nPc = {actual_pc/1e5:.0f} bar')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.grid(which='minor', alpha=0.15)
ax3.minorticks_on()

plt.tight_layout()
plt.show()
