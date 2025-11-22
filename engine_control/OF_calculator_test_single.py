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
cstar_constant = 1504.71

def solve_OF(actual_OF: float, actual_pc: float) -> float:
    """
    Solve for O/F ratio using the cubic equation.
    
    Parameters
    ----------
    actual_OF : float
        Actual O/F ratio (used to calculate mdot_f/At)
    actual_pc : float
        Actual chamber pressure in Pa
        
    Returns
    -------
    float
        Solved O/F ratio
    """
    # Calculate actual mdot_f/At from actual conditions
    cstar_actual = cea.get_Cstar(Pc=actual_pc, MR=actual_OF)
    actual_mdot_ov_At = actual_pc / cstar_actual
    actual_mdot_o_ov_At = (actual_OF / (1 + actual_OF)) * actual_mdot_ov_At
    actual_mdot_f_ov_At = actual_mdot_ov_At / (1 + actual_OF)
    
    cstar_assumed = cstar_constant
    
    calc_mdot_ov_At = actual_pc / cstar_assumed
    calc_mdot_o_ov_At = calc_mdot_ov_At - actual_mdot_f_ov_At

    calc_OF = calc_mdot_o_ov_At / actual_mdot_f_ov_At

    target_OF = 3

    demand_mdot_f_ov_At = calc_mdot_o_ov_At / target_OF
    required_mdot_f_ov_At = actual_mdot_o_ov_At / target_OF

    return calc_OF, calc_mdot_o_ov_At, demand_mdot_f_ov_At, required_mdot_f_ov_At, actual_mdot_f_ov_At

# Test multiple O/F values
test_OFs = np.linspace(1.5, 5, 30)
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
        solved, calc_mdot_o_ov_At, demand_mdot_f_ov_At, required_mdot_f_ov_At, actual_mdot_f_ov_At = solve_OF(test_OF, actual_pc)
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
        calc_mdot_o_ov_At_list.append(np.nan)
        demand_mdot_f_ov_At_list.append(np.nan)
        required_mdot_f_ov_At_list.append(np.nan)
        actual_mdot_f_ov_At_list.append(np.nan)
        percent_errors.append(np.nan)

print("="*60)
print(f"Successful: {len(test_OFs) - len(failed_OFs)}/{len(test_OFs)}")
print(f"Failed: {len(failed_OFs)}/{len(test_OFs)}")
print("="*60)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'OF Ratio Solver Test - Assumed constant c* = {cstar_constant:.2f} m/s')

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

ax1.set_xlabel('Actual OF')
ax1.set_ylabel('Solved OF')
ax1.set_title(f'Solved OF vs Actual OF\nPc = {actual_pc/1e5:.0f} bar')
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

ax2.set_xlabel('Actual OF')
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
actual_mdot_f_ov_At_valid = np.array(actual_mdot_f_ov_At_list)[valid_mask]

ax3.plot(test_OFs_valid, demand_mdot_f_ov_At_valid, 'b-o', label='Demand mdot_f/At (using calculated mdot_o for target OF = 3)')
ax3.plot(test_OFs_valid, required_mdot_f_ov_At_valid, 'r-o', label='Required mdot_f/At (using real mdot_o for target OF = 3)')
ax3.plot(test_OFs_valid, actual_mdot_f_ov_At_valid, 'g-o', label='Input mdot_f/At')

# Mark failed points
if len(failed_OFs) > 0:
    ax3.plot(failed_OFs, [0] * len(failed_OFs), 'rx', 
            label='Failed Solutions')

ax3.set_xlabel('Actual OF')
ax3.set_ylabel('mdot_f / At (kg/s per m²)')
ax3.set_title(f'Fuel mdot / unit throat area\nPc = {actual_pc/1e5:.0f} bar')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.grid(which='minor', alpha=0.15)
ax3.minorticks_on()

plt.tight_layout()
plt.show()
