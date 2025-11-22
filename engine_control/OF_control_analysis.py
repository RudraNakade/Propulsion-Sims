from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters
pc_min = 20e5  # Pa
pc_max = 35e5  # Pa
n_points = 50

OF_ratios = np.arange(2, 5, 1)

pc_array = np.linspace(pc_min, pc_max, n_points)

# Initialize CEA object
cea = CEA_Obj(
    oxName='N2O',
    fuelName='Isopropanol',
    isp_units='sec',
    cstar_units='m/s',
    pressure_units='Pa',
    temperature_units='K',
    make_debug_prints=False
)

# Store results
cstar_results = np.zeros((len(OF_ratios), n_points))

for i, OF in enumerate(OF_ratios):
    for j, pc in enumerate(pc_array):
        cstar_results[i, j] = cea.get_Cstar(Pc=pc, MR=OF)

# Plot results
fig, ax = plt.subplots(figsize=(12, 8))

for i, OF in enumerate(OF_ratios):
    ax.plot(pc_array/1e5, cstar_results[i, :], 
            label=f'O/F = {OF:.1f}')

ax.set_xlabel('Pc (bar)')
ax.set_ylabel('c* (m/s)')
ax.set_title('c* vs Pc')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.grid(which='minor', alpha=0.15)
ax.minorticks_on()

plt.tight_layout()

# Print average c* for each OF ratio
print("\n" + "="*60)
print("Average c* Values")
print("="*60)
for i, OF in enumerate(OF_ratios):
    avg_cstar = np.mean(cstar_results[i, :])
    print(f"O/F = {OF:.1f}: Average c* = {avg_cstar:.2f} m/s")
print("="*60)

OF_array = np.linspace(1.5, 4.5, 50)
pc_low = 20e5
pc_high = 35e5
pc_step = 5e5
    
pc_values = np.arange(pc_low, pc_high + pc_step, pc_step)

cstar_vs_OF = np.zeros((len(pc_values), len(OF_array)))

for i, pc in enumerate(pc_values):
    for j, OF in enumerate(OF_array):
        cstar_vs_OF[i, j] = cea.get_Cstar(Pc=pc, MR=OF)

fig2, ax2 = plt.subplots(figsize=(12, 8))

for i, pc in enumerate(pc_values):
    ax2.plot(OF_array, cstar_vs_OF[i, :], 
            label=f'Pc = {pc/1e5:.0f} bar',
            alpha=0.5)


cstar_avg = np.mean(cstar_vs_OF, axis=0)
ax2.plot(OF_array, cstar_avg, 'k--', label='Average')

ax2.set_xlabel('OF Ratio')
ax2.set_ylabel('c* (m/s)')
ax2.set_title('c* vs OF Ratio')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.grid(which='minor', alpha=0.15)
ax2.minorticks_on()

plt.tight_layout()

# Calculate R² values function
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Fit curves to average c*
def logarithmic(x, a, b, c):
    """y = a * ln(x + b) + c"""
    return a * np.log(x + b) + c

def power(x, a, b, c):
    """y = a * x^b + c"""
    return a * np.power(x, b) + c

def polynomial_2(x, a, b, c):
    """y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def square_root(x, a, b, c):
    """y = a * sqrt(x + b) + c"""
    return a * np.sqrt(x + b) + c

# Fit the curves with error handling
fits_successful = {}

try:
    popt_log, _ = curve_fit(logarithmic, OF_array, cstar_avg, p0=[50, 0.1, 1450], maxfev=2000)
    cstar_log = logarithmic(OF_array, *popt_log)
    r2_log = r_squared(cstar_avg, cstar_log)
    fits_successful['logarithmic'] = True
except RuntimeError as e:
    print(f"Warning: Logarithmic fit failed: {e}")
    fits_successful['logarithmic'] = False
    popt_log = [50, 0.1, 1450]
    cstar_log = logarithmic(OF_array, *popt_log)
    r2_log = 0

try:
    popt_pow, _ = curve_fit(power, OF_array, cstar_avg, p0=[1400, 0.05, 100], 
                            bounds=([0, 0, 0], [2000, 1, 1000]), maxfev=3000)
    cstar_pow = power(OF_array, *popt_pow)
    r2_pow = r_squared(cstar_avg, cstar_pow)
    fits_successful['power'] = True
except RuntimeError as e:
    print(f"Warning: Power fit failed: {e}")
    fits_successful['power'] = False
    popt_pow = [1400, 0.05, 100]
    cstar_pow = power(OF_array, *popt_pow)
    r2_pow = 0

try:
    popt_poly2, _ = curve_fit(polynomial_2, OF_array, cstar_avg, p0=[-5, 30, 1450], maxfev=2000)
    cstar_poly2 = polynomial_2(OF_array, *popt_poly2)
    r2_poly2 = r_squared(cstar_avg, cstar_poly2)
    fits_successful['polynomial'] = True
except RuntimeError as e:
    print(f"Warning: Polynomial fit failed: {e}")
    fits_successful['polynomial'] = False
    popt_poly2 = [-5, 30, 1450]
    cstar_poly2 = polynomial_2(OF_array, *popt_poly2)
    r2_poly2 = 0

try:
    popt_sqrt, _ = curve_fit(square_root, OF_array, cstar_avg, p0=[150, 0.1, 1300], maxfev=2000)
    cstar_sqrt = square_root(OF_array, *popt_sqrt)
    r2_sqrt = r_squared(cstar_avg, cstar_sqrt)
    fits_successful['square_root'] = True
except RuntimeError as e:
    print(f"Warning: Square root fit failed: {e}")
    fits_successful['square_root'] = False
    popt_sqrt = [150, 0.1, 1300]
    cstar_sqrt = square_root(OF_array, *popt_sqrt)
    r2_sqrt = 0

# Plot curve fits
fig3, ax3 = plt.subplots(figsize=(14, 10))

ax3.plot(OF_array, cstar_avg, 'ko-', label='Average c* (Data)', zorder=10)
ax3.plot(OF_array, cstar_log, '-', label=f'Logarithmic: R²={r2_log:.6f}')
ax3.plot(OF_array, cstar_pow, '-', label=f'Power: R²={r2_pow:.6f}')
ax3.plot(OF_array, cstar_poly2, '-', label=f'Polynomial (2nd): R²={r2_poly2:.6f}')
ax3.plot(OF_array, cstar_sqrt, '-', label=f'Square Root: R²={r2_sqrt:.6f}')

ax3.set_xlabel('OF Ratio')
ax3.set_ylabel('c* (m/s)')
ax3.set_title('c* vs OF Ratio Curve Fitting')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.grid(which='minor', alpha=0.15)
ax3.minorticks_on()

plt.tight_layout()

# Print fit parameters
print("\n" + "="*60)
print("Curve Fit Parameters")
print("="*60)
if fits_successful['logarithmic']:
    print(f"\nLogarithmic: y = {popt_log[0]:.4f} * ln(x + {popt_log[1]:.4f}) + {popt_log[2]:.4f}")
    print(f"R² = {r2_log:.6f}")
else:
    print(f"\nLogarithmic: FIT FAILED (using default parameters)")
    
if fits_successful['power']:
    print(f"\nPower: y = {popt_pow[0]:.4f} * x^{popt_pow[1]:.4f} + {popt_pow[2]:.4f}")
    print(f"R² = {r2_pow:.6f}")
else:
    print(f"\nPower: FIT FAILED (using default parameters)")
    
if fits_successful['polynomial']:
    print(f"\nPolynomial (2nd): y = {popt_poly2[0]:.4f} * x² + {popt_poly2[1]:.4f} * x + {popt_poly2[2]:.4f}")
    print(f"R² = {r2_poly2:.6f}")
    
    # Calculate error between data and polynomial fit
    poly_error = cstar_avg - cstar_poly2
    poly_percent_error = (poly_error / cstar_avg) * 100
    max_percent_error = np.max(np.abs(poly_percent_error))
    max_error_idx = np.argmax(np.abs(poly_percent_error))
    
    print(f"Maximum % error: {max_percent_error:.4f}% at O/F = {OF_array[max_error_idx]:.2f}")
else:
    print(f"\nPolynomial (2nd): FIT FAILED (using default parameters)")
    
if fits_successful['square_root']:
    print(f"\nSquare Root: y = {popt_sqrt[0]:.4f} * sqrt(x + {popt_sqrt[1]:.4f}) + {popt_sqrt[2]:.4f}")
    print(f"R² = {r2_sqrt:.6f}")
else:
    print(f"\nSquare Root: FIT FAILED (using default parameters)")
print("="*60)

# Print c* at OF = 3
OF_3_idx = np.argmin(np.abs(OF_array - 3.0))
cstar_at_OF_3 = cstar_avg[OF_3_idx]
print(f"\nc* at O/F = 3.0: {cstar_at_OF_3:.2f} m/s")

plt.show()