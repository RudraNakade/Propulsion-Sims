import matplotlib.pyplot as plt
import numpy as np

a = -51.5878
b = 428.9313
c = 685.0469

Dt = 42.41e-3
At = np.pi * (Dt / 2)**2
mdot_f = 0.76

Pc = 30e5
mdot_f_ov_At = mdot_f / At

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
    
    b_norm = b / a
    c_norm = c / a
    d_norm = d / a
    
    # Calculate intermediate values
    p = c_norm - (b_norm**2) / 3
    q = 2 * (b_norm**3) / 27 - (b_norm * c_norm) / 3 + d_norm
    
    # Calculate discriminant
    discriminant = (q**2) / 4 + (p**3) / 27
    
    if discriminant > 0:
        # One real root
        u = np.cbrt(-q/2 + np.sqrt(discriminant))
        v = np.cbrt(-q/2 - np.sqrt(discriminant))
        x = u + v - b_norm/3
        return x
    elif discriminant == 0:
        # Multiple roots (return the distinct real root)
        u = np.cbrt(-q/2)
        x1 = 2*u - b_norm/3
        x2 = -u - b_norm/3
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
        
        x1 = r * np.cos(theta / 3) - b_norm/3
        x2 = r * np.cos((theta + 2*np.pi) / 3) - b_norm/3
        x3 = r * np.cos((theta + 4*np.pi) / 3) - b_norm/3
        
        # Return the root in the valid range (1 to 6)
        roots = [x1, x2, x3]
        valid_roots = [r for r in roots if 1 <= r <= 6]
        if valid_roots:
            return valid_roots[0]
        return x1

# Plot the cubic equation
OF_min = -10
OF_max = 10
OF_range = np.linspace(OF_min, OF_max, 1000)
y_values = cubic_equation(OF_range, a, b, c, Pc, mdot_f_ov_At)

# Solve for the root
coeff_a = a
coeff_b = a + b
coeff_c = b + c
coeff_d = c - (Pc / mdot_f_ov_At)

solved_OF = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d)
print(f"Solved O/F: {solved_OF:.4f}")

plt.figure(figsize=(12, 8))
plt.plot(OF_range, y_values, 'b-')
plt.axhline(y=0, color='r', linestyle='--', label='y=0')
plt.axvline(x=solved_OF, color='g', linestyle='--', label=f'Solved O/F = {solved_OF:.4f}')
plt.xlabel('O/F Ratio')
plt.ylabel('f(O/F)')
plt.title(f'Cubic Equation vs O/F Ratio\nPc = {Pc/1e5:.0f} bar, mdot_f/At = {mdot_f_ov_At:.1f}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot for array of mdot_f values
mdot_f_array = np.linspace(0.2, 1, 10)

plt.figure(figsize=(12, 8))

for mdot_f_val in mdot_f_array:
    mdot_f_ov_At_val = mdot_f_val / At
    y_vals = cubic_equation(OF_range, a, b, c, Pc, mdot_f_ov_At_val)
    
    # Solve for this mdot_f
    coeff_d_val = c - (Pc / mdot_f_ov_At_val)
    solved_OF_val = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d_val)
    
    plt.plot(OF_range, y_vals, label=f'mdot_f = {mdot_f_val:.2f} kg/s (OF = {solved_OF_val:.2f})')

plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('O/F Ratio')
plt.ylabel('f(O/F)')
plt.title(f'Cubic Equation for Different Fuel Flow Rates\nPc = {Pc/1e5:.0f} bar')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
