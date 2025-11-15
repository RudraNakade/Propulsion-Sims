import numpy as np

od = 130e-3
h = 850e-3

V_cyl = np.pi * (od/2)**2 * h

m = 6.35 # kg

rho = m / V_cyl

print(f"Cyl Volume: {V_cyl*1e3:.2f} L")
print(f"Cyl Density: {rho:.2f} kg/m^3")