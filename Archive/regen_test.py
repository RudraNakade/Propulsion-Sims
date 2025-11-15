import numpy as np

mdot = 0.15
v = 7.932
rho = 786

area = mdot / (rho * v) # = 0.25 * pi * (od**2 - id**2)

id = (45 + 2*(0.8)) * 1e-3
od = np.sqrt((4 * area) / np.pi + id**2)
gap = (od - id) / 2

print(f"OD = {od*1e3:.4f} mm")
print(f"Gap = {gap*1e3:.4f} mm")

Cd = 0.7
CdA = 75.208e-6
d_up = 25.4e-3 * (0.5 - 2*0.036)
mdot = 2.4259
rho = 935.62
dp = 5.56e5

# dp = (((np.sqrt(1 - (d/d_up)**4) * mdot) / (Cd * A))**2) / (2 * rho)

beta = 2 * rho * dp / (CdA**2)
d = d_up / beta

print(f"Diameter = {d*1e3:.4f} mm")