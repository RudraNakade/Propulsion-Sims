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