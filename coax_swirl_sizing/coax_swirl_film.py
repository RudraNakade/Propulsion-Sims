import numpy as np

dp = 9e5
mdot = 0.61379
rho = 786

film_perc = 0.2 # 20%

film_mdot = film_perc * mdot

film_CdA = film_mdot / np.sqrt(2 * rho * dp)

Cd = 0.65

film_A = film_CdA / Cd

hole_A = np.pi * 0.25 * (0.4e-3) ** 2  # 0.4 mm diameter
n_holes = film_A / hole_A

print(f"Number of holes: {n_holes:.2f}")