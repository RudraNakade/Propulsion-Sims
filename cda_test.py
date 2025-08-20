import numpy as np
from os import system

system('cls')

mdot = 0.760
pc = 31.8e5
rho = 790

annulus_id = 17.27e-3
annulus_od = 18.06e-3
annulus_area_init = np.pi * (annulus_od**2 - annulus_id**2) * 0.25

film_Cd = 0.65
film_d = 0.4e-3
film_n = 42
film_area_init = 0.25 * np.pi * (film_d**2) * film_n
film_CdA_init = film_area_init * film_Cd

total_CdA_init = 17.4e-6
total_Cd_init = total_CdA_init / (annulus_area_init + film_area_init)

annulus_CdA_init = total_CdA_init - film_Cd * film_area_init
annulus_Cd_init = annulus_CdA_init / annulus_area_init

film_frac_init = film_CdA_init / annulus_CdA_init

annulus_Cd_min = annulus_Cd_init
annulus_Cd_max = 0.75

annulus_id = 17.27e-3
annulus_od = 18.25e-3
annulus_area = np.pi * (annulus_od**2 - annulus_id**2) * 0.25

film_Cd = 0.65
film_d = 0.4e-3
film_n = 56
film_area = 0.25 * np.pi * (film_d**2) * film_n
film_CdA = film_area * film_Cd

annulus_CdA_min = annulus_area * annulus_Cd_min
total_CdA_min = annulus_CdA_min + film_CdA
total_Cd_min = total_CdA_min / (annulus_area + film_area)
dP_max = (mdot / total_CdA_min)**2 / (2 * rho) # Pa
stiffness_max = dP_max / pc
film_frac_max = film_CdA / annulus_CdA_min

annulus_CdA_max = annulus_area * annulus_Cd_max
total_CdA_max = annulus_CdA_max + film_CdA
total_Cd_max = total_CdA_max / (annulus_area + film_area)
dP_min = (mdot / total_CdA_max)**2 / (2 * rho) # Pa
stiffness_min = dP_min / pc
film_frac_min = film_CdA / annulus_CdA_max

dP_init = (mdot / total_CdA_init)**2 / (2 * rho) # Pa
stiffness_init = dP_init / pc

print("="*60)
print(f"Initial values:")
print(f"Cd's: Total: {total_Cd_init:.6f}, Annulus: {annulus_Cd_init:.6f}, Film: {film_Cd:.6f}")
print(f"dP: {dP_init/1e5:.4f} bar, Stiffness: {stiffness_init:.2%}")
print(f"Film cooling: {film_frac_init:.2%}")

print("\n" + "="*60)
print(f"Resized (min Cd):")
print(f"Cd's: Total: {total_Cd_min:.6f}, Annulus: {annulus_Cd_min:.6f}, Film: {film_Cd:.6f}")
print(f"dP: {dP_max/1e5:.4f} bar, Stiffness: {stiffness_max:.2%}")
print(f"Film cooling: {film_frac_max:.2%}")

print("\n" + "="*60)
print(f"Resized (max Cd):")
print(f"Cd's: Total: {total_Cd_max:.6f}, Annulus: {annulus_Cd_max:.6f}, Film: {film_Cd:.6f}")
print(f"dP: {dP_min/1e5:.4f} bar, Stiffness: {stiffness_min:.2%}")
print(f"Film cooling: {film_frac_min:.2%}")