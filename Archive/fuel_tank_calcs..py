import numpy as np
from os import system
system('cls')

water_rho = 1000
ipa_rho = 786

ipa_mass = 7
water_mass = 6

total_mass = water_mass + ipa_mass

total_volume = (water_mass / water_rho) + (ipa_mass / ipa_rho)

avg_density = total_mass / total_volume

tank_volume = 7.418e-3 # m^3
ullage_frac = 0.1

fill_volume = tank_volume * (1 - ullage_frac)
fill_mass = fill_volume * avg_density

print(f"Mixed Mass: {total_mass:.2f} kg")
print(f"Total Volume: {total_volume*1e3:.2f} L")
print(f"Avg Density: {avg_density:.2f} kg/m³")
print(f"Fill Volume: {fill_volume*1e3:.2f} L")
print(f"Fill Mass: {fill_mass:.2f} kg")