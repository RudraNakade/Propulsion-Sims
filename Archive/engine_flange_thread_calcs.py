import numpy as np
from os import system

system('cls')

shear_strength = 200 # AlSi10Mg shear strength lower bound

tensile_yield_strength = 450 # A2-70
tensile_ultimate_strength = 700 # A2-70
yield_strength = tensile_yield_strength

bolt_force = 5000 / 12

thread_major_d = 3
thread_minor_d = 2.53
thread_pitch = 0.5

engagement_length = 3

threads_engaged = engagement_length / thread_pitch

shear_area = np.pi * 0.25 * (thread_major_d**2 - thread_minor_d**2) * threads_engaged

shear_force = bolt_force / shear_area
thread_shear_sf = shear_strength / shear_force

tensile_area = np.pi * 0.25 * thread_major_d**2
tensile_stress = bolt_force / tensile_area
tensile_yield_sf = tensile_yield_strength / tensile_stress
tensile_ultimate_sf = tensile_ultimate_strength / tensile_stress

print(f"Thread Shear Area: {shear_area:.2f} mm^2")
print(f"Thread Tensile Area: {tensile_area:.2f} mm^2\n")
print(f"Bolt Shear Stress: {shear_force:.2f} MPa")
print(f"Bolt Tensile Stress: {tensile_stress:.2f} MPa\n")
print(f"Thread Shear Safety Factor: {thread_shear_sf:.2f}")
print(f"Bolt Tensile Yield Safety Factor: {tensile_yield_sf:.2f}")
print(f"Bolt Tensile Ultimate Safety Factor: {tensile_ultimate_sf:.2f}")