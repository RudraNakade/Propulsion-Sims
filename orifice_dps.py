from matplotlib.pylab import f
import numpy as np
from os import system

system('cls')

d_restriction = 3.3 # mm
d_pipe = 3.5 # mm
ox_hole_d = 0.8 # mm
ox_hole_n = 24

Cd_low_dp = 0.4
Cd_inj = 0.4

rho = 860 # kg/m^3
mdot = 0.177 # kg/s

A_restriction = 0.25e-6 * np.pi * d_restriction**2
A_inj = 0.25e-6 * np.pi * (ox_hole_d)**2 * ox_hole_n

CdA_1 = A_restriction * Cd_low_dp
CdA_2 = A_inj * Cd_inj
CdA_3 = (CdA_1**-2 + CdA_2**-2)**-0.5

beta = d_restriction/d_pipe
A3 = CdA_3 / Cd_inj

dp_restriction = (((np.sqrt(1 - beta**4) * mdot) / (Cd_low_dp * A_restriction))**2) / (2 * rho)
dp_inj = ((mdot / (Cd_inj * A_inj))**2) / (2 * rho)
dp_3 = ((mdot / (CdA_3))**2) / (2 * rho)

print(f'beta            = {beta:.3f}') 
print(f'restriction dp  = {dp_restriction/1e5:.3f} bar')
print(f'inj dp          = {dp_inj/1e5:.3f} bar')
print(f'total dp        = {(dp_restriction + dp_inj)/1e5:.3f} bar')
print(f'ratio           = {dp_restriction/dp_inj:.3f}')

print(f'Orifice CdA     = {CdA_1:.3e} m^2')
print(f'Injector CdA    = {CdA_2:.3e} m^2')
print(f'Combined CdA    = {CdA_3:.3e} m^2')