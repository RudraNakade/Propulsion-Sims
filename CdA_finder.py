import numpy as np
from os import system
# system("cls")

calc_CdA = lambda mdot, dP, rho: mdot / np.sqrt(2 * rho * dP)
calc_mdot = lambda CdA, dP, rho: CdA * np.sqrt(2 * rho * dP)

rho = 1000 # kg/m^3
tank_vol = 0.25 * np.pi * 44**2 * 230 * 1e-9 # m^3
flow_time = 8*60 + 45 # seconds

mdot = rho * tank_vol / flow_time
dP = 6.4e5 # Pa

id = 0.2e-3 # m
A = np.pi * (id / 2) ** 2 # m^2

CdA = calc_CdA(mdot, dP, rho)
Cd = CdA / A

dP_firing = (25-11.80)*1e5 # Pa, pressure drop during firing
# dP_firing = (25-1)*1e5 # Pa
rho_ipa = 786 # kg/m^3
mdot_firing = calc_mdot(CdA, dP_firing, rho_ipa)
mdot_flowtest = calc_mdot(CdA, dP, rho)

ox_A = 4.3982297150257107e-07 # m^2
ox_Cd = 0.5
ox_CdA = Cd * A

ox_flow_time = tank_vol * rho / mdot_flowtest

print("CdA Calculated Results:")
print(f"{'CdA:':<5} {CdA:.5e} mm^2")
print(f"{'Cd:':<5} {Cd:.5f}")

print(f"{'mdot:':<5} {mdot_firing*1e3:.4f} g/s")

print(f"\nOx estimated flow time: {ox_flow_time:.2f} s")