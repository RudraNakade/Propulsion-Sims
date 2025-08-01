import numpy as np
from pyfluids import Fluid, FluidsList, Input
from scipy.optimize import root_scalar
from os import system
from two_phase_models import *

system("cls")

def calc_NHNE_mdot(tank_p, vapor_p, injector_p, chamber_p, CdA):
    NHNE_CdA = CdA # / 1.1 # 10% lower to account for combustion heat transfer in orifice
    nitrous_up = Fluid(FluidsList.NitrousOxide)
    nitrous_up_sat = Fluid(FluidsList.NitrousOxide)

    nitrous_up.update(Input.pressure(vapor_p), Input.quality(0)) # Loading at saturation
    nitrous_up = nitrous_up.isentropic_compression_to_pressure(tank_p) # Supercharge to tank pressure
    nitrous_up = nitrous_up.isenthalpic_expansion_to_pressure(injector_p)  # Isenthalpic expansion to injector pressure
    nitrous_up_sat.update(Input.temperature(nitrous_up.temperature), Input.quality(100))

    spi_mdot = spi_model(injector_p, chamber_p, nitrous_up.density, NHNE_CdA, 1)
    hem_mdot = hem_model(nitrous_up, chamber_p, NHNE_CdA, 1)
    k = np.sqrt((injector_p - chamber_p) / (nitrous_up_sat.pressure - chamber_p))
    nhne_mdot = nhne_model(spi_mdot, hem_mdot, k)
    return (nhne_mdot, hem_mdot, spi_mdot, nitrous_up.density, nitrous_up_sat.density)

def calc_NHNE_injector_p(mdot, tank_p, vapor_p, chamber_p, CdA):
    def NHNE_mdot_func(injector_p, tank_p, vapor_p, chamber_p, CdA, mdot):
        return calc_NHNE_mdot(tank_p, vapor_p, injector_p, chamber_p, CdA)[0] - mdot
    injector_p = sp.optimize.root_scalar(NHNE_mdot_func, args=(tank_p, vapor_p, chamber_p, CdA, mdot), bracket=[(chamber_p+1e3), (tank_p-1e3)]).root
    return injector_p

ox_mdot = 2.331 # kg/s
fuel_mdot = 0.761 # kg/s
total_mdot = ox_mdot + fuel_mdot
OF = ox_mdot / fuel_mdot

film = 0.20031
core_OF = OF * (1 + film)

tank_p = 75e5
tank_vapor_p = 31.2e5
chamber_p = 31.8e5

Cd = 0.75
A_inj = np.pi * 60 * (1.5e-3 / 2) ** 2
CdA = Cd * A_inj

injector_p = calc_NHNE_injector_p(ox_mdot, tank_p, tank_vapor_p, chamber_p, CdA)
nitrous_inj_density = calc_NHNE_mdot(tank_p, tank_vapor_p, injector_p, chamber_p, CdA)[3]

eff_Cd = ox_mdot / (A_inj * np.sqrt(2 * (injector_p - chamber_p) * nitrous_inj_density))

print(f"Calculated Injector Pressure: {injector_p/1e5:.4f} Bar, Effective Cd: {eff_Cd:.4f}, Cd Ratio: {eff_Cd/Cd:.4f}")
print(f"Injector Density: {nitrous_inj_density:.4f} kg/m^3")
print(f"Injector dP: {(injector_p - chamber_p)/1e5:.4f} Bar")
print(f"Injector Stiffness: {100*(injector_p - chamber_p)/chamber_p:.2f} %")
print(f"OF Ratio (Total) = {OF:.4f}")
print(f"Core OF Ratio = {core_OF:.4f}")