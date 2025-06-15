import numpy as np
from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt
from os import system

system("cls")

def spi_model(P_up, P_down, rho_up, A_eff, Cd):
    """Single-Phase Incompressible flow model"""
    delta_p = P_up - P_down
    return Cd * A_eff * np.sqrt(2 * rho_up * delta_p)

def hem_model(nitrous_up, nitrous_down, A_eff, Cd):
    """Homogeneous Equilibrium Model"""
    return Cd * A_eff * nitrous_down.density * np.sqrt(2 * (nitrous_up.enthalpy - nitrous_down.enthalpy))

def nhne_model(mdot_spi, mdot_hem, k):
    """Non-Homogeneous Non-Equilibrium Model"""
    return (mdot_spi * k / (1 + k)) + (mdot_hem / (1 + k))

tank_p = 40e5
tank_vapor_p = 32e5
injector_p = 30e5
chamber_p = 24e5

Cd = 0.65

pintle_d = 17.27e-3
pintle_id = 12e-3
hole_d = 1.5e-3
n_holes_row = 12
n_rows = 5

inlet_area = np.pi * (pintle_id / 2) ** 2
pintle_circumference = np.pi * pintle_d
blockage_ratio = (n_holes_row * hole_d) / pintle_circumference
A_inj =  n_rows * n_holes_row * np.pi * (hole_d / 2) ** 2
# A_inj = 118.609 * 0.75 * 1e-6
A_nhne = A_inj/ 1.1

nitrous = Fluid(FluidsList.NitrousOxide) # Random nitrous instance to get properties
nitrous_tank = Fluid(FluidsList.NitrousOxide)
nitrous_injector = Fluid(FluidsList.NitrousOxide)
nitrous_chamber = Fluid(FluidsList.NitrousOxide)

# Setup nitrous tank state, modelling filling as saturated liquid and then pressurisation (supercharging)
nitrous_tank.update(Input.pressure(tank_vapor_p), Input.quality(0))  # Saturated liquid
fill_temp = nitrous_tank.temperature
nitrous_tank.update(Input.entropy(nitrous_tank.entropy), Input.pressure(tank_p)) # Supercharged liquid from saturation

# Setup nitrous injector state
nitrous_injector.update(Input.pressure(injector_p), Input.entropy(nitrous_tank.entropy))  # Isentropic expansion to below saturation (two phase)
injector_T = nitrous_injector.temperature
nitrous.update(Input.temperature(injector_T), Input.quality(0))
injector_vapor_p = nitrous.pressure
injector_sat_density = nitrous.density

# Setup nitrous chamber state
nitrous_chamber.update(Input.pressure(chamber_p), Input.entropy(nitrous_injector.entropy))  # Isentropic expansion to chamber pressure

# Calculate mass flow rates
mdot_spi = spi_model(injector_p, chamber_p, nitrous_injector.density, A_nhne, Cd)
mdot_hem = hem_model(nitrous_injector, nitrous_chamber, A_nhne, Cd)
k = np.sqrt((injector_p - chamber_p) / (injector_vapor_p - chamber_p))

mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

print(f"Tank: p: {nitrous_tank.pressure/1e5:.2f} Bar, vp: {tank_vapor_p/1e5:.2f} Bar, T: {nitrous_tank.temperature:.2f} deg C, fill T: {fill_temp:.2f} deg C, phase: {nitrous_tank.phase}")

print(f"\nInjector: p: {nitrous_injector.pressure/1e5:.2f} Bar, vp: {injector_vapor_p/1e5:.2f} Bar, temperature: {injector_T:.2f} deg C") #, quality: {nitrous_injector.quality:.2f} %, ")
print(f"          density: {nitrous_injector.density:.2f} kg/m^3, saturated density: {injector_sat_density:.2f} kg/m^3, phase: {nitrous_injector.phase}")

print(f"\nNHNE mdot: {mdot_nhne:.4f} kg/s, SPI mdot: {mdot_spi:.4f} kg/s, HEM mdot: {mdot_hem:.4f} kg/s")