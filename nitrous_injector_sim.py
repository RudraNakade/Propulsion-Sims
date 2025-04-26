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

def calculate_flow_rates(T, P_up, vp, P_down_array, A_eff, Cd, Cd_SPI):
    """Calculate flow rates for all models"""
    nitrous_up = Fluid(FluidsList.NitrousOxide)
    nitrous_down = Fluid(FluidsList.NitrousOxide)
    
    # Get properties at upstream conditions
    nitrous_up.update(Input.pressure(P_up), Input.temperature(T))
    rho_up = nitrous_up.density
    upstream_entropy = nitrous_up.entropy
    
    # Initialize result arrays
    mdot_spi = np.zeros_like(P_down_array)
    mdot_spi_adj = np.zeros_like(P_down_array)
    mdot_hem = np.zeros_like(P_down_array)
    mdot_nhne = np.zeros_like(P_down_array)
    rho_down = np.zeros_like(P_down_array)
    quality_down = np.zeros_like(P_down_array)
    T_down = np.zeros_like(P_down_array)
    
    # Calculate flow rates for each downstream pressure
    for i, P_down in enumerate(P_down_array):
        # Update properties for downstream conditions
        nitrous_down.update(Input.pressure(P_down), Input.entropy(upstream_entropy))
        rho_down[i] = nitrous_down.density
        quality_down[i] = nitrous_down.quality
        T_down[i] = nitrous_down.temperature

        mdot_spi[i] = spi_model(P_up, P_down, rho_up, A_eff, Cd)
        mdot_spi_adj[i] = spi_model(P_up, P_down, rho_up, A_eff, Cd_SPI)
        
        mdot_hem[i] = hem_model(nitrous_up, nitrous_down, A_eff, Cd)

        if np.isnan(mdot_hem[i]):
            mdot_hem[i] = 0.0

        if mdot_hem[i] < np.max(mdot_hem):
            mdot_hem[i] = np.max(mdot_hem)
        
        # NHNE Model calculations
        k = np.sqrt((P_up - P_down) / (vp - P_down)) if P_down < vp else 1.0
        mdot_nhne[i] = nhne_model(mdot_spi[i], mdot_hem[i], k)
    
    return mdot_spi, mdot_spi_adj, mdot_hem, mdot_nhne, vp, rho_down, quality_down, T_down

# Setup parameters
vp = 32e5
n_holes = 24
d_hole = 0.8e-3  # 1.5 mm holes
A_inj = n_holes * np.pi * (d_hole / 2) ** 2
# A_inj = 1.75e-6 / 0.2
Cd = 0.65
Cd_SPI = 0.45

# Get vapor pressure for the temperature
nitrous = Fluid(FluidsList.NitrousOxide)
nitrous.update(Input.pressure(vp), Input.quality(0))
T = nitrous.temperature
upstream_density = nitrous.density
P_up = vp + 10

P_down_array = np.linspace(P_up, 1e5, 5000)  # From 1 bar to just below vapor pressure

# Calculate flow rates
mdot_spi, mdot_spi_adj, mdot_hem, mdot_nhne, P_vapor, rho_down, quality_down, T_down = calculate_flow_rates(
    T, P_up, vp, P_down_array, A_inj, Cd, Cd_SPI
)

# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# # Plot 1: Mass flow rates (top-left)
# axs[0, 0].plot(P_down_array / 1e5, mdot_spi, 'b-', label='SPI Model')
# axs[0, 0].plot(P_down_array / 1e5, mdot_spi_adj, 'b--', label=f'SPI Model (Cd={Cd_SPI})')
# axs[0, 0].plot(P_down_array / 1e5, mdot_hem, 'r-', label='HEM Model')
# axs[0, 0].plot(P_down_array / 1e5, mdot_nhne, 'g-', label='NHNE Model')
# axs[0, 0].axvline(x=P_vapor/1e5, color='k', linestyle='--', label=f'VP ({P_vapor/1e5:.2f} bar)')
# axs[0, 0].set_title(f'Mass Flow Rate\nUpstream: T = {T:.2f}°C, P_up={P_up/1e5:.2f} bar, VP={P_vapor/1e5:.2f} bar')
# axs[0, 0].set_xlabel('Downstream Pressure (bar)')
# axs[0, 0].set_ylabel('Mass Flow Rate (kg/s)')
# axs[0, 0].grid(True)
# axs[0, 0].legend()

# # Plot 2: Density (top-right)
# axs[0, 1].plot(P_down_array / 1e5, rho_down, 'm-', linewidth=2)
# axs[0, 1].axvline(x=P_vapor/1e5, color='k', linestyle='--', label=f'VP ({P_vapor/1e5:.2f} bar)')
# axs[0, 1].axhline(y=upstream_density, color='b', linestyle='--', label=f'Upstream Density = {upstream_density:.2f} kg/m^3')
# axs[0, 1].set_title(f'Density\nUpstream: T = {T:.2f}°C, P_up={P_up/1e5:.2f} bar, VP={P_vapor/1e5:.2f} bar')
# axs[0, 1].set_xlabel('Downstream Pressure (bar)')
# axs[0, 1].set_ylabel('Density (kg/m³)')
# axs[0, 1].grid(True)
# axs[0, 1].legend()

# # Plot 3: Downstream Quality (bottom-left)
# axs[1, 0].plot(P_down_array / 1e5, quality_down, 'c-', linewidth=2, label='Quality')
# axs[1, 0].axvline(x=P_vapor/1e5, color='k', linestyle='--', label=f'VP ({P_vapor/1e5:.2f} bar)')
# axs[1, 0].set_title(f'Downstream Quality\nUpstream: T = {T:.2f}°C, P_up={P_up/1e5:.2f} bar, VP={P_vapor/1e5:.2f} bar')
# axs[1, 0].set_xlabel('Downstream Pressure (bar)')
# axs[1, 0].set_ylabel('Quality (Vapor Mass Fraction)')
# axs[1, 0].grid(True)
# axs[1, 0].legend()

# # Plot 4: Downstream Temperature (bottom-right)
# axs[1, 1].plot(P_down_array / 1e5, T_down, 'r-', linewidth=2, label='Temperature')
# axs[1, 1].axvline(x=P_vapor/1e5, color='k', linestyle='--', label=f'VP ({P_vapor/1e5:.2f} bar)')
# axs[1, 1].set_title(f'Downstream Temperature\nUpstream: T = {T:.2f}°C, P_up={P_up/1e5:.2f} bar, VP={P_vapor/1e5:.2f} bar')
# axs[1, 1].set_xlabel('Downstream Pressure (bar)')
# axs[1, 1].set_ylabel('Temperature (deg C)')
# axs[1, 1].grid(True)
# axs[1, 1].legend()

P_up = 38.6e5
vp = 38.5e5
pc = 30e5

Cd = 0.7
A_inj = 48 * np.pi * (1.7e-3 / 2) ** 2  # 1.5 mm holes

nitrous.update(Input.pressure(vp), Input.quality(0))
T = nitrous.temperature
nitrous.update(Input.pressure(P_up), Input.temperature(T))
nitrous_down = Fluid(FluidsList.NitrousOxide)
nitrous_down.update(Input.pressure(pc), Input.entropy(nitrous.entropy))
upstream_density = nitrous.density
mdot_spi = spi_model(P_up, pc, upstream_density, A_inj, Cd)
mdot_hem = hem_model(nitrous, nitrous_down, A_inj, Cd)
k = np.sqrt((P_up - pc) / (vp - pc))
mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

print(f"NHNE Mdot: {mdot_nhne:.2f} kg/s, @ vp = {vp/1e5:.2f} bar, P_up = {P_up/1e5:.2f} bar, P_down = {pc/1e5:.2f} bar")

P_up = 38.6e5
vp = 32e5
pc = 30e5
nitrous.update(Input.pressure(vp), Input.quality(0))
T = nitrous.temperature
nitrous.update(Input.pressure(P_up), Input.temperature(T))
nitrous_down = Fluid(FluidsList.NitrousOxide)
nitrous_down.update(Input.pressure(pc), Input.entropy(nitrous.entropy))
upstream_density = nitrous.density
mdot_spi = spi_model(P_up, pc, upstream_density, A_inj, Cd)
mdot_hem = hem_model(nitrous, nitrous_down, A_inj, Cd)
k = np.sqrt((P_up - pc) / (vp - pc))
mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

print(f"NHNE Mdot: {mdot_nhne:.2f} kg/s, @ vp = {vp/1e5:.2f} bar, P_up = {P_up/1e5:.2f} bar, P_down = {pc/1e5:.2f} bar")

# Adjust layout and spacing
plt.tight_layout()
plt.show()