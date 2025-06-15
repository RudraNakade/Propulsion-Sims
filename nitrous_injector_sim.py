from matplotlib.pylab import f
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
vp = 40e5
n_holes = 48
d_hole = 1.5e-3  # 1.5 mm holes
A_inj = n_holes * np.pi * (d_hole / 2) ** 2
# A_inj = 1.75e-6 / 0.2
Cd = 0.65
Cd_SPI = 0.4

# Get vapor pressure for the temperature
nitrous_up = Fluid(FluidsList.NitrousOxide)
nitrous_up.update(Input.pressure(vp), Input.quality(0))
T = nitrous_up.temperature
upstream_density = nitrous_up.density
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


# R2S Estimated pressures - 6 kN 3.5 OF - mdot = 2.14827 kg/s
# P_up = 40.8e5
# vp = P_up - 100
# pc = 30e5


# R2S Estimated pressures - 4.7 kN 3.5 OF - mdot = 1.71922 kg/s

# P_up = 30.18e5
# vp = P_up - 100
# pc = 24e5

# R2S Estimated pressures - 4.7 kN 3 OF - mdot = 1.70405 kg/s

# P_up = 30.03e5
# vp = P_up - 100
# pc = 24e5

# R2S Estimated pressures - 4.7 kN 2.5 OF - mdot = 1.69344 kg/s

# P_up = 29.93e5
# vp = P_up - 100
# pc = 24e5

# Rocket estimated pressures - 6 kN 3.5 OF - mdot = 2.14827 kg/s

# P_up = 39.03e5
# vp = 36e5
# pc = 30e5


# Rocket estimated pressures - 4.7 kN

P_up = 36.01e5
vp = 36e5
pc = 24e5

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

p_inj_max = 50e5

pintle_structural_area = (np.pi * (pintle_d / 2) ** 2 - np.pi * (pintle_id / 2) ** 2)  - (0.5 * (pintle_d - pintle_id) * hole_d * n_holes_row)
pressure_force_cold = 0.25 * np.pi * (pintle_id ** 2) * p_inj_max
pressure_force_hot = 0.25 * np.pi * (pintle_d ** 2) * (p_inj_max - pc)

stress_cold = pressure_force_cold / pintle_structural_area
stress_hot = pressure_force_hot / pintle_structural_area



# print(0.5 * (pintle_d - pintle_id) * hole_d * 1e6)

nitrous_up.update(Input.pressure(vp), Input.quality(0))
T = nitrous_up.temperature
nitrous_up.update(Input.pressure(P_up), Input.temperature(T))
nitrous_down = Fluid(FluidsList.NitrousOxide)
nitrous_down.update(Input.pressure(pc), Input.entropy(nitrous_up.entropy))
upstream_density = nitrous_up.density
mdot_spi = spi_model(P_up, pc, upstream_density, A_nhne, Cd)
mdot_hem = hem_model(nitrous_up, nitrous_down, A_nhne, Cd)
k = np.sqrt((P_up - pc) / (vp - pc))
mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

Cd_effective = mdot_nhne / (A_inj * np.sqrt(2 * upstream_density * (P_up - pc)))

print(f"Pintle Structural Area: {pintle_structural_area*1e6:.4f} mm^2")
print(f"Force Cold: {pressure_force_cold:.1f} N, Force Hot: {pressure_force_hot:.1f} N")
print(f"Stress Cold: {stress_cold/1e6:.2f} MPa, Stress Hot: {stress_hot/1e6:.2f} MPa\n")
print(f"Inlet Area: {inlet_area*1e6:.4f} mm^2, Injector Area: {A_inj*1e6:.4f} mm^2")

print(f"Blockage Ratio: {blockage_ratio:.4f}")
print(f"SPI Mdot: {mdot_spi:.3f} kg/s, HEM Mdot: {mdot_hem:.3f} kg/s, k = {k:.2f}")
print(f"NHNE Mdot: {mdot_nhne:.4f} kg/s\n")

print(f"Upstream Properties: T = {T:.2f}°C, P = {P_up/1e5:.2f} bar, VP = {vp/1e5:.2f} bar, Density = {upstream_density:.2f} kg/m³")
print(f"Effective Cd: {Cd_effective:.4f}, Cd ratio: {Cd_effective / Cd:.4f}")
# P_up = 38.6e5
# vp = 32e5
# pc = 30e5
# nitrous.update(Input.pressure(vp), Input.quality(0))
# T = nitrous.temperature
# nitrous.update(Input.pressure(P_up), Input.temperature(T))
# nitrous_down = Fluid(FluidsList.NitrousOxide)
# nitrous_down.update(Input.pressure(pc), Input.entropy(nitrous.entropy))
# upstream_density = nitrous.density
# mdot_spi = spi_model(P_up, pc, upstream_density, A_inj, Cd)
# mdot_hem = hem_model(nitrous, nitrous_down, A_inj, Cd)
# k = np.sqrt((P_up - pc) / (vp - pc))
# mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

# print(f"NHNE Mdot: {mdot_nhne:.2f} kg/s, @ vp = {vp/1e5:.2f} bar, P_up = {P_up/1e5:.2f} bar, P_down = {pc/1e5:.2f} bar")

# # Adjust layout and spacing
# plt.tight_layout()
# plt.show()