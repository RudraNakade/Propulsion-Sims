import numpy as np
from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt

def calculate_flow_rates(T, P_tank, P_chamber, A_eff, Cd, Cd_SPI):
    nitrous_up = Fluid(FluidsList.NitrousOxide)
    nitrous_down = Fluid(FluidsList.NitrousOxide)
    
    # Initialize arrays
    mdot_spi_up = np.zeros_like(P_tank)
    mdot_spi_adj_up = np.zeros_like(P_tank)
    mdot_hem_up = np.zeros_like(P_tank)
    mdot_nhne_up = np.zeros_like(P_tank)
    
    mdot_spi_down = np.zeros_like(P_chamber)
    mdot_spi_adj_down = np.zeros_like(P_chamber)
    mdot_hem_down = np.zeros_like(P_chamber)
    mdot_nhne_down = np.zeros_like(P_chamber)
    
    # Get vapor pressure at this temperature
    nitrous_up.update(Input.temperature(T), Input.quality(0))
    nitrous_vp = nitrous_up.pressure
    P_up_fixed = nitrous_vp + 5e5  # 10 bar above vapor pressure
    P_down_fixed = 25e5
    
    # Calculate upstream varying flow rates
    for i, P_up in enumerate(P_tank):
        nitrous_up.update(Input.pressure(P_up), Input.quality(0))
        P_vapour = nitrous_up.pressure
        nitrous_up.update(Input.pressure(P_up), Input.temperature(T))
        upstream_entropy = nitrous_up.entropy

        # SPI Model
        rho_up = nitrous_up.density
        mdot_spi_up[i] = Cd * A_eff * np.sqrt(2 * rho_up * (P_up - P_down_fixed))
        mdot_spi_adj_up[i] = Cd_SPI * A_eff * np.sqrt(2 * rho_up * (P_up - P_down_fixed))
        
        nitrous_down.update(Input.pressure(P_down_fixed), Input.entropy(upstream_entropy))
        k = np.sqrt((P_up - P_down_fixed) / (P_vapour - P_down_fixed))

        # HEM Model
        mdot_hem_up[i] = Cd * A_eff * nitrous_down.density * np.sqrt(2 * (nitrous_up.enthalpy - nitrous_down.enthalpy))
        if mdot_hem_down[i] < max(mdot_hem_down):
            mdot_hem_down[i] = max(mdot_hem_down)

        # NHNE Model
        mdot_nhne_up[i] = (mdot_spi_up[i] * k / (1 + k)) + (mdot_hem_up[i] / (1 + k))
    
    # Calculate downstream varying flow rates
    for i, P_down in enumerate(P_chamber):
        nitrous_up.update(Input.pressure(P_up_fixed), Input.quality(0))
        P_vapour = nitrous_up.pressure
        nitrous_up.update(Input.pressure(P_up_fixed), Input.temperature(T))
        upstream_entropy = nitrous_up.entropy

        # SPI Model
        rho_up = nitrous_up.density
        mdot_spi_down[i] = Cd * A_eff * np.sqrt(2 * rho_up * (P_up_fixed - P_down))
        mdot_spi_adj_down[i] = Cd_SPI * A_eff * np.sqrt(2 * rho_up * (P_up_fixed - P_down))
        
        nitrous_down.update(Input.pressure(P_down), Input.entropy(upstream_entropy))
        k = np.sqrt((P_up_fixed - P_down) / (P_vapour - P_down))

        # HEM Model
        mdot_hem_down[i] = Cd * A_eff * nitrous_down.density * np.sqrt(2 * (nitrous_up.enthalpy - nitrous_down.enthalpy))
        if mdot_hem_down[i] < max(mdot_hem_down):
            mdot_hem_down[i] = max(mdot_hem_down)

        # NHNE Model
        mdot_nhne_down[i] = (mdot_spi_down[i] * k / (1 + k)) + (mdot_hem_down[i] / (1 + k))
        
    return (mdot_spi_up, mdot_spi_adj_up, mdot_hem_up, mdot_nhne_up,
            mdot_spi_down, mdot_spi_adj_down, mdot_hem_down, mdot_nhne_down,
            P_up_fixed, P_down_fixed, nitrous_vp)

# Setup
temperatures = [-10, -5, 0, 5, 10, 15]
N = 100
Cd = 0.6
Cd_SPI = 0.4
n_holes = 24
d_hole = 1e-3 * 0.7
A_Inj = n_holes * np.pi * (d_hole / 2) ** 2
A_orifice = np.pi * (3.3e-3 / 2) ** 2
A_eff = 1/(1/A_Inj + 1/A_orifice)

# Create figures for each temperature
for T in temperatures:
    nitrous_up = Fluid(FluidsList.NitrousOxide)
    nitrous_up.update(Input.temperature(T), Input.quality(0))
    nitrous_vp = nitrous_up.pressure
    
    P_tank = np.linspace(nitrous_vp + 1e5, 50e5, N)
    P_chamber = np.linspace(1e5, nitrous_vp - 1e5, N)
    
    results = calculate_flow_rates(T, P_tank, P_chamber, A_eff, Cd, Cd_SPI)
    (mdot_spi_up, mdot_spi_adj_up, mdot_hem_up, mdot_nhne_up,
     mdot_spi_down, mdot_spi_adj_down, mdot_hem_down, mdot_nhne_down,
     P_up_fixed, P_down_fixed, nitrous_vp) = results
    
    # Create figure for this temperature
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'N2O Flow Characteristics at {T}°C (Vapor Pressure: {nitrous_vp/1e5:.1f} bar)')
    
    # Plot all four subplots
    # First subplot - Mass flow vs Upstream pressure
    ax1.plot(P_tank / 1e5, mdot_spi_up, 'b-', label='SPI Model')
    ax1.plot(P_tank / 1e5, mdot_spi_adj_up, 'b-.', label='SPI Model (Adjusted Cd)')
    ax1.plot(P_tank / 1e5, mdot_hem_up, 'r-', label='HEM Model')
    ax1.plot(P_tank / 1e5, mdot_nhne_up, 'g-', label='NHNE Model')
    ax1.set_title(f'Mass Flow Rate vs Upstream (P_down = {P_down_fixed/1e5:.1f} bar)')
    ax1.set_xlabel('Upstream Pressure (bar)')
    ax1.set_ylabel('Mass Flow Rate (kg/s)')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot - Mass flow vs Downstream pressure
    ax2.plot(P_chamber / 1e5, mdot_spi_down, 'b-', label='SPI Model')
    ax2.plot(P_chamber / 1e5, mdot_spi_adj_down, 'b-.', label='SPI Model (Adjusted Cd)')
    ax2.plot(P_chamber / 1e5, mdot_hem_down, 'r-', label='HEM Model')
    ax2.plot(P_chamber / 1e5, mdot_nhne_down, 'g-', label='NHNE Model')
    ax2.set_title(f'N2O Mass Flow Rate with {P_up_fixed/1e5:.2f} bar Upstream')
    ax2.set_xlabel('Downstream Pressure (bar)')
    ax2.set_ylabel('Mass Flow Rate (kg/s)')
    ax2.legend()
    ax2.grid(True)

    # Third subplot - Mass flow vs Pressure drop (Upstream varying)
    dP_up = P_tank - P_down_fixed
    ax3.plot(dP_up / 1e5, mdot_spi_up, 'b-', label='SPI Model')
    ax3.plot(dP_up / 1e5, mdot_spi_adj_up, 'b-.', label='SPI Model (Adjusted Cd)')
    ax3.plot(dP_up / 1e5, mdot_hem_up, 'r-', label='HEM Model')
    ax3.plot(dP_up / 1e5, mdot_nhne_up, 'g-', label='NHNE Model')
    ax3.set_title('Mass Flow vs Pressure Drop (Fixed Downstream)')
    ax3.set_xlabel('Pressure Drop (bar)')
    ax3.set_ylabel('Mass Flow Rate (kg/s)')
    ax3.legend()
    ax3.grid(True)

    # Fourth subplot - Mass flow vs Pressure drop (Downstream varying)
    dP_down = P_up_fixed - P_chamber
    ax4.plot(dP_down / 1e5, mdot_spi_down, 'b-', label='SPI Model')
    ax4.plot(dP_down / 1e5, mdot_spi_adj_down, 'b-.', label='SPI Model (Adjusted Cd)')
    ax4.plot(dP_down / 1e5, mdot_hem_down, 'r-', label='HEM Model')
    ax4.plot(dP_down / 1e5, mdot_nhne_down, 'g-', label='NHNE Model')
    ax4.set_title('Mass Flow vs Pressure Drop (Fixed Upstream)')
    ax4.set_xlabel('Pressure Drop (bar)')
    ax4.set_ylabel('Mass Flow Rate (kg/s)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)
plt.show(block=True)