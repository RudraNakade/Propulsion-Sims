import numpy as np
from scipy.optimize import minimize_scalar
from pyfluids import Input, Fluid
from scipy.optimize import root_scalar
from scipy.constants import gas_constant
from unit_converter import K_to_degC

def spc_mdot(CdA: float, input_fluid: Fluid, t_0: float, p_0: float, p_down: float) -> tuple[float, float, bool]:
    """Single-Phase Compressible flow model
    \nAssumes ideal gas behaviour.
    \nCalculates specific gas constant using fluid molecular weight.
    \nCalculates ratio of specific heats using specific gas constant and constant pressure specific heat.
    Args:
        CdA (float): Cd * Area (m²)
        input_fluid (Fluid): Pyfluids fluid object
        t_0 (float): Upstream stagnation temperature (K)
        p_0 (float): Upstream stagnation pressure (Pa)
        p_down (float): Downstream pressure (Pa)
    Returns:
        tuple[float, float, bool]: (mdot (kg/s), critical pressure (Pa), choked status)
    """

    if p_down > p_0:
        raise ValueError("Warning: Downstream pressure greater than upstream pressure in SPC model.")

    fluid = Fluid(input_fluid.name).with_state(Input.temperature(K_to_degC(t_0)), Input.pressure(p_0))  # stagnation

    sp_gas_const = gas_constant / fluid.molar_mass
    gamma = fluid.specific_heat / (fluid.specific_heat - sp_gas_const)

    critical_p = p_0 * (2 / (gamma + 1))**(gamma / (gamma - 1))

    if p_down > p_0:
        raise ValueError("Downstream pressure must be less than or equal to upstream pressure for SPC model.")
    elif p_down < critical_p:
        # Choked
        choked = True
        mdot = (p_0 * CdA / np.sqrt(t_0)) * np.sqrt(gamma / sp_gas_const) * ((2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))))
    else:
        # Not choked
        choked = False
        mdot = (p_0 * CdA / np.sqrt(t_0)) * np.sqrt((2 * gamma / (sp_gas_const * (gamma - 1))) * ((p_down / p_0)**(2 / gamma) - (p_down / p_0)**((gamma + 1) / gamma)))

    return (mdot, critical_p, choked)

def spc_p_0(CdA: float, mdot: float, fluid: Fluid, t_0: float, p_down: float) -> tuple[float, float, bool]:
    """Calculates the upstream stagnation pressure (p_0) for a given mass flow rate (mdot) using the SPC model.
    Assumes ideal gas behaviour.
    Calculates specific gas constant using fluid molecular weight.
    Calculates ratio of specific heats using calculated specific gas constant and constant pressure specific heat.
    Args:
        CdA (float): Cd * Area (m²)
        mdot (float): Mass flow rate (kg/s)
        fluid (Fluid): Pyfluids fluid object
        t_0 (float): Upstream stagnation temperature (K)
        p_down (float): Downstream pressure (Pa)
    Returns:
        tuple[float, float, bool]: (upstream pressure (p_0) (Pa), critical pressure (Pa), choked status)
    """

    def mdot_func(p_0, target_mdot, CdA, fluid, t_0, p_down):
        return spc_mdot(CdA, fluid, t_0, p_0, p_down)[0] - target_mdot

    result = root_scalar(mdot_func, args=(mdot, CdA, fluid, t_0, p_down), method='brentq', bracket=[p_down, 1000e5]) # max 1000 bar
    if not result.converged:
        raise ValueError("Could not find root for feed pressure calculation.")
    
    (_, downstream_choking_p, choked) = spc_mdot(CdA, fluid, t_0, result.root, p_down)

    return (result.root, downstream_choking_p, choked)

def spc_p_down(CdA: float, mdot: float, fluid: Fluid, t_0: float, p_0: float) -> tuple[float, float, bool]: # TODO: Update (won't work due to choked flow)
    """Calculates the downstream pressure (p_down) for a given mass flow rate (mdot) using the SPC model.
    Assumes ideal gas behaviour.
    Calculates specific gas constant using fluid molecular weight.
    Calculates ratio of specific heats using calculated specific gas constant and constant pressure specific heat.
    Args:
        CdA (float): Cd * Area (m²)
        mdot (float): Mass flow rate (kg/s)
        fluid (Fluid): Pyfluids fluid object
        t_0 (float): Upstream (stagnation) temperature (K)
        p_0 (float): Upstream (stagnation) pressure (Pa)
    Returns:
        tuple[float, float, bool]: (downstream pressure (p_down) (Pa), critical pressure (Pa), choked status)
    """

    def mdot_func(p_down, target_mdot, CdA, fluid, t_0, p_0):
        return spc_mdot(CdA, fluid, t_0, p_0, p_down)[0] - target_mdot

    result = root_scalar(mdot_func, args=(mdot, CdA, fluid, t_0, p_0), method='brentq', bracket=[0, p_0])
    if not result.converged:
        raise ValueError("Could not find root for downstream pressure calculation.")

    (_, downstream_choking_p, choked) = spc_mdot(CdA, fluid, t_0, p_0, result.root)

    return (result.root, downstream_choking_p, choked)

def spi_mdot(CdA: float, P_up: float, P_down: float, rho_up: float) -> float:
    """Single-Phase Incompressible flow model
    Args:
        CdA (float): Cd * Area (m²)
        P_up (float): Upstream pressure (Pa)
        P_down (float): Downstream pressure (Pa)
        rho_up (float): Upstream density (kg/m³)
    Returns:
        float: Mass flow rate (kg/s)
    """
    
    dP = P_up - P_down
    
    if dP < 0:
        print("Warning: Negative pressure drop specified for SPI model.")
        return 0

    mdot = CdA * np.sqrt(2 * rho_up * dP)
    return mdot

def spi_CdA(mdot: float, P_up: float, P_down: float, rho_up: float) -> float:
    """Calculates Cd * Area (CdA) for Single-Phase Incompressible flow model
    Args:
        mdot (float): Mass flow rate (kg/s)
        P_up (float): Upstream pressure (Pa)
        P_down (float): Downstream pressure (Pa)
        rho_up (float): Upstream density (kg/m³)
    Returns:
        float: Cd * Area (m²)
    """

    dP = P_up - P_down

    if dP < 0:
        print("Warning: Negative pressure drop specified for SPI model.")
        return 0

    CdA = mdot / np.sqrt(2 * rho_up * dP)
    return CdA

def hem_mdot(CdA: float, upstream: Fluid, downstream_p: float) -> float:
    """Homogeneous Equilibrium Model (HEM)
    Args:
        CdA (float): Cd * Area (m²)
        upstream_fluid (Fluid): Upstream fluid state
        downstream_p (float): Downstream pressure (Pa)
    Returns:
        float: Mass flow rate (kg/s)
    """

    def HEMfunc(CdA: float, upstream: Fluid, downstream_p: float) -> float:
        downstream = upstream.clone()
        downstream.update(Input.pressure(downstream_p), Input.entropy(upstream.entropy))
        mdot = CdA * downstream.density * np.sqrt(2 * (upstream.enthalpy - downstream.enthalpy)) if upstream.enthalpy > downstream.enthalpy else 0
        return mdot

    sol = minimize_scalar(lambda downstream_p: -HEMfunc(CdA, upstream, downstream_p), bounds=[9e4,upstream.pressure], method='bounded')

    choked_p = sol.x
    choked_mdot = -sol.fun

    if (choked_p > downstream_p):
        mdot = choked_mdot
    else:
        mdot = HEMfunc(CdA, upstream, downstream_p)

    return mdot

def nhne_mdot(CdA: float, upstream_fluid: Fluid, downstream_p: float) -> float:
    """Non-Homogeneous Non-Equilibrium Model (NHNE)
    Args:
        CdA (float): Cd * Area (m²)
        upstream_fluid (Fluid): Upstream Pyfluids fluid object
        downstream_p (float): Downstream pressure (Pa)
    Returns:
        float: Mass flow rate (kg/s)
    """
    
    P_up = upstream_fluid.pressure
    P_down = downstream_p
    rho_up = upstream_fluid.density

    vp_instance = upstream_fluid.clone()
    vp_instance.update(Input.temperature(upstream_fluid.temperature), Input.quality(0))
    vp = vp_instance.pressure

    if P_down >= vp:
        mdot = spi_mdot(CdA, P_up, P_down, rho_up)
    else:
        mdot_spi = spi_mdot(CdA, P_up, P_down, rho_up)
        mdot_hem = hem_mdot(CdA, upstream_fluid, P_down)

        k = np.sqrt((P_up - vp) / (vp - P_down))

        mdot = (mdot_spi * k / (1 + k)) + (mdot_hem / (1 + k))
    return mdot