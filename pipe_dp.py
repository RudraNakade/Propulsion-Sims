import numpy as np
from scipy.optimize import root_scalar
from pyfluids import Fluid, FluidsList, Input
from os import system

system('cls')

in2mm = lambda x: x * 25.4
mm2in = lambda x: x / 25.4

def pipe_dp(id, L, mdot, rho, dynamic_viscosity, roughness = -1, rel_roughness = -1):
    """
    Calculate the pressure drop in a pipe using the Darcy-Weisbach equation.
    Parameters
    ----------
    id : float
        Inner diameter of the pipe in millimeters.
    L : float
        Length of the pipe in meters.
    mdot : float
        Mass flow rate in kg/s.
    rho : float
        Fluid density in kg/m^3.
    dynamic_viscosity : float
        Dynamic viscosity of the fluid in Pa·s.
    roughness : float, optional
        Absolute roughness of the pipe in mm. Default is -1.
    rel_roughness : float, optional
        Relative roughness of the pipe (dimensionless). Default is -1.
    Notes
    -----
    Either roughness or rel_roughness must be specified, but not both.
    Returns
    -------
    float
        Pressure drop in Pascal.
    Raises
    ------
    ValueError
        If neither roughness nor rel_roughness is defined or if both are defined.
    Examples
    --------
    >>> pipe_dp(id=10, L=1, mdot=0.1, rho=1000, dynamic_viscosity=0.001, roughness=1e-5)
    >>> pipe_dp(id=10, L=1, mdot=0.1, rho=1000, dynamic_viscosity=0.001, rel_roughness=1e-3)
    """

    id *= 1e-3

    def turb_f(f, Re, rel_roughness):
        return (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))

    if (roughness == -1) and (rel_roughness == -1):
        raise ValueError('Either roughness or relative roughness must be defined')
    elif (roughness != -1) and (rel_roughness != -1):
        raise ValueError('Only one of roughness or relative roughness must be defined')
    elif roughness != -1:
        rel_roughness = 1e-3 * roughness / id
    else:
        roughness = rel_roughness * id

    print(f'Relative roughness = {rel_roughness:.2e}, Roughness = {roughness:.2e}')

    A = 0.25 * np.pi * id**2
    v = mdot / (rho * A)
    Re = rho * v * id / dynamic_viscosity

    print(f'A = {A*1e6:.2f} mm^2, v = {v:.2f} m/s')

    if Re < 3000:
        f = 64 / Re
        print(f'Laminar flow, Re = {Re:.0f}, f = {f:.5f}')
    else:
        f = root_scalar(turb_f, args=(Re, rel_roughness), bracket=[0.00001, 1]).root
        print(f'Turbulent flow, Re = {Re:.0f}, f = {f:.5f}')
    
    return f * L * rho * v**2 / (2 * id)

A = 1.5*0.4
dh = np.sqrt(4*A/np.pi)

fuel = Fluid(FluidsList.Ethanol)
fuel.update(Input.temperature(60), Input.pressure(40e5))
dyn_visc_f = fuel.dynamic_viscosity

rho = 900
mu = dyn_visc_f
id = in2mm(0.5 - 2 * 0.036)
L = 1e-3 * 730
pipe_rough = 1e-3 * 0.5
mdot = 2.4

line_dp = pipe_dp(id, L, mdot, rho, mu, roughness=pipe_rough)

print(f'Line ID: {id:.3f} mm, Length: {L:.3f} m, Flow rate: {mdot:.4f} kg/s')
print(f'Line DP: {line_dp/1e5:.3f} Bar')

CdA = 1e6 * mdot / np.sqrt(2 * line_dp * rho)

print(f'CdA: {CdA:.3f} mm^2')