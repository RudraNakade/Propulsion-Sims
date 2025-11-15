import numpy as np
from pyfluids import Fluid, FluidsList, Input
from flow_models import *
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from os import system
system('cls')

def dV_dt_ullage(prop_mdot: float, prop_rho: float) -> float:
    prop_vdot = prop_mdot / prop_rho
    return -prop_vdot

def pressurant_mdot(press_orifice_CdA: float, pressurant_fluid: Fluid, pressurant_temp: float, pressurant_tank_p: float, prop_tank_p: float) -> float:
    return spc_mdot(press_orifice_CdA, pressurant_fluid, pressurant_temp, pressurant_tank_p, prop_tank_p)[0]

def prop_tank_p_t_from_enthalpy(pressurant_reference: Fluid, ullage_vol: float, ullage_mass: float, ullage_specific_enthalpy: float) -> tuple[float, float]:
    """Calculate ullage pressure and temperature from mass, volume, and specific enthalpy"""
    if ullage_mass <= 0 or ullage_vol <= 0:
        return (101325, 273.15)  # Return atmospheric conditions if no ullage
    
    ullage_density = ullage_mass / ullage_vol
    ullage = pressurant_reference.clone()
    ullage.update(Input.density(ullage_density), Input.enthalpy(ullage_specific_enthalpy))
    return (ullage.pressure, ullage.temperature + 273.15)

def pressurant_tank_p_t(pressurant_fluid: Fluid, pressurant_mass: float, pressurant_tank_vol: float) -> tuple[float, float]:
    pressurant = pressurant_fluid.clone()
    pressurant_density = pressurant_mass / pressurant_tank_vol
    pressurant.update(Input.density(pressurant_density), Input.entropy(pressurant.entropy))
    return (pressurant.pressure, (pressurant.temperature + 273.15))

def incoming_gas_enthalpy(pressurant_reference: Fluid, pressurant_tank_temp: float, pressurant_tank_p: float, prop_tank_p: float) -> float:
    """Calculate specific enthalpy of gas after isenthalpic expansion through orifice"""
    upstream_gas = pressurant_reference.clone()
    upstream_gas.update(Input.temperature(pressurant_tank_temp - 273.15), Input.pressure(pressurant_tank_p+1e-4))
    
    # Isenthalpic process to propellant tank pressure
    downstream_gas = upstream_gas.clone()
    downstream_gas.update(Input.pressure(prop_tank_p), Input.enthalpy(upstream_gas.enthalpy))

    if downstream_gas.pressure > upstream_gas.pressure:
        print(f"Warning: Downstream pressure ({downstream_gas.pressure/1e5:.2f} bar) exceeds upstream pressure ({upstream_gas.pressure/1e5:.2f} bar). Check input conditions.")
    
    return downstream_gas.enthalpy

def chamber_p_model(prop_mdot: float) -> float:
    sample_mdot = 2.33
    sample_pc = 31.8e5
    zero_pc = 101325

    calculated_pc = ((sample_pc - zero_pc) / sample_mdot) * prop_mdot + zero_pc

    if calculated_pc < 5e5:
        calculated_pc = zero_pc

    # return calculated_pc
    return 101325

# press_orifice_id = 2e-3
# outlet_orifice_id = 8.2e-3
# orifice_Cd = 0.7

# press_tank_total_vol = 6.8e-3 # m^3
# prop_tank_total_vol = 19.25e-3
# prop_tank_init_ullage = 0.2

press_orifice_id = 3e-3
outlet_orifice_id = 7e-3
orifice_Cd = 0.65

press_tank_total_vol = 9e-3 # m^3
prop_tank_total_vol = 20e-3
prop_tank_init_ullage = 0.1

press_orifice_CdA = orifice_Cd * 0.25 * np.pi * press_orifice_id**2
outlet_orifice_CdA = orifice_Cd * 0.25 * np.pi * outlet_orifice_id**2

# initial conditions
pressurant_initial_p = 250e5
pressurant_initial_t = 273.15 + 25
prop_tank_init_p = 50e5
# prop_density = 950
prop_density = 1000
pressurant_fluid = Fluid(FluidsList.Nitrogen).with_state(Input.temperature(pressurant_initial_t-273.15), Input.pressure(pressurant_initial_p))
total_pressurant_mass = pressurant_fluid.density * press_tank_total_vol

prop_vol = prop_tank_total_vol * (1 - prop_tank_init_ullage)
ullage_vol = prop_tank_total_vol * prop_tank_init_ullage

prop_init_mass = prop_density * prop_vol

# ullage_press_fluid = pressurant_fluid.clone().isenthalpic_expansion_to_pressure(prop_tank_init_p)
# ullage_press_fluid = pressurant_fluid.clone().with_state(Input.temperature(0), Input.pressure(prop_tank_init_p))
ullage_press_fluid = pressurant_fluid.clone().with_state(Input.pressure(prop_tank_init_p), Input.temperature(pressurant_initial_t-273.15))
ullage_init_mass = ullage_press_fluid.density * ullage_vol
ullage_init_specific_enthalpy = ullage_press_fluid.enthalpy  # Initial specific enthalpy

pressurant_tank_mass = total_pressurant_mass - ullage_init_mass
pressurant_tank_density = pressurant_tank_mass / press_tank_total_vol

# pressurant_fluid.update(Input.density(pressurant_tank_density), Input.entropy(pressurant_fluid.entropy))

time = 0

def system_ode(t, y) -> list[float]:
    """
    ODE system for tank pressurization with energy conservation.
    y = [prop_mass, ullage_vol, ullage_mass, pressurant_tank_mass, ullage_total_enthalpy]
    """
    prop_mass, ullage_vol, ullage_mass, pressurant_tank_mass, ullage_total_enthalpy = y
    
    # print(f"Time: {t:.2f} s, Prop Mass: {prop_mass:.3f} kg, Ullage Vol: {ullage_vol*1e3:.3f} L, Ullage Mass: {ullage_mass*1e3:.3f} g, Pressurant Tank Mass: {pressurant_tank_mass:.3f} kg")

    # Calculate current ullage specific enthalpy
    ullage_specific_enthalpy = ullage_total_enthalpy / ullage_mass if ullage_mass > 0 else 0

    # Current states
    ullage_pressure, ullage_temperature = prop_tank_p_t_from_enthalpy(pressurant_fluid, ullage_vol, ullage_mass, ullage_specific_enthalpy)
    pressurant_pressure, pressurant_temperature = pressurant_tank_p_t(pressurant_fluid, pressurant_tank_mass, press_tank_total_vol)
    
    if ullage_pressure > pressurant_pressure:
        print(f"Warning: Ullage pressure ({ullage_pressure/1e5:.2f} bar) exceeds pressurant tank pressure ({pressurant_pressure/1e5:.2f} bar) at time {t:.2f} s.")
    
    # Solve for outlet_p and prop_mdot simultaneously
    def coupled_equations(vars):
        outlet_p, prop_mdot_abs = vars
        
        # Calculate prop_mdot from flow equation
        calculated_prop_mdot = -spi_mdot(outlet_orifice_CdA, ullage_pressure, outlet_p, prop_density)
        
        # Calculate outlet_p from chamber pressure model
        calculated_outlet_p = chamber_p_model(prop_mdot_abs)
        
        return [
            calculated_prop_mdot + prop_mdot_abs,  # prop_mdot should be negative, so this equals zero
            calculated_outlet_p - outlet_p         # outlet pressures should match
        ]
    
    # Initial guess: atmospheric pressure and small flow rate
    initial_guess = [15e5, 1]
    
    if pressurant_pressure > ullage_pressure:
        pressurant_mdot = spc_mdot(press_orifice_CdA, pressurant_fluid, pressurant_temperature, pressurant_pressure, ullage_pressure)[0]
    else:
        pressurant_mdot = 0

    if prop_mass > 0: # liquid flow
        dullage_mass_dt = pressurant_mdot  # pressurant flows into ullage
        outgoing_enthalpy = 0
        ullage_mdot_out = 0
        try:
            solution = fsolve(coupled_equations, initial_guess)
            outlet_p, prop_mdot_abs = solution
            prop_mdot = -prop_mdot_abs  # Make it negative (outflow)
        except Exception as e:
            print(f"Error in solving coupled equations: {e}")
            outlet_p = 101325
            prop_mdot = -spi_mdot(outlet_orifice_CdA, ullage_pressure, outlet_p, prop_density)

    else: # gas flow#
        prop_mass = 0
        prop_mdot = 0
        outlet_p = 101325
        ullage_mdot_out = spc_mdot(outlet_orifice_CdA, pressurant_fluid, ullage_temperature, ullage_pressure, 101325)[0]
        dullage_mass_dt = pressurant_mdot - ullage_mdot_out
        outgoing_enthalpy = pressurant_fluid.clone().with_state(Input.temperature(ullage_temperature - 273.15), Input.pressure(ullage_pressure)).enthalpy
        # print(f"Gas outflow at time {t:.2f} s, with mdot {pressurant_mdot:.4f} kg/s")
        # print(f"Pressurant p: {pressurant_pressure/1e5:.2f} bar, T: {pressurant_temperature-273.15:.2f} C, ullage p: {ullage_pressure/1e5:.2f} bar, T: {ullage_temperature-273.15:.2f} C")

    # Calculate enthalpy of incoming gas after isenthalpic expansion
    incoming_enthalpy = incoming_gas_enthalpy(pressurant_fluid, pressurant_temperature, pressurant_pressure, ullage_pressure)
    
    # Derivatives
    dprop_mass_dt = prop_mdot
    dullage_vol_dt = dV_dt_ullage(prop_mdot, prop_density)
    dpressurant_tank_mass_dt = -pressurant_mdot

    # Energy conservation: rate of change of total enthalpy in ullage
    # H_total = m * h_specific
    # dH_total/dt = h_specific * dm/dt + m * dh_specific/dt
    # For mixing: dH_total/dt = h_incoming * dm_incoming/dt (assuming no heat transfer)
    dullage_total_enthalpy_dt = incoming_enthalpy * pressurant_mdot - outgoing_enthalpy * ullage_mdot_out

    return [dprop_mass_dt, dullage_vol_dt, dullage_mass_dt, dpressurant_tank_mass_dt, dullage_total_enthalpy_dt]

# Initial conditions vector - now includes total enthalpy
y0 = [prop_init_mass, ullage_vol, ullage_init_mass, pressurant_fluid.density * press_tank_total_vol, ullage_init_mass * ullage_init_specific_enthalpy]

t_span = (0, 8)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE system
solution = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8)

# Extract results
time = solution.t
prop_mass = solution.y[0]
ullage_vol = solution.y[1]
ullage_mass = solution.y[2]
pressurant_tank_mass = solution.y[3]
ullage_total_enthalpy = solution.y[4]
gas_mass = ullage_mass + pressurant_tank_mass

# Calculate pressures over time
ullage_pressures = np.zeros_like(time)
ullage_temps = np.zeros_like(time)
pressurant_pressures = np.zeros_like(time)
pressurant_temps = np.zeros_like(time)
prop_mdots = np.zeros_like(time)
pressurant_mdots = np.zeros_like(time)
chamber_pressures = np.zeros_like(time)
ullage_specific_enthalpies = np.zeros_like(time)
ullage_mdot_out = np.zeros_like(time)
ullage_mdot = np.zeros_like(time)

for i in range(len(time)):
    ullage_specific_enthalpies[i] = ullage_total_enthalpy[i] / ullage_mass[i] if ullage_mass[i] > 0 else ullage_init_specific_enthalpy
    ullage_pressures[i], ullage_temps[i] = prop_tank_p_t_from_enthalpy(pressurant_fluid, ullage_vol[i], ullage_mass[i], ullage_specific_enthalpies[i])
    pressurant_pressures[i], pressurant_temps[i] = pressurant_tank_p_t(pressurant_fluid, pressurant_tank_mass[i], press_tank_total_vol)
    
    # Calculate pressurant mass flow rate
    if pressurant_pressures[i] > ullage_pressures[i]:
        pressurant_mdots[i] = pressurant_mdot(press_orifice_CdA, pressurant_fluid, pressurant_temps[i], pressurant_pressures[i], ullage_pressures[i])
    else:
        pressurant_mdots[i] = 0
    
    # Handle liquid vs gas flow
    if prop_mass[i] > 0:  # Liquid flow case
        # Solve coupled equations for chamber pressure and propellant flow
        ullage_mdot_out[i] = 0
        def coupled_equations_plot(vars):
            outlet_p, prop_mdot_abs = vars
            calculated_prop_mdot = -spi_mdot(outlet_orifice_CdA, ullage_pressures[i], outlet_p, prop_density)
            calculated_outlet_p = chamber_p_model(prop_mdot_abs)
            return [calculated_prop_mdot + prop_mdot_abs, calculated_outlet_p - outlet_p]
        
        try:
            solution = fsolve(coupled_equations_plot, [15e5, 1])
            chamber_pressures[i], prop_mdot_abs = solution
            prop_mdots[i] = -prop_mdot_abs
        except:
            chamber_pressures[i] = 101325
            prop_mdots[i] = -spi_mdot(outlet_orifice_CdA, ullage_pressures[i], chamber_pressures[i], prop_density)
    else:  # Gas flow case
        chamber_pressures[i] = 101325  # Atmospheric pressure
        prop_mdots[i] = 0  # No liquid propellant flow
        # Calculate ullage mass flow rate out
        ullage_mdot_out[i] = spc_mdot(outlet_orifice_CdA, pressurant_fluid, ullage_temps[i], ullage_pressures[i], 101325)[0]

    ullage_mdot[i] = pressurant_mdots[i] - ullage_mdot_out[i]
    
# Plot results
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

def grid_with_minor(ax):
    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.3)
    ax.minorticks_on()

ax1.plot(time, ullage_pressures/1e5, 'b-', label='Propellant Tank')
ax1.plot(time, pressurant_pressures/1e5, 'm-', label='Pressurant Tank')
ax1.plot(time, chamber_pressures/1e5, 'r-', label='Chamber Pressure')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pressure (bar)')
ax1.set_title('Pressures')
ax1.legend()
grid_with_minor(ax1)

ax2.plot(time, -prop_mdots, 'g-', label='Propellant Massflow')
ax2.plot(time, ullage_mdot_out, 'r-', label='Ullage Massflow Out')
ax2.plot(time, pressurant_mdots, 'orange', label='Pressurant Massflow')
ax2.plot(time, ullage_mdot, 'b-', label='d(Ullage Mass)/dt')
ax2.legend(loc='best')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Propellant Mass Flow Rate (kg/s)', color='green')
ax2.set_title('Propellant Mass Flow Rate and Mass')
grid_with_minor(ax2)

# ax2_twin = ax2.twinx()
# ax2_twin.plot(time, prop_mass, 'b-')
# ax2_twin.set_ylabel('Propellant Mass (kg)', color='blue')
# ax2_twin.tick_params(axis='y', labelcolor='blue')

ax3.plot(time, ullage_vol * 1e3, 'orange', label='Ullage Volume')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Ullage Volume (L)', color='orange')
ax3.set_title('Ullage Volume and Mass')
grid_with_minor(ax3)

ax3_twin = ax3.twinx()
ax3_twin.plot(time, pressurant_tank_mass*1e3, 'c-', label='Pressurant Mass')
ax3_twin.plot(time, ullage_mass*1e3, 'purple', label='Ullage Mass')
ax3_twin.plot(time, gas_mass*1e3, 'k-', label='Total Gas Mass')
ax3_twin.set_ylabel('Ullage Mass (g)', color='purple')
ax3_twin.tick_params(axis='y', labelcolor='purple')

lns = ax3_twin.get_lines() + ax3.get_lines()
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc="lower right")

ax4.plot(time, ullage_temps - 273.15, 'b-', label='Ullage Temperature')
ax4.plot(time, pressurant_temps - 273.15, 'r-', label='Pressurant Tank Temperature')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Temperature (°C)')
ax4.set_title('Tank Temperatures')
ax4.legend(loc='best')
grid_with_minor(ax4)

ax5.plot(time, ullage_specific_enthalpies / 1e3, 'b-', label='Specific Enthalpy')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Specific Enthalpy (kJ/kg)', color='blue')
ax5.set_title('Ullage Enthalpies')
ax5.tick_params(axis='y', labelcolor='blue')
grid_with_minor(ax5)

ax5_twin = ax5.twinx()
ax5_twin.plot(time, ullage_total_enthalpy / 1e3, 'r-', label='Total Enthalpy')
ax5_twin.set_ylabel('Total Enthalpy (kJ)', color='red')
ax5_twin.tick_params(axis='y', labelcolor='red')

lns5 = ax5.get_lines() + ax5_twin.get_lines()
labs5 = [l.get_label() for l in lns5]
ax5.legend(lns5, labs5, loc="best")

# Leave ax6 empty or add another plot if needed
ax6.axis('off')

fig.tight_layout()



plt.show()