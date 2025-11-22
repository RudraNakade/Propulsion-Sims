from thermo import PRZEDZIECKI_SRIDHAR
from propulsion_system_sim import *
from pyfluids import Fluid, FluidsList
import unit_converter as uc
import numpy as np
import custom_fluids
from os import system

system('cls')

# Define fluids
n2o = custom_fluids.thermo_fluid("nitrous oxide", temperature = 273.15-8, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
ipa = custom_fluids.thermo_fluid("isopropanol", temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")
water = custom_fluids.pyfluid(Fluid(FluidsList.Water), 290, 10e5, "Water", "H2O")

# tank_p = 50e5
fitting_Cd = 0.6

fuel_tank_p = 55e5  # Pa
# ox_tank_p_ael = 50e5  # Pa

ox_tank_p_rocket = 50e5 # Pa

annulus_id = 17.29e-3
annulus_od = 18.18e-3
annulus_area = np.pi * (annulus_od**2 - annulus_id**2) * 0.25

film_area = 0.25 * np.pi * (0.4e-3)**2 * 56

fuel_inj_area = annulus_area + film_area

old_annulus_od = 17.96e-3
old_annulus_area = np.pi * (old_annulus_od**2 - annulus_id**2) * 0.25
old_film_area = 0.25 * np.pi * (0.4e-3)**2 * 44
old_fuel_inj_area = old_annulus_area + old_film_area

# fuel_inj_Cd = 17.4e-6 / old_fuel_inj_area
fuel_inj_Cd = 0.723

fuel_inj_CdA = fuel_inj_area * fuel_inj_Cd  # m²
print(f"Fuel Injector CdA: {fuel_inj_CdA * 1e6:.3f} mm², Cd: {fuel_inj_Cd:.3f}")

fuel_feed_ael = feed_system(fuel_tank_p, "Fuel Feed System")
# ox_feed_ael = feed_system(ox_tank_p_ael, "Ox Feed System")

pipe_id_3_4 = uc.in_to_m(0.75 - 2*0.036)
pipe_id_1_2 = uc.in_to_m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)
pipe_area_1_2 = np.pi * (pipe_id_1_2 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2
pipe_roughness = 0.005e-3 # m

fuel_pipes_length = 1.5
ox_pipes_length = 0.5

fuel_hose_length = 0.8 # m
ox_hose_length = 1.2 # m

fuel_engine_pipe_length = 0.5
ox_engine_pipe_length = 0.35

# fuel_valve_angle = 60
# ox_valve_angle = 70

fuel_valve_angle = 60
ox_valve_angle = 95

fuel_valve_position = (fuel_valve_angle - 22) / (100 - 22)
ox_valve_position = (ox_valve_angle - 14) / (95 - 14)

fuel_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name = "Fuel Tank Outlet")
fuel_raceway = pipe(id = pipe_id_1_2, L=fuel_pipes_length, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")
fuel_hose = pipe(id = pipe_id_1_2, L=fuel_hose_length, abs_roughness = pipe_roughness, name = "Fuel Feed System Hose")
fuel_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Slok Ball Valve')
fuel_engine_pipes_ael = pipe(id = pipe_id_1_2, L=fuel_engine_pipe_length, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
fuel_engine_pipes_rocket = pipe(id = pipe_id_1_2, L=0.6, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
regen_channels = orifice(CdA = 24.4e-6, name = "Regen Channels") # Measured
fuel_injector = orifice(CdA = fuel_inj_CdA, name = "Fuel Injector")
fuel_feed_ael.add_component(fuel_tank_outlet, fuel_raceway, fuel_hose, fuel_valve, fuel_engine_pipes_ael, regen_channels, fuel_injector)

fuel_feed_ael.set_fluid(ipa)
# fuel_feed.set_fluid(water)

ox_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name = "Ox Tank Outlet")
ox_pipes_ael = pipe(id = pipe_id_1_2, L=ox_pipes_length, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_pipes_rocket = pipe(id = pipe_id_1_2, L=0.6, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_hose = pipe(id = pipe_id_3_4, L=ox_hose_length, abs_roughness = pipe_roughness, name = "Ox Feed System Hose")
ox_engine_pipes = pipe(id = pipe_id_1_2, L=ox_engine_pipe_length, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Slok Ball Valve')
ox_injector_liquid = orifice(CdA = 79e-6, name = "N2O Injector")
ox_injector_two_phase = orifice(CdA = 57e-6, name = "N2O Injector")
# ox_feed_ael.add_component(ox_tank_outlet, ox_pipes_ael, ox_hose, ox_valve, ox_engine_pipes, ox_injector_two_phase)

# ox_feed_CdA = 45e-6
# ox_feed_orifice = orifice(CdA = ox_feed_CdA, name = "Ox Feed Orifice")
# ox_feed.add_component(ox_feed_orifice, ox_injector)

# ox_feed_ael.set_fluid(n2o)

def calc_CdA(dp, mdot, rho):
    """Calculate the required CdA for a given pressure drop and mass flow rate."""
    return mdot / np.sqrt(2 * rho * dp)

ox_valve.set_position(ox_valve_position)
fuel_valve.set_position(fuel_valve_position)

main_engine = engine("configs/l9.yaml", cstar_eff=0.96, cf_eff=0.905)

# coupled_system_ael = propulsion_system(fuel_feed_ael, ox_feed_ael, main_engine)
# coupled_system_ael.solve(True)

# ox_hose_dp_ael = ox_hose.get_inlet_pressure() - ox_hose.get_outlet_pressure()
# ox_engine_pipe_dp_ael = ox_engine_pipes.get_inlet_pressure() - ox_engine_pipes.get_outlet_pressure()

# fuel_hose_dp_ael = fuel_hose.get_inlet_pressure() - fuel_hose.get_outlet_pressure()
# fuel_engine_pipe_dp_ael = fuel_engine_pipes_ael.get_inlet_pressure() - fuel_engine_pipes_ael.get_outlet_pressure()

# ox_system_dP_ael = ox_tank_outlet.inlet_pressure - ox_injector_two_phase.inlet_pressure
# ox_mdot_ael = ox_feed_ael._mdot
# ox_system_CdA_ael = calc_CdA(ox_system_dP_ael, ox_mdot_ael, n2o.density())

fuel_feed_rocket = feed_system(fuel_tank_p, "Rocket Fuel Feed System")
fuel_feed_rocket.add_component(fuel_tank_outlet, fuel_raceway, fuel_engine_pipes_rocket, fuel_valve, regen_channels, fuel_injector)
fuel_feed_rocket.set_fluid(ipa)

ox_feed_rocket = feed_system(ox_tank_p_rocket, "Rocket Ox Feed System")
ox_feed_rocket.add_component(ox_tank_outlet, ox_pipes_rocket, ox_valve, ox_injector_liquid)
ox_feed_rocket.set_fluid(n2o)

coupled_system_rocket = propulsion_system(fuel_feed_rocket, ox_feed_rocket, main_engine)
# coupled_system_rocket.solve(True)

fuel_valve_angles = np.linspace(30, 100, 20)
fuel_tank_p_arr = np.arange(45e5, 60e5, 5e5)

valve_CdA_array = np.zeros(len(fuel_valve_angles))
pc_arr = np.zeros((len(fuel_tank_p_arr), len(fuel_valve_angles)))
OF_arr = np.zeros((len(fuel_tank_p_arr), len(fuel_valve_angles)))
thrust_arr = np.zeros((len(fuel_tank_p_arr), len(fuel_valve_angles)))

for i, tank_p in enumerate(fuel_tank_p_arr):
    for j, angle in enumerate(fuel_valve_angles):
        fuel_valve_position = (angle - 22) / (100 - 22)
        fuel_valve.set_position(fuel_valve_position)
        valve_CdA_array[j] = fuel_valve.get_effective_CdA()
        fuel_feed_rocket.set_inlet_pressure(tank_p)
        coupled_system_rocket.solve(False)
        pc_arr[i, j] = main_engine._pc
        OF_arr[i, j] = main_engine._OF
        thrust_arr[i, j] = main_engine.thrust
        print(f"Tank P: {tank_p/1e5:.1f} bar, Fuel Valve Angle: {angle:.1f} deg - Pc: {main_engine._pc/1e5:.2f} bar, OF: {main_engine._OF:.2f}, Thrust: {main_engine.thrust:.2f} N")

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(12, 8))

for i, tank_p in enumerate(fuel_tank_p_arr):
    ax1.plot(fuel_valve_angles, OF_arr[i, :], '-o', label=f'{tank_p/1e5:.0f} bar')
ax1.set_xlabel('Fuel Valve Angle (deg)')
ax1.set_ylabel('OF Ratio', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)
ax1.legend(title='Fuel Tank Pressure', loc='upper left')

ax2 = ax1.twinx()
ax2.plot(fuel_valve_angles, valve_CdA_array * 1e6, '-s', color='tab:red', label='Valve CdA')
ax2.set_ylabel('Valve CdA (mm²)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('OF Ratio and Valve CdA vs Fuel Valve Angle')
fig.tight_layout()
plt.show()

# ox_engine_pipe_dp_rocket = ox_engine_pipes.get_inlet_pressure() - ox_engine_pipes.get_outlet_pressure()
# fuel_engine_pipe_dp_rocket = fuel_engine_pipes_rocket.get_inlet_pressure() - fuel_engine_pipes_rocket.get_outlet_pressure()

# ox_system_dP_rocket = ox_tank_outlet.inlet_pressure - ox_injector_liquid.inlet_pressure
# ox_mdot_rocket = ox_feed_rocket._mdot
# ox_system_CdA_rocket = calc_CdA(ox_system_dP_rocket, ox_mdot_rocket, n2o.density())

# print(f"AEL: Ox System DP: {ox_system_dP_ael/1e5:.2f} Bar, CdA: {ox_system_CdA_ael*1e6:.2f} mm² at {ox_mdot_ael:.2f} kg/s")
# print(f"  Ox Hose DP: {ox_hose_dp_ael/1e5:.2f} Bar, Ox Engine Pipe DP: {ox_engine_pipe_dp_ael/1e5:.2f} Bar")
# print(f"  Fuel Hose DP: {fuel_hose_dp_ael/1e5:.2f} Bar, Fuel Engine Pipe DP: {fuel_engine_pipe_dp_ael/1e5:.2f} Bar")
# print(f"Rocket: Ox System DP: {ox_system_dP_rocket/1e5:.2f} Bar, CdA: {ox_system_CdA_rocket*1e6:.2f} mm² at {ox_mdot_rocket:.2f} kg/s")
# print(f"  Ox Engine Pipe DP: {ox_engine_pipe_dp_rocket/1e5:.2f} Bar")
# print(f"  Fuel Engine Pipe DP: {fuel_engine_pipe_dp_rocket/1e5:.2f} Bar")

pipe_id_1_4 = uc.in_to_m(0.25 - 2*0.036)

# press_system = feed_system(50e5, "Test Pressurant Feed System")
# outlet_pipe = pipe(id = pipe_id_1_4, L = 0.1, abs_roughness = pipe_roughness, name = "COPV Outlet Pipe")
# ereg
# ereg_valve = orifice(CdA = )


# actual_fuel_sys_CdA = 13.5e-6
# mdot = 0.65

# fuel_feed_to_injector = feed_system(fuel_tank_p, "Fuel Feed to Injector")
# fuel_feed_to_injector.add_component(fuel_tank_outlet, fuel_pipes, fuel_hose, fuel_valve, fuel_engine_pipes, regen_channels)
# fuel_feed_to_injector.set_fluid(ipa)

# fuel_feed_to_injector.solve_pressures(inlet_pressure=50e5, mdot=mdot)
# fuel_feed_to_injector.print_pressures()
# fuel_feed_to_injector.print_components()

# inj_dp = fuel_injector.dp(ipa, 0.76)
# regen_dp = regen_channels.dp(ipa, 0.76)
# print(f"Injector DP for 0.76 kg/s: {inj_dp/1e5:.2f} Bar")
# print(f"Regen DP for 0.76 kg/s: {regen_dp/1e5:.2f} Bar")
# print(f"Total DP for 0.76 kg/s: {(inj_dp + regen_dp)/1e5:.2f} Bar")

# def spi_CdA(mdot, dP, rho):
#     """Calculate CdA of a single-phase injector."""
#     return mdot / np.sqrt(2 * rho * dP)

# dP = fuel_feed_to_injector._inlet_pressure - fuel_feed_to_injector.line[-1].outlet_pressure
# rho = ipa.density()

# print(f"Using: mdot = {mdot:.2f} kg/s, dP = {dP/1e5:.2f} Bar, rho = {rho:.2f} kg/m³")

# spi_CdA_value = spi_CdA(mdot, dP, rho)
# print(f"Single-Phase Injector CdA: {spi_CdA_value*1e6:.3f} mm²")

# ox_feed.solve_mdot(cold_flow_tank_p, 101325)
# ox_feed.print_pressures()
# ox_feed.print_components()

print(f"N2O Vapor Pressure: {n2o.vapor_pressure()/1e5:.2f} Bar, N2O Density: {n2o.density():.2f} kg/m³")