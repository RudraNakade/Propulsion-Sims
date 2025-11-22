from propulsion_system_sim import *
from pyfluids import Fluid, FluidsList
import unit_converter as uc
import numpy as np
import custom_fluids
from os import system

system('cls')

# Define fluids
n2o = custom_fluids.thermo_fluid("nitrous oxide", temperature = 273.15 - 8, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
ipa = custom_fluids.thermo_fluid("isopropanol", temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")
water = custom_fluids.pyfluid(Fluid(FluidsList.Water), 290, 10e5, "Water", "H2O")

tank_p = 80e5
delivery_p = 60e5

fitting_Cd = 0.6

pipe_id_1 = uc.in_to_m(1 - 2*0.036)
pipe_id_3_4 = uc.in_to_m(0.75 - 2*0.036)
pipe_id_1_2 = uc.in_to_m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)

pipe_area_1 = np.pi * (pipe_id_1 / 2) ** 2
pipe_area_3_4 = np.pi * (pipe_id_3_4 / 2) ** 2
pipe_area_1_2 = np.pi * (pipe_id_1_2 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2

pipe_roughness = 0.005e-3 # m

fuel_feed_system = feed_system(tank_p, "Fuel Feed System")
fuel_feed_system.set_fluid(ipa)

fuel_tank_outlet = orifice(CdA = pipe_area_3_4 * fitting_Cd, name="Fuel Tank Outlet")
fuel_tank_valve_pipe = pipe(id=pipe_id_3_4, L=1, abs_roughness=pipe_roughness, name="Fuel Tank to Valve Pipe")
fuel_run_valve = ball_valve(open_CdA= uc.Cv_to_CdA(13.6), name ="Fuel Run Valve - 3/4\" Slok 3-piece ball valve")
fuel_run_line = pipe(id=pipe_id_3_4, L=1, abs_roughness=pipe_roughness, name="Fuel Run Line")
fuel_hose = pipe(id=pipe_id_3_4 * 0.85, L=1, abs_roughness=pipe_roughness, name="Fuel Hose")

# fuel_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name="Fuel Tank Outlet")
# fuel_tank_valve_pipe = pipe(id=pipe_id_1_2, L=1, abs_roughness=pipe_roughness, name="Fuel Tank to Valve Pipe")
# fuel_run_valve = ball_valve(open_CdA= uc.Cv_to_CdA(7.5), name ="Fuel Run Valve - 1/2\" Slok 3-piece ball valve")
# fuel_run_line = pipe(id=pipe_id_1_2, L=1, abs_roughness=pipe_roughness, name="Fuel Run Line")
# fuel_hose = pipe(id=pipe_id_1_2, L=1, abs_roughness=pipe_roughness, name="Fuel Hose")

fuel_feed_system.add_component(fuel_tank_outlet, fuel_tank_valve_pipe, fuel_run_valve, fuel_run_line, fuel_hose)

fuel_feed_system.solve_mdot(inlet_pressure=tank_p, outlet_pressure=delivery_p)
fuel_feed_system.print_pressures()

#################################################################################################################################

ox_feed_system = feed_system(tank_p, "Oxidizer Feed System")
ox_feed_system.set_fluid(n2o)

ox_tank_outlet = orifice(CdA = pipe_area_1 * fitting_Cd, name="Oxidizer Tank Outlet")
ox_tank_valve_pipe = pipe(id=pipe_id_1, L=1, abs_roughness=pipe_roughness, name="Oxidizer Tank to Valve Pipe")
ox_run_valve = ball_valve(open_CdA= uc.Cv_to_CdA(40), name ="Oxidizer Run Valve - 1\" Slok 3-piece ball valve")
ox_run_line = pipe(id=pipe_id_1, L=1, abs_roughness=pipe_roughness, name="Oxidizer Run Line")
ox_hose = pipe(id=pipe_id_1*0.85, L=1, abs_roughness=pipe_roughness, name="Oxidizer Hose")

# ox_tank_outlet = orifice(CdA = pipe_area_3_4 * fitting_Cd, name="Oxidizer Tank Outlet")
# ox_tank_valve_pipe = pipe(id=pipe_id_3_4, L=1, abs_roughness=pipe_roughness, name="Oxidizer Tank to Valve Pipe")
# ox_run_valve = ball_valve(open_CdA= uc.Cv_to_CdA(13.6), name ="Oxidizer Run Valve - 3/4\" Slok 3-piece ball valve")
# ox_run_line = pipe(id=pipe_id_3_4, L=1, abs_roughness=pipe_roughness, name="Oxidizer Run Line")
# ox_hose = pipe(id=pipe_id_3_4, L=1, abs_roughness=pipe_roughness, name="Oxidizer Hose")

ox_feed_system.add_component(ox_tank_outlet, ox_tank_valve_pipe, ox_run_valve, ox_run_line, ox_hose)

ox_feed_system.solve_mdot(inlet_pressure=tank_p, outlet_pressure=delivery_p)
ox_feed_system.print_pressures()

fuel_vdot = fuel_feed_system.get_mdot() / ipa.density()
ox_vdot = ox_feed_system.get_mdot() / n2o.density()

R = 287
gamma = 1.4
T = 300

fuel_pressurant_mdot = tank_p * fuel_vdot / (R * T)
ox_pressurant_mdot = tank_p * ox_vdot / (R * T)

print(f"Fuel Pressurant Mass Flow Rate: {fuel_pressurant_mdot*1e3:.2f} g/s")
print(f"Oxidizer Pressurant Mass Flow Rate: {ox_pressurant_mdot*1e3:.2f} g/s")