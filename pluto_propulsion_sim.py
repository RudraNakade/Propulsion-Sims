from propulsion_system_sim import *
import unit_converter as uc
import numpy as np
from os import system

system('cls')

# Define fluids
n2o = thermo_fluid(Chemical("nitrous oxide"), temperature = 268, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
ipa = thermo_fluid(Chemical("isopropanol"), temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")

tank_p = 55e5
tank_outlet_Cd = 0.7

fuel_tank_p = tank_p  # Pa
ox_tank_p = tank_p  # Pa

annulus_id = 17.27e-3
annulus_od = 18.25e-3
annulus_area = np.pi * (annulus_od**2 - annulus_id**2) * 0.25

film_area = 0.25 * np.pi * (0.4e-3)**2 * 56

fuel_inj_area = annulus_area + film_area

# fuel_inj_Cd = 17.4e-6 / fuel_inj_area
fuel_inj_Cd = 0.7

# fuel_inj_CdA = 17.4e-6  # m²
# fuel_inj_CdA = 21e-6  # m²
fuel_inj_CdA = fuel_inj_area * fuel_inj_Cd  # m²
print(f"Fuel Injector CdA: {fuel_inj_CdA * 1e6:.3f} mm²")

fuel_feed = feed_system(fuel_tank_p, "Fuel Feed System")
ox_feed = feed_system(ox_tank_p, "Ox Feed System")

pipe_id_3_4 = uc.in_to_m(0.75 - 2*0.036)
pipe_id_1_2 = uc.in_to_m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)
pipe_area_1_2 = np.pi * (pipe_id_1_2 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2
pipe_roughness = 0.01e-3 # m

fuel_tank_outlet = orifice(CdA = pipe_area_1_2 * tank_outlet_Cd, name = "Fuel Tank Outlet")
fuel_pipes = pipe(id = pipe_id_1_2, L=2, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")

# fuel_tank_outlet = orifice(CdA = pipe_area_3_8 * tank_outlet_Cd, name = "Fuel Tank Outlet")
# fuel_pipes = pipe(id = pipe_id_3_8, L=2, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")

fuel_hose = pipe(id = pipe_id_1_2, L=1, abs_roughness = pipe_roughness, name = "Fuel Feed System Hose")
fuel_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Slok Ball Valve')
fuel_inlet_fitting = diameter_change(Cd = 0.7, D = pipe_id_3_8, D_up = pipe_id_1_2, name = "Fuel Inlet Fitting")
regen_channels = orifice(CdA = 24.4e-6, name = "Regen Channels")
fuel_injector = orifice(CdA = fuel_inj_CdA, name = "Fuel Injector") # Measured
fuel_feed.add_component(fuel_tank_outlet, fuel_pipes, fuel_hose, fuel_valve, fuel_inlet_fitting, regen_channels, fuel_injector)

fuel_feed.set_fluid(ipa)

ox_tank_outlet = orifice(CdA = pipe_area_1_2 * tank_outlet_Cd, name = "Ox Tank Outlet")
ox_pipes = pipe(id = pipe_id_1_2, L=0.5, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_hose = pipe(id = pipe_id_3_4, L=1, abs_roughness = pipe_roughness, name = "Ox Feed System Hose")
ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Slok Ball Valve')
ox_injector = orifice(CdA = 79e-6, name = "N2O Injector")
ox_feed.add_component(ox_tank_outlet, ox_pipes, ox_hose, ox_valve, ox_injector)

ox_feed.set_fluid(n2o)

ox_valve.set_position(0.65)

main_engine = engine("configs/l9.yaml", cstar_eff=0.96, cf_eff=0.905)

coupled_system = propulsion_system(fuel_feed, ox_feed, main_engine)

coupled_system.solve(True)

print(f"N2O Vapor Pressure: {n2o.vapor_pressure()/1e5:.2f} Bar")