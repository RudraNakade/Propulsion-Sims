from propulsion_system_sim import *
import unit_converter as uc
import numpy as np
import custom_fluids
from pyfluids import FluidsList, Fluid
from os import system

system('cls')

# Define fluids
n2o = custom_fluids.thermo_fluid("nitrous oxide", temperature = 269, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
# ipa = custom_fluids.thermo_fluid("isopropanol", temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")
ipa = custom_fluids.pyfluid(Fluid(FluidsList.Ethanol), 290, 40e5, "Ethanol", "Ethanol")

tank_p = 40e5
fitting_Cd = 0.7

fuel_tank_p = tank_p  # Pa
ox_tank_p = tank_p  # Pa

annulus_id = 16.9e-3
annulus_od = 18.00e-3
annulus_area = np.pi * (annulus_od**2 - annulus_id**2) * 0.25

film_area = 0.25 * np.pi * (0.4e-3)**2 * 44

fuel_inj_area = annulus_area + film_area

fuel_inj_Cd = 0.75

fuel_inj_CdA = fuel_inj_area * fuel_inj_Cd  # m²
print(f"Fuel Injector CdA: {fuel_inj_CdA * 1e6:.3f} mm²")

ox_inj_Cd = 79e-6 / (60 * np.pi * 0.25 * (1.5e-3)**2)
ox_inj_Cd = 0.5
ox_inj_A = 48 * 0.25 * np.pi * (1.5e-3)**2
ox_inj_CdA = ox_inj_A * ox_inj_Cd

print(f"Ox Injector Cd: {ox_inj_Cd:.3f}, Area: {ox_inj_A*1e6:.3f} mm², CdA: {ox_inj_CdA*1e6:.3f} mm²")

fuel_feed = feed_system(fuel_tank_p, "Fuel Feed System")
ox_feed = feed_system(ox_tank_p, "Ox Feed System")

pipe_id_3_4 = uc.in_to_m(0.75 - 2*0.036)
pipe_id_1_2 = uc.in_to_m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)
pipe_area_1_2 = np.pi * (pipe_id_1_2 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2
pipe_roughness = 0.01e-3 # m

fuel_pipes_length = 1.5
fuel_hose_length = 1
ox_pipes_length = 0.3
ox_hose_length = 1

fuel_tank_outlet = orifice(CdA = pipe_area_3_8 * fitting_Cd, name = "Fuel Tank Outlet")
fuel_pipes = pipe(id = pipe_id_3_8, L=fuel_pipes_length, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")
fuel_hose = pipe(id = pipe_id_3_8, L=fuel_hose_length, abs_roughness = pipe_roughness, name = "Fuel Feed System Hose")
fuel_valve = ball_valve(open_CdA = uc.Cv_to_CdA(6), name = '3/8" Slok Ball Valve')
regen_channels = orifice(CdA = 24.4e-6*2, name = "Regen Channels") # Measured
fuel_injector = orifice(CdA = fuel_inj_CdA, name = "Fuel Injector")
fuel_feed.add_component(fuel_tank_outlet, fuel_pipes, fuel_hose, fuel_valve, regen_channels, fuel_injector)
# fuel_feed.add_component(fuel_tank_outlet, fuel_pipes, fuel_valve, regen_channels, fuel_injector)

fuel_feed.set_fluid(ipa)

ox_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name = "Ox Tank Outlet")
ox_pipes = pipe(id = pipe_id_1_2, L=ox_pipes_length, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_hose = pipe(id = pipe_id_1_2, L=ox_hose_length, abs_roughness = pipe_roughness, name = "Ox Feed System Hose")
ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Slok Ball Valve')
ox_injector = orifice(CdA = ox_inj_CdA, name = "N2O Injector")
ox_feed.add_component(ox_tank_outlet, ox_pipes, ox_hose, ox_valve, ox_injector)
# ox_feed.add_component(ox_tank_outlet, ox_pipes, ox_valve, ox_injector)

ox_feed.set_fluid(n2o)

main_engine = engine("configs/thanos-r.yaml", cstar_eff=0.9, cf_eff=0.95)

coupled_system = propulsion_system(fuel_feed, ox_feed, main_engine)

coupled_system.solve(True)

print(f"N2O Vapor Pressure: {n2o.vapor_pressure()/1e5:.2f} Bar, N2O Density: {n2o.density():.2f} kg/m³")