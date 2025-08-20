from propulsion_system_sim import *
import unit_converter as uc
import numpy as np

# Define fluids
n2o = thermo_fluid(Chemical("nitrous oxide"), temperature = 263, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
ipa = thermo_fluid(Chemical("isopropanol"), temperature = 290, pressure = 50e5, name = "IPA", cea_name = "Isopropanol")

tank_p = 40e5
tank_outlet_Cd = 0.8

fuel_tank_p = tank_p  # Pa
ox_tank_p = tank_p  # Pa

annulus_Cd = 0.7
annulus_id = 7.7e-3
annulus_od = 8e-3
annulus_area = np.pi * (annulus_od**2 - annulus_id**2) * 0.25
annulus_CdA = annulus_area * annulus_Cd

film_Cd = 0.65
film_area = 0.25 * np.pi * (0.2e-3)**2 * 22
film_CdA = film_area * film_Cd

fuel_inj_CdA = annulus_CdA + film_CdA

print(f"Fuel Injector CdA: {fuel_inj_CdA * 1e6:.3f} mm²")

fuel_feed = feed_system(fuel_tank_p, "Fuel Feed System")
fuel_feed.set_fluid(ipa)
ox_feed = feed_system(ox_tank_p, "Ox Feed System")
ox_feed.set_fluid(n2o)

pipe_id_1_4 = uc.in_to_m(0.25 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)
pipe_area_1_4 = np.pi * (pipe_id_1_4 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2
pipe_roughness = 0.01e-3 # m

fuel_tank_outlet = orifice(CdA = pipe_area_1_4 * tank_outlet_Cd, name = "Fuel Tank Outlet")
fuel_pipes = pipe(id = pipe_id_1_4, L=0.5, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")

fuel_valve = ball_valve(open_CdA = uc.Cv_to_CdA(0.6), name = '1/4" Slok Ball Valve')
regen_channels = orifice(CdA = 7.924e-6, name = "Regen Channels")
fuel_injector = orifice(CdA = fuel_inj_CdA, name = "Fuel Injector")
fuel_feed.add_component(fuel_tank_outlet, fuel_pipes, fuel_valve, regen_channels, fuel_injector)

ox_tank_outlet = orifice(CdA = pipe_area_3_8 * tank_outlet_Cd, name = "Ox Tank Outlet")
ox_pipes = pipe(id = pipe_id_3_8, L=0.5, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(0.6), name = '1/4" Slok Ball Valve')
ox_injector = orifice(CdA = 6.78e-6, name = "N2O Injector")
ox_feed.add_component(ox_tank_outlet, ox_pipes, ox_valve, ox_injector)

fuel_valve.set_position(1)
ox_valve.set_position(1)

main_engine = engine("configs/csj_v2.yaml", cstar_eff=0.95, cf_eff=1)

coupled_system = propulsion_system(fuel_feed, ox_feed, main_engine)

coupled_system.solve(True)

channel_area = 0.5e-3 * 1.2e-3
channel_dh = 2 * np.sqrt(channel_area / np.pi)
regen_channel = pipe(id = channel_dh, L=0.15, abs_roughness=pipe_roughness, name = "Regen Channel")
channel_mdot = 0.088 / 40

print(f"N2O Vapor Pressure: {n2o.vapor_pressure()/1e5:.2f} Bar")