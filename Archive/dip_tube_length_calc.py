import numpy as np
from pyfluids import Fluid, FluidsList, Input
from os import system
system('cls')

ox_tank_vol = 19.25e-3 # m^3
fuel_tank_vol = 7.311e-3 # m^3

fuel_vol = fuel_tank_vol * 0.9
fuel_rho = 790
fuel_mass = fuel_rho * fuel_vol

tank_id = 177.9e-3
tank_area = 0.25 * np.pi * tank_id**2

# ox_dip_tube_length = 450e-3
ox_dip_tube_length = 708e-3 - 10e-3
ox_vol = ox_dip_tube_length * tank_area

ox_ullage_vol = ox_tank_vol - ox_vol
ox_ullage_frac = ox_ullage_vol / ox_tank_vol

fuel_ullage_vol = fuel_tank_vol - fuel_vol
fuel_ullage_frac = fuel_ullage_vol / fuel_tank_vol

ox_fluid = FluidsList.NitrousOxide
# ox_fluid = FluidsList.CarbonDioxide
ox = Fluid(ox_fluid).with_state(Input.pressure(26e5), Input.quality(0))

ox_rho = ox.density
ox_mass = ox_rho * ox_vol
ox_temp = ox.temperature

hotfire = True

if hotfire:
    prop_mass = ox_mass + fuel_mass

    fuel_mdot = 0.76
    ox_mdot = 2.33
    engine_OF = ox_mdot / fuel_mdot

    fuel_flow_time = fuel_mass / fuel_mdot
    ox_flow_time = ox_mass / ox_mdot

    firing_time = min(fuel_flow_time, ox_flow_time)

    useful_prop_mass = ox_mdot * firing_time + fuel_mdot * firing_time

    isp = 195
    impulse = useful_prop_mass * isp * 9.81

tank_OF = ox_mass / fuel_mass

print(f"Ox Tank Volume: {ox_tank_vol*1e3:.2f} L, Fuel Tank Volume: {fuel_tank_vol*1e3:.2f} L")
print(f"Ox Density: {ox_rho:.2f} kg/m³, Ox Temp: {ox_temp:.2f} °C")
print(f"Ox Volume: {ox_vol*1e3:.2f} L, Fuel Volume: {fuel_vol*1e3:.2f} L")
print(f"Ox Mass: {ox_mass:.2f} kg, Fuel Mass: {fuel_mass:.2f} kg")
print(f"Tank O/F: {tank_OF:.2f}, Engine O/F: {engine_OF:.2f}")
print(f"Ox Ullage Fraction: {ox_ullage_frac*100:.2f} %, Fuel Ullage Fraction: {fuel_ullage_frac*100:.2f} %")

if hotfire:
    print(f"Total Prop Mass: {prop_mass:.2f} kg, Firing Time: {firing_time:.2f} s")
    print(f"Useful Prop Mass: {useful_prop_mass:.2f} kg, Total Impulse: {impulse:.0f} Ns")