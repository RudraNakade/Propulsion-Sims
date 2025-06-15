ox_mdot = 1.91
fuel_mdot = 0.664

fuel_rho = 790
ox_rho = 900

thrust = 4705.7
impulse = 40960


ullage_factor = 1.1

total_mdot = ox_mdot + fuel_mdot
of = ox_mdot / fuel_mdot

isp = thrust / total_mdot / 9.81

prop_mass = total_mdot * impulse / thrust
fuel_mass = prop_mass * fuel_mdot / total_mdot
ox_mass = prop_mass - fuel_mass

fuel_volume = (fuel_mass / fuel_rho) * ullage_factor
ox_volume = (ox_mass / ox_rho) * ullage_factor


print(f"Thrust: {thrust:.2f} N, Isp: {isp:.2f} s, Ox mdot: {ox_mdot:.2f} kg/s, Fuel mdot: {fuel_mdot:.2f} kg/s")
print(f"Total impulse: {impulse:.2f} Ns, Burn time: {impulse / thrust:.2f} s, OF Ratio: {of:.2f}")
print(f"Fuel mass: {fuel_mass:.2f} kg, density: {fuel_rho:.2f} kg/m^3, volume: {fuel_volume*1e3:.2f} L")
print(f"Oxidizer mass: {ox_mass:.2f} kg, density: {ox_rho:.2f} kg/m^3, volume: {ox_volume*1e3:.2f} L")
print(f"Volume OF ratio: {ox_volume / fuel_volume:.2f}")