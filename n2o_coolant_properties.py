import CoolProp.CoolProp as CP

fluid = 'NitrousOxide'
T = CP.PropsSI('T', 'Q', 0, 'P', CP.PropsSI('Pcrit', fluid), fluid)  # Saturation temp at critical pressure

# Get saturation pressure at a given temperature (e.g., 250 K)
T_sat = 273.15   # Kelvin
P_sat = CP.PropsSI('P', 'T', T_sat, 'Q', 0, fluid)

# Get transport properties at saturation (liquid)
# viscosity = CP.PropsSI('VISCOSITY', 'T', T_sat, 'Q', 0, fluid)      # [Pa.s]
# thermal_conductivity = CP.PropsSI('CONDUCTIVITY', 'T', T_sat, 'Q', 0, fluid)  # [W/m/K]
density = CP.PropsSI('D', 'T', T_sat, 'Q', 0, fluid)                # [kg/m^3]
heat_capacity = CP.PropsSI('C', 'T', T_sat, 'Q', 0, fluid)                # [J/kg/K]

print(f"Saturation Pressure at {T_sat} K: {P_sat} Pa")
print(f"Heat Capacity (liquid): {heat_capacity} J/kg/K")
# print(f"Viscosity (liquid): {viscosity} Pa.s")
# print(f"Thermal Conductivity (liquid): {thermal_conductivity} W/m/K")
print(f"Density (liquid): {density} kg/m^3")