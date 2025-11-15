from pyfluids import Fluid, Input, FluidsList

pressurant = Fluid(FluidsList.Nitrogen)

# Initial parameters:
cylinder_p = 300e5
cylinder_T = 20
cylinder_V = 50e-3

tank_V = 6.8e-3

def fill_tank(pressurant: Fluid, cylinder_p, cylinder_T, cylinder_V, tank_V):
    pressurant.update(Input.temperature(cylinder_T), Input.pressure(cylinder_p))
    sp_vol_init = pressurant.specific_volume
    m = cylinder_V / sp_vol_init
    sp_vol_final = (cylinder_V + tank_V) / m
    pressurant.update(Input.specific_volume(sp_vol_final), Input.entropy(pressurant.entropy))
    return pressurant.pressure, pressurant.temperature, pressurant.density

def cylinder_heating(pressurant: Fluid, cylinder_p, cylinder_T_init, cylinder_T_final):
    pressurant.update(Input.temperature(cylinder_T_init), Input.pressure(cylinder_p))
    pressurant.update(Input.density(pressurant.density), Input.temperature(cylinder_T_final))
    return pressurant.pressure, pressurant.temperature, pressurant.density

filling_cycles = 3

for cycle in range(filling_cycles):
    cylinder_p, cylinder_T, cylinder_rho = fill_tank(pressurant, cylinder_p, cylinder_T, cylinder_V, tank_V)
    cylinder_p_heated, cylinder_T_heated, cylinder_rho_heated = cylinder_heating(pressurant, cylinder_p, cylinder_T, 20)
    print(f"Cycle {cycle+1}: Post Filling, Post Heating")
    print(f"  Cylinder Pressure: {cylinder_p/1e5:.2f} bar, {cylinder_p_heated/1e5:.2f} bar")
    print(f"  Cylinder Temperature: {cylinder_T:.2f} °C, {cylinder_T_heated:.2f} °C")
    print(f"  Cylinder Density: {cylinder_rho:.2f} kg/m³, {cylinder_rho_heated:.2f} kg/m³\n")
    cylinder_p = cylinder_p_heated
    cylinder_T = cylinder_T_heated
    cylinder_rho = cylinder_rho_heated