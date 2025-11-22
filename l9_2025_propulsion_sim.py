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
fitting_Cd = 0.7

fuel_tank_p = 50e5  # Pa
ox_tank_p = 45e5 # Pa

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

pipe_id_3_4 = uc.in_to_m(0.75 - 2*0.036)
pipe_id_1_2 = uc.in_to_m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in_to_m(0.375 - 2*0.036)
pipe_id_1_4 = uc.in_to_m(0.25 - 2*0.036)
pipe_area_3_4 = np.pi * (pipe_id_3_4 / 2) ** 2
pipe_area_1_2 = np.pi * (pipe_id_1_2 / 2) ** 2
pipe_area_3_8 = np.pi * (pipe_id_3_8 / 2) ** 2
pipe_area_1_4 = np.pi * (pipe_id_1_4 / 2) ** 2

pipe_roughness = 0.005e-3 # m

fuel_pipes_length = 1

fuel_engine_pipe_length = 0.5
ox_engine_pipe_length = 0.35


fuel_feed_rocket = feed_system(fuel_tank_p, "Rocket Fuel Feed System")
fuel_feed_rocket.set_fluid(ipa)

ox_feed_rocket = feed_system(ox_tank_p, "Rocket Ox Feed System")
ox_feed_rocket.set_fluid(n2o)

fuel_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name = "Fuel Tank Outlet")
fuel_raceway = pipe(id = pipe_id_1_2, L=fuel_pipes_length, abs_roughness = pipe_roughness, name = "Fuel Feed System Pipes")
# fuel_reducer = diameter_change(Cd = fitting_Cd, D_down=pipe_id_3_8, D_up=pipe_id_1_2, name = "Fuel 1/2\" to 3/8\" Reducer")
fuel_valve = ball_valve(open_CdA = uc.Cv_to_CdA(4), name = '1/2" Slok Ball Valve')
# fuel_valve = needle_valve(open_CdA=uc.Cv_to_CdA(4), name = '1/2" Needle Valve')
fuel_engine_pipes_rocket = pipe(id = pipe_id_1_2, L=0.5, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
regen_channels = orifice(CdA = 24.4e-6, name = "Regen Channels") # Measured
fuel_injector = orifice(CdA = fuel_inj_CdA, name = "Fuel Injector")

fuel_feed_rocket.add_component(fuel_tank_outlet, fuel_raceway, fuel_engine_pipes_rocket, fuel_valve, regen_channels, fuel_injector)


ox_tank_outlet = orifice(CdA = pipe_area_1_2 * fitting_Cd, name = "Ox Tank Outlet")
ox_pipes = pipe(id = pipe_id_1_2, L=0.3, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(12), name = '1/2" Ball Valve')
ox_engine_pipes = pipe(id = pipe_id_1_2, L=0.2, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
ox_injector_liquid = orifice(CdA = 79e-6, name = "N2O Injector")
ox_feed_rocket.add_component(ox_tank_outlet, ox_pipes, ox_valve, ox_engine_pipes, ox_injector_liquid)

# ox_tank_outlet = orifice(CdA = pipe_area_3_4 * fitting_Cd, name = "Ox Tank Outlet")
# ox_pipes = pipe(id = pipe_id_3_4, L=0.3, abs_roughness = pipe_roughness, name = "Ox Feed System Pipes")
# ox_valve = ball_valve(open_CdA = uc.Cv_to_CdA(24), name = '1/2" Ball Valve')
# ox_engine_pipes = pipe(id = pipe_id_3_4, L=0.2, abs_roughness = pipe_roughness, name = "Engine Feed Pipes")
# ox_injector_stepdown = diameter_change(Cd = fitting_Cd, D_down=pipe_id_1_2, D_up=pipe_id_3_4, name = "Ox Injector Stepdown")
# ox_injector_liquid = orifice(CdA = 79e-6, name = "N2O Injector")
# ox_feed_rocket.add_component(ox_tank_outlet, ox_pipes, ox_valve, ox_engine_pipes, ox_injector_stepdown, ox_injector_liquid)

def calc_CdA(dp, mdot, rho):
    """Calculate the required CdA for a given pressure drop and mass flow rate."""
    return mdot / np.sqrt(2 * rho * dp)


main_engine = engine("configs/l9.yaml", cstar_eff=0.96, cf_eff=0.905)

prop_system = propulsion_system(fuel_feed_rocket, ox_feed_rocket, main_engine)
prop_system.solve(True)

max_error = 3e5

fuel_tank_setpoint = 55e5
ox_tank_setpoint = 47e5

# Create pressure arrays
fuel_tank_p_arr = np.arange(fuel_tank_setpoint - max_error, fuel_tank_setpoint + max_error + 1e5, 1e5)
ox_tank_p_arr = np.arange(ox_tank_setpoint - max_error, ox_tank_setpoint + max_error + 1e5, 1e5)
# Initialize result arrays
OF_arr = np.zeros((len(fuel_tank_p_arr), len(ox_tank_p_arr)))
pc_arr = np.zeros((len(fuel_tank_p_arr), len(ox_tank_p_arr)))
thrust_arr = np.zeros((len(fuel_tank_p_arr), len(ox_tank_p_arr)))

print("\n" + "="*80)
print("Running Coupled System Simulations")
print("="*80)

for i, fuel_p in enumerate(fuel_tank_p_arr):
    for j, ox_p in enumerate(ox_tank_p_arr):
        fuel_feed_rocket.set_inlet_pressure(fuel_p)
        ox_feed_rocket.set_inlet_pressure(ox_p)
        
        try:
            prop_system.solve(False)
            pc_arr[i, j] = main_engine._pc
            OF_arr[i, j] = main_engine._OF
            thrust_arr[i, j] = main_engine.thrust
            print(f"Fuel P: {fuel_p/1e5:.1f} bar, Ox P: {ox_p/1e5:.1f} bar - Pc: {main_engine._pc/1e5:.2f} bar, OF: {main_engine._OF:.2f}, Thrust: {main_engine.thrust:.2f} N")
        except ValueError as e:
            pc_arr[i, j] = np.nan
            OF_arr[i, j] = np.nan
            thrust_arr[i, j] = np.nan
            print(f"Fuel P: {fuel_p/1e5:.1f} bar, Ox P: {ox_p/1e5:.1f} bar - FAILED: {e}")

print("="*80)

# Create contour plots
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Create meshgrid for plotting
fuel_p_mesh, ox_p_mesh = np.meshgrid(fuel_tank_p_arr / 1e5, ox_tank_p_arr / 1e5, indexing='ij')

# Plot 1: O/F Ratio contour
contour1 = ax1.contourf(fuel_p_mesh, ox_p_mesh, OF_arr, levels=15, cmap='viridis')
contour1_lines = ax1.contour(fuel_p_mesh, ox_p_mesh, OF_arr, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax1.clabel(contour1_lines, inline=True, fontsize=8)

# Find and mark min/max values
valid_OF = OF_arr[~np.isnan(OF_arr)]
if len(valid_OF) > 0:
    min_OF = np.nanmin(OF_arr)
    max_OF = np.nanmax(OF_arr)
    min_idx = np.unravel_index(np.nanargmin(OF_arr), OF_arr.shape)
    max_idx = np.unravel_index(np.nanargmax(OF_arr), OF_arr.shape)
    
    ax1.scatter([fuel_p_mesh[min_idx]], [ox_p_mesh[min_idx]], 
               color='blue', s=100, marker='o', label=f'Min: OF={min_OF:.2f}', zorder=5)
    ax1.scatter([fuel_p_mesh[max_idx]], [ox_p_mesh[max_idx]], 
               color='orange', s=100, marker='o', label=f'Max: OF={max_OF:.2f}', zorder=5)

ax1.set_xlabel('Fuel Tank Pressure (bar)')
ax1.set_ylabel('Oxidizer Tank Pressure (bar)')
ax1.set_title('O/F Ratio vs Tank Pressures')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig.colorbar(contour1, ax=ax1, label='O/F Ratio')

# Plot 2: Thrust contour
contour2 = ax2.contourf(fuel_p_mesh, ox_p_mesh, thrust_arr, levels=15, cmap='plasma')
contour2_lines = ax2.contour(fuel_p_mesh, ox_p_mesh, thrust_arr, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax2.clabel(contour2_lines, inline=True, fontsize=8)

# Find and mark min/max values
valid_thrust = thrust_arr[~np.isnan(thrust_arr)]
if len(valid_thrust) > 0:
    min_thrust = np.nanmin(thrust_arr)
    max_thrust = np.nanmax(thrust_arr)
    min_thrust_idx = np.unravel_index(np.nanargmin(thrust_arr), thrust_arr.shape)
    max_thrust_idx = np.unravel_index(np.nanargmax(thrust_arr), thrust_arr.shape)
    
    ax2.scatter([fuel_p_mesh[min_thrust_idx]], [ox_p_mesh[min_thrust_idx]], 
               color='blue', s=100, marker='o', label=f'Min: Thrust={min_thrust:.2f} N', zorder=5)
    ax2.scatter([fuel_p_mesh[max_thrust_idx]], [ox_p_mesh[max_thrust_idx]], 
               color='orange', s=100, marker='o', label=f'Max: Thrust={max_thrust:.2f} N', zorder=5)

ax2.set_xlabel('Fuel Tank Pressure (bar)')
ax2.set_ylabel('Oxidizer Tank Pressure (bar)')
ax2.set_title('Thrust vs Tank Pressures')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.colorbar(contour2, ax=ax2, label='Thrust (N)')

fuel_valve_angle_arr = np.arange(30, 91, 5)

OF_arr2 = np.zeros_like(fuel_valve_angle_arr)
pc_arr2 = np.zeros_like(fuel_valve_angle_arr)
valve_CdA_arr = np.zeros_like(fuel_valve_angle_arr)

for i, angle in enumerate(fuel_valve_angle_arr):
    fuel_valve.set_position(angle / 90)  # Scale to 0-90 degrees
    valve_CdA_arr[i] = fuel_valve.get_flow_coeff(angle / 90)
    try:
        prop_system.solve(False)
        pc_arr2[i] = main_engine._pc
        OF_arr2[i] = main_engine._OF
        print(f"Fuel Valve Angle: {angle}° - Pc: {main_engine._pc/1e5:.2f} bar, OF: {main_engine._OF:.2f}")
    except ValueError as e:
        pc_arr2[i] = np.nan
        OF_arr2[i] = np.nan
        print(f"Fuel Valve Angle: {angle}° - FAILED: {e}")

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
ax1.plot(fuel_valve_angle_arr, OF_arr2, 'b-o')
ax1.set_xlabel('Fuel Valve Angle (degrees)')
ax1.set_ylabel('O/F Ratio')
ax1.set_title('O/F Ratio vs Fuel Valve Angle')
ax1.grid(True, alpha=0.3)
ax2.plot(fuel_valve_angle_arr, pc_arr2 / 1e5, 'r-o')
ax2.set_xlabel('Fuel Valve Angle (degrees)')
ax2.set_ylabel('Chamber Pressure Pc (bar)')
ax2.set_title('Chamber Pressure Pc vs Fuel Valve Angle')
ax2.grid(True, alpha=0.3)

ax3.plot(fuel_valve_angle_arr, valve_CdA_arr, 'g-o')
ax3.set_xlabel('Fuel Valve Angle (degrees)')
ax3.set_ylabel('Valve Flow Coefficient')
ax3.set_title('Valve Flow Coefficient vs Fuel Valve Angle')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
