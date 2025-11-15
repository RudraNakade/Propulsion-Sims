from pyfluids import FluidsList, Fluid, Input
import numpy as np

air = Fluid(FluidsList.Nitrogen).with_state(Input.temperature(20), Input.pressure(101325))

id = 25.4e-3 * 3/4
length = 1.5 # m
v = 11 # m/s

area = np.pi * 0.25 * id**2
pipe_vol = area * length

rho_init = air.density
m_init = rho_init * pipe_vol

# air = air.isentropic_compression_to_pressure(50e5)
air.update(Input.temperature(air.temperature), Input.pressure(50e5))
print(f"Air temperature after compression: {air.temperature:.2f} C")

rho_pressed = air.density
V_pressed = m_init / rho_pressed
l_pressed = V_pressed / area

time = l_pressed / v

print(f"Initial density: {rho_init:.2f} kg/m3")
print(f"Final density: {rho_pressed:.2f} kg/m3")
print(f"Length of compressed air column: {l_pressed:.3f} m")
print(f"Volume Ratio: {V_pressed / pipe_vol:.4f}")
print(f"Air mass: {m_init*1e3:.2f} g")
print(f"Time for air to travel gas length at {v:.2f} m/s: {time:.4f} s")