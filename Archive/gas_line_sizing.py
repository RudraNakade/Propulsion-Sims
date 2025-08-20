import numpy as np
from pyfluids import Fluid, FluidsList, Input

pressure = 20e5 # 20 bar
temperature = 25 # deg C

line_id = 25.4e-3 * (0.25 - 2*0.036) # 1/4" Tube, 0.036" wall thickness
mdot = 30e-3 # 30 g/s

fluid = Fluid(FluidsList.Oxygen)
fluid.update(Input.temperature(temperature), Input.pressure(pressure))
rho = fluid.density

area = np.pi * (line_id / 2)**2
velocity = mdot / (rho * area)
c = fluid.sound_speed
M = velocity / c

print(f"Density: {rho:.2f} kg/m^3")
print(f"Line Velocity: {velocity:.2f} m/s")
print(f"Mach Number: {M:.2f}")