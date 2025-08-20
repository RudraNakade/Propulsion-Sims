import numpy as np
from pyfluids import Fluid, Input, FluidsList

propane = Fluid(FluidsList.nPropane)
methane = Fluid(FluidsList.Methane)
lin = Fluid(FluidsList.Nitrogen)

propane_min_temp = propane.min_temperature
methane_min_temp = methane.min_temperature

lin.update(Input.temperature(propane_min_temp), Input.quality(0))
propane_lin_pressure = lin.pressure
lin.update(Input.temperature(methane_min_temp), Input.quality(0))
methane_lin_pressure = lin.pressure

print(f"Propane min temperature: {propane_min_temp:.3f} °C, {propane_min_temp+273.15:.3f} K")
print(f"Methane min temperature: {methane_min_temp:.3f} °C, {methane_min_temp+273.15:.3f} K")
print(f"Propane Nitrogen pressure: {propane_lin_pressure/1e5:.3f} bar")
print(f"Methane Nitrogen pressure: {methane_lin_pressure/1e5:.3f} bar")