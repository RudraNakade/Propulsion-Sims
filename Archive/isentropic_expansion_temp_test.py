from pyfluids import Fluid, FluidsList, Input

n2_stagnation = Fluid(FluidsList.Nitrogen).with_state(Input.temperature(20), Input.pressure(100e5))

n2_expanded = Fluid(FluidsList.Nitrogen).with_state(Input.entropy(n2_stagnation.entropy), Input.pressure(101325))

print(f"N2 Stagnation Temp: {n2_stagnation.temperature:.2f} °C, Pressure: {n2_stagnation.pressure/1e5:.2f} bar")
print(f"N2 Expanded Temp: {n2_expanded.temperature:.2f} °C, Pressure: {n2_expanded.pressure/1e5:.2f} bar")
print(n2_expanded.quality)