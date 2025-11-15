from pyfluids import Fluid, FluidsList, Input

fluid = Fluid(FluidsList.NitrousOxide)
fluid.update(Input.pressure(25e5), Input.quality(0))

print(fluid.temperature, fluid.density)