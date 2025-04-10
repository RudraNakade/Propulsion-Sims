import enginesim as es

test = es.injector()

test.size_fuel_holes(Cd = 0.985, d = 1.03, n = 3)

element_mdot = test.spi_fuel_core_mdot(
    dp = 1.327454,
    fuel_rho = 786, # kg/m^3
)

print(f"mdot = {element_mdot*15:.3f} kg/s")