from os import system
import flow_models
import enginesim
from pyfluids import Fluid, FluidsList, Input
import numpy as np
from scipy.optimize import root_scalar
import warnings

system('cls')

# ox_tank_p = 52e5
# ipa_tank_p = 53e5
# ox_vp = ox_tank_p - 1e3
# annulus_od = 0.9e-3

ox_tank_p = 50e5
ipa_tank_p = 50e5
ox_vp = 25e5

ox_inj_Cd = 0.65
fuel_inj_Cd = 0.6

needle_id = 0.2e-3
annulus_id = 0.5e-3

fuel_inj_A = 0.25 * np.pi * needle_id**2


# Orifice or annulus as restriction

ox_orifice_id = 0.5e-3
ox_orifice_A = 0.25 * np.pi * ox_orifice_id**2


annulus_od = 0.7e-3
ox_annulus_A = 0.25 * np.pi * (annulus_od**2 - annulus_id**2)
ox_inj_A = ox_annulus_A

# annulus_od = 1e-3
# ox_annulus_A = 0.25 * np.pi * (annulus_od**2 - annulus_id**2)
# ox_inj_A = ox_orifice_A

ox_inj_CdA = ox_inj_Cd * ox_inj_A

fuel_inj_CdA = fuel_inj_Cd * fuel_inj_A


ox_inj_ratio = ox_annulus_A / ox_orifice_A

n2o = Fluid(FluidsList.NitrousOxide)
n2o.update(Input.pressure(ox_vp), Input.quality(0))
n2o = n2o.isentropic_compression_to_pressure(ox_tank_p)

ipa_rho = 790

igniter = enginesim.engine("configs/igniter.yaml")

cstar_eff = 0.7

def pc_residual(pc: float, engine: enginesim.engine, fuel_inj_p: float, ox_inj_CdA: float, fuel_inj_CdA: float, ox_fluid: Fluid, fuel_rho: float, cstar_eff: float) -> float:
    ox_mdot = flow_models.nhne_mdot(ox_inj_CdA, ox_fluid, pc)
    fuel_mdot = flow_models.spi_mdot(fuel_inj_CdA, fuel_inj_p, pc, fuel_rho)

    engine.mdot_combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        cstar_eff = cstar_eff,
        fuel_mdot = fuel_mdot,
        ox_mdot = ox_mdot
    )

    return pc - engine.pc

def solve_pc(cstar_eff: float, ox_tank_p: float, ipa_tank_p: float, ox_inj_CdA: float, fuel_inj_CdA: float, ox_fluid: Fluid, fuel_rho: float, engine: enginesim.engine, ox_inj_A: float):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = root_scalar(
            pc_residual,
            args=(engine, ipa_tank_p, ox_inj_CdA, fuel_inj_CdA, ox_fluid, fuel_rho, cstar_eff),
            bracket=[101325, (np.max([ox_tank_p, ipa_tank_p]) - 1e3)],
            method='brentq'
        )

    pc = sol.root

    ox_mdot = flow_models.nhne_mdot(ox_inj_CdA, ox_fluid, pc)
    fuel_mdot = flow_models.spi_mdot(fuel_inj_CdA, ipa_tank_p, pc, fuel_rho)

    ox_eff_Cd = flow_models.spi_CdA(ox_mdot, ox_tank_p, pc, ox_fluid.density) / ox_inj_A

    OF = ox_mdot / fuel_mdot

    engine.combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        pc = pc,
        OF = OF,
        cstar_eff = cstar_eff
    )
    
    return pc, OF, ox_eff_Cd

pc, OF, ox_eff_Cd = solve_pc(cstar_eff, ox_tank_p, ipa_tank_p, ox_inj_CdA, fuel_inj_CdA, n2o, ipa_rho, igniter, ox_inj_A)

igniter.combustion_sim(
    fuel = 'Isopropanol',
    ox = 'N2O',
    pc = pc,
    OF = OF,
    cstar_eff = cstar_eff
)
igniter.print_data()

n2o_downstream = n2o.clone().with_state(Input.pressure(pc), Input.enthalpy(n2o.enthalpy))
n2o_rho = n2o_downstream.density
v_ox = igniter.ox_mdot / (n2o_rho * ox_annulus_A)
v_fuel = igniter.fuel_mdot / (ipa_rho * fuel_inj_A)

velocity_ratio = v_ox / v_fuel

print(f"Ox effective Cd: {ox_eff_Cd:.4f}, Annulus / Orifice Area Ratio: {ox_inj_ratio:.2f}")
print(f"Ox Velocity: {v_ox:.2f} m/s, Fuel Velocity: {v_fuel:.2f} m/s, Velocity Ratio: {velocity_ratio:.2f}")