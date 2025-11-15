from pyfluids import Fluid, FluidsList, Input
from scipy.optimize import root_scalar

from os import system
system('cls')

initial_mass = 10  # kg

initial_vp = 40e5
final_vp = 28e5
fluid = FluidsList.NitrousOxide

initial_state = Fluid(fluid).with_state(Input.pressure(initial_vp), Input.quality(0))
final_state = Fluid(fluid).with_state(Input.pressure(final_vp), Input.quality(0))

enthalpy_drop = initial_state.enthalpy - final_state.enthalpy

def venting_mass_loss_eq(mass_lost: float, initial_mass: float, initial_state: Fluid, final_state: Fluid) -> float:
    remaining_mass = initial_mass - mass_lost
    
    required_enthalpy_drop = (initial_state.enthalpy - final_state.enthalpy) * remaining_mass
    
    initial_vapor = Fluid(fluid).with_state(Input.pressure(initial_vp), Input.quality(100))
    final_vapor = Fluid(fluid).with_state(Input.pressure(final_vp), Input.quality(100))

    vaporization_heat_initial = initial_vapor.enthalpy - initial_state.enthalpy
    vaporization_heat_final = final_vapor.enthalpy - final_state.enthalpy

    average_vaporization_heat = 0.5 * (vaporization_heat_initial + vaporization_heat_final)
    
    boiling_enthalpy_loss = average_vaporization_heat * mass_lost
    
    return boiling_enthalpy_loss - required_enthalpy_drop

mass_lost_solution = root_scalar(venting_mass_loss_eq, bracket=[0, initial_mass], args=(initial_mass, initial_state, final_state), method='brentq')
mass_lost = mass_lost_solution.root
print(f"Mass Lost During Venting: {mass_lost:.2f} kg ({mass_lost/initial_mass*100:.1f}% of initial mass)")

print(f"Initial VP: {initial_vp/1e5:.2f} Bar, Final VP: {final_vp/1e5:.2f} Bar")
print(f"Initial Temp: {initial_state.temperature:.2f} °C, Final Temp: {final_state.temperature:.2f} °C")
print(f"Enthalpy Drop: {enthalpy_drop/1e3:.2f} kJ/kg")