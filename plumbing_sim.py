from typing import List, Optional
from scipy.optimize import root_scalar
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
import numpy as np
import unit_conversion as uc
from os import system

system('cls')

def colebrook(f: float, Re: float, rel_roughness: float) -> float:
    """
    Colebrook-White equation for turbulent friction factor
    Returns the residual that should equal zero when f is correct
    """
    return (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))

class feed_system_component:
    """Base class for plumbing components"""
    def __init__(self, name: str = ""):
        self.name = name
        self.inlet_pressure = None
        self.outlet_pressure = None
    
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop across component"""
        raise NotImplementedError
    
    def get_inlet_pressure(self) -> float:
        """Get inlet pressure"""
        if self.inlet_pressure is None:
            raise ValueError("Inlet pressure not set")
        return self.inlet_pressure
    
    def get_outlet_pressure(self) -> float:
        """Get outlet pressure"""
        if self.outlet_pressure is None:
            raise ValueError("Outlet pressure not set")
        return self.outlet_pressure

class pipe(feed_system_component):
    """Pipe component with friction losses
    Parameters
    ----------
    id : float
        Inner diameter of the pipe
        Units: m
    L : float
        Length of the pipe
        Units: m
    abs_roughness : float
        Absolute roughness of the pipe
        Units: m
    name : str, optional
        Name of the pipe, by default "Pipe"
    """
    def __init__(self, id: float, L: float, abs_roughness: float, name: str = "Pipe"):
        super().__init__(name)
        self.id = id  # m
        self.L = L  # m
        self.abs_roughness = abs_roughness  # m
        self.area = 0.25 * np.pi * id ** 2
        self._velocity = None  # m/s
    
    def re(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate Reynolds number"""
        rho = fluid.density()
        visc = fluid.viscosity()
        return rho * self.velocity(mdot, rho) * self.id / visc

    def friction_factor(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate Darcy friction factor using Colebrook equation approximation"""
        Re = self.re(fluid, mdot)
        rel_roughness = self.abs_roughness / self.id
        
        is_laminar = Re < 3000

        if is_laminar:
            return 64 / Re
        else:  # Turbulent flow - colebrook equation
            return root_scalar(colebrook, args=(Re, rel_roughness), bracket=[0.00001, 1]).root
    
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using Darcy-Weisbach equation"""
        f = self.friction_factor(fluid, mdot)
        rho = fluid.density()
        velocity = self.velocity(mdot, rho)
        return f * (self.L / self.id) * (rho * velocity ** 2) * 0.5

    def velocity(self, mdot: float, rho: float) -> float:
        """Calculate velocity based on mass flow rate and density"""
        self._velocity = mdot / (rho * self.area)
        return self._velocity

class orifice(feed_system_component):
    """Orifice component with flow coefficient
    
    Parameters
    ----------
    CdA : float, optional
        Discharge coefficient * Area (m²)
    Cv : float, optional
        Flow coefficient in US units (GPM at 1 psi drop)
    Kv : float, optional
        Flow coefficient in metric units (m³/h at 1 bar drop)
    name : str, optional
        Name of the orifice, by default "Orifice"
    
    Note: Only one of CdA, Cv, or Kv should be provided
    """
    def __init__(self, CdA: float = None, Cv: float = None, Kv: float = None, name: str = "Orifice"):
        super().__init__(name)
        
        # Check that only one parameter is provided
        params_provided = sum(x is not None for x in [CdA, Cv, Kv])
        if params_provided != 1:
            raise ValueError("Exactly one of CdA, Cv, or Kv must be provided")
        
        if CdA is not None:
            self.CdA = CdA  # m²
        elif Cv is not None:
            self.CdA = Cv / 29.84 * 6.309e-5  # Convert to m²
        elif Kv is not None:
            self.CdA = Kv / 36 * 2.778e-4  # Convert to m²
    
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using orifice equation"""
        rho = fluid.density()
        # ΔP = (ṁ / (Cd * A))² / (2 * ρ)
        return (mdot / self.CdA) ** 2 / (2 * rho)

class base_fluid_class:
    """Base class for fluid properties"""
    def __init__(self, name: str = "Generic Fluid"):
        self.name = name
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at given conditions"""
        raise NotImplementedError
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid viscosity at given conditions"""
        raise NotImplementedError
    
    def get_properties(self, temperature: float = None, pressure: float = None) -> dict:
        """Get all fluid properties at given conditions"""
        return {
            'density': self.density(temperature, pressure),
            'viscosity': self.viscosity(temperature, pressure),
            'name': self.name
        }

class incompressible_fluid(base_fluid_class):
    """Incompressible fluid with constant properties"""
    def __init__(self, density: float, visc: float = 1e-6, name: str = "Incompressible Fluid"):
        super().__init__(name)
        self._density = density  # kg/m³ (private attribute)
        self._visc = visc  # Pa·s (private attribute)
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density - constant for incompressible fluid"""
        return self._density
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid viscosity - constant for incompressible fluid"""
        return self._visc
    
class pyfluid(base_fluid_class):
    """Fluid class for interfacing with pyfluids library"""
    def __init__(self, input_fluid: Fluid, temperature: float, pressure: float, name: str = "PyFluid"):
        super().__init__(name)
        self._temperature = temperature  # K
        self._pressure = pressure  # Pa
        self.fluid = input_fluid
        self.fluid.update(Input.temperature(uc.K2degC(temperature)), Input.pressure(pressure))
    
    def update_state(self, temperature: float = None, pressure: float = None):
        """Update fluid state with new temperature and/or pressure"""
        if temperature is not None:
            self._temperature = temperature
        if pressure is not None:
            self._pressure = pressure
        self.fluid.update(Input.temperature(uc.K2degC(self._temperature)), Input.pressure(self._pressure))
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.fluid.density
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid dynamic viscosity at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.fluid.dynamic_viscosity

class thermo_fluid(base_fluid_class):
    """Fluid class for interfacing with thermo library"""
    def __init__(self, chemical: Chemical, temperature: float, pressure: float, name: str = "Thermo Fluid"):
        super().__init__(name)
        self.chemical = chemical
        self._temperature = temperature  # K
        self._pressure = pressure  # Pa
        self.update_state(temperature, pressure)
    
    def update_state(self, temperature: float = None, pressure: float = None):
        """Update fluid state with new temperature and/or pressure"""
        if temperature is not None:
            self._temperature = temperature
        if pressure is not None:
            self._pressure = pressure
        self.chemical.calculate(self._temperature, self._pressure)
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.chemical.rho
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid dynamic viscosity at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.chemical.mu

class fluid_system:
    """Fluids system with series components only."""
    def __init__(self, name: str = "Fluid System"):
        self.name = name
        self.line: List[feed_system_component] = []
        self.fluid: Optional[base_fluid_class] = None
        self.mdot: Optional[float] = None  # Store current mass flow rate
    
    def add_component(self, *components: feed_system_component):
        """Add one or more components to the system"""
        for component in components:
            self.line.append(component)
    
    def set_fluid(self, fluid: base_fluid_class):
        """Set the fluid properties"""
        self.fluid = fluid
    
    def calc_total_dp(self, mdot: float) -> float:
        """Calculate total pressure drop for given mass flow rate"""
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")
        
        total_dp = 0

        for component in self.line:
            dp = component.dp(self.fluid, mdot)
            total_dp += dp
            
        return total_dp
    
    def solve_pressures(self, inlet_pressure: float = None, mdot: float = None, outlet_pressure: float = None) -> None:
        """
        Solve for pressure at each point given inlet pressure and mass flow rate,
        or outlet pressure and mass flow rate.

        Specify exactly one of inlet_pressure or outlet_pressure.
        """
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")
        if (inlet_pressure is None and outlet_pressure is None) or (inlet_pressure is not None and outlet_pressure is not None):
            raise ValueError("Specify exactly one of inlet_pressure or outlet_pressure")

        # Store the mass flow rate
        self.mdot = mdot
        
        # Calculate total pressure drop
        total_dp = self.calc_total_dp(mdot)

        if inlet_pressure is not None:
            current_pressure = inlet_pressure
            for i, component in enumerate(self.line):
                dp = component.dp(self.fluid, mdot)
                component.inlet_pressure = current_pressure
                current_pressure -= dp
                component.outlet_pressure = current_pressure
                if current_pressure < 0:
                    raise ValueError(f"Negative pressure encountered in feed system component {i}: '{component.name}'.")
        else:
            # outlet_pressure is specified, solve for inlet_pressure
            inlet_pressure = outlet_pressure + total_dp
            current_pressure = inlet_pressure
            for i, component in enumerate(self.line):
                dp = component.dp(self.fluid, mdot)
                component.inlet_pressure = current_pressure
                current_pressure -= dp
                component.outlet_pressure = current_pressure
                if current_pressure < 0:
                    raise ValueError(f"Negative pressure encountered in feed system component {i}: '{component.name}'.")

    def solve_mdot(self, inlet_pressure: float, outlet_pressure: float) -> float:
        """Solve for mass flow rate given inlet and outlet pressures"""
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")
        
        target_dp = inlet_pressure - outlet_pressure
        
        def dp_func(mdot):
            return self.calc_total_dp(mdot) - target_dp
        
        # Solve for mdot
        mdot = root_scalar(dp_func, bracket=[1e-5, 100], method='brentq').root
        
        # Store and update pressures
        self.mdot = mdot
        self.solve_pressures(inlet_pressure, mdot)
        
        return mdot
    
    def print_pressures(self) -> None:
        """Print pressures at each component"""
        if self.mdot is None:
            raise ValueError("Mass flow rate not set. Run solve_pressures() or solve_mdot() first.")
            
        print(self.name + " - Component Pressures:")
        print("-" * 50)
        total_dp = self.line[0].inlet_pressure - self.line[-1].outlet_pressure
        for i, component in enumerate(self.line, 1):
            if component.inlet_pressure is not None:
                pressure_drop = component.inlet_pressure - component.outlet_pressure
                print(f"{i}. {component.name} ({type(component).__name__}):")
                print(f"   {'Inlet Pressure':<18}: {component.inlet_pressure/1e5:<6.3f} Bar")
                print(f"   {'Outlet Pressure':<18}: {component.outlet_pressure/1e5:<6.3f} Bar")
                print(f"   {'Pressure Drop':<18}: {pressure_drop/1e5:<6.3f} Bar")
                if isinstance(component, pipe):
                    print(f"   {'Velocity':<18}: {component.velocity(self.mdot, self.fluid.density()):<6.3f} m/s")
            if i < len(self.line):
                print("")
        print(f"Total Pressure Drop: {total_dp/1e5:.3f} Bar")
        print("-" * 50 + "\n")

    def print_components(self, show_parameters: bool = True) -> None:
        """Print component layout and optionally their parameters"""
        if show_parameters:
            print(self.name + " - Component Parameters:")
            print("-" * 50)
            for i, component in enumerate(self.line, 1):
                print(f"{i}. {component.name} ({type(component).__name__}):")
                if isinstance(component, pipe):
                    print(f"   {'Inner Diameter':<18}: {component.id*1e3:<6.3f} mm")
                    print(f"   {'Length':<18}: {component.L:<6.4f} m")
                    print(f"   {'Roughness':<18}: {component.abs_roughness*1e3:<6.4f} mm")

                elif isinstance(component, orifice):
                    print(f"   {'CdA':<18}: {component.CdA*1e6:<6.3f} mm²")

                if i < len(self.line):
                    print("")
        else:
            print(self.name + " - Components:")
            for i, component in enumerate(self.line, 1):
                print(f"{i}. {component.name} ({type(component).__name__})")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    # Define fluids
    n2o = thermo_fluid(Chemical("nitrous oxide"), temperature = 273, pressure = 32e5, name="N2O") # Cold nitrous
    ipa = thermo_fluid(Chemical("isopropanol"), temperature = 290, pressure = 40e5, name="IPA")
    ethanol = pyfluid(Fluid(FluidsList.Ethanol), temperature = 290, pressure = 40e5, name="Ethanol")

    # Create fuel and ox feed systems
    fuel_feed = fluid_system("Fuel Feed System")
    ox_feed = fluid_system("Ox Feed System")

    # Pipe dimensions
    pipe_id_1_2 = uc.in2m(0.5 - 2*0.036)
    pipe_id_3_8 = uc.in2m(0.375 - 2*0.036)

    # Setup fuel feed system
    fuel_pipes = pipe(id = pipe_id_3_8, L=0.5, abs_roughness = 0.015e-3, name = "Fuel Feed System Plumbing")
    regen_channels = orifice(CdA = 24.4e-6, name = "Regen Channels")
    fuel_injector = orifice(CdA = 21e-6, name = "Fuel Injector")
    # fuel_injector = orifice(CdA = 17.4e-6, name = "Fuel Injector")
    fuel_feed.add_component(fuel_pipes, regen_channels, fuel_injector)

    fuel_feed.set_fluid(ipa)

    # Add ox components
    ox_pipes = pipe(id = pipe_id_1_2, L=1.5, abs_roughness = 0.015e-3, name = "Ox Feed System Plumbing")
    ox_injector = orifice(CdA = 78e-6, name = "Ox Injector")
    ox_feed.add_component(ox_pipes, ox_injector)

    ox_feed.set_fluid(n2o)

    tank_p = 50e5
    pc = 32e5
    ox_mdot = 2.335
    fuel_mdot = 0.762

    # fuel_feed.solve_pressures(outlet_pressure = pc, mdot = fuel_mdot)
    fuel_feed.solve_mdot(inlet_pressure = tank_p, outlet_pressure = pc)
    fuel_feed.print_pressures()
    fuel_feed.print_components()

    # ox_feed.solve_pressures(outlet_pressure = pc, mdot = ox_mdot)
    ox_feed.solve_mdot(inlet_pressure = tank_p, outlet_pressure = pc)
    ox_feed.print_pressures()
    ox_feed.print_components()

    print(f"Fuel mdot: {fuel_feed.mdot:.4f} kg/s, Ox mdot: {ox_feed.mdot:.4f} kg/s, OF Ratio: {ox_feed.mdot / fuel_feed.mdot:.2f}")

    # # Example 2: Solve for mass flow rate given pressures
    # outlet_pressure = 1.01325e5  # Pa (1 bar)
    # calculated_flow_rate = system.solve_mdot(inlet_pressure, outlet_pressure)
    # print(f"Calculated mass flow rate: {calculated_flow_rate:.4f} kg/s")
    
    # # Print system summary
    # system.print_pressures()
    # system.print_components(True)