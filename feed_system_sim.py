from typing import List, Optional
from scipy.optimize import root_scalar, fsolve
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
import numpy as np
import unit_conversion as uc
import enginesim as es
from os import system
import time

system('cls')

def colebrook(f: float, Re: float, rel_roughness: float) -> float:
    """
    Colebrook-White equation for turbulent friction factor
    Returns the residual that should equal zero when f is correct
    """
    return (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))

# Feed system components
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
        rho = fluid.density(pressure=self.inlet_pressure)
        visc = fluid.viscosity(pressure=self.inlet_pressure)
        u = self.velocity(mdot, rho)
        return rho * u * self.id / visc

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
        rho = fluid.density(pressure=self.inlet_pressure)
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
        rho = fluid.density(pressure=self.inlet_pressure)
        # ΔP = (ṁ / (Cd * A))² / (2 * ρ)
        return (mdot / self.CdA) ** 2 / (2 * rho)

# Valve types
class valve(feed_system_component):
    """Base valve component with variable CdA"""
    
    def __init__(self, open_CdA: float, name: str = "Valve"):
        super().__init__(name)
        self.open_CdA = open_CdA  # m²
        self.position = 1.0  # Fully open by default
    
    def set_position(self, position: float) -> None:
        """Set valve position (0 = closed, 1 = fully open)"""
        self.position = np.clip(position, 0, 1)
    
    def get_flow_coeff(self, position: float) -> float:
        """Get flow multiplier based on valve position - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement get_flow_coeff")
    
    def get_effective_CdA(self) -> float:
        """Get effective CdA based on current position and valve type"""
        flow_coeff = self.get_flow_coeff(self.position)
        return self.open_CdA * flow_coeff
    
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using current effective CdA"""
        effective_CdA = self.get_effective_CdA()
        if effective_CdA <= 0:
            return float('inf')  # Closed valve = infinite pressure drop

        rho = fluid.density(pressure=self.inlet_pressure)
        return (mdot / effective_CdA) ** 2 / (2 * rho)

class ball_valve(valve):    
    def __init__(self, open_CdA: float, name: str = "Ball Valve"):
        super().__init__(open_CdA, name)
    
    def get_flow_coeff(self, position: float) -> float:
        flow_arr = [0, 3, 5, 9, 15, 23, 35, 58, 77, 90] / 90
        pos_arr  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] / 90

        return np.interp(position, pos_arr, flow_arr)

class needle_valve(valve):
    def __init__(self, open_CdA: float, name: str = "Needle Valve"):
        super().__init__(open_CdA, name)
    
    def get_flow_coeff(self, position: float) -> float:
        return position

# Fluids
class base_fluid_class:
    """Base class for fluid properties"""
    def __init__(self, name: str = None, cea_name: str = None):
        self.name = name
        self.cea_name = cea_name

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
    def __init__(self, density: float, visc: float = 1e-6, name: str = "Incompressible Fluid", cea_name: str = None):
        """
        Initialize incompressible fluid with constant properties.
        """
        super().__init__(name, cea_name)
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
    def __init__(self, input_fluid: Fluid, temperature: float = None, pressure: float = None, name: str = "PyFluid", cea_name: str = None):
        super().__init__(name, cea_name)
        self.fluid = input_fluid
        self._temperature = temperature
        self._pressure = pressure
        self.update_state(temperature, pressure)

    def update_state(self, temperature: float, pressure: float):
        """Update fluid state with new temperature and/or pressure"""
        if temperature is not None:
            self._temperature = temperature
        if pressure is not None:
            self._pressure = pressure
        if temperature is not None or pressure is not None:
            self.fluid.update(Input.temperature(uc.K2degC(self._temperature)), Input.pressure(self._pressure))
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at current or specified conditions"""
        # self.update_state(temperature, pressure)
        return self.fluid.density
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid dynamic viscosity at current or specified conditions"""
        self.update_state(temperature, pressure)
        return self.fluid.dynamic_viscosity

class thermo_fluid(base_fluid_class):
    """Fluid class for interfacing with thermo library"""
    def __init__(self, chemical: Chemical, temperature: float = None, pressure: float = None, name: str = "Thermo Fluid", cea_name: str = None):
        super().__init__(name, cea_name)
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
        # if temperature is not None or pressure is not None:
        #     self.update_state(temperature, pressure)
        return self.chemical.rho
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid dynamic viscosity at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.chemical.mu

# Feed system
class feed_system:
    """Fluids system with series components only."""
    def __init__(self, inlet_pressure: float, name: str = "Fluid System"):
        self.name = name
        self.line: List[feed_system_component] = []
        self.fluid: Optional[base_fluid_class] = None
        self._mdot: Optional[float] = None  # Store current mass flow rate
        self._inlet_pressure = inlet_pressure

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
    
    def get_mdot(self) -> Optional[float]:
        """Get the current mass flow rate"""
        if self._mdot is None:
            raise ValueError("Mass flow rate not set. Run solve_pressures() or solve_mdot() first.")
        return self._mdot
    
    def set_mdot(self, mdot: float) -> None:
        if mdot <= 0:
            raise ValueError("Mass flow rate must be positive")
        self._mdot = mdot

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
        self._mdot = mdot
        
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
        self._mdot = mdot
        self.solve_pressures(inlet_pressure, mdot)
        
        return mdot
    
    def print_pressures(self) -> None:
        """Print pressures at each component"""
        if self._mdot is None:
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
                    print(f"   {'Velocity':<18}: {component.velocity(self._mdot, self.fluid.density()):<6.3f} m/s")
            if i < len(self.line):
                print("")
        print(f"\nTotal Pressure Drop: {total_dp/1e5:.3f} Bar")
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

                elif isinstance(component, valve):
                    print(f"   {'Open CdA':<18}: {component.open_CdA*1e6:<6.3f} mm²")
                    print(f"   {'Position':<18}: {component.position:.2f} (0 = closed, 1 = fully open)")
                    print(f"   {'Effective CdA':<18}: {component.get_effective_CdA()*1e6:<6.3f} mm²")

                if i < len(self.line):
                    print("")
        else:
            print(self.name + " - Components:")
            for i, component in enumerate(self.line, 1):
                print(f"{i}. {component.name} ({type(component).__name__})")
        print("-" * 50 + "\n")

# Engine
class engine:
    """Engine class to store engine properties and functions"""
    def __init__(self, file: str, name: str = "Rocket Engine", pamb: float = 1.01325, cstar_eff: float = 1):
        self.name = name
        self.file = file
        self.es_engine = es.engine(self.file)
        self.cstar_eff = cstar_eff
        self.pamb = pamb

    def set_props(self, fuel: str, ox: str) -> None:
        """Set engine propellants"""
        self.fuel = fuel
        self.ox = ox
        self.es_engine.set_props(fuel=fuel, ox=ox)

    def pc_mdot_func(self, fuel_mdot: float, ox_mdot: float) -> float:
        self.mdot_combustion_sim(
            fuel_mdot=fuel_mdot,
            ox_mdot=ox_mdot,
            full_sim=False
        )
        return self.es_engine.pc * 1e5 # Pa

    def mdot_combustion_sim(self, fuel_mdot: float, ox_mdot: float, full_sim: bool = True) -> None:
        self.es_engine.mdot_combustion_sim(
            fuel=self.fuel,
            ox=self.ox,
            fuel_mdot=fuel_mdot,
            ox_mdot=ox_mdot,
            pamb=self.pamb,
            cstar_eff=self.cstar_eff,
            full_sim=full_sim
        )

    def simple_combustion_sim(self, pc: float = None, OF: float = None) -> None:
        self.set_pc_of(pc, OF)
        
        self.es_engine.combustion_sim(
            fuel=self.fuel,
            ox=self.ox,
            pc=self._pc / 1e5,  # Convert to bar
            OF=self._OF,
            pamb=self.pamb,
            cstar_eff=self.cstar_eff,
            simplified=True
        )
        self._pc = pc
        self._OF = OF

        self.thrust = self.es_engine.thrust
        self.isp = self.es_engine.isp
        self.cstar = self.es_engine.cstar

    def initialise(self) -> None:
        """Initialise engine parameters"""
        self.es_engine.initialise()

    def pc_of_mdot_calc(self, pc: float, OF: float) -> tuple:
        return self.es_engine.pc_of_mdot_calc(self.fuel, self.ox, pc/1e5, OF, cstar_eff=self.cstar_eff)

    def print_data(self) -> None:
        self.es_engine.print_data()

    def get_pc(self) -> float:
        return self._pc

    def set_pc_of(self, pc: float = None, OF: float = None) -> None:
        if pc is not None:
            self._pc = pc
        if OF is not None:
            self._OF = OF

# Coupled system
class coupled_feed_system:
    """Container for coupled fuel and oxidizer feed systems"""
    def __init__(self, fuel_system: feed_system, ox_system: feed_system, engine: engine):
        self.fuel_system = fuel_system
        self.ox_system = ox_system
        self.engine = engine
        self.engine.set_props(fuel_system.fluid.cea_name, ox_system.fluid.cea_name)
    
    def solve(self) -> None:
        """Solve coupled systems and store results"""
        [fuel_mdot, ox_mdot] = solve_coupled_system(
            self.fuel_system,
            self.ox_system,
            self.engine,
        )
        self.fuel_system.set_mdot(fuel_mdot)
        self.ox_system.set_mdot(ox_mdot)

    def calc_final_values(self, full_engine_sim: bool = True) -> None:
        self.fuel_system.solve_pressures(inlet_pressure=self.fuel_system._inlet_pressure, mdot=self.fuel_system.get_mdot())
        self.ox_system.solve_pressures(inlet_pressure=self.ox_system._inlet_pressure, mdot=self.ox_system.get_mdot())

        self.engine.mdot_combustion_sim(self.fuel_system.get_mdot(), self.ox_system.get_mdot(), full_sim=full_engine_sim)

    def print_summary(self, full_engine_sim: bool = True) -> None:
        self.calc_final_values(full_engine_sim=full_engine_sim)

        print("=" * 60)
        print(f"COUPLED FEED SYSTEM ANALYSIS")
        print("=" * 60)
        
        self.fuel_system.print_pressures()
        self.fuel_system.print_components()
        
        self.ox_system.print_pressures() 
        self.ox_system.print_components()
        
        if full_engine_sim:
            print(f"ENGINE PERFORMANCE:")
            self.engine.print_data()

        print("=" * 60)

def solve_coupled_system(fuel_system: feed_system, ox_system: feed_system, engine: engine) -> tuple:
    # Calculate initial guess
    fuel_system.solve_mdot(fuel_system._inlet_pressure, 101325)
    ox_system.solve_mdot(ox_system._inlet_pressure, 101325)

    fuel_mdot = fuel_system.get_mdot()
    ox_mdot = ox_system.get_mdot()
    total_mdot = fuel_mdot + ox_mdot
    OF = ox_mdot / fuel_mdot
    cstar = 1000
    pc = cstar * total_mdot / engine.es_engine.at

    initial_guess = [pc, OF]  # Initial guess for pressure and OF
    # print(f"Initial guess: pc = {pc/1e5:.2f} Bar, OF = {OF:.2f}")

    def system_equations(pc_OF):
        pc, OF = pc_OF

        [fuel_mdot, ox_mdot] = engine.pc_of_mdot_calc(pc, OF)

        fuel_feed_residual = fuel_system.calc_total_dp(fuel_mdot) - (fuel_system._inlet_pressure - pc)
        ox_feed_residual = ox_system.calc_total_dp(ox_mdot) - (ox_system._inlet_pressure - pc)

        return fuel_feed_residual, ox_feed_residual

    tol = 1e-5
    sol = fsolve(system_equations, initial_guess, xtol=tol, full_output=False)

    residual_tol = 1e2
    fuel_residual, ox_residual = system_equations(sol)
    if np.max(np.abs([fuel_residual, ox_residual])) > residual_tol:
        raise ValueError(f"Solver did not converge: fuel residual = {fuel_residual:.4f}, ox residual = {ox_residual:.4f}")

    pc, OF = sol
    engine.set_pc_of(pc, OF)
    engine.simple_combustion_sim(pc, OF)
    return engine.pc_of_mdot_calc(pc, OF)

if __name__ == "__main__":
    # Define fluids
    n2o = thermo_fluid(Chemical("nitrous oxide"), temperature = 273, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
    lox = pyfluid(Fluid(FluidsList.Oxygen), temperature = 110, pressure = 40e5, name = "LOX", cea_name = "LOX")
    ipa = thermo_fluid(Chemical("isopropanol"), temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")
    ethanol = pyfluid(Fluid(FluidsList.Ethanol), temperature = 290, pressure = 40e5, name = "Ethanol", cea_name = "Ethanol")
    methanol = pyfluid(Fluid(FluidsList.Methanol), temperature = 290, pressure = 40e5, name = "Methanol", cea_name = "Methanol")
    propane = pyfluid(Fluid(FluidsList.nPropane), temperature = 273, pressure = 40e5, name = "Propane", cea_name = "Propane")

    fuel_tank_p = 50e5  # Pa
    ox_tank_p = 50e5  # Pa

    fuel_feed = feed_system(fuel_tank_p, "Fuel Feed System")
    ox_feed = feed_system(ox_tank_p, "Ox Feed System")

    pipe_id_1_2 = uc.in2m(0.5 - 2*0.036)
    pipe_id_3_8 = uc.in2m(0.375 - 2*0.036)
    abs_roughness = 0.015e-3  # m

    fuel_pipes = pipe(id = pipe_id_3_8, L=0.5, abs_roughness = abs_roughness, name = "Feed System Pipes")
    fuel_valve = needle_valve(open_CdA = uc.Cv2CdA(1.8), name = '1/2" Needle Valve')
    regen_channels = orifice(CdA = 24.4e-6, name = "Regen Channels")
    fuel_injector = orifice(CdA = 17.4e-6, name = "Fuel Injector") # Measured
    fuel_feed.add_component(fuel_pipes, fuel_valve, regen_channels, fuel_injector)

    fuel_feed.set_fluid(ipa)
    # fuel_feed.set_fluid(ethanol)

    ox_pipes = pipe(id = pipe_id_1_2, L=1.5, abs_roughness = abs_roughness, name = "Feed System Pipes")
    ox_valve = needle_valve(open_CdA = uc.Cv2CdA(2.4), name = '3/4" Needle Valve')
    ox_injector = orifice(CdA = 78e-6, name = "N2O Injector")
    ox_feed.add_component(ox_pipes, ox_valve, ox_injector)

    ox_feed.set_fluid(n2o)
    # ox_feed.set_fluid(lox)
    
    engine = engine("configs/l9.cfg")
    
    coupled_system = coupled_feed_system(fuel_feed, ox_feed, engine)
    
    # t = time.time()
    # coupled_system.solve()
    # print(f"Solved coupled systems in {1e3*(time.time() - t):.2f} ms")
    # coupled_system.print_summary()

    valve_positions = np.arange(0.1, 1.01, 0.01)
    ox_mdot = np.zeros_like(valve_positions)
    fuel_mdot = np.zeros_like(valve_positions)
    pc = np.zeros_like(valve_positions)
    OF = np.zeros_like(valve_positions)
    isp = np.zeros_like(valve_positions)
    thrust = np.zeros_like(valve_positions)
    cstar = np.zeros_like(valve_positions)

    for i, pos in enumerate(valve_positions):
        t = time.time()
        fuel_valve.set_position(pos)
        ox_valve.set_position(pos)
        coupled_system.solve()
        fuel_mdot[i] = fuel_feed.get_mdot()
        ox_mdot[i] = ox_feed.get_mdot()
        pc[i] = engine.get_pc()
        OF[i] = ox_mdot[i] / fuel_mdot[i]
        isp[i] = engine.isp
        thrust[i] = engine.thrust
        cstar[i] = engine.cstar
        print(f"Iteration {i+1}/{len(valve_positions)} solved in {1e3*(time.time() - t):.2f} ms\n")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 2, 1)
    plt.plot(valve_positions, fuel_mdot, label='Fuel')
    plt.plot(valve_positions, ox_mdot, label='Ox')
    plt.xlabel('Valve Position')
    plt.ylabel('Mass Flow Rate (kg/s)')
    plt.title('Mass Flow Rates vs Valve Position')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(valve_positions, pc / 1e5)
    plt.xlabel('Valve Position')
    plt.ylabel('Chamber Pressure (bar)')
    plt.title('Chamber Pressure vs Valve Position')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(valve_positions, OF)
    plt.xlabel('Valve Position')
    plt.ylabel('OF Ratio')
    plt.title('OF Ratio vs Valve Position')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(valve_positions, isp)
    plt.xlabel('Valve Position')
    plt.ylabel('Specific Impulse (s)')
    plt.title('Specific Impulse vs Valve Position')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(valve_positions, thrust)
    plt.xlabel('Valve Position')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs Valve Position')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(valve_positions, cstar)
    plt.xlabel('Valve Position')
    plt.ylabel('Characteristic Velocity (m/s)')
    plt.title('C* vs Valve Position')
    plt.grid()

    plt.tight_layout()
    plt.show()