from typing import List, Optional
from scipy.optimize import root_scalar, fsolve
import numpy as np
import enginesim as es
from custom_fluids import base_fluid_class
import time

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
        self.abs_roughness = abs_roughness # m
        self.rel_roughness = abs_roughness / id
        self.area = 0.25 * np.pi * id ** 2
        self._velocity = None  # m/s
    
    def re(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate Reynolds number"""
        rho = fluid.density()#pressure = self.inlet_pressure)
        visc = fluid.viscosity()#pressure = self.inlet_pressure)
        u = self.velocity(mdot, rho)
        return rho * u * self.id / visc

    def friction_factor(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate Darcy friction factor using Colebrook equation approximation"""
        Re = self.re(fluid, mdot)
        
        is_laminar = Re < 4000

        if is_laminar:
            return 64 / Re
        else:  # Turbulent flow - colebrook equation
            # f = root_scalar(colebrook, args=(Re, self.rel_roughness), bracket=[0.00001, 1]).root # Colebrook solver
            # f = 1.325 / (np.log((self.rel_roughness / 3.7) + (5.74 / Re ** 0.9))) ** 2  # Swamee-Jain
            f = (-1.8 * np.log10((self.rel_roughness/3.7)**1.11 + (6.9 / Re))) ** -2 # Haaland
            return f
    
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using Darcy-Weisbach equation"""
        f = self.friction_factor(fluid, mdot)
        rho = fluid.density()#pressure = self.inlet_pressure)
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
    name : str, optional
        Name of the orifice, by default "Orifice"
    """
    def __init__(self, CdA: float, name: str = "Orifice"):
        super().__init__(name)
        self.CdA = CdA
        
    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using orifice equation"""
        rho = fluid.density()#pressure = self.inlet_pressure)
        # ΔP = (ṁ / (Cd * A))² / (2 * ρ)
        return (mdot / self.CdA) ** 2 / (2 * rho)

class diameter_change(feed_system_component):
    def __init__(self, Cd: float, D_down: float, D_up: float, name: str = "Orifice (Diameter Change)"):
        super().__init__(name)
        self.Cd = Cd
        self.D = D_down
        self.D_up = D_up
        self.A = 0.25 * np.pi * D_down ** 2
        self.A_up = 0.25 * np.pi * D_up ** 2
        self.beta = D_down / D_up
        self.Cd_eff = Cd / np.sqrt(1 - self.beta**4)
        self.CdA_eff = self.Cd_eff * self.A

    def dp(self, fluid: 'base_fluid_class', mdot: float) -> float:
        """Calculate pressure drop using orifice equation"""
        rho = fluid.density()#pressure = self.inlet_pressure)
        # ΔP = (ṁ / (Cd * A))² / (2 * ρ)
        return (mdot / self.CdA_eff) ** 2 / (2 * rho)

# Regulator
class regulator(feed_system_component):
    def __init__(self, open_CdA: float, opening_dP_range: float, name: str = "Regulator"):
        super().__init__(name)
        self.open_CdA = open_CdA
        self.opening_const = open_CdA / opening_dP_range # CdA per unit dP

# Valve types
class valve(feed_system_component):
    """Base valve component with variable CdA"""

    def __init__(self, open_CdA: float, name: str, max_rate: float):
        super().__init__(name)
        self.open_CdA = open_CdA  # m²
        self.position = 1.0  # Fully open by default
        self.max_rate = max_rate

    def set_position(self, position: float) -> None:
        """Set valve position (0 = closed, 1 = fully open)"""
        self.position = np.clip(position, 0, 1)
    
    def get_position(self) -> float:
        """Get current valve position"""
        return self.position
    
    def get_max_rate(self) -> float:
        """Get maximum rate of change for valve position"""
        return self.max_rate
    
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

        rho = fluid.density()#pressure=self.inlet_pressure)
        return (mdot / effective_CdA) ** 2 / (2 * rho)

class ball_valve(valve):    
    def __init__(self, open_CdA: float, name: str = "Ball Valve", max_rate: float = 10):
        super().__init__(open_CdA, name, max_rate)

    def get_flow_coeff(self, position: float) -> float:
        # flow_arr = np.array([0, 3, 5, 9, 15, 23, 35, 58, 77, 90]) / 90
        flow_arr = np.array([0, 3, 4, 8, 13, 19, 30, 50, 67, 78]) / 78
        pos_arr  = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) / 90

        return np.interp(position, pos_arr, flow_arr)

class needle_valve(valve):
    def __init__(self, open_CdA: float, max_rate: float = 5, name: str = "Needle Valve"):
        super().__init__(open_CdA, name, max_rate)

    def get_flow_coeff(self, position: float) -> float:
        return position

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

    def calc_total_dp(self, mdot: float = None) -> float:
        """Calculate total pressure drop for given mass flow rate"""
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")
        
        if mdot is None:
            mdot = self._mdot
        
        total_dp = 0

        current_pressure = self._inlet_pressure
        for component in self.line:
            dp = component.dp(self.fluid, mdot)
            total_dp += dp
            component.inlet_pressure = current_pressure
            current_pressure -= dp
            component.outlet_pressure = current_pressure

            if current_pressure < 0:
                print("Warning: Negative pressure encountered")
                return np.inf

        return total_dp

    def get_total_dp(self) -> float:
        return self._total_dp

    def get_mdot(self) -> Optional[float]:
        """Get the current mass flow rate"""
        if self._mdot is None:
            raise ValueError("Mass flow rate not set. Run solve_pressures() or solve_mdot() first.")
        return self._mdot
    
    def set_mdot(self, mdot: float) -> None:
        if mdot <= 0:
            raise ValueError("Mass flow rate must be positive")
        self._mdot = mdot

    def set_inlet_pressure(self, inlet_pressure: float) -> None:
        """Set the inlet pressure of the feed system"""
        self._inlet_pressure = inlet_pressure

    def solve_pressures(self, inlet_pressure: float = None, mdot: float = None) -> None:
        """
        Solve for pressure at each point given inlet pressure and mass flow rate,
        or outlet pressure and mass flow rate.

        Specify exactly one of inlet_pressure or outlet_pressure.
        """
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")

        # Store the mass flow rate
        if mdot is not None:
            self._mdot = mdot

        # print(f"Feed system: {self.name} - Solving pressures with inlet pressure: {self._inlet_pressure/1e5:.3f}, mass flow rate: {self._mdot:.3f}")

        if inlet_pressure is not None:
            self._inlet_pressure = inlet_pressure

        self._total_dp = self.calc_total_dp(self._mdot)

    def solve_mdot(self, inlet_pressure: float, outlet_pressure: float) -> float:
        """Solve for mass flow rate given inlet and outlet pressures"""
        if not self.fluid:
            raise ValueError("Fluid must be set before solving")

        self._inlet_pressure = inlet_pressure

        target_dp = inlet_pressure - outlet_pressure
        
        def dp_func(mdot):
            return self.calc_total_dp(mdot) - target_dp
        
        # Solve for mdot
        tol = 1e-4
        mdot = root_scalar(dp_func, bracket=[1e-5, 100], method='brentq', xtol=tol).root
        
        # Store and update pressures
        self._mdot = mdot
        self.solve_pressures(inlet_pressure, mdot)
        
        return mdot
    
    def print_pressures(self) -> None:
        """Print pressures at each component"""
        if self._mdot is None:
            raise ValueError("Mass flow rate not set. Run solve_pressures() or solve_mdot() first.")

        print(f"{self.name} - Component Pressures:")
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
                    print(f"\n   {'Velocity':<18}: {component.velocity(self._mdot, self.fluid.density()):<6.3f} m/s")
                    print(f"   {'Friction Factor':<18}: {component.friction_factor(self.fluid, self._mdot):<6.4f}")
                    print(f"   {'Re':<18}: {component.re(self.fluid, self._mdot):<6.3e}")
            if i < len(self.line):
                print("")
        print(f"\nMass Flow Rate: {self._mdot:.4f} kg/s")
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

                elif isinstance(component, valve):
                    print(f"   {'Fully Open CdA':<18}: {component.open_CdA*1e6:<6.3f} mm²")
                    print(f"   {'Current CdA':<18}: {component.get_effective_CdA()*1e6:<6.3f} mm²")
                    print(f"   {'Position':<18}: {component.position:.2f} (0 = closed, 1 = fully open)")

                elif isinstance(component, diameter_change):
                    print(f"   {'Cd':<18}: {component.Cd:.3f}")
                    print(f"   {'Effective Cd':<18}: {component.Cd_eff:.3f}")
                    print(f"   {'Diameter':<18}: {component.D*1e3:.3f} mm")
                    print(f"   {'Upstream Diameter':<18}: {component.D_up*1e3:.3f} mm")
                    print(f"   {'CdA':<18}: {component.Cd * component.A * 1e6:.3f} mm²")
                    print(f"   {'Effective CdA':<18}: {component.CdA_eff*1e6:.3f} mm²")

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
    def __init__(self, file: str, name: str = "Rocket Engine", pamb: float = 101325, cstar_eff: float = 1, cf_eff: float = 1):
        self.name = name
        self.file = file
        self.es_engine = es.engine(self.file)
        self.pamb = pamb
        self.cstar_eff = cstar_eff
        self.cf_eff = cf_eff

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
        return self.es_engine.pc

    def mdot_combustion_sim(self, fuel_mdot: float, ox_mdot: float, full_sim: bool = True) -> None:
        self.es_engine.mdot_combustion_sim(
            fuel=self.fuel,
            ox=self.ox,
            fuel_mdot=fuel_mdot,
            ox_mdot=ox_mdot,
            pamb=self.pamb,
            cstar_eff=self.cstar_eff,
            cf_eff=self.cf_eff,
            simplified=not(full_sim)
        )

    def simple_combustion_sim(self, pc: float = None, OF: float = None) -> None:
        self.set_pc_of(pc, OF)
        
        self.es_engine.combustion_sim(
            fuel=self.fuel,
            ox=self.ox,
            pc=self._pc,
            OF=self._OF,
            pamb=self.pamb,
            cstar_eff=self.cstar_eff,
            cf_eff=self.cf_eff,
            simplified=True
        )
        self._pc = pc
        self._OF = OF

        self.thrust = self.es_engine.thrust
        self.isp = self.es_engine.isp
        self.cstar = self.es_engine.cstar

    def pc_of_mdot_calc(self, pc: float, OF: float) -> tuple:
        return self.es_engine.pc_of_mdot_calc(self.fuel, self.ox, pc, OF, cstar_eff=self.cstar_eff)

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
class propulsion_system:
    """Container for coupled fuel and oxidizer feed systems"""
    def __init__(self, fuel_system: feed_system, ox_system: feed_system, engine: engine):
        self.fuel_system = fuel_system
        self.ox_system = ox_system
        self.engine = engine
        self.engine.set_props(fuel_system.fluid.cea_name, ox_system.fluid.cea_name)
    
    def solve(self, print_results = False) -> None:
        """Solve coupled systems and store results"""
        [fuel_mdot, ox_mdot] = solve_coupled_system(
            self.fuel_system,
            self.ox_system,
            self.engine,
        )
        # print(f"Calculated mass flow rates: Fuel = {fuel_mdot:.6f} kg/s, Oxidizer = {ox_mdot:.6f} kg/s")
        self.fuel_system.set_mdot(fuel_mdot)
        self.ox_system.set_mdot(ox_mdot)
        if print_results:
            self.print_summary()
        else:
            self.calc_final_values(False)

    def calc_final_values(self, full_engine_sim: bool = True) -> None:
        engine_pc = self.engine.get_pc()

        self.fuel_system.solve_pressures(mdot=self.fuel_system.get_mdot())
        self.ox_system.solve_pressures(mdot=self.ox_system.get_mdot())

        # Verify that outlet pressures match engine pc
        fuel_residual = self.fuel_system.line[-1].outlet_pressure - engine_pc
        ox_residual = self.ox_system.line[-1].outlet_pressure - engine_pc

        residual_tol = 1
        if np.max(np.abs([fuel_residual, ox_residual])) > residual_tol:
            raise ValueError(f"Solver did not converge: fuel residual = {fuel_residual:.4f}, ox residual = {ox_residual:.4f}")

        self.engine.mdot_combustion_sim(self.fuel_system.get_mdot(), self.ox_system.get_mdot(), full_sim=full_engine_sim)

    def print_summary(self) -> None:
        self.calc_final_values(True)

        print("=" * 60)
        print(f"COUPLED FEED SYSTEM ANALYSIS")
        print("=" * 60)
        
        self.fuel_system.print_pressures()
        self.fuel_system.print_components()
        
        self.ox_system.print_pressures() 
        self.ox_system.print_components()
        
        # if full_engine_sim:
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
    OF_guess = ox_mdot / fuel_mdot
    cstar = 800
    pc_guess = cstar * total_mdot / engine.es_engine.at

    initial_guess = [pc_guess, OF_guess]  # Initial guess for pressure and OF
    print(f"Initial guess: pc = {pc_guess/1e5:.2f} Bar, OF = {OF_guess:.2f}, ox_mdot = {ox_mdot:.6f} kg/s, fuel_mdot = {fuel_mdot:.6f} kg/s")

    def system_equations(pc_OF):
        pc, OF = pc_OF

        [fuel_mdot, ox_mdot] = engine.pc_of_mdot_calc(pc, OF)
        fuel_system.solve_pressures(mdot=fuel_mdot)
        ox_system.solve_pressures(mdot=ox_mdot)

        fuel_feed_residual = fuel_system.get_total_dp() - (fuel_system._inlet_pressure - pc)
        ox_feed_residual = ox_system.get_total_dp() - (ox_system._inlet_pressure - pc)

        # print(f"""Solver iteration: pc = {pc/1e5:.6f} Bar, OF = {OF:.6f}
        #       fuel residual: {fuel_feed_residual:.4f}
        #       ox residual: {ox_feed_residual:.4f}\n""")
        
        return fuel_feed_residual, ox_feed_residual

    tol = 1e-7
    t = time.time()
    sol = fsolve(system_equations, initial_guess, xtol=tol, full_output=False)
    print(f"Solver converged in {(time.time() - t)*1e3:.2f} ms")

    pc, OF = sol
    engine.simple_combustion_sim(pc, OF)
    return engine.pc_of_mdot_calc(pc, OF)