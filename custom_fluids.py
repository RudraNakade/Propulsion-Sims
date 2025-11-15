from pyfluids import Fluid, Input
from thermo import Chemical

class base_fluid_class:
    """Base class for fluid properties"""
    def __init__(self, name: str = None, cea_name: str = None):
        self.name = name
        self.cea_name = cea_name

    def update_state(self, temperature: float = None, pressure: float = None):
        """Update fluid state with new temperature and/or pressure"""
        raise NotImplementedError

    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at given conditions"""
        raise NotImplementedError
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid viscosity at given conditions"""
        raise NotImplementedError

    def conductivity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid thermal conductivity at given conditions"""
        raise NotImplementedError

    def heat_capacity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid heat capacity at given conditions"""
        raise NotImplementedError

    def vapor_pressure(self, temperature: float = None) -> float:
        """Get fluid vapor pressure at given conditions"""
        raise NotImplementedError

    def saturation_temperature(self, pressure: float = None) -> float:
        """Get fluid saturation temperature at given conditions"""
        raise NotImplementedError

class incompressible_fluid(base_fluid_class):
    """Incompressible fluid with constant properties"""
    def __init__(self, density: float, visc: float = 1e-6, name: str = "Incompressible Fluid", cea_name: str = None):
        """
        Initialize incompressible fluid with constant properties.
        """
        super().__init__(name, cea_name)
        self._density = density  # kg/m³ (private attribute)
        self._visc = visc  # Pa·s (private attribute)

    def update_state(self, temperature: float = None, pressure: float = None):
        """Update fluid state with new temperature and/or pressure"""
        pass

    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density - constant for incompressible fluid"""
        return self._density
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid viscosity - constant for incompressible fluid"""
        return self._visc

    def conductivity(self, temperature: float = None, pressure: float = None) -> None:
        """Get fluid thermal conductivity - constant for incompressible fluid"""
        return None

    def heat_capacity(self, temperature: float = None, pressure: float = None) -> None:
        """Get fluid heat capacity - constant for incompressible fluid"""
        return None

    def vapor_pressure(self, temperature = None):
        return None

    def saturation_temperature(self, pressure = None):
        return None

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
            self.fluid.update(Input.temperature(self._temperature - 273.15), Input.pressure(self._pressure))
    
    def density(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid density at current or specified conditions"""
        self.update_state(temperature, pressure)
        return self.fluid.density
    
    def viscosity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid dynamic viscosity at current or specified conditions"""
        self.update_state(temperature, pressure)
        return self.fluid.dynamic_viscosity

    def conductivity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid thermal conductivity at current or specified conditions"""
        self.update_state(temperature, pressure)
        return self.fluid.conductivity

    def heat_capacity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid heat capacity at current or specified conditions"""
        self.update_state(temperature, pressure)
        return self.fluid.specific_heat

    def vapor_pressure(self, temperature: float = None) -> float | None:
        """Get fluid vapor pressure at current or specified conditions"""
        if temperature is None:
            temperature = self._temperature
        fluid_clone = self.fluid.clone()

        if temperature > fluid_clone.critical_temperature:
            return None
        
        fluid_clone.update(Input.temperature(temperature - 273.15), Input.quality(0))
        return fluid_clone.pressure
    
    def saturation_temperature(self, pressure: float = None) -> float | None:
        """Get fluid saturation temperature at current or specified conditions"""
        if pressure is None:
            pressure = self._pressure
        fluid_clone = self.fluid.clone()

        if pressure > fluid_clone.critical_pressure:
            return None
        
        fluid_clone.update(Input.pressure(pressure), Input.quality(0))
        return fluid_clone.temperature + 273.15

class thermo_fluid(base_fluid_class):
    """Fluid class for interfacing with thermo library"""
    def __init__(self, chemical_name: str, temperature: float = None, pressure: float = None, name: str = "Thermo Fluid", cea_name: str = None):
        super().__init__(name, cea_name)
        self.chemical = Chemical(chemical_name)
        self._temperature = temperature  # K
        self._pressure = pressure  # Pa
        if temperature is not None and pressure is not None:
            self.update_state(temperature=temperature, pressure=pressure)

    def update_state(self, temperature: float = None, pressure: float = None, enthalpy: float = None, entropy: float = None):
        """Update fluid state with new temperature and/or pressure"""
        # Count how many parameters are specified
        params = [temperature, pressure, enthalpy, entropy]
        specified_count = sum(1 for param in params if param is not None)

        if specified_count != 2:
            raise ValueError("Exactly two parameters must be specified")
        
        # Update instance variables for specified parameters
        if temperature is not None:
            self._temperature = temperature
        if pressure is not None:
            self._pressure = pressure
        if enthalpy is not None:
            self._enthalpy = enthalpy
        if entropy is not None:
            self._entropy = entropy
        
        # Handle all possible pairs
        if temperature is not None and pressure is not None:
            self.chemical.calculate(T=temperature, P=pressure)
            self._enthalpy = self.chemical.H
            self._entropy = self.chemical.S
        elif temperature is not None and enthalpy is not None:
            self.chemical.calculate_TH(T=temperature, H=enthalpy)
            self._pressure = self.chemical.P
            self._entropy = self.chemical.S
        elif temperature is not None and entropy is not None:
            self.chemical.calculate_TS(T=temperature, S=entropy)
            self._pressure = self.chemical.P
            self._enthalpy = self.chemical.H
        elif pressure is not None and enthalpy is not None:
            self.chemical.calculate_PH(P=pressure, H=enthalpy)
            self._temperature = self.chemical.T
            self._entropy = self.chemical.S
        elif pressure is not None and entropy is not None:
            self.chemical.calculate_PS(P=pressure, S=entropy)
            self._temperature = self.chemical.T
            self._enthalpy = self.chemical.H
        elif enthalpy is not None and entropy is not None:
            raise NotImplementedError("Enthalpy and entropy pair not implemented")
    
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

    def conductivity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid thermal conductivity at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.chemical.k

    def heat_capacity(self, temperature: float = None, pressure: float = None) -> float:
        """Get fluid heat capacity at current or specified conditions"""
        if temperature is not None or pressure is not None:
            self.update_state(temperature, pressure)
        return self.chemical.Cp

    def vapor_pressure(self, temperature: float = None) -> float:
        """Get fluid vapor pressure at current or specified conditions"""
        if temperature is not None:
            self.update_state(temperature, None)
        return self.chemical.Psat
    
    def saturation_temperature(self, pressure = None):
        if pressure is None:
            pressure = self._pressure
        return self.chemical.Tsat(pressure)