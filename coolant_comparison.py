import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
from rocketcea.cea_obj_w_units import CEA_Obj

class FluidProperty:
    def __init__(self, name, temperatures):
        self.name = name
        self.temps = temperatures
        self.rho = np.zeros_like(self.temps)
        self.cp = np.zeros_like(self.temps)
        self.visc = np.zeros_like(self.temps)
        self.k = np.zeros_like(self.temps)
        self.h = None
        
    def calculate_heat_transfer(self, D_h):
        """Calculate heat transfer coefficient"""
        self.calc_of_isp()
        thrust = 4000 / 40  # N (example value)
        mdot_total = thrust / (9.81 * self.isp)  # kg/s (example value)
        if self.isox:
            mdot = self.OF * mdot_total / (1 + self.OF)
        else:
            mdot = mdot_total / (1 + self.OF)
        v = mdot / (self.rho * np.pi * (D_h / 2)**2)
        Re = self.rho * v * D_h / self.visc
        Pr = self.cp * self.visc / self.k
        Nu = 0.023 * Re**0.8 * Pr**0.4 # Dittus-Boelter
        self.h = Nu * self.k / D_h
        
        return self.h

    def calc_of_isp(self):
        self.cea = CEA_Obj(
            oxName = self.ox_name,
            fuelName = self.fuel_name,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            make_debug_prints=False)
        
        def cstar_func(OF):
            return -self.cea.get_Cstar(Pc = self.pc/1e5, MR = OF)
        
        result = sp.minimize_scalar(cstar_func, bounds=[0.1, 10])
        self.OF = result.x if result.success else np.nan
        eps = self.cea.get_eps_at_PcOvPe(Pc=self.pc/1e5, MR=self.OF, PcOvPe=(self.pc/self.pe))
        [self.isp, _] = self.cea.estimate_Ambient_Isp(Pc=self.pc/1e5, MR=self.OF, eps=eps, Pamb=1.01325, frozen=0, frozenAtThroat=0)


class PyFluidsProperty(FluidProperty):
    def __init__(self, name, fluid_type, temps, fuel_name, ox_name, pc=25e5, pe=1e5, isox=False):
        super().__init__(name, temps)
        self.fluid = Fluid(fluid_type)
        self.ox_name = ox_name
        self.fuel_name = fuel_name
        self.pc = pc
        self.pe = pe
        self.isox = isox
        self.calculate_properties()

    def calculate_properties(self):
        """Calculate fluid properties for each temperature"""
        for i, T in enumerate(self.temps):
            try:
                self.fluid.update(Input.temperature(T-273.15), Input.pressure(self.pc))
                self.rho[i] = self.fluid.density
                self.cp[i] = self.fluid.specific_heat
                self.visc[i] = self.fluid.dynamic_viscosity if self.fluid.dynamic_viscosity > 0 else np.nan
                self.k[i] = self.fluid.conductivity
            except ValueError:
                # Set values to Np.nan if calculation fails
                self.rho[i] = np.nan
                self.cp[i] = np.nan
                self.visc[i] = np.nan
                self.k[i] = np.nan

class IsopropanolProperty(FluidProperty):
    def __init__(self, temps, fuel_name, ox_name, pc=25e5, pe=1e5):
        super().__init__("Isopropanol", temps)
        self.ipa = Chemical("isopropanol")
        self.vp = np.zeros_like(self.temps)
        self.ox_name = ox_name
        self.fuel_name = fuel_name
        self.pc = pc
        self.pe = pe
        self.isox = False
        self.calculate_properties()
        
    def calculate_properties(self):
        """Calculate isopropanol properties using thermo library"""
        for i, T in enumerate(self.temps):
            self.ipa.calculate(T, self.pc)
            self.rho[i] = self.ipa.rho
            self.cp[i] = self.ipa.Cp
            self.k[i] = self.ipa.k
            self.vp[i] = self.ipa.Psat
            self.visc[i] = self.ipa.mu

temperatures = np.linspace(270, 450, 100)
pc = 20e5  # Pressure in Pa
pe = 1.01325e5  # Pressure in Pa

ox = "N2O"

# Setup geometrical parameters
D_h = 2e-3  # hydraulic diameter in m

# Create fluid objects
fluids = [
    PyFluidsProperty("Methanol", FluidsList.Methanol, temperatures, "Methanol", ox, pc=pc, pe=pe),
    PyFluidsProperty("Ethanol", FluidsList.Ethanol, temperatures, "Ethanol", ox, pc=pc, pe=pe),
    IsopropanolProperty(temperatures, "Isopropanol", ox, pc=pc, pe=pe),
    PyFluidsProperty("Propane", FluidsList.nPropane, temperatures, "Propane", ox, pc=pc, pe=pe),
    # PyFluidsProperty("Methane", FluidsList.Methane, temperatures, "CH4", ox, pc=pc, pe=pe),
    # PyFluidsProperty("LOX cooled - Propane", FluidsList.Oxygen, temperatures, "Propane", "LOX", pc=pc, pe=pe, isox=True),
    # PyFluidsProperty("LOX cooled - Ethanol", FluidsList.Oxygen, temperatures, "Ethanol", "LOX", pc=pc, pe=pe, isox=True),
    # PyFluidsProperty("LOX cooled - Methanol", FluidsList.Oxygen, temperatures, "Methanol", "LOX", pc=pc, pe=pe, isox=True),
    # PyFluidsProperty("LOX cooled - IPA", FluidsList.Oxygen, temperatures, "Isopropanol", "LOX", pc=pc, pe=pe, isox=True),
    # PyFluidsProperty("LOX cooled - Methane", FluidsList.Oxygen, temperatures, "CH4", "LOX", pc=pc, pe=pe, isox=True),
]

for fluid in fluids:
    fluid.calculate_heat_transfer(D_h)
    print(f"{fluid.name} - OF: {fluid.OF:.2f}, Isp: {fluid.isp:.2f} sec")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:cyan']
ylabels = [
    ('Density (kg/m^3)', 'rho'),
    ('Specific Heat (J/kg-K)', 'cp'),
    ('Thermal Conductivity (W/m-K)', 'k'),
    ('Heat Transfer Coefficient (kW/m²-K)', 'h')
]
titles = ['Density', 'Specific Heat', 'Thermal Conductivity', 'Heat Transfer Coefficient']

for idx, (ylabel, key) in enumerate(ylabels):
    ax = axs[idx//2, idx%2]
    for i, fluid in enumerate(fluids):
        values = fluid.h*1e-3 if key == 'h' else getattr(fluid, key)
        ax.plot(fluid.temps, values, label=fluid.name, color=colors[i % len(colors)])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(ylabel)
    ax.set_title(titles[idx])
    ax.legend()
    ax.grid()
    ax.grid(which='minor', alpha=0.5)
    ax.set_ylim(0, None)
    ax.set_xlim(min(fluid.temps), max(fluid.temps))

htc_ax = axs[1, 1]
htc_ax.set_title(f'Heat Transfer Coefficient\n(D_h = {D_h*1e3:.2f}mm) - Density, OF, ISP corrected')

plt.tight_layout()
plt.show()