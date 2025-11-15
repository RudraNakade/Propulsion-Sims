from matplotlib.pylab import f, gamma
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import fuelCards, add_new_fuel
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import simpson
from os import path
import numpy as np
from scipy.constants import g, gas_constant
import warnings
import yaml

# Custom modules
from unit_converter import *
from flow_models import *
import custom_materials
import custom_fluids

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def OFsweep(fuel, ox, start, end, pc, pe = None, cr = None, pamb=101325, eps = None, show_vac=False, show_frozen=False, cstar_eff = 1, cf_eff = 1):
        __doc__ = """
            Performs a sweep analysis of the engine performance across a range of OF ratios and plots the ISP and chamber temperature.
            
            Parameters
            ----------
            fuel : str
                Fuel name (e.g., 'Isopropanol', 'Ethanol')
            ox : str
                Oxidizer name (e.g., 'N2O', 'LOX')
            OFstart : float
                Starting value for OF ratio
            OFend : float
                Ending value for OF ratio
            pc : float
                Chamber pressure in bar
            pe : float
                Exit pressure in bar
            pamb : float
                Ambient pressure in bar (default: 1.01325)
            cr : float
                Contraction ratio
            showvacisp : bool, optional
                Whether to display vacuum ISP on the plot (default: False)
            filmcooled : bool, optional
                Whether film cooling is being used (default: False)
            film_perc : float, optional
                Film cooling percentage (default: 0)
                
            Returns
            -------
            None
                Creates and displays a plot with OF ratio on x-axis,
                ISP values on primary y-axis and chamber temperature on secondary y-axis.
                
            Notes
            -----
            This method dynamically adjusts the expansion ratio (eps) for each OF value
            to maintain the specified pressure ratio between chamber and exit.
            If filmcooled=True, it also calculates and plots the corrected ISP accounting
            for the performance loss due to film cooling.
        """
        if eps is not None and pe is not None:
            raise ValueError("Cannot specify both eps and pe.")
        
        if eps is None and pe is None:
            raise ValueError("Must specify either eps or pe.")

        vary_eps = True

        if eps:
            vary_eps = False

        if cr is None:
            cr_string = 'CR = inf'
            ceaObj = CEA_Obj(
                oxName=ox,
                fuelName=fuel,
                isp_units='sec',
                cstar_units = 'm/s',
                pressure_units='Pa',
                temperature_units='K',
                sonic_velocity_units='m/s',
                enthalpy_units='J/g',
                density_units='kg/m^3',
                specific_heat_units='J/kg-K',
                viscosity_units='centipoise', # stored value in pa-s
                thermal_cond_units='W/cm-degC', # stored value in W/m-K
                make_debug_prints=False)
        else:
            cr_string = f'CR = {cr:.2f}'
            ceaObj = CEA_Obj(
                oxName=ox,
                fuelName=fuel,
                isp_units='sec',
                cstar_units = 'm/s',
                pressure_units='Pa',
                temperature_units='K',
                sonic_velocity_units='m/s',
                enthalpy_units='J/g',
                density_units='kg/m^3',
                specific_heat_units='J/kg-K',
                viscosity_units='centipoise', # stored value in pa-s
                thermal_cond_units='W/cm-degC', # stored value in W/m-K
                fac_CR=cr,
                make_debug_prints=False)
        
        OFs = np.linspace(start,end,100)
        ispseas = np.zeros_like(OFs)
        ispvacs = np.zeros_like(OFs)
        Tcs = np.zeros_like(OFs)
        for i, OF in enumerate(OFs):
            if vary_eps:
                eps = ceaObj.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
            [ispsea, _] = ceaObj.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=pamb, frozen=0, frozenAtThroat=0)
            [ispvac, _, Tc] = ceaObj.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=0, frozenAtThroat=0)
            ispseas[i] = ispsea * cstar_eff * cf_eff
            ispvacs[i] = ispvac * cstar_eff * cf_eff
            Tcs[i] = Tc

        if show_frozen:
            ispseas_frozen = np.zeros_like(OFs)
            ispvacs_frozen = np.zeros_like(OFs)
            for i, OF in enumerate(OFs):
                if vary_eps:
                    eps = ceaObj.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
                [ispsea_frozen, _] = ceaObj.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=pamb, frozen=1, frozenAtThroat=0)
                [ispvac_frozen, _, _] = ceaObj.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=1, frozenAtThroat=0)
                ispseas_frozen[i] = ispsea_frozen * cstar_eff * cf_eff
                ispvacs_frozen[i] = ispvac_frozen * cstar_eff * cf_eff

        # Find peak values and their corresponding OFs
        peak_sl_isp_idx = np.argmax(ispseas)
        peak_sl_isp_OF = OFs[peak_sl_isp_idx]
        if show_frozen:
            peak_sl_isp_frozen_idx = np.argmax(ispseas_frozen)
            peak_sl_isp_frozen_OF = OFs[peak_sl_isp_frozen_idx]
            peak_vac_isp_frozen_idx = np.argmax(ispvacs_frozen)
            peak_vac_isp_frozen_OF = OFs[peak_vac_isp_frozen_idx]
        peak_vac_isp_idx = np.argmax(ispvacs)
        peak_vac_isp_OF = OFs[peak_vac_isp_idx]
        peak_tc_idx = np.argmax(Tcs)
        peak_tc_OF = OFs[peak_tc_idx]

        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax1.set_xlabel('OF Ratio')
        ax1.set_ylabel('ISP (s)', color='b')
        ax1.plot(OFs, ispseas, 'b', label='SL ISP')
        if show_vac == True:
            ax1.plot(OFs, ispvacs, 'm', label='Vac ISP')
        if show_frozen:
            ax1.plot(OFs, ispseas_frozen, 'b--', label='SL ISP (frozen)')
            if show_vac == True:
                ax1.plot(OFs, ispvacs_frozen, 'm--', label='Vac ISP (frozen)')
        ax2 = ax1.twinx()
        ax2.plot(OFs, Tcs, 'r',label='Chamber Temp')
        ax2.set_ylabel('Chamber Temp (K)', color='r')
        ax1.grid()
        ax1.grid(which="minor", alpha=0.5)
        ax1.minorticks_on()
        ax1.set_ylim(bottom=0)
        plt.xlim(start, end)

        height = 0.8
        ax1.annotate(f"Peak SL ISP: {ispseas[peak_sl_isp_idx]:.1f}s, OF={peak_sl_isp_OF:.2f}",
            xy=(peak_sl_isp_OF, ispseas[peak_sl_isp_idx]),
            xytext=(peak_sl_isp_OF, ispseas[peak_sl_isp_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
            ha='center')
        ax1.plot([peak_sl_isp_OF, peak_sl_isp_OF], 
             [ispseas[peak_sl_isp_idx] * height, ispseas[peak_sl_isp_idx]], 
             'b-', linewidth=0.8)

        if show_frozen == True:
            height = 0.6
            ax1.annotate(f"Peak SL ISP (frozen): {ispseas_frozen[peak_sl_isp_frozen_idx]:.1f}s, OF={peak_sl_isp_frozen_OF:.2f}",
                xy=(peak_sl_isp_frozen_OF, ispseas_frozen[peak_sl_isp_frozen_idx]),
                xytext=(peak_sl_isp_frozen_OF, ispseas_frozen[peak_sl_isp_frozen_idx] * height),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                ha='center')
            ax1.plot([peak_sl_isp_frozen_OF, peak_sl_isp_frozen_OF], 
                 [ispseas_frozen[peak_sl_isp_frozen_idx] * height, ispseas_frozen[peak_sl_isp_frozen_idx]], 
                 'b--', linewidth=0.8)

        if show_vac == True:
            height = 0.9
            ax1.annotate(f"Peak Vac ISP: {ispvacs[peak_vac_isp_idx]:.1f}s, OF={peak_vac_isp_OF:.2f}",
            xy=(peak_vac_isp_OF, ispvacs[peak_vac_isp_idx]),
            xytext=(peak_vac_isp_OF, ispvacs[peak_vac_isp_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
            ha='center')
            ax1.plot([peak_vac_isp_OF, peak_vac_isp_OF], 
                 [ispvacs[peak_vac_isp_idx] * height, ispvacs[peak_vac_isp_idx]], 
                 'b-', linewidth=0.8)
        
        height = 0.7
        ax2.annotate(f"Peak Tc: {Tcs[peak_tc_idx]:.0f}K, OF={peak_tc_OF:.2f}",
            xy=(peak_tc_OF, Tcs[peak_tc_idx]),
            xytext=(peak_tc_OF, Tcs[peak_tc_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
            ha='center')
        # Add a line instead of an arrow
        ax2.plot([peak_tc_OF, peak_tc_OF], 
             [Tcs[peak_tc_idx] * height, Tcs[peak_tc_idx]], 
             'r-', linewidth=0.8)
        
        # Create a combined legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        
        # Only create legend if there are labeled items
        if lines:
            ax1.legend(lines, labels, loc='best')
        if vary_eps:
            plt.title(f'OF Sweep - {fuel} / {ox}\nPc = {pc/1e5:.2f} bar, Pe = {pe/1e5:.2f} bar, Pamb = {pamb/1e5:.2f} bar, {cr_string}\nη_c* = {cstar_eff:.2f}, η_Cf = {cf_eff:.2f}')
        else:
            plt.title(f'OF Sweep - {fuel} / {ox}\nPc = {pc/1e5:.2f} bar, Pamb = {pamb/1e5:.2f} bar, eps = {eps:.2f}, {cr_string}\nη_c* = {cstar_eff:.2f}, η_Cf = {cf_eff:.2f}')
        fig.tight_layout()

def machfunc(mach, area, gamma, at):
    area_ratio = area / at
    if mach == 0:
        mach = 1e-7
    return (area_ratio - ((1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) / ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))))

class cooling_channel_geometry:
    def __init__(self, material: custom_materials.material, n_channels: int, wall_thickness: float, rib_height: float):
        self.material = material
        self.n_channels = n_channels
        self.wall_thickness = wall_thickness
        self.rib_height = rib_height

    def area(self, r) -> float:
        """Calculate cross-sectional area based on channel geometry."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def d_h(self, r) -> float:
        """Calculate hydraulic diameter based on channel geometry."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def cold_flow(self, engine: 'engine', coolant: custom_fluids.base_fluid_class, coolant_mdot: float, p_in: float, T_in: float) -> None:
        coolant.update_state(T_in, p_in)

        n = engine.stations
        mdot = coolant_mdot / self.n_channels
        pressure = p_in
        total_dP = 0

        d_h = self.d_h(engine.r[0])
        area = 0.25 * np.pi * d_h**2
        v = mdot / (coolant.density() * area)

        Re = coolant.density() * v * d_h / coolant.viscosity()

        d_h_prev = d_h

        for i in range(n-1):
            coolant.update_state(T_in, pressure)
            dx = engine.x[i] - engine.x[i+1]
            dr = engine.r[i] - engine.r[i+1]
            dl = np.sqrt(dx**2 + dr**2)
            angle = np.rad2deg(np.arctan2(dr, dl))
            abs_roughness = self.material.Ra(np.abs(angle))

            d_h = self.d_h(engine.r[i])
            avg_dh = (d_h + d_h_prev) / 2
            d_h_prev = d_h

            area = 0.25 * np.pi * avg_dh**2
            v = mdot / (coolant.density() * area)
            Re = coolant.density() * v * d_h / coolant.viscosity()

            rel_roughness = abs_roughness / avg_dh

            if Re < 3400: 
                f = 64 / Re
            else:
                f = (-1.8 * np.log10((rel_roughness/3.7)**1.11 + (6.9 / Re))) ** -2
            dP = f * (dl / avg_dh) * (coolant.density() * v **2) * 0.5
            pressure -= dP

            total_dP += dP
        
        print(f"Total Pressure Drop: {total_dP/1e5:.2f} bar")

class arc_channels(cooling_channel_geometry):
    def __init__(self, material: custom_materials.material, n_channels: int, wall_thickness: float, rib_height: float, arc_angle: float):
        super().__init__(material, n_channels, wall_thickness, rib_height)
        self.arc_angle = arc_angle

    def area(self, r) -> float:
        channel_width = (2 * r + 2 * self.wall_thickness + self.rib_height) * np.sin(np.deg2rad(self.arc_angle / 2))
        channel_area = channel_width * self.rib_height
        return channel_area

    def d_h(self, r) -> float:
        channel_width = (2 * r + 2 * self.wall_thickness + self.rib_height) * np.sin(np.deg2rad(self.arc_angle / 2))
        channel_area = channel_width * self.rib_height
        d_h = np.sqrt(4 * channel_area / np.pi)
        return d_h

class constant_channels(cooling_channel_geometry):
    def __init__(self, material: custom_materials.material, n_channels: int, wall_thickness: float, rib_height: float, channel_width: float):
        super().__init__(material, n_channels, wall_thickness, rib_height)
        self.channel_width = channel_width

    def area(self, r) -> float:
        channel_area = self.channel_width * self.rib_height
        return channel_area

    def d_h(self, r) -> float:
        channel_area = self.channel_width * self.rib_height
        d_h = np.sqrt(4 * channel_area / np.pi)
        return d_h

class engine:
    def __init__(self, file):
        self.file = file
        self.fuel = None
        self.ox = None
        
        if not (path.exists(self.file) and path.getsize(self.file) > 0):
            raise FileNotFoundError(f"Config file {self.file} not found or empty")
            
        self.load_config()
        self.update()

    def load_config(self):
        """Load YAML format config file"""
        with open(self.file, 'r') as f:
            config = yaml.safe_load(f)

        if not config or 'metadata' not in config:
            raise ValueError("Missing required 'metadata' section in YAML config")

        self.name = config['metadata'].get('name', 'Unnamed Engine')

        if 'geometry' not in config:
            raise ValueError("Missing required 'geometry' section in YAML config")
        
        geom = config['geometry']
        
        geom_params = [
            'chamber_diameter', 'throat_diameter', 'exit_diameter',
            'chamber_length', 'cyl_conv_radius', 'converging_angle'
        ]
        
        for param in geom_params:
            if param not in geom:
                raise ValueError(f"Missing required geometry parameter: '{param}'")
        
        self.dc = geom['chamber_diameter'] * 1e-3
        self.dt = geom['throat_diameter'] * 1e-3
        self.de = geom['exit_diameter'] * 1e-3
        self.lc = geom['chamber_length'] * 1e-3
        if geom['cyl_conv_radius'] is None or (isinstance(geom['cyl_conv_radius'], str) and geom['cyl_conv_radius'].lower() == 'max'):
            self.r_cyl = -1
        else:
            self.r_cyl = geom['cyl_conv_radius'] * 1e-3
        self.conv_angle = geom['converging_angle']
        
        # Handle nozzle type
        if 'nozzle_type' not in geom:
            raise ValueError("Missing required geometry parameter: 'nozzle_type'")
        
        nozzle_type = geom['nozzle_type'].lower()
        self.rao = nozzle_type == 'rao' or nozzle_type == 'bell'
        
        if self.rao:
            if 'diverging_angle' not in geom:
                raise ValueError("Missing required parameter 'diverging_angle' for rao/bell nozzle")
            if 'exit_angle' not in geom:
                raise ValueError("Missing required parameter 'exit_angle' for rao/bell nozzle")
            if 'exit_length' not in geom:
                raise ValueError("Missing required parameter 'exit_length' for rao/bell nozzle")
            self.div_angle = geom['diverging_angle']
            self.exit_angle = geom['exit_angle']
            self.le = geom['exit_length'] * 1e-3
            self.lt = 0
        else:
            if 'throat_conv_radius' not in geom:
                raise ValueError("Missing required parameter 'throat_conv_radius' for conical nozzle")
            if 'throat_div_radius' not in geom:
                raise ValueError("Missing required parameter 'throat_div_radius' for conical nozzle")
            if 'diverging_angle' not in geom:
                raise ValueError("Missing required parameter 'diverging_angle' for conical nozzle")
            if 'throat_length' not in geom:
                raise ValueError("Missing required parameter 'throat_length' for conical nozzle")
            
            self.r_conv = geom['throat_conv_radius'] * 1e-3
            self.r_div = geom['throat_div_radius'] * 1e-3
            self.div_angle = geom['diverging_angle']
            self.lt = geom['throat_length'] * 1e-3

    def update(self):
        self.rc = self.dc/2
        self.rt = self.dt/2
        self.re = self.de/2
        self.ac = np.pi*self.rc**2
        self.at = np.pi*self.rt**2
        self.ae = np.pi*self.re**2
        self.cr = self.ac / self.at
        self.eps = (self.de/self.dt)**2

        if self.rao:
            self.rao_frac = np.tan(np.deg2rad(15)) * self.le / ((np.sqrt(self.eps)-1) * self.rt)
            self.r_conv = 1.5 * self.rt
            self.r_div = 0.382 * self.rt

        self.r_cyl_max = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - self.r_conv
        if (self.r_cyl < 0)|(self.r_cyl > self.r_cyl_max):
            self.r_cyl = self.r_cyl_max

        self.generate_contour()

    def set_props(self, fuel=None, ox=None):
        """Set the fuel and oxidizer for the engine."""
        if fuel is not None:
            self.fuel = fuel
        if ox is not None:
            self.ox = ox
        if self.fuel is None or self.ox is None:
            raise ValueError("Fuel and / or oxidizer must be specified.")
        self.gen_cea_obj()

    def gen_cea_obj(self):
        self.cea = CEA_Obj(
            oxName = self.ox,
            fuelName = self.fuel,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Pa',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/kg',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            fac_CR=self.cr,
            make_debug_prints=False)

    def combustion_sim(self, fuel, ox, OF, pc, pamb = 101325, cstar_eff = 1, cf_eff = 1, frozen=False, simplified=False):
        __doc__ = """Calls CEA to get performance values for a given chamber pressure and OF ratio.
            Parameters:
            -----------
            fuel : str
                Fuel propellant name
            ox : str 
                Oxidizer propellant name
            OF : float
                OF Ratio
            pc : float
                Chamber pressure in Pa
            pamb : float, optional
                Ambient pressure in Pa, default is 101325 (atm pressure)
            cstar_eff : float, optional
                Characteristic velocity efficiency, default: 1.00
            cf_eff : float, optional
                Thrust coefficient efficiency, default: 1.00
            --------
            None
                Updates various engine performance attributes:
                - Performance: isp, cstar, cf, thrust, etc.
                - Mass Flow rates: mdot, ox_mdot, fuel_mdot
                - Gas properties: temperatures, pressures, transport properties (for regen cooling simulation)"""
        self.OF = OF
        self.pc = pc
        self.pamb = pamb
        self.cstar_eff = cstar_eff
        self.cf_eff = cf_eff

        self.set_props(fuel=fuel, ox=ox)
        self.gen_cea_obj()

        if not frozen:
            self.cstar = self.cea.get_Cstar(Pc=self.pc, MR=self.OF) * cstar_eff
            [_, self.cf, self.exitcond] = self.cea.get_PambCf(Pamb=self.pamb, Pc=self.pc, MR=self.OF, eps=self.eps)
        else:
            self.cstar = self.cea.getFrozen_IvacCstrTc(Pc=self.pc, MR=self.OF, frozenAtThroat=0)[1] * cstar_eff
            [_, self.cf, self.exitcond] = self.cea.getFrozen_PambCf(Pamb=self.pamb, Pc=self.pc, MR=self.OF, eps=self.eps)


        self.cf = self.cf * self.cf_eff

        if not simplified:
            self.ispvac = self.cea.get_Isp(Pc=self.pc, MR=self.OF, eps=self.eps) * cstar_eff * self.cf_eff
            [self.ispsea, _] = self.cea.estimate_Ambient_Isp(Pc=self.pc, MR=self.OF, eps=self.eps, Pamb=101325, frozen=+(frozen), frozenAtThroat=0)
            self.ispsea = self.ispsea * self.cstar_eff * self.cf_eff
            [self.Tg_c, self.Tg_t, self.Tg_e] = self.cea.get_Temperatures(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen), frozenAtThroat=0)
            self.pt = self.pc / self.cea.get_Throat_PcOvPe(Pc=self.pc, MR=self.OF)
            self.PinjPcomb = self.cea.get_Pinj_over_Pcomb(Pc=self.pc, MR=self.OF)
            self.Me = self.cea.get_MachNumber(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen), frozenAtThroat=0)
            [self.cp_c, self.mu_c, self.k_c, self.pr_c] = self.cea.get_Chamber_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
            [self.cp_t, self.mu_t, self.k_t, self.pr_t] = self.cea.get_Throat_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
            [self.cp_e, self.mu_e, self.k_e, self.pr_e] = self.cea.get_Exit_Transport(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen), frozenAtThroat=0)
            [self.mw_t, self.gam_t] = self.cea.get_Throat_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen))
            [self.mw_c, self.gam_c] = self.cea.get_Chamber_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps)
            [self.mw_e, self.gam_e] = self.cea.get_exit_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen))
            self.mu_c = self.mu_c * 1e-3 # now pa-s
            self.k_c = self.k_c * 1e2 # now W/m-K

            # Estimate residence time
            chamber_idxs = np.where(self.x <= 0)
            x = self.x[chamber_idxs]
            r = self.r[chamber_idxs]
            area = np.pi * r**2

            mach = np.zeros_like(area)
            for i, a in enumerate(area):
                if a > self.at:
                    mach[i] = root_scalar(machfunc, args=(a, self.gam_c, self.at), bracket=[0, 1]).root
                else:
                    mach[i] = 1.0
            
            gas_temp = self.Tg_c / (1 + 0.5 * (self.gam_c - 1) * (mach**2))
            sp_gas_const = gas_constant / (self.mw_c * 1e-3)
            sound_speed = np.sqrt(self.gam_c * sp_gas_const * gas_temp)
            v = sound_speed * mach

            self.residence_time = simpson(1 / v, x)

        self.mdot = self.pc * self.at / self.cstar
        self.ox_mdot = self.mdot * self.OF / (1 + self.OF)
        self.fuel_mdot = self.mdot / (1 + self.OF)

        self.thrust = self.cstar * self.cf * self.mdot
        self.isp = self.thrust / (self.mdot * g)         

        self.pe = self.pc / self.cea.get_PcOvPe(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=+(frozen), frozenAtThroat=0)

    def pressure_combustion_sim(self, fuel, ox, fuel_CdA, ox_CdA, fuel_upstream_p, ox_upstream_p, pamb = 101325,
                              fuel_rho = None, ox_rho = None,
                              ox_gas=None, fuel_gas=None,
                              ox_temp = None, fuel_temp = None,
                              cstar_eff = 1, cf_eff = 1, frozen=False, n_max=100):
        __doc__ = """Combustion sim based on injector pressures.\n
            Required Inputs: fuel, ox, fuel_upstream_p, ox_upstream_p, film_frac, fuel_rho, ox_rho\n
            Optional Inputs: oxclass, ox_gas, ox_temp, fuelclass, fuel_gas, fuel_temp"""

        self.fuel = fuel
        self.ox = ox

        ox_is_gas = ox_gas is not None
        fuel_is_gas = fuel_gas is not None

        def pcfunc(pc_guess, cstar,
                   fuel_upstream_p, ox_upstream_p,
                   fuel_CdA, ox_CdA,
                   fuel_rho = None, ox_rho = None,
                   fuel_gas = None, ox_gas = None,
                   fuel_temp = None, ox_temp = None):
            if ox_is_gas:
                mdot_o = spc_mdot(ox_CdA, ox_gas, degC_to_K(ox_temp), ox_upstream_p*1e5, pc_guess)[0]
            else:
                mdot_o = ox_CdA * np.sqrt(2 * ox_rho * (ox_upstream_p - pc_guess) * 1e5)

            if fuel_is_gas:
                mdot_f = spc_mdot(fuel_CdA, fuel_gas, degC_to_K(fuel_temp), fuel_upstream_p*1e5, pc_guess)[0]
            else:
                mdot_f = fuel_CdA * np.sqrt(2 * fuel_rho * (fuel_upstream_p - pc_guess) * 1e5)

            return ((cstar / self.at) * ((mdot_f + mdot_o) * 1e-5) - pc_guess)
        
        self.gen_cea_obj()

        min_inj_p = min(fuel_upstream_p, ox_upstream_p)

        cstar = 1450 * cstar_eff # initial c* guess
        rel_diff = 1

        n = 0
        while rel_diff > 5e-4:
            n += 1
            converged = True
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", category=RuntimeWarning)
                try:
                    pc = root_scalar(pcfunc, bracket=[0, min_inj_p], args=(cstar * cstar_eff, fuel_upstream_p, ox_upstream_p, fuel_CdA, ox_CdA, fuel_rho, ox_rho, fuel_gas, ox_gas, fuel_temp, ox_temp), method='brentq', rtol=1e-4).root
                except ValueError as e:
                    print(f"ValueError occurred: {e}")
                    converged = False
                    pass
                except Exception as e:
                    print(f"Error occurred: {e}")
                    converged = False
                    pass

            if ox_is_gas:
                mdot_o = spc_mdot(ox_CdA, ox_gas, degC_to_K(ox_temp), ox_upstream_p*1e5, pc)[0]
            else:
                mdot_o = ox_CdA * np.sqrt(2 * ox_rho * (ox_upstream_p - pc) * 1e5)

            if fuel_is_gas:
                mdot_f = spc_mdot(fuel_CdA, fuel_gas, degC_to_K(fuel_temp), fuel_upstream_p*1e5, pc)[0]
            else:
                mdot_f = fuel_CdA * np.sqrt(2 * fuel_rho * (fuel_upstream_p-pc) * 1e5)

            OF = mdot_o / mdot_f
            cstar_old = cstar
            cstar = self.cea.get_Cstar(Pc=pc, MR=OF) * cstar_eff
            rel_diff = abs((cstar - cstar_old) / cstar_old)
            
            if n > n_max:
                print(f"{bcolors.WARNING}Warning: Max iterations exceeded")
                converged = False
                break
        
        if not converged:
            print(f"{bcolors.FAIL}Error: Convergence failed: fuel inj p: {self.fuel_inj_p:.2f}, ox inj p: {self.ox_inj_p:.2f}, n: {n}{bcolors.ENDC}")

        self.combustion_sim(fuel, ox, OF, pc, pamb, cstar_eff, cf_eff)

    def pc_of_mdot_calc(self, fuel, ox, pc, OF, cstar_eff = 1):
        self.fuel = fuel
        self.ox = ox
        self.pc = pc
        self.OF = OF
        self.cstar_eff = cstar_eff

        cstar = self.cea.get_Cstar(pc, OF) * cstar_eff
        total_mdot = pc * self.at / cstar
        fuel_mdot = total_mdot / (1 + OF)
        ox_mdot = total_mdot * OF / (1 + OF)

        return fuel_mdot, ox_mdot

    def mdot_solver_func(self, pc, OF, cstar_eff, total_mdot):
        cstar = self.cea.get_Cstar(Pc=pc, MR=OF) * cstar_eff
        mdot = pc * self.at / cstar
        return (mdot - total_mdot)

    def mdot_combustion_sim(self, fuel, ox, fuel_mdot, ox_mdot, pamb = 101325, cstar_eff = 1, cf_eff = 1, frozen=False, simplified=True):
        total_mdot = fuel_mdot + ox_mdot
        self.OF = ox_mdot / fuel_mdot
        self.fuel = fuel
        self.ox = ox

        self.gen_cea_obj()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", category=RuntimeWarning)
            try:
                pc = root_scalar(self.mdot_solver_func, bracket=[1, 1000e5], args=(self.OF, cstar_eff, total_mdot), method='brentq', rtol=1e-3).root
            except ValueError:
                print(f"{bcolors.FAIL}Error: Could not find a solution.{bcolors.ENDC}")
                return

        self.combustion_sim(fuel, ox, self.OF, pc, pamb, cstar_eff, cf_eff, frozen, simplified)

    def generate_contour(self):
        self.n_points = 20
        step = 0.1e-3

        cyl_step = 4 * step
        conv_step = 2 * step
        throat_step = 1 * step
        div_step = 2 * step

        l_conv_cone = (self.rc - self.rt - ((self.r_cyl + self.r_conv) * (1 - np.cos(np.deg2rad(self.conv_angle))))) / np.tan(np.deg2rad(self.conv_angle))

        self.l_cyl = self.lc - ((self.r_cyl + self.r_conv) * np.sin(np.deg2rad(self.conv_angle))) - l_conv_cone

        # Cylindrical section
        n = int(np.ceil(self.l_cyl / cyl_step)) + 1
        self.x = np.linspace(0, self.l_cyl, n)
        self.r = np.ones_like(self.x) * self.rc

        conv_arc_ax_length = (self.r_cyl * np.sin(np.deg2rad(self.conv_angle)))
        conv_arc_end_r = self.rc - self.r_cyl * (1 - np.cos(np.deg2rad(self.conv_angle)))

        # Converging arc
        if self.r_cyl > 0:
            n = int(np.ceil(conv_arc_ax_length / conv_step)) + 1
            l = np.linspace(0, conv_arc_ax_length, n)
            r = np.sqrt(self.r_cyl**2 - l**2) + self.rc - self.r_cyl
            l = l + self.l_cyl
            self.r = np.append(self.r, r[1:])
            self.x = np.append(self.x, l[1:])

        # Conical converging section
        if l_conv_cone > 0:
            n = int(np.ceil(l_conv_cone / conv_step)) + 1
            l = np.linspace(0, l_conv_cone, n)
            r = conv_arc_end_r - l * np.tan(np.deg2rad(self.conv_angle))
            l = l + self.l_cyl + conv_arc_ax_length
            self.r = np.append(self.r, r[1:])
            self.x = np.append(self.x, l[1:])

        # Throat upstream arc
        n = int(np.ceil((self.r_conv*np.sin(np.deg2rad(self.conv_angle))) / throat_step))
        l = np.linspace(0, (self.r_conv*np.sin(np.deg2rad(self.conv_angle))), n)
        r = (self.r_conv + self.rt) - np.sqrt((self.r_conv)**2 - (l - self.r_conv * np.sin(np.deg2rad(self.conv_angle)))**2)
        l = l + self.r_cyl * np.sin(np.deg2rad(self.conv_angle)) + (self.lc - (self.r_cyl + self.r_conv)*np.sin(np.deg2rad(self.conv_angle)))
        self.r = np.append(self.r, r[1:])
        self.x = np.append(self.x, l[1:])

        # Straight throat section
        if self.lt > 0:
            n = int(np.ceil(self.lt / throat_step)) + 1
            l = np.linspace(0, self.lt, n)
            r = np.ones_like(l) * self.rt
            l = l + self.lc
            self.r = np.append(self.r, r[1:])
            self.x = np.append(self.x, l[1:])

        # Throat downstream arc
        n = int(np.ceil((self.r_div * np.sin(np.deg2rad(self.div_angle))) / throat_step)) + 1
        l3 = np.linspace(0, (self.r_div * np.sin(np.deg2rad(self.div_angle))), n)
        r3 = (self.r_div + self.rt) - np.sqrt((self.r_div)**2 - l3**2)
        l3 = l3 + self.lc + self.lt

        self.r = np.append(self.r, r3[1:])
        self.x = np.append(self.x, l3[1:])

        if self.rao:
            # t = np.concatenate((np.linspace(0, 0.2, int(1*self.n_points)), np.linspace(0.2, 1, int(1*self.n_points))))
            t = np.linspace(0, 1, int(2*self.n_points))

            Nx = l3[-1]
            Ny = r3[-1]
            Ex = self.lc + self.le
            Ey = self.re

            m1 = np.tan(np.deg2rad(self.div_angle))
            m2 = np.tan(np.deg2rad(self.exit_angle))
            c1 = Ny - m1*Nx
            c2 = Ey - m2*Ex
            Qx = (c2 - c1)/(m1 - m2)
            Qy = (m1*c2 - m2*c1)/(m1 - m2)

            l4 = Nx*(1-t)**2 + 2*(1-t)*t*Qx + Ex*t**2
            r4 = Ny*(1-t)**2 + 2*(1-t)*t*Qy + Ey*t**2

            self.r = np.append(self.r, r4[1:])
            self.x = np.append(self.x, l4[1:])

        else:
            # Conical diverging section
            dr = self.re - r3[-1]
            dl = dr / np.tan(np.deg2rad(self.div_angle))
            n = int(np.ceil(dl / div_step)) + 1
            l = np.linspace(0, dl, n)
            r = r3[-1] + l * np.tan(np.deg2rad(self.div_angle))
            l = l + l3[-1]
            self.r = np.append(self.r, r[1:])
            self.x = np.append(self.x, l[1:])

        self.le = self.x[-1] - self.lc - self.lt
        self.ltotal = self.lc + self.lt + self.le

        self.x = self.x - self.lc
        self.stations = len(self.r)

        # Calculate chamber volume
        chamber_end_idx = np.where(self.x >= 0)[0][0] if np.any(self.x >= 0) else len(self.x)
        x_chamber = self.x[:chamber_end_idx]
        r_chamber = self.r[:chamber_end_idx]
        self.chamber_volume = np.pi * simpson(r_chamber**2, x_chamber)
        self.lstar = self.chamber_volume / (self.at)

    def show_contour(self):
        plt.figure()
        plt.plot(self.x*1e3, self.r*1e3, 'b', label='Contour')
        plt.plot(self.x*1e3, -self.r*1e3, 'b')
        plt.plot(self.x*1e3, -self.r*1e3, 'xk')
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.grid()
        plt.xlabel('Axial Distance (mm)')
        plt.ylabel('Radius (mm)')
        plt.title('Chamber Contour')
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlim(left=self.x[0]*1e3, right=self.x[-1]*1e3)
        plt.legend()

    def save(self):
        with open(self.file, 'w+') as output_file:
            output_file.write(f'dc: {self.dc*1e3}')
            output_file.write(f'\ndt: {self.dt*1e3}')
            output_file.write(f'\nde: {self.de*1e3}')
            output_file.write(f'\neps: {self.eps}')
            output_file.write(f'\nconv_angle: {self.conv_angle}')
            output_file.write(f'\nlc: {self.lc*1e3}')
            output_file.write(f'\nle: {self.le*1e3}')
            output_file.write(f'\ntheta_n: {self.div_angle}')
            output_file.write(f'\ntheta_e: {self.exit_angle}')

    def print_data(self):
        print('-----------------------------------')
        print(f'{self.name}')
        print('Combustion Sim')
        print(f'{self.ox} / {self.fuel}\n')
        print(f'{"Parameter":<20} {"Value":<10} {"Unit"}')
        print(f'{"OF:":<20} {self.OF:<10.3f}')
        print(f'{"Chamber Pressure:":<20} {self.pc/1e5:<10.2f} bar')
        print(f'{"Ambient Pressure:":<20} {self.pamb/1e5:<10.2f} bar')
        print(f'{"Thrust:":<20} {self.thrust:<10.2f} N')
        print(f'{"ISP:":<20} {self.isp:<10.2f} s')
        print(f'{"SL ISP:":<20} {self.ispsea:<10.2f} s')
        print(f'{"Vac ISP:":<20} {self.ispvac:<10.2f} s')
        print(f'{"C*:":<20} {self.cstar:<10.2f} m/s')
        print(f'{"Cf:":<20} {self.cf:<10.4f}\n')
        print(f'{"η_C*:":<20} {self.cstar_eff:<10.4f}')
        print(f'{"η_Cf:":<20} {self.cf_eff:<10.4f}\n')
        print(f'{"η_ideal:":<20} {self.cstar_eff*self.cf_eff:<10.4f}\n')
        if(self.mdot >= 1e-1):
            print(f'{"Total mdot:":<20} {self.mdot:<10.4f} kg/s')
            print(f'{"Ox mdot:":<20} {self.ox_mdot:<10.4f} kg/s')
            print(f'{"Fuel mdot:":<20} {self.fuel_mdot:<10.4f} kg/s\n')
        else:
            print(f'{"Total mdot:":<20} {self.mdot*1e3:<10.4f} g/s')
            print(f'{"Ox mdot:":<20} {self.ox_mdot*1e3:<10.4f} g/s')
            print(f'{"Fuel mdot:":<20} {self.fuel_mdot*1e3:<10.4f} g/s\n')
        
        print(f'{"Residence Time:":<20} {self.residence_time*1e3:<10.4f} ms\n')
        print(f'{"Chamber Temp:":<20} {self.Tg_c:<10.1f} K')
        print(f'{"Throat Temp:":<20} {self.Tg_t:<10.1f} K')
        print(f'{"Exit Temp:":<20} {self.Tg_e:<10.1f} K\n')
        print(f'{"Throat Pressure:":<20} {self.pt/1e5:<10.2f} bar')
        print(f'{"Exit Pressure:":<20} {self.pe/1e5:<10.2f} bar')
        print(f'{"Exit Condition:":<20} {self.exitcond:<10}')
        print(f'{"Pc loss ratio:":<20} {self.PinjPcomb:<10.3f}\n')
        print(f'{"Exit Mach Number:":<20} {self.Me:<10.3f}')
        print(f'{"Contraction Ratio:":<20} {self.cr:<10.3f}')
        print(f'{"Expansion Ratio:":<20} {self.eps:<10.3f}\n')
        print(f'{"Converging Angle:":<20} {self.conv_angle:<10.3f}°')
        print(f'{"Diverging Angle:":<20} {self.div_angle:<10.3f}°')
        if self.rao:
            print(f'{"Exit Angle:":<20} {self.exit_angle:<10.3f}°')
        print(f'\n{"Chamber Diameter:":<20} {self.dc*1e3:<10.3f} mm')
        print(f'{"Throat Diameter:":<20} {self.dt*1e3:<10.3f} mm')
        print(f'{"Exit Diameter:":<20} {self.de*1e3:<10.3f} mm\n')
        # print(f'{"Chamber Volume:":<20} {self.chamber_volume*1e9:<10.3f} mm³')
        print(f'{"L*:":<20} {self.lstar*1e3:<10.3f} mm')
        print(f'{"Chamber Length:":<20} {self.lc*1e3:<10.3f} mm')
        print(f'{"Cylindrical Length:":<20} {self.l_cyl*1e3:<10.3f} mm')
        if (self.lt > 0):
            print(f'{"Nozzle Length:":<20} {self.lt*1e3:<10.3f} mm')
        if self.rao:
            print(f'{"Exit Length:":<20} {self.le*1e3:<10.3f} mm ({self.rao_frac:.2%})')
        else:
            print(f'{"Exit Length:":<20} {self.le*1e3:<10.3f} mm')
        print(f'{"Total Length:":<20} {self.ltotal*1e3:<10.3f} mm\n')
        print('-----------------------------------')

    def thermal_sim(self, cooling_channel: cooling_channel_geometry, coolant: custom_fluids.base_fluid_class, coolant_mdot, coolant_T_in, coolant_p_in, rev_flow):
        # Seems way off at higher pc with way too high wall temp
        self.generate_contour()

        self.coolant = coolant
        self.n_channels = cooling_channel.n_channels
        self.coolant_mdot = coolant_mdot

        self.pr              = np.zeros(self.stations)
        self.gamma           = np.zeros(self.stations)
        self.M               = np.zeros(self.stations)
        self.hg              = np.zeros(self.stations)
        self.q               = np.zeros(self.stations)
        self.pg              = np.zeros(self.stations)
        self.Tg              = np.zeros(self.stations)
        self.Tg_aw           = np.zeros(self.stations)
        self.Twg             = np.zeros(self.stations)
        self.Twc             = np.zeros(self.stations)
        self.Tc              = np.zeros(self.stations)
        self.v_coolant       = np.zeros(self.stations)
        self.rho_coolant     = np.zeros(self.stations)
        self.mu_coolant      = np.zeros(self.stations)
        self.k_coolant       = np.zeros(self.stations)
        self.p_coolant       = np.zeros(self.stations)
        self.p_sat_coolant   = np.zeros(self.stations)
        self.t_sat_coolant   = np.zeros(self.stations)
        self.cp_coolant      = np.zeros(self.stations)
        self.Re_coolant      = np.zeros(self.stations)
        self.Pr_coolant      = np.zeros(self.stations)
        self.Nu_coolant      = np.zeros(self.stations)
        self.fr_coolant      = np.zeros(self.stations)
        self.hc              = np.zeros(self.stations)
        self.d_h             = np.zeros(self.stations)
        self.angle           = np.zeros(self.stations)
        self.abs_roughness   = np.zeros(self.stations)
        self.rel_roughness   = np.zeros(self.stations)
        self.total_heat_flux = 0

        t_next = coolant_T_in

        self.coolant.update_state(coolant_T_in, coolant_p_in)
        coolant_heat = 0

        first = True

        def colebrook(f: float, Re: float, rel_roughness: float) -> float:
            """
            Colebrook-White equation for turbulent friction factor
            Returns the residual that should equal zero when f is correct
            """
            residual = (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))
            return residual

        def friction_factor(Re, rel_roughness):
            if Re < 2300:
                return 64 / Re
            else:
                # return (-1.8 * np.log10((rel_roughness/3.7)**1.11 + (6.9 / Re))) ** -2 # 
                return root_scalar(colebrook, bracket=[1e-10, 1], args=(Re, rel_roughness), xtol=1e-6, method='brentq').root
            
        def Nu_gnielinski(Re, Pr, fr):
            return (fr/8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(fr/8) * (Pr**(2/3) - 1))
        
        def Nu_dittus_boelter(Re, Pr):
            return 0.023 * Re**0.8 * Pr**0.4

        for j in range(self.stations):
            if rev_flow == True:
                i = -(j+1)
                # i = -j
                inext = i - 1
                iprev = i + 1
            else:
                i = j
                inext = i + 1
                iprev = i - 1
            area = np.pi * self.r[i]**2
            self.Tc[i] = t_next
            self.d_h[i] = cooling_channel.d_h(self.r[i])
            tw = cooling_channel.wall_thickness

            if first:
                self.p_coolant[i] = coolant_p_in
                self.rho_coolant[i] = self.coolant.density()
                self.cp_coolant[i] = self.coolant.heat_capacity()
                self.mu_coolant[i] = self.coolant.viscosity()
                self.k_coolant[i] = self.coolant.conductivity()
                self.v_coolant[i] = self.coolant_mdot / (self.rho_coolant[i] * cooling_channel.area(self.r[i]) * self.n_channels)
                self.Re_coolant[i] = self.rho_coolant[i] * self.v_coolant[i] * self.d_h[i] / self.mu_coolant[i]
                self.Pr_coolant[i] = self.mu_coolant[i] * self.cp_coolant[i] / self.k_coolant[i]

                dx = self.x[i] - self.x[inext]
                dr = self.r[i] - self.r[inext]
                dl = np.sqrt(dx**2 + dr**2)
                self.angle[i] = np.rad2deg(np.arctan(dr / dx)) # diverging angle

                self.abs_roughness[i] = cooling_channel.material.Ra(np.abs(self.angle[i])) # downskin roughness
                self.rel_roughness[i] = self.abs_roughness[i] / self.d_h[i]
                
                self.fr_coolant[i] = friction_factor(self.Re_coolant[i], self.rel_roughness[i])

                # self.Nu_coolant[i] = Nu_dittus_boelter(self.Re_coolant[i], self.Pr_coolant[i])
                self.Nu_coolant[i] = Nu_gnielinski(self.Re_coolant[i], self.Pr_coolant[i], self.fr_coolant[i])

                self.hc[i] = self.Nu_coolant[i] * self.k_coolant[i] / self.d_h[i]
                first = False

            # TODO: Account for total and static pressure of coolant (total decrease due to friction, static change due to channel size)

            else:
                dx = self.x[i] - self.x[iprev]
                dr = self.r[i] - self.r[iprev]
                dl = np.sqrt(dx**2 + dr**2)
                self.angle[i] = np.rad2deg(np.arctan(dr / dx)) # diverging angle

                avg_dh = (self.d_h[i] + self.d_h[iprev]) / 2

                self.abs_roughness[i] = cooling_channel.material.Ra(np.abs(self.angle[i])) # downskin roughness
                self.rel_roughness[i] = self.abs_roughness[i] / avg_dh

                self.fr_coolant[i] = friction_factor(self.Re_coolant[iprev], self.rel_roughness[i])

                dp = self.fr_coolant[i] * (dl / avg_dh) * (self.rho_coolant[iprev] * self.v_coolant[iprev] ** 2) * 0.5

                if np.isnan(dp) or dp < 0:
                    print(f"Warning: Negative pressure drop ({dp:.4f} bar) in station {i}")
                    print(f"dx: {dx*1e3:.4f} mm, dr: {dr*1e3:.4f} mm, dl: {dl*1e3:.4f} mm, angle: {self.angle[i]:.4f}")
                    print(f"Location: x: ({self.x[i]*1e3:.4f}, r: {self.r[i]*1e3:.4f}) mm")

                self.p_coolant[i] = self.p_coolant[iprev] - dp
                if self.p_coolant[i] < 0:
                    raise ValueError(f"Warning: Negative coolant pressure in station {i}")
                self.coolant.update_state(self.Tc[i], self.p_coolant[i])

            self.p_sat_coolant[i] = self.coolant.vapor_pressure()
            self.t_sat_coolant[i] = self.coolant.saturation_temperature()

            self.rho_coolant[i] = self.coolant.density()
            self.cp_coolant[i] = self.coolant.heat_capacity()
            self.mu_coolant[i] = self.coolant.viscosity()
            self.k_coolant[i] = self.coolant.conductivity()
            self.v_coolant[i] = self.coolant_mdot / (self.rho_coolant[i] * cooling_channel.area(self.r[i]) * self.n_channels)
            self.Re_coolant[i] = self.rho_coolant[i] * self.v_coolant[i] * self.d_h[i] / self.mu_coolant[i]
            self.Pr_coolant[i] = self.mu_coolant[i] * self.cp_coolant[i] / self.k_coolant[i]

            # self.Nu_coolant[i] = Nu_dittus_boelter(self.Re_coolant[i], self.Pr_coolant[i])
            self.Nu_coolant[i] = Nu_gnielinski(self.Re_coolant[i], self.Pr_coolant[i], self.fr_coolant[i])

            self.hc[i] = self.Nu_coolant[i] * self.k_coolant[i] / self.d_h[i]

            throat_rcurv =  (self.r_conv + self.r_div) / 2
            if self.x[i] < 0: # Converging section
                self.gamma[i] = (self.gam_t - self.gam_c)/(self.rt - self.rc) * (self.r[i] - self.rc) + self.gam_c
                self.pr[i] = (self.pr_t - self.pr_c)/(self.rt - self.rc) * (self.r[i] - self.rc) + self.pr_c
                if area == self.at: # Throat
                    self.M[i] = 1.0
                else:
                    self.M[i] = root_scalar(machfunc, args=(area, self.gamma[i], self.at), bracket=[0, 1]).root
            else: # Diverging section
                self.gamma[i] = (0.5*(0.8*self.gam_e+1.2*self.gam_t) - self.gam_t)/(self.re - self.rt) * (self.r[i] - self.rt) + self.gam_t
                self.pr[i] = (self.pr_e - self.pr_t)/(self.re - self.rt) * (self.r[i] - self.rt) + self.pr_t
                if area < self.at:
                    print(f"Warning: Area ({area*1e6:.2f} mm²) is less than throat area ({self.at*1e6:.2f} mm²) in diverging section, setting to throat area")
                    area = self.at
                self.M[i] = root_scalar(machfunc, args=(area, self.gamma[i], self.at), bracket=[1, 5]).root

            self.Tg_aw[i] = self.Tg_c * ((1 + (0.5 * (self.pr[i]**(1/3)) * (self.gamma[i] - 1) * (self.M[i]**2))) / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))
            self.Tg[i] = self.Tg_c / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)
            self.pg[i] = self.pc * (self.Tg[i] / self.Tg_c)**(self.gamma[i] / (self.gamma[i] - 1))

            bartz = ((0.026 / (self.dt**0.2))) * ((self.mu_c**0.2) * self.cp_c / (self.pr_c**0.6)) * ((self.pc / self.cstar)**0.8) * ((self.dt / throat_rcurv)**0.1) * ((self.at / area)**0.9)

            if j != self.stations - 1:
                dA = np.pi * (self.r[i] + self.r[inext]) * np.sqrt((self.r[i] - self.r[inext]) ** 2 + (self.x[inext] - self.x[i]) ** 2)

            def wall_temp_func(Twg, tw) -> float:
                correction_factor = (((0.5 * (Twg / self.Tg_c) * (1 + (0.5 * (self.gamma[i] - 1)) * (self.M[i]**2)) + 0.5)**0.68) * ((1 + 0.5 * (self.gamma[i] - 1) * (self.M[i]**2))**0.12))**-1
                hg = bartz * correction_factor
                q = hg * (self.Tg_aw[i] - Twg)
                Twc = (q / self.hc[i]) + self.Tc[i]
                Twg_new = (q * tw / cooling_channel.material.k((Twg+Twc)/2)) + Twc
                return Twg_new - Twg

            bracket = [self.Tc[i], self.Tg_aw[i]]
            sol = root_scalar(wall_temp_func, args=(tw), bracket=bracket, method='brentq')
            if not sol.converged:
                print(f"Warning: Wall temp solver did not converge at station {i}")
                self.Twg[i] = 0
            else:
                self.Twg[i] = sol.root

            correction_factor = (((0.5 * (self.Twg[i] / self.Tg_c) * (1 + (0.5 * (self.gamma[i] - 1) * self.M[i]**2)) + 0.5)**0.68) * ((1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))**0.12)**-1
            self.hg[i] = bartz * correction_factor
            self.q[i] = self.hg[i] * (self.Tg_aw[i] - self.Twg[i])
            self.Twc[i] = (self.q[i] / self.hc[i]) + self.Tc[i]
            self.Twg[i] = (self.q[i] * tw / cooling_channel.material.k((self.Twg[i]+self.Twc[i])/2)) + self.Twc[i]

            coolant_heat = self.q[i] * dA / self.coolant_mdot
            # t_next = (coolant_heat / self.cp_coolant[i]) + self.Tc[i]
            Hprev = self.coolant.chemical.H
            coolant.update_state(enthalpy=(Hprev+coolant_heat), pressure=self.p_coolant[i])
            t_next = coolant.chemical._temperature
            self.total_heat_flux += self.q[i] * dA
        return
            
    def plot_thermals(self, title):
        self.thermalsplot = subplot(3, 5, title, self)
        xlabel = 'Axial Distance (mm)'
        self.thermalsplot.plt(1, self.x*1e3, self.hg*1e-3,'Gas Side Conv. Coeff.', xlabel, 'Gas Side Conv. Coeff. (kW/m^2/K)','r', True)
        self.thermalsplot.plt(2, self.x*1e3, self.q*1e-3,'Heat Flux', xlabel, 'Heat Flux (kW/m^2)','r', True)
        self.thermalsplot.plt(3, self.x*1e3, self.Tg_aw, 'Gas Temperatures', xlabel, 'Gas Temperature (K)', 'r', True, label='Adiabatic Wall Recovery')
        self.thermalsplot.addline(3, self.x*1e3, self.Tg, 'm', label='Hot Gas')
        self.thermalsplot.plt(4, self.x*1e3, self.pg/1e5, 'Pressures', xlabel, 'Pressure (bar)', 'r', True, label='Hot Gas')
        self.thermalsplot.addline(4, self.x*1e3, self.p_coolant/1e5, 'b', label='Coolant')
        self.thermalsplot.addline(4, self.x*1e3, self.p_sat_coolant/1e5, 'b--', label='Coolant Saturation')
        self.thermalsplot.plt(5, self.x*1e3, self.M, 'Mach', xlabel, 'Mach', 'b', True)
        self.thermalsplot.plt(6, self.x*1e3, self.gamma, 'Gamma', xlabel, 'Gamma', 'b', True)
        self.thermalsplot.plt(7, self.x*1e3, self.pr, 'Prandtl Number', xlabel, 'Prandtl Number', 'b', True)
        self.thermalsplot.plt(8, self.x*1e3, self.Twg, 'Wall Temp', xlabel, 'Wall Temp (K)', 'r', True, label='Twg')
        self.thermalsplot.addline(8, self.x*1e3, self.Twc, 'm', label='Twc')
        self.thermalsplot.addline(8, self.x*1e3, self.Tc, 'b', label='Tc')
        self.thermalsplot.plt(9, self.x*1e3, self.Tc, 'Coolant Temperature', xlabel, 'Coolant Temperature (K)', 'r', True, label='Coolant')
        self.thermalsplot.addline(9, self.x*1e3, self.t_sat_coolant, 'b--', label='Coolant Saturation')
        self.thermalsplot.plt(10, self.x*1e3, self.rho_coolant, 'Coolant Density', xlabel, 'Coolant Density (kg/m^3)', 'b', True)
        self.thermalsplot.plt(11, self.x*1e3, self.hc*1e-3,'Coolant Side Conv. Coeff.', xlabel, 'Coolant Side Conv. Coeff. (kW/m^2/K)', 'r', True)
        self.thermalsplot.plt(12, self.x*1e3, self.v_coolant, 'Coolant Velocity', xlabel, 'Coolant Velocity (m/s)', 'b', True)
        self.thermalsplot.plt(13, self.x*1e3, self.cp_coolant, 'Coolant Specific Heat', xlabel, 'Coolant Specific Heat (J/kg/K)', 'b', True)
        self.thermalsplot.plt(14, self.x*1e3, self.k_coolant, 'Coolant Thermal Conductivity', xlabel, 'Coolant Thermal Conductivity (W/m/K)', 'b', True)
        self.thermalsplot.plt(15, self.x*1e3, self.mu_coolant*1e3, 'Coolant Viscosity', xlabel, 'Coolant Viscosity (mPa.s)', 'b', True)

class subplot:
    def __init__(self, yn: int, xn: int, title: str, engine: engine):
        self.fig = plt.figure(figsize=(xn * 4, yn * 4), layout="compressed")
        self.fig.suptitle(title)
        self.xn = xn
        self.yn = yn
        self.ax = {}
        self.ax2 = {}
        self.x = engine.x
        self.r = engine.r
        self.max_r = np.max(engine.r) 

    def plt(self, loc, x, y, title, xlabel, ylabel, colour, draw_engine_contour=True, **label):
        if 'label' in label:
            label = label['label']
        else:
            label = None
        self.ax[loc] = self.fig.add_subplot(self.yn, self.xn, loc)
        self.ax[loc].plot(x, y, colour, label=label)
        self.ax[loc].set_title(title)
        self.ax[loc].set_xlabel(xlabel)
        self.ax[loc].set_ylabel(ylabel)
        self.ax[loc].grid(alpha=1)
        self.ax[loc].set_xlim(self.x[0]*1e3, self.x[-1]*1e3)
        # self.ax[loc].set_aspect('equal', adjustable='datalim')
        self.ax[loc].xaxis.grid(AutoMinorLocator())
        self.ax[loc].yaxis.grid(AutoMinorLocator())
        if draw_engine_contour == True:
            self.ax2[loc] = self.ax[loc].twinx()
            self.ax2[loc].plot(self.x*1e3, self.r*1e3, color='gray')
            self.ax2[loc].set_ylim(0, self.max_r*5e3)
            self.ax2[loc].set_yticks([])
            self.ax2[loc].set_ylabel('')

    def addline(self, loc, x, y, colour, label = None):
        self.ax[loc].plot(x, y, colour, label=label)
        self.ax[loc].legend()   

if __name__ == '__main__':
    from os import system
    system('cls')

    ipa = custom_fluids.thermo_fluid('isopropanol', name = "IPA", cea_name = "Isopropanol")
    n2o = custom_fluids.thermo_fluid('n2o', name = "N2O", cea_name = "Nitrous Oxide")

    igniter = engine('configs/igniter.yaml')
    csj = engine('configs/csj.yaml')
    pluto = engine('configs/l9.yaml')
    l9_csj = engine('configs/l9_csj.yaml')

    csj_channels = constant_channels(
        material=custom_materials.Al6082_T6,
        n_channels = 40,
        wall_thickness = 0.8e-3,
        rib_height = 1.2e-3,
        channel_width = 0.5e-3
    )

    printed_channels = arc_channels(
        material=custom_materials.GRCop42,
        n_channels = 200,
        wall_thickness = 1e-3,
        rib_height = 2.5e-3,
        arc_angle = 3.8,
    )

    card_str = """
    fuel C2H5OH(L)   C 2 H 6 O 1     wt%=80.00
    h,cal=-66370.0      t(k)=298.15
    fuel water H 2.0 O 1.0   wt%=20.00
    h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
    """

    add_new_fuel(card_str=card_str, name='Ethanol80')

    # OFsweep(
    #     fuel = 'RP-1',
    #     ox = 'LOX',
    #     pc = 200e5,
    #     eps = 110,
    #     start = 0.1,
    #     end = 4,
    #     show_vac=True
    # )

    # pluto.dt = 35.5e-3
    # pluto.de = 2.35*pluto.dt
    # pluto.update()

    # card_str = """
    # oxid N2O4(L)   N 2 O 4   wt%=96.5
    # h,cal=-4676.0     t(k)=298.15
    # oxid SiO2  Si 1.0 O 2.0    wt%=3.5
    # h,cal=-216000.0     t(k)=298.15  rho=1.48
    # """

    # l9_csj.combustion_sim(
    #     fuel = 'Ethanol',
    #     ox = 'LOX',
    #     OF = 1.35,
    #     pc = 30e5,
    #     cf_eff = 0.98,
    # )
    # l9_csj.print_data()

    # pluto.dt = 43e-3
    # pluto.de = np.sqrt(4.5)*pluto.dt
    # pluto.lc = 190e-3
    # pluto.le = 80e-3
    pluto.update()

    pluto.combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        OF = 3.06,
        pc = 31.8e5,
        cstar_eff = 0.96,
        # cf_eff = 0.95,
        cf_eff = 0.907,
    )
    pluto.print_data()

    csj.combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        OF = 3.5,
        pc = 25e5,
    )

    def calc_Re_throat(engine: engine):
        cp = engine.cp_t
        gamma = engine.gam_t
        cv = cp / gamma
        R = cp - cv

        a_throat = np.sqrt(gamma * R * engine.Tg_t)
        mu = engine.mu_t
        rho_throat = engine.pt / (R * engine.Tg_t)

        return (rho_throat * a_throat * engine.dt) / mu
    
    print(f'Throat Reynolds Number: {calc_Re_throat(pluto):.2f}')
    print(f'Throat Reynolds Number: {calc_Re_throat(csj):.2f}')

    # igniter.de = igniter.dt
    # igniter.update()

    # igniter.combustion_sim(
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     OF = 6.0,
    #     pc = 15e5,
    #     cstar_eff=0.7,
    #     cf_eff=1,
    # )
    # igniter.print_data()

    # csj.combustion_sim(
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     OF = 3.5,
    #     pc = 25e5,
    #     cstar_eff=0.95,
    #     cf_eff=0.92
    # )
    # csj.print_data()

    # pluto.combustion_sim(
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     OF = 3.066,
    #     pc = 31.8e5,
    #     # cstar_eff = 0.9592,
    #     # cf_eff = 0.9064
    # )
    # pluto.print_data()

    # pluto.thermal_sim(
    #     cooling_channel = printed_channels,
    #     coolant = lox,
    #     coolant_mdot = pluto.ox_mdot,
    #     coolant_T_in = 95,
    #     coolant_p_in = 80e5,
    #     rev_flow = True,
    #     roughness_mult = 1,
    # )
    # pluto.plot_thermals('Engine Thermals')

    # plt.figure()
    # plt.plot(pluto.x*1e3, pluto.abs_roughness*1e6, label='Roughness')
    # plt.xlabel('Axial Position (mm)')
    # plt.ylabel('Ra (µm)')
    # plt.title('Channel Absolute Roughness')
    # plt.legend()
    # plt.grid()
    # plt.figure()
    # plt.plot(pluto.x*1e3, pluto.angle, label='Angle')
    # plt.xlabel('Axial Position (mm)')
    # plt.ylabel('Angle (degrees)')
    # plt.title('Channel Angle')
    # plt.legend()
    # plt.grid()
    # plt.figure()
    # plt.plot(pluto.x*1e3, pluto.Re_coolant, label='Reynolds Number')
    # plt.axhline(y=2300, color='r', linestyle='--', label='Re = 2300')
    # plt.xlabel('Axial Position (mm)')
    # plt.ylabel('Re (dimensionless)')
    # plt.title('Channel Reynolds Number')
    # plt.legend()
    # plt.grid()
    # plt.figure()
    # plt.plot(pluto.x*1e3, pluto.fr_coolant, label='Friction Factor')
    # plt.xlabel('Axial Position (mm)')
    # plt.ylabel('Friction Factor (dimensionless)')
    # plt.title('Channel Friction Factor')
    # plt.legend()
    # plt.grid()

    plt.show()