from matplotlib.pylab import f
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
from pyfluids import Fluid, FluidsList, Mixture, Input
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from os import path, system
import numpy as np
import scipy as sp
import csv
import warnings

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

g = sp.constants.g

with open('film_stability.csv') as data: # [Re, eta]
    film_stab_coeff = []
    film_stab_re = []
    filtered_data = csv.reader(data)
    for line in filtered_data:
        film_stab_re.append(float(line[0]))
        film_stab_coeff.append(float(line[1]))

in2mm = lambda x: x * 25.4
mm2in = lambda x: x / 25.4
psi2bar = lambda x: x / 14.5038
bar2psi = lambda x: x * 14.5038

def OFsweep(fuel, ox, OFstart, OFend, pc, pe, cr, pamb=1.01325, showvacisp=False, filmcooled=False, film_perc=0):
        """
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
        film_frac = film_perc * 1e-2

        ceaObj = CEA_Obj(
            oxName=ox,
            fuelName=fuel,
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
            fac_CR=cr,
            make_debug_prints=False)
        OFs = np.linspace(OFstart,OFend,100)
        ispseas = []
        ispseas_true = []
        ispvacs = []
        Tcs = []
        for OF in OFs:
            eps = ceaObj.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
            [ispsea, _] = ceaObj.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=pamb, frozen=0, frozenAtThroat=0)
            if filmcooled == True:
                ispsea_true = ispsea / (1 + (film_frac * (1/(OF + 1))))
                ispseas_true.append(ispsea_true)
            [ispvac, _, Tc] = ceaObj.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=0, frozenAtThroat=0)
            ispseas.append(ispsea)
            ispvacs.append(ispvac)
            Tcs.append(Tc)

        # Find peak values and their corresponding OFs
        peak_sl_isp_idx = np.argmax(ispseas)
        peak_sl_isp_OF = OFs[peak_sl_isp_idx]
        peak_vac_isp_idx = np.argmax(ispvacs)
        peak_vac_isp_OF = OFs[peak_vac_isp_idx]
        peak_tc_idx = np.argmax(Tcs)
        peak_tc_OF = OFs[peak_tc_idx]

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('OF Ratio')
        ax1.set_ylabel('ISP (s)', color='b')
        ax1.plot(OFs, ispseas, 'b', label='SL ISP')
        if showvacisp == True:
            ax1.plot(OFs, ispvacs, 'b--', label='Vac ISP')
        if filmcooled == True:
            ax1.plot(OFs, ispseas_true, 'b--', label='True SL ISP')
        ax2 = ax1.twinx()
        ax2.plot(OFs, Tcs, 'r',label='Chamber Temp')
        ax2.set_ylabel('Chamber Temp (K)', color='r',)
        ax1.grid()
        ax1.set_ylim(bottom=0)
        plt.xlim(OFstart, OFend)
        
        # Add vertical lines for peak values
        ax1.axvline(x=peak_sl_isp_OF, color='k', linestyle='--', alpha=0.7)
        ax1.text(peak_sl_isp_OF + 0.1, ax1.get_ylim()[1] * 0.95, f"Peak SL ISP OF: {peak_sl_isp_OF:.2f}", 
                 verticalalignment='top', color='k')
        
        if showvacisp == True:
            ax1.axvline(x=peak_vac_isp_OF, color='k', linestyle='--', alpha=0.7)
            ax1.text(peak_vac_isp_OF + 0.1, ax1.get_ylim()[1] * 0.85, f"Peak Vac ISP OF: {peak_vac_isp_OF:.2f}", 
                     verticalalignment='top', color='k')
        
        ax2.axvline(x=peak_tc_OF, color='r', linestyle='--', alpha=0.7)
        ax2.text(peak_tc_OF + 0.1, ax2.get_ylim()[1] * 0.95, f"Stoichiometric OF: {peak_tc_OF:.2f}", 
                 verticalalignment='top', color='r')
        
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.title(f'OF Sweep\n{fuel} / {ox}, Pc = {pc:.2f} bar, Pe = {pe:.2f} bar, Pamb = {pamb:.2f} bar, CR = {cr:.2f}')
        fig.tight_layout()

class engine:
    def __init__(self, file):
        self.file = file
        if path.exists(self.file) and path.getsize(self.file) > 0:
            with open(self.file, 'r') as input_file:
                data = input_file.readlines()
                self.dc = float(data[0].split()[1]) * 1e-3
                self.dt = float(data[1].split()[1]) * 1e-3
                self.de = float(data[2].split()[1]) * 1e-3
                self.eps = (self.de/self.dt)**2
                self.conv_angle = float(data[4].split()[1])
                self.lc = float(data[5].split()[1]) * 1e-3
                self.le = float(data[6].split()[1]) * 1e-3
                self.theta_n = float(data[7].split()[1])
                self.theta_e = float(data[8].split()[1])

            self.rc = self.dc/2
            self.rt = self.dt/2
            self.re = self.de/2
            self.ac = np.pi*self.rc**2
            self.at = np.pi*self.rt**2
            self.ae = np.pi*self.re**2
            self.cr = self.ac/self.at      
            self.rao_percentage = 100 * np.tan(np.deg2rad(15)) * self.le / ((np.sqrt(self.eps)-1)*self.rt)
            self.R2 = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - 1.5*self.rt
            self.ltotal = self.lc + self.le

            self.cstar_eff = 1

    def update(self):
        self.rc = self.dc/2
        self.rt = self.dt/2
        self.re = self.de/2
        self.ac = np.pi*self.rc**2
        self.at = np.pi*self.rt**2
        self.ae = np.pi*self.re**2
        self.cr = self.ac/self.at      
        self.rao_percentage = 100 * np.tan(np.deg2rad(15)) * self.le / ((np.sqrt(self.eps)-1)*self.rt)
        self.R2 = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - 1.5*self.rt
        self.ltotal = self.lc + self.le

    def gen_cea_obj(self):
        self.cea = CEA_Obj(
            oxName = self.ox,
            fuelName = self.fuel,
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
            fac_CR=self.cr,
            make_debug_prints=False)

    def combustion_sim(self, fuel, ox, OF, pc, cstar_eff = 1, sizing=False, **kwargs):
        self.fuel = fuel
        self.ox = ox
        self.OF = OF
        self.pc = pc
        self.pamb = 1.01325
        self.cstar_eff = cstar_eff
        if 'pamb' in kwargs:
            self.pamb = kwargs['pamb']

        if sizing == True:
            self.thrust = kwargs['thrust']
            self.pe = kwargs['pe']
            self.cr = kwargs['cr']
            self.conv_angle = kwargs['conv_angle']
            self.lstar = kwargs['lstar']
            self.rao_percentage = kwargs['rao_percentage']

        self.gen_cea_obj()

        if sizing == True:
            self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.pc, MR=self.OF, PcOvPe=(self.pc/self.pe))

        [self.ispvac, self.cstar, _] = self.cea.get_IvacCstrTc(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.ispsea, self.exitcond] = self.cea.estimate_Ambient_Isp(Pc=self.pc, MR=self.OF, eps=self.eps, Pamb=self.pamb, frozen=0, frozenAtThroat=0)
        [self.Tg_c, self.Tg_t, self.Tg_e] = self.cea.get_Temperatures(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        self.ispsea = self.ispsea * self.cstar_eff
        self.ispvac = self.ispvac * self.cstar_eff
        self.ispthroat = self.cea.get_Throat_Isp(Pc=self.pc, MR=self.OF, frozen=0) * self.cstar_eff
        self.cstar = self.cstar * self.cstar_eff
        self.pt = self.pc/self.cea.get_Throat_PcOvPe(Pc=self.pc, MR=self.OF)
        self.cf = (self.cea.get_PambCf(Pamb=self.pamb, Pc=self.pc, MR=self.OF, eps=self.eps))[0]
        if sizing == True:
            self.at = self.thrust / (self.pc * self.cf * 1e5)
        self.mdot = self.pc * 1e5 * self.at / self.cstar
        self.ox_mdot = self.mdot * self.OF / (1 + self.OF)
        self.fuel_mdot = self.mdot / (1 + self.OF)
        if sizing == False:
            self.thrust = self.at * self.pc * self.cf * 1e5
        self.thrust_thoat = self.mdot * self.ispthroat * g
        self.PinjPcomb = self.cea.get_Pinj_over_Pcomb(Pc=self.pc, MR=self.OF)
        self.Me = self.cea.get_MachNumber(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        [self.cp_c, self.mu_c, self.k_c, self.pr_c] = self.cea.get_Chamber_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.cp_t, self.mu_t, self.k_t, self.pr_t] = self.cea.get_Throat_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.cp_e, self.mu_e, self.k_e, self.pr_e] = self.cea.get_Exit_Transport(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        [_, self.gam_t] = self.cea.get_Throat_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0)
        [_, self.gam_c] = self.cea.get_Chamber_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps)
        [_, self.gam_e] = self.cea.get_exit_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        self.mu_c = self.mu_c * 1e-3 # now pa-s
        self.k_c = self.k_c * 100 # now W/m-K

        if sizing == True:
            self.rt = np.sqrt(self.at/np.pi)
            self.dt = 2*self.rt

            self.ae = self.at * self.eps
            self.re = np.sqrt(self.ae/np.pi)
            self.de = 2*self.re

            self.ac = self.at * self.cr
            self.rc = np.sqrt(self.ac/np.pi)
            self.dc = 2*self.rc

            self.R2 = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - 1.5*self.rt

            self.n_points = 100
            l = np.linspace(0, (self.R2*np.sin(np.deg2rad(self.conv_angle))), self.n_points)
            r = np.sqrt(self.R2**2 - l**2) + self.rc - self.R2
            l2 = np.linspace(0, (1.5*self.rt*np.sin(np.deg2rad(self.conv_angle))), self.n_points)
            r2 = 2.5*self.rt - np.sqrt((1.5*self.rt)**2 - (l2 - 1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))**2)
            
            self.vc = self.lstar*self.at
            self.vcyl = self.vc - np.sum(r[0:-1]*r[0:-1])*np.pi*(self.R2*np.sin(np.deg2rad(self.conv_angle)))/(self.n_points-1) - np.sum(r2[0:-1]*r2[0:-1])*np.pi*(1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))/(self.n_points-1)

            if self.vcyl < 0:
                raise ValueError('L* too short / Contraction ratio too high') 

            self.lcyl = self.vcyl/self.ac
            self.lc = self.lcyl + (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle))
            self.le = (np.sqrt(self.eps)-1)*self.rt*self.rao_percentage/(100*np.tan(np.deg2rad(15)))

            # self.nozzle = Nozzle(
            #     Rt = self.rt * 1000 / 25.4,
            #     CR = self.cr,
            #     eps = self.eps,
            #     pcentBell = self.rao_percentage,
            #     Rup = 1.5,
            #     Rd = 0.382,
            #     Rc = self.R2/self.rt,
            #     cham_conv_ang = self.conv_angle,
            #     theta = None,
            #     exitAng = None,
            #     forceCone = 0,
            #     use_huzel_angles = True)

            self.theta_n = 22 # self.nozzle.theta
            self.theta_e = 14 # self.nozzle.exitAng

            self.generate_contour()

            self.save()
        else:
            self.pe = self.pc/self.cea.get_PcOvPe(Pc=self.pc, MR=self.OF, eps=self.eps)

        # self.print_data()

    def size_injector(self, injector, fuel_Cd, fuel_stiffness, fuel_rho, ox_Cd, ox_stiffness, ox_rho):
        self.fuel_Cd = fuel_Cd
        self.ox_Cd = ox_Cd
        self.fuel_stiffness = fuel_stiffness
        self.fuel_dp = self.fuel_stiffness * self.pc
        self.ox_stiffness = ox_stiffness
        self.ox_dp = self.ox_stiffness * self.pc

        injector.fuel_CdA = self.fuel_mdot / np.sqrt(2*fuel_rho*self.fuel_dp*1e5)
        injector.fuel_A = self.fuel_CdA / self.fuel_Cd
        injector.ox_CdA = self.ox_mdot / np.sqrt(2*ox_rho*self.ox_dp*1e5)
        injector.ox_A = injector.ox_CdA / self.ox_Cd


        self.fuel_inj_p = self.pc + self.fuel_dp
        self.ox_inj_p = self.pc + self.ox_dp

    def inj_p_combustion_sim(self, injector, fuel, ox, fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, ox_temp=15, fuel_gas_class=None, fuel_temp=15, cstar_eff=1, n_max=100):
        """Combustion sim based on injector pressures.\n
        Required Inputs: fuel, ox, fuel_inj_p, ox_inj_p, fuel_rho, ox_rho\n
        Optional Inputs: oxclass, ox_gas, ox_temp, fuelclass, fuel_gas, fuel_temp"""

        self.fuel = fuel
        self.ox = ox
        self.fuel_inj_p = fuel_inj_p
        self.ox_inj_p = ox_inj_p
        self.fuel_rho = fuel_rho
        self.ox_rho = ox_rho
        
        if ox_gas_class is None:
            ox_gas = False
        else:
            ox_gas = True

        if fuel_gas_class is None:
            fuel_gas = False
        else:
            fuel_gas = True

        if ox_gas:
            ox_gas_class.update(Input.temperature(ox_temp), Input.pressure(ox_inj_p*1e5))
            ox_R = 8.31447/ox_gas_class.molar_mass
            ox_gamma = (ox_gas_class.specific_heat)/(ox_gas_class.specific_heat-ox_R)
            ox_rho = ox_gas_class.density
            # choking_ratio = ((gamma + 1)/2)**(gamma/(gamma-1))
            ox_k = (2/(ox_gamma+1))**((ox_gamma+1)/(ox_gamma-1))
        else:
            ox_gamma = 0
            ox_k = 0

        if fuel_gas:
            fuel_gas_class.update(Input.temperature(fuel_temp), Input.pressure(fuel_inj_p*1e5))
            fuel_R = 8.31447/fuel_gas_class.molar_mass
            fuel_gamma = (fuel_gas_class.specific_heat)/(fuel_gas_class.specific_heat-fuel_R)
            fuel_rho = fuel_gas_class.density
            fuel_k = (2/(fuel_gamma+1))**((fuel_gamma+1)/(fuel_gamma-1))
        else:
            fuel_gamma = 0
            fuel_k = 0

        # def pcfunc(pc, cstar, fuel_CdA, fuel_inj_p, fuel_rho, ox_CdA, ox_inj_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k):
        #     if ox_gas:
        #         min_choked_p = 2 * pc / (2-ox_gamma*ox_k)
        #         if (ox_inj_p >= min_choked_p): #((ox_inj_p/pc) >= choking_ratio)&(ox_can_choke == True):
        #             return ((cstar / self.at) * ((fuel_CdA*np.sqrt(2*fuel_rho*(fuel_inj_p-pc)*1e5)) + (ox_CdA*np.sqrt(ox_gamma*ox_rho*ox_inj_p*1e5*ox_k))) * 1e-5) - pc
        #         else:
        #             return ((cstar / self.at) * ((fuel_CdA*np.sqrt(2*fuel_rho*(fuel_inj_p-pc)*1e5)) + (ox_CdA*np.sqrt(2*ox_rho*(ox_inj_p-pc)*1e5))) * 1e-5) - pc
        #     else:
        #         return ((cstar / self.at) * ((fuel_CdA*np.sqrt(2*fuel_rho*(fuel_inj_p-pc)*1e5)) + (ox_CdA*np.sqrt(2*ox_rho*(ox_inj_p-pc)*1e5))) * 1e-5) - pc
            
        def pcfunc(pc, cstar, fuel_CdA, fuel_inj_p, fuel_rho, ox_CdA, ox_inj_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k):
            if ox_gas:
                ox_min_choked_p = 2 * pc / (2-ox_gamma*ox_k)
                if ox_inj_p >= ox_min_choked_p:
                    mdot_o = ox_CdA * np.sqrt(ox_gamma * ox_rho * ox_inj_p * 1e5 * ox_k)
                else:
                    mdot_o = ox_CdA * np.sqrt(2 * ox_rho * (ox_inj_p - pc) * 1e5)
            else:
                mdot_o = ox_CdA*np.sqrt(2*ox_rho*(ox_inj_p-pc)*1e5)

            if fuel_gas:
                fuel_min_choked_p = 2 * pc / (2-fuel_gamma*fuel_k)
                if fuel_inj_p >= fuel_min_choked_p:
                    mdot_f = fuel_CdA * np.sqrt(fuel_gamma * fuel_rho * fuel_inj_p * 1e5 * fuel_k)
                else:
                    mdot_f = fuel_CdA * np.sqrt(2 * fuel_rho * (fuel_inj_p - pc) * 1e5)
            else:
                mdot_f = fuel_CdA*np.sqrt(2*fuel_rho*(fuel_inj_p-pc)*1e5)

            # return ((cstar / self.at) * (mdot_f + mdot_o) * 1e-5) - pc
            return ((cstar / self.at) * (mdot_f + mdot_o) * 1e-5) - pc
        
        self.gen_cea_obj()

        min_inj_p = min(self.fuel_inj_p, self.ox_inj_p)

        cstar_init = 1500 * cstar_eff
        cstar = cstar_init
        rel_diff = 1

        n = 0
        while rel_diff > 1e-4:
            n += 1
            converged = True
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                try:
                    # pc = sp.optimize.fsolve(pcfunc, x0=(1), args=(cstar*cstar_eff, injector.fuel_CdA, self.fuel_inj_p, fuel_rho, injector.ox_CdA, self.ox_inj_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k))[0]
                    pc = sp.optimize.root_scalar(pcfunc, bracket=[1, min_inj_p], args=(cstar*cstar_eff, injector.fuel_CdA, self.fuel_inj_p, fuel_rho, injector.ox_CdA, self.ox_inj_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k), method='brentq').root
                except ValueError:
                    converged = False
                    pass
                except Exception:
                    converged = False
                    pass
                    
            mdot_f = injector.fuel_CdA * np.sqrt(2 * fuel_rho * (self.fuel_inj_p - pc) * 1e5)
            if ox_gas:
                min_choked_p = 2 * pc / (2-ox_gamma*ox_k)
                if ox_inj_p >= min_choked_p:
                    mdot_o = injector.ox_CdA * np.sqrt(ox_gamma * ox_rho * ox_inj_p * 1e5 * ox_k)
                else:
                    mdot_o = injector.ox_CdA * np.sqrt(2 * ox_rho * (self.ox_inj_p - pc) * 1e5)
            else:
                mdot_o = injector.ox_CdA*np.sqrt(2*ox_rho*(self.ox_inj_p-pc)*1e5)

            if fuel_gas:
                min_choked_p = 2 * pc / (2-fuel_gamma*fuel_k)
                if fuel_inj_p >= min_choked_p:
                    mdot_f = injector.fuel_CdA * np.sqrt(fuel_gamma * fuel_rho * fuel_inj_p * 1e5 * fuel_k)
                else:
                    mdot_f = injector.fuel_CdA * np.sqrt(2 * fuel_rho * (fuel_inj_p - pc) * 1e5)
            else:
                mdot_f = injector.fuel_CdA*np.sqrt(2*fuel_rho*(fuel_inj_p-pc)*1e5)

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
        # else:
        #     print(f"Converged: fuel inj p: {self.fuel_inj_p:.2f}, ox inj p: {self.ox_inj_p:.2f}, n: {n}")

        self.combustion_sim(fuel, ox, OF, pc, cstar_eff)

        self.OF = OF
        self.pc = pc

    def generate_contour(self):
        self.n_points = 20

        # Converging arc
        l = np.linspace(0, (self.R2*np.sin(np.deg2rad(self.conv_angle))), int(1.5*self.n_points))
        r = np.sqrt(self.R2**2 - l**2) + self.rc - self.R2
        l = l + (self.lc - (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle)))

        # Throat upstream arc

        l2 = np.linspace(0, (1.5*self.rt*np.sin(np.deg2rad(self.conv_angle))), int(self.n_points))
        r2 = 2.5*self.rt - np.sqrt((1.5*self.rt)**2 - (l2 - 1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))**2)
        l2 = l2 + self.R2*np.sin(np.deg2rad(self.conv_angle)) + (self.lc - (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle)))
        
        # Throat downstream arc

        l3 = np.linspace(0, (0.382*self.rt*np.sin(np.deg2rad(self.theta_n))), int(0.3*self.n_points))
        r3 = 1.382*self.rt - np.sqrt((0.382*self.rt)**2 - l3**2)
        l3 = l3 + self.lc

        # Parabolic nozzle

        t = np.concatenate((np.linspace(0, 0.2, int(1*self.n_points)), np.linspace(0.2, 1, int(1*self.n_points))))

        Nx = l3[-1]
        Ny = r3[-1]
        Ex = self.lc + self.le
        Ey = self.re

        m1 = np.tan(np.deg2rad(self.theta_n))
        m2 = np.tan(np.deg2rad(self.theta_e))
        c1 = Ny - m1*Nx
        c2 = Ey - m2*Ex
        Qx = (c2 - c1)/(m1 - m2)
        Qy = (m1*c2 - m2*c1)/(m1 - m2)

        l4 = Nx*(1-t)**2 + 2*(1-t)*t*Qx + Ex*t**2
        r4 = Ny*(1-t)**2 + 2*(1-t)*t*Qy + Ey*t**2

        contour = [np.concatenate((np.array(self.rc*np.ones(2*self.n_points)), r[1:], r2[1:], r3[1:], r4[1:])), np.concatenate((np.array(np.linspace(0, l[0], 2*self.n_points)), l[1:], l2[1:], l3[1:], l4[1:]))] # [radius, axial length]
        self.r = contour[0]
        self.x = contour[1]
        self.stations = len(self.r)

    def show_contour(self):
        self.generate_contour()
        plt.figure()
        plt.plot(self.x*1e3, self.r*1e3, 'b', label='Contour')
        plt.plot(self.x*1e3, -self.r*1e3, 'b')
        plt.plot(self.x*1e3, -self.r*1e3, 'xk')
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.grid()
        plt.xlabel('Axial Distance (mm)')
        plt.ylabel('Radius (mm)')
        plt.title('Chamber Contour')
        plt.axis('equal')
        plt.xlim(left=0)
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
            output_file.write(f'\ntheta_n: {self.theta_n}')
            output_file.write(f'\ntheta_e: {self.theta_e}')

    def print_data(self):
        print('-----------------------------------')
        print(f'{self.fuel} / {self.ox}\n')
        print(f'OF:                 {self.OF:.3f}')
        print(f'Chamber Pressure:   {self.pc:.2f} bar')
        print(f'Thrust:             {self.thrust:.2f} N')
        print(f'Throat Thrust:      {self.thrust_thoat:.2f} N\n')
        print(f'Sea Level ISP:      {self.ispsea:.2f} s')
        print(f'Throat ISP:         {self.ispthroat:.2f} s')
        print(f'Vac ISP:            {self.ispvac:.2f} s')
        print(f'C*:                 {self.cstar:.2f} m/s')
        print(f'Ct:                 {self.cf:.4f}\n')
        print(f'Chamber Temp:       {self.Tg_c:.2f} K')
        print(f'Throat Temp:        {self.Tg_t:.2f} K')
        print(f'Exit Temp:          {self.Tg_e:.2f} K\n')
        print(f'Throat Pressure:    {self.pt:.2f} bar')
        print(f'Exit Pressure:      {self.pe:.2f} bar\n')
        print(f'Exit Mach Number:   {self.Me:.3f}')
        print(f'Expansion Ratio:    {self.eps:.3f}')
        print(f'Exit Condition:     {self.exitcond}\n')
        if(self.mdot >= 1e-1):
            print(f'Total mdot:         {self.mdot:.3f} kg/s')
            print(f'Ox mdot:            {self.ox_mdot:.3f} kg/s')
            print(f'Fuel mdot:          {self.fuel_mdot:.3f} kg/s\n')
        else:
            print(f'Total mdot:         {self.mdot*1e3:.2f} g/s')
            print(f'Ox mdot:            {self.ox_mdot*1e3:.2f} g/s')
            print(f'Fuel mdot:          {self.fuel_mdot*1e3:.2f} g/s\n')
        print(f'Chamber Diameter:   {self.dc*1e3:.3f} mm')
        print(f'Throat Diameter:    {self.dt*1e3:.3f} mm')
        print(f'Exit Diameter:      {self.de*1e3:.3f} mm\n')
        print(f'Chamber Length      {self.lc*1e3:.3f} mm')
        print(f'Exit Length         {self.le*1e3:.3f} mm')
        print(f'Total Length:       {(self.lc+self.le)*1e3:.3f} mm\n')
        print(f'Contraction Ratio:  {self.cr:.3f}')
        print(f'Nozzle Inlet Angle: {self.theta_n:.3f}°')
        print(f'Nozzle Exit Angle:  {self.theta_e:.3f}°')
        print(f'Pinj/Pcomb:         {self.PinjPcomb:.3f}\n')
        print('-----------------------------------')

    def thermal_sim(self, wall_k, n_channels, h_rib, tw, channel_arc_angle, coolant, coolant_mdot, coolant_T_in, coolant_p_in, rev_flow):
        self.generate_contour()

        self.wall_k = wall_k
        self.coolant = coolant
        self.n_channels = n_channels
        self.h_rib = h_rib
        self.tw = tw
        self.channel_arc_angle = channel_arc_angle
        self.channel_width = (2*self.r+2*self.tw+self.h_rib)*np.sin(np.deg2rad(channel_arc_angle/2))
        self.channel_area = self.channel_width * self.h_rib
        self.d_h = 2*self.channel_area / (self.h_rib+self.channel_width)
        self.coolant_mdot = coolant_mdot

        self.pr          = np.zeros(self.stations)
        self.gamma       = np.zeros(self.stations)
        self.M           = np.zeros(self.stations)
        self.hg          = np.zeros(self.stations)
        self.q           = np.zeros(self.stations)
        self.pg          = np.zeros(self.stations)
        self.Tg          = np.zeros(self.stations)
        self.Twg         = np.zeros(self.stations)
        self.Twc         = np.zeros(self.stations)
        self.Tc          = np.zeros(self.stations)
        self.v_coolant   = np.zeros(self.stations)
        self.rho_coolant = np.zeros(self.stations)
        self.mu_coolant  = np.zeros(self.stations)
        self.k_coolant   = np.zeros(self.stations)
        self.cp_coolant  = np.zeros(self.stations)
        self.Re_coolant  = np.zeros(self.stations)
        self.Pr_coolant  = np.zeros(self.stations)
        self.Nu_coolant  = np.zeros(self.stations)
        self.hc          = np.zeros(self.stations)
        self.total_heat_flux = 0

        def machfunc(mach, area, gamma):
            area_ratio = area / self.at
            if mach == 0:
                mach = 1e-7
            return (area_ratio - ((1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) / ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))))

        self.throat_rcurv =  self.rt * 0.941
        t_next = coolant_T_in

        for j in range(self.stations):
            if rev_flow == True:
                i = -(j+1)
                inext = i - 1
            else:
                i = j
                inext = i + 1
            r = self.r[i]
            A = np.pi * r * r
            self.Tc[i] = t_next

            self.coolant_class = Fluid(self.coolant)
            self.coolant_class.update(Input.pressure(coolant_p_in*1e5),  Input.temperature(self.Tc[i] - 273.15))

            self.rho_coolant[i] = self.coolant_class.density 
            self.cp_coolant[i] = self.coolant_class.specific_heat
            self.mu_coolant[i] = self.coolant_class.dynamic_viscosity
            self.k_coolant[i] = self.coolant_class.conductivity
            self.v_coolant[i] = self.coolant_mdot / (self.rho_coolant[i] * self.channel_area[i] * self.n_channels)
            self.Re_coolant[i] = self.rho_coolant[i] * self.v_coolant[i] * self.d_h[i] / self.mu_coolant[i]
            self.Pr_coolant[i] = self.mu_coolant[i] * self.cp_coolant[i] / self.k_coolant[i]
            self.Nu_coolant[i] = 0.023 * self.Re_coolant[i]**0.8 * self.Pr_coolant[i]**0.4
            self.hc[i] = self.Nu_coolant[i] * self.k_coolant[i] / self.d_h[i]

            if self.x[i] == self.lc:
                self.gamma[i] = self.gam_t
                self.pr[i] = self.pr_t
                self.M[i] = 1
            elif self.x[i] < self.lc:
                self.gamma[i] = (self.gam_t - self.gam_c)/(self.rt - self.rc) * (r - self.rc) + self.gam_c
                self.pr[i] = (self.pr_t - self.pr_c)/(self.rt - self.rc) * (r - self.rc) + self.pr_c
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[0, 1]).root
            else:
                self.gamma[i] = (0.5*(0.8*self.gam_e+1.2*self.gam_t) - self.gam_t)/(self.re - self.rt) * (r - self.rt) + self.gam_t
                self.pr[i] = (self.pr_e - self.pr_t)/(self.re - self.rt) * (r - self.rt) + self.pr_t
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[1, 5]).root

            self.Tg[i] = self.Tg_c * ((1 + (0.5 * (self.pr[i]**(1/3)) * (self.gamma[i] - 1) * (self.M[i]**2))) / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))
            # self.Tg[i] = self.Tg_c / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)
            self.pg[i] = self.pc * (self.Tg[i] / self.Tg_c)**(self.gamma[i] / (self.gamma[i] - 1))
            bartz = ((0.026 / (self.dt**0.2))) * ((self.mu_c**0.2) * self.cp_c / (self.pr_c**0.6)) * ((self.pc *1e5 / self.cstar)**0.8) * ((self.dt / self.throat_rcurv)**0.1) * ((self.at / A)**0.9)

            if j == self.stations-1:
                pass
            else:
                dA = np.pi * (self.r[i] + self.r[inext]) * np.sqrt((self.r[i] - self.r[inext]) ** 2 + (self.x[inext] - self.x[i]) ** 2)
                rel_diff = 1
                self.Twg[i] = (self.Tg[i] + self.Tc[i])/2
                while rel_diff > 5e-4:
                    twg_old = self.Twg[i]
                    correction_factor = 1 / (((0.5 * (self.Twg[i] / self.Tg_c) * (1 + (0.5 * (self.gamma[i] - 1)) * (self.M[i]**2)) + 0.5)**0.68) * ((1 + 0.5 * (self.gamma[i] - 1) * (self.M[i]**2))**0.12))
                    self.hg[i] = bartz * correction_factor
                    self.q[i] = self.hg[i] * (self.Tg[i] - self.Twg[i])
                    self.Twc[i] = (self.q[i] / self.hc[i]) + self.Tc[i]
                    self.Twg[i] = (self.q[i] * self.tw / self.wall_k) + self.Twc[i]

                    rel_diff = abs((self.Twg[i] - twg_old) / twg_old)
                    # print(bartz,correction_factor)
                    # print(f'Iteration: {j}, Station: {i}')
                    # print(f'Twg_old: {twg_old:.2f}, Twg_new: {self.Twg[i]:.2f}, Tg_c: {self.Tg_c:.2f}, Tg: {self.Tg[i]:.2f}, M: {self.M[i]:.2f}, gamma: {self.gamma[i]:.2f}')
                    # print(f'hg: {self.hg[i]:.2f}, hc: {self.hc[i]:.2f}, q: {self.q[i]:.2f}, Twc: {self.Twc[i]:.2f}, Twg: {self.Twg[i]:.2f}, Tc: {self.Tc[i]:.2f}, t_next: {t_next:.2f}')
                    # print(f'Relative Difference: {rel_diff:.2f}\n')
                # def tempsolver():
                #     solverVals = {}
                #     def qfunc(twg, bartz, M, gamma, tg_c, tg, tc, hc):
                #         correction_factor = 1 / ((0.5 * (twg / tg_c) * (1 + (0.5 * (gamma - 1)) * (M**2)) + 0.5)**0.68 * (1 + 0.5 * (gamma - 1) * (M**2))**0.12)
                #         solverVals['hg'] = bartz * correction_factor
                #         solverVals['q'] = solverVals['hg'] * (tg_c - twg)
                #         solverVals['twc'] = (solverVals['q'] / hc) + tc
                #         return (((solverVals['twc'] - tc) * hc) - (solverVals['hg'] * (tg - twg)))
                #     return solverVals, qfunc

                # out, qfunc = tempsolver()
                # self.Twg[i] = sp.optimize.fsolve(qfunc, x0=(0.5*(self.Tg[i]+self.Tc[i])), args=(bartz, self.M[i], self.gamma[i], self.Tg_c, self.Tg[i], self.Tc[i], self.hc[i]))
                # self.hg[i] = out['hg']
                # self.q[i] = out['q']
                # self.Twc[i] = out['twc']

                # print(f'Iteration: {j}')
                # print(f'Twg: {self.Twg[i]:.2f}, Hg: {self.hg[i]:.2f}, Q: {self.q[i]:.2f}, Twc: {self.Twc[i]:.2f}, Tc: {self.Tc[i]:.2f}')

                t_next = (self.q[i] * dA / (self.coolant_mdot * self.cp_coolant[i])) + self.Tc[i]

            self.total_heat_flux += self.q[i] * dA

        if rev_flow == True:
            self.hg[0] = self.hg[1]
            self.q[0] = self.q[1]
            self.Twg[0] = self.Twg[1]
            self.Twc[0] = self.Twc[1]
            self.Tc[0] = t_next
        else:
            self.hg[-1] = self.hg[-2]
            self.q[-1] = self.q[-2]
            self.Twg[-1] = self.Twg[-2]
            self.Twc[-1] = self.Twc[-2]
            self.Tc[-1] = t_next

    def thermal_sim_fc(self, wall_k, n_channels, h_rib, tw, channel_arc_angle, coolant, coolant_mdot, coolant_T_in, coolant_p_in, film_mdot, film_T_in):
        global film_stab_coeff, film_stab_re

        self.generate_contour()

        self.wall_k = wall_k
        self.coolant = coolant
        self.coolant_class = Fluid(FluidsList[self.coolant])
        self.film_class = Fluid(FluidsList[self.coolant])
        self.n_channels = n_channels
        self.h_rib = h_rib
        self.tw = tw
        self.channel_arc_angle = channel_arc_angle
        self.channel_width = (2 * self.r + 2 * self.tw + self.h_rib)*np.sin(np.deg2rad(channel_arc_angle / 2))
        self.channel_area = self.channel_width * self.h_rib
        self.d_h = 2 * self.channel_area / (self.h_rib+self.channel_width)
        self.coolant_mdot = coolant_mdot
        self.throat_rcurv =  self.rt * 0.941
        t_next = coolant_T_in
        self.film_mdot_in = film_mdot
        t_f_next = film_T_in

        self.pr          = np.zeros(self.stations)
        self.gamma       = np.zeros(self.stations)
        self.M           = np.zeros(self.stations)
        self.hg          = np.zeros(self.stations)
        self.q           = np.zeros(self.stations)
        self.pg          = np.zeros(self.stations)
        self.Tg          = np.zeros(self.stations)
        self.Twg         = np.zeros(self.stations)
        self.Twc         = np.zeros(self.stations)
        self.Tc          = np.zeros(self.stations)
        self.v_coolant   = np.zeros(self.stations)
        self.rho_coolant = np.zeros(self.stations)
        self.mu_coolant  = np.zeros(self.stations)
        self.k_coolant   = np.zeros(self.stations)
        self.cp_coolant  = np.zeros(self.stations)
        self.Re_coolant  = np.zeros(self.stations)
        self.Pr_coolant  = np.zeros(self.stations)
        self.Nu_coolant  = np.zeros(self.stations)
        self.hc          = np.zeros(self.stations)
        self.hg_f        = np.zeros(self.stations)
        self.T_f         = np.zeros(self.stations)
        self.Re_f        = np.zeros(self.stations)
        self.film_mdot_l = np.zeros(self.stations)
        self.film_mdot_g = np.zeros(self.stations)
        self.filmstate   = np.zeros(self.stations) # 0 - liquid, 1 - vaporization, 2 - gas
        self.q_f         = np.zeros(self.stations)
        self.total_heat_flux = 0


        def machfunc(mach, area, gamma):
            area_ratio = area / self.at
            if mach == 0:
                mach = 1e-7
            return (area_ratio - ((1.0 / mach) * ((1 + 0.5*(gamma - 1)*mach*mach) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))))

        f_state_next = 0
        f_mdot_l_next = self.film_mdot_in

        for j in range(self.stations):
            i = j
            inext = i + 1
            r = self.r[i]
            A = np.pi * r * r
            self.Tc[i] = t_next
            self.T_f[i] = t_f_next
            self.filmstate[i] = f_state_next
            self.film_mdot_l[i] = f_mdot_l_next
            self.film_mdot_g[i] = self.film_mdot_in - self.film_mdot_l[i]

            self.coolant_class.update(Input.pressure(coolant_p_in*1e5),  Input.temperature(self.Tc[i] - 273.15))

            self.rho_coolant[i] = self.coolant_class.density
            self.cp_coolant[i] = self.coolant_class.specific_heat
            self.mu_coolant[i] = self.coolant_class.dynamic_viscosity
            self.k_coolant[i] = self.coolant_class.conductivity
            self.v_coolant[i] = self.coolant_mdot / (self.rho_coolant[i] * self.channel_area[i] * self.n_channels)
            self.Re_coolant[i] = self.rho_coolant[i] * self.v_coolant[i] * self.d_h[i] / self.mu_coolant[i]
            self.Pr_coolant[i] = self.mu_coolant[i] * self.cp_coolant[i] / self.k_coolant[i]
            self.Nu_coolant[i] = 0.023 * self.Re_coolant[i]**0.8 * self.Pr_coolant[i]**0.4
            self.hc[i] = self.Nu_coolant[i] * self.k_coolant[i] / self.d_h[i]

            if self.x[i] == self.lc:
                self.gamma[i] = self.gam_t
                self.pr[i] = self.pr_t
                self.M[i] = 1
            elif self.x[i] < self.lc:
                self.gamma[i] = (self.gam_t - self.gam_c)/(self.rt - self.rc) * (r - self.rc) + self.gam_c
                self.pr[i] = (self.pr_t - self.pr_c)/(self.rt - self.rc) * (r - self.rc) + self.pr_c
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[0, 1]).root
            else:
                self.gamma[i] = (0.5*(0.8*self.gam_e+1.2*self.gam_t) - self.gam_t)/(self.re - self.rt) * (r - self.rt) + self.gam_t
                self.pr[i] = (self.pr_e - self.pr_t)/(self.re - self.rt) * (r - self.rt) + self.pr_t
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[1, 5]).root

            self.Tg[i] = (self.Tg_c / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))
            self.pg[i] = self.pc * (self.Tg[i] / self.Tg_c)**(self.gamma[i] / (self.gamma[i] - 1))
            bartz = ((0.026 / (self.dt**0.2))) * (self.mu_c**0.2 * self.cp_c / (self.pr_c**0.6)) * ((self.pc *1e5 / self.cstar)**0.8) * ((self.dt / self.throat_rcurv)**0.1) * ((self.at / A)**0.9)

            if j == self.stations-1:
                pass
            else:
                dA = np.pi * (self.r[i] + self.r[inext]) * np.sqrt((self.r[i] - self.r[inext]) ** 2 + (self.x[inext] - self.x[i]) ** 2)
                rel_diff = 1
                self.Twg[i] = self.Tg[i]
                while rel_diff > 5e-4:
                    twg_old = self.Twg[i]

                    if self.filmstate[i] == 0:
                        rel_diff = 0
                        f_mdot_l_next = self.film_mdot_in
                        correction_factor = 1 / ((0.5 * self.T_f[i] / self.Tg_c * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)+0.5)**0.68 * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)**0.12)
                        self.hg_f[i] = bartz * correction_factor
                        self.hg[i] = 0
                        self.q_f[i] = self.hg_f[i] * (self.Tg[i] - self.T_f[i])
                        self.Twg[i] = self.Twc[i] = self.Tc[i]

                        self.film_class.update(Input.pressure(self.pg[i]*1e5),  Input.temperature(self.T_f[i] - 273.15))

                        mu_f = self.film_class.dynamic_viscosity
                        cp_f = self.film_class.specific_heat
                        tvap = self.film_class.bubble_point_at_pressure(self.pg[i]*1e5).temperature + 273.15
                        Re_f = self.film_mdot_l[i] / (2 * np.pi * self.r[i] * mu_f)
                        eta = np.interp(Re_f, film_stab_re, film_stab_coeff)
                        t_f_next = self.T_f[i] + dA * self.q_f[i] / (eta * self.film_mdot_l[i] * cp_f)
                        if t_f_next >= tvap:
                            rel_diff = 1
                            t_f_next = tvap
                            self.filmstate[i] = f_state_next = 1

                    elif self.filmstate[i] == 1:
                        rel_diff = 0
                        Qvap = self.coolant_class.two_phase_point_at_pressure(self.pg[i]*1e5,100).enthalpy-self.coolant_class.two_phase_point_at_pressure(self.pg[i]*1e5,0).enthalpy
                        tvap = self.film_class.bubble_point_at_pressure(self.pg[i]*1e5).temperature + 273.15
                        correction_factor = 1 / ((0.5 * self.T_f[i] / self.Tg_c * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)+0.5)**0.68 * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)**0.12)
                        self.hg_f[i] = bartz * correction_factor
                        self.hg[i] = 0
                        self.q_f[i] = self.hg_f[i] * (self.Tg[i] - self.T_f[i])
                        self.Twg[i] = self.Twc[i] = self.Tc[i]
                        t_f_next = tvap
                        f_mdot_l_next = self.film_mdot_l[i] - dA * (self.q_f[i] / Qvap)
                        if f_mdot_l_next <= 0:
                            f_mdot_l_next = 0
                            rel_diff = 1
                            self.filmstate[i] = f_state_next = 2
                    else:
                        correction_factor = 1 / ((0.5 * self.Twg[i] / self.Tg_c * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)+0.5)**0.68 * (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)**0.12)
                        self.hg[i] = bartz * correction_factor
                        self.q[i] = self.hg[i] * (self.Tg[i] - self.Twg[i])
                        self.Twc[i] = (self.q[i] / self.hc[i]) + self.Tc[i]
                        self.Twg[i] = (self.q[i] * self.tw / self.wall_k) + self.Twc[i]
                        self.T_f[i] = self.Tg[i]

                        t_next = (self.q[i] * dA / (self.coolant_mdot * self.cp_coolant[i])) + self.Tc[i]
                        rel_diff = abs((self.Twg[i] - twg_old) / twg_old)
                        f_state_next = 2
            self.total_heat_flux += self.q[i] * dA

        self.hg[-1] = self.hg[-2]
        self.q[-1] = self.q[-2]
        self.Twg[-1] = self.Twg[-2]
        self.Twc[-1] = self.Twc[-2]
        self.Tc[-1] = t_next
        self.T_f[-1] = self.T_f[-2]

    def plot_thermals(self, title):
        self.thermalsplot = subplot(3, 5, title, self)
        self.thermalsplot.plt(1, self.x*1e3, self.hg*1e-3,'Gas Side Conv. Coeff.','Axial Distance (mm)','Gas Side Conv. Coeff. (kW/m^2/K)','r', True)
        self.thermalsplot.plt(2, self.x*1e3, self.q*1e-3,'Heat Flux','Axial Distance (mm)','Heat Flux (kW/m^2)','r', True)
        self.thermalsplot.plt(3, self.x*1e3, self.Tg, 'Gas Temperature', 'Axial Distance (mm)', 'Gas Temperature (K)', 'r', True)
        self.thermalsplot.plt(4, self.x*1e3, self.pg, 'Pressure', 'Axial Distance (mm)', 'Pressure (bar)', 'b', True)
        self.thermalsplot.plt(5, self.x*1e3, self.M, 'Mach', 'Axial Distance (mm)', 'Mach', 'b', True)
        self.thermalsplot.plt(6, self.x*1e3, self.gamma, 'Gamma', 'Axial Distance (mm)', 'Gamma', 'b', True)
        self.thermalsplot.plt(7, self.x*1e3, self.pr, 'Prandtl Number', 'Axial Distance (mm)', 'Prandtl Number', 'b', True)
        self.thermalsplot.plt(8, self.x*1e3, self.Twg, 'Wall Temp', 'Axial Distance (mm)', 'Wall Temp (K)', 'r', True, label='Twg')
        self.thermalsplot.addline(8, self.x*1e3, self.Twc, 'm', label='Twc')
        self.thermalsplot.addline(8, self.x*1e3, self.Tc, 'b', label='Tc')
        self.thermalsplot.plt(9, self.x*1e3, self.Tc, 'Coolant Temperature', 'Axial Distance (mm)', 'Coolant Temperature (K)', 'r', True)
        self.thermalsplot.plt(10, self.x*1e3, self.rho_coolant, 'Coolant Density', 'Axial Distance (mm)', 'Coolant Density (kg/m^3)', 'b', True)
        self.thermalsplot.plt(11, self.x*1e3, self.hc*1e-3,'Coolant Side Conv. Coeff.','Axial Distance (mm)','Coolant Side Conv. Coeff. (kW/m^2/K)','r', True)
        self.thermalsplot.plt(12, self.x*1e3, self.v_coolant, 'Coolant Velocity', 'Axial Distance (mm)', 'Coolant Velocity (m/s)', 'b', True)
        self.thermalsplot.plt(13, self.x*1e3, self.cp_coolant, 'Coolant Specific Heat', 'Axial Distance (mm)', 'Coolant Specific Heat (J/kg/K)', 'b', True)
        self.thermalsplot.plt(14, self.x*1e3, self.k_coolant, 'Coolant Thermal Conductivity', 'Axial Distance (mm)', 'Coolant Thermal Conductivity (W/m/K)', 'b', True)
        self.thermalsplot.plt(15, self.x*1e3, self.mu_coolant*1e3, 'Coolant Viscosity', 'Axial Distance (mm)', 'Coolant Viscosity (mPa.s)', 'b', True)

    def plot_film(self):
        self.filmplot = subplot(2,2,'Film Cooling Data', self)
        self.filmplot.plt(1, self.x*1e3, self.T_f, 'Film Temp','Axial Distance (mm)','Film Temperature (K)','r', True)
        self.filmplot.plt(2, self.x*1e3, self.q_f*1e-3, 'Film Heat Flux','Axial Distance (mm)','Film Heat Flux (kW/m^2)','r', True)
        self.filmplot.plt(3, self.x*1e3, self.film_mdot_l, 'Liquid Film mdot','Axial Distance (mm)','Liquid Film mdot (kg/s)','b', True,label='Liquid Film mdot (kg/s)')
        self.filmplot.addline(3, self.x*1e3, self.film_mdot_g, 'r', label='Gas Film mdot (kg/s)')
        self.filmplot.plt(4, self.x*1e3, self.filmstate, 'Film State','Axial Distance (mm)','Film State','b', True)#

    def combustion_sim_sens_study(self, param, param_range, fuel, ox, fuel_inj_p, ox_inj_p, fuel_rho, ox_rho, oxclass, ox_can_choke=False, ox_temp=15, fuel_can_choke=False, fuel_temp=15):
        pc = np.array([])
        thrust = np.array([])
        OF = np.array([])
        ox_mdot = np.array([])
        fuel_mdot = np.array([])
        ox_inj_p_arr = np.array([])
        fuel_inj_p_arr = np.array([])

        for val in param_range:
            if param == "Ox Inj Pressure":
                ox_inj_p = val
            elif param == "Fuel Inj Pressure":
                fuel_inj_p = val
            elif param == "Ox Temp":
                ox_temp = val
            elif param == "Ox CdA":
                self.ox_CdA = val
            elif param == "Fuel CdA":
                self.fuel_CdA = val
            elif param == "C* Eff":
                self.cstar_eff = val
            try:
                self.inj_p_combustion_sim(
                    fuel = fuel,
                    ox = ox,
                    fuel_inj_p = fuel_inj_p,
                    ox_inj_p = ox_inj_p,
                    fuel_rho = fuel_rho,
                    ox_rho = ox_rho,
                    ox_gas_class = oxclass,
                    ox_gas = ox_can_choke,
                    ox_temp = ox_temp,
                    fuel_gas = fuel_can_choke,
                )
                pc = np.append(pc, self.pc)
                thrust = np.append(thrust, self.thrust)
                OF = np.append(OF, self.OF)
                ox_mdot = np.append(ox_mdot, self.ox_mdot)
                fuel_mdot = np.append(fuel_mdot, self.fuel_mdot)
            except ValueError:
                pc = np.append(pc, 0)
                thrust = np.append(thrust, 0)
                OF = np.append(OF, 0)
                ox_mdot = np.append(ox_mdot, 0)
                fuel_mdot = np.append(fuel_mdot, 0)
            ox_inj_p_arr = np.append(ox_inj_p_arr, self.ox_inj_p)
            fuel_inj_p_arr = np.append(fuel_inj_p_arr, self.fuel_inj_p)

        total_mdot = ox_mdot + fuel_mdot
        sens_study = plt.figure(constrained_layout=True)
        sens_study.suptitle(f'{param} Sensitivity Study')
        ax1 = sens_study.add_subplot(2, 2, 1)
        ax1.plot(param_range, pc, 'k')
        ax1.plot(param_range, (fuel_inj_p_arr-pc), 'r')
        # ax1.plot(param_range, (ox_inj_p_arr-pc), 'b')
        ax1.set_title('Pressures')
        ax1.set_xlabel(param)
        ax1.set_ylabel('Chamber Pressure (bar)')
        ax1.legend(['Chamber', 'Fuel dP', 'Ox dP'])
        ax1.grid(alpha=1)
        ax2 = sens_study.add_subplot(2, 2, 2)
        ax2.plot(param_range, thrust, 'r')
        ax2.set_title('Thrust')
        ax2.set_xlabel(param)
        ax2.set_ylabel('Thrust (N)')
        ax2.grid(alpha=1)
        ax3 = sens_study.add_subplot(2, 2, 3)
        ax3.plot(param_range, OF, 'r')
        ax3.set_title('O/F Ratio')
        ax3.set_xlabel(param)
        ax3.set_ylabel('O/F Ratio')
        ax3.grid(alpha=1)
        ax4 = sens_study.add_subplot(2, 2, 4)
        ax4.plot(param_range, total_mdot*1e3, 'k')
        ax4.plot(param_range, ox_mdot*1e3, 'b')
        ax4.plot(param_range, fuel_mdot*1e3, 'r')
        ax4.set_title('Mass Flow Rate')
        ax4.set_xlabel(param)
        ax4.set_ylabel('Mass Flow Rate (g/s)')
        ax4.grid(alpha=1)
        ax4.legend(['Total', 'Ox', 'Fuel'])

class injector():
    """A class representing a bipropellant injector for a rocket engine.
    The injector class provides methods to size both fuel and oxidizer injector elements 
    and calculate propellant flow rates through these elements. It supports various injector 
    configurations including annular holes and circular holes.

    Attributes
    fuel_A : float
        Fuel injector total area in square meters.

    fuel_Cd : float
        Fuel injector discharge coefficient, default 0.75.

    ox_A : float
        Oxidizer injector total area in square meters.

    ox_Cd : float
        Oxidizer injector discharge coefficient, default 0.4.

    fuel_CdA : float
        Product of fuel discharge coefficient and area.

    ox_CdA : float
        Product of oxidizer discharge coefficient and area.

    Methods

    size_fuel_anulus(Cd, ID, OD, n=1)
        Sizes fuel injector for annular holes.

    size_ox_anulus(Cd, ID, OD, n=1)
        Sizes oxidizer injector for annular holes.

    size_fuel_holes(Cd, d, n=1)
        Sizes fuel injector for circular holes.

    size_ox_holes(Cd, d, n=1)
        Sizes oxidizer injector for circular holes.

    spi_fuel_mdot(dp, fuel_rho)
        Calculates fuel mass flow rate using single phase incompressible model.

    spi_ox_mdot(dp, ox_rho)
        Calculates oxidizer mass flow rate using single phase incompressible model.

    calc_start_mdot(fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, 
                   ox_temp=15, fuel_gas_class=None, fuel_temp=15)
        Calculates starting mass flow rates for injector venting to atmosphere.
        Supports both incompressible liquid and compressible gas calculations.
    Notes
    -----
    The class converts input dimensions in millimeters to meters internally.
    For gas propellants, the class can calculate choked flow conditions."""

    def __init__(self):
        self.fuel_A = 0
        self.fuel_Cd = 0.75
        self.ox_A = 0
        self.ox_Cd = 0.4

    def set_fuel_CdA(self, CdA):
        """
        Sets the fuel injector CdA.

        Parameters
        ----------
        CdA : float
            Product of fuel discharge coefficient and area.
        """
        self.fuel_CdA = CdA

    def set_ox_CdA(self, CdA):
        """
        Sets the oxidizer injector CdA.

        Parameters
        ----------
        CdA : float
            Product of oxidizer discharge coefficient and area.
        """
        self.ox_CdA = CdA

    def size_fuel_anulus(self, Cd, ID, OD, n = 1):
        """
        Sizes fuel injector for a number of identical annular holes.

        Parameters
        ----------
        Cd : float
            Discharge coefficient for the fuel annulus.
        ID : float
            Inner diameter of the annulus in millimeters.
        OD : float
            Outer diameter of the annulus in millimeters.
        n : int, optional
            Number of annular holes (default 1).
        """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25e-6 * np.pi * (OD**2 - ID**2) * n
        self.fuel_CdA = self.fuel_A * Cd
    
    def size_ox_anulus(self, Cd, ID, OD, n = 1):
        """
        Sizes oxidizer injector for a number of identical annular holes.

        Parameters
        ----------
        Cd : float
            Discharge coefficient for the oxidizer annulus.
        ID : float
            Inner diameter of the annulus in millimeters.
        OD : float
            Outer diameter of the annulus in millimeters.
        n : int, optional
            Number of annular holes (default 1).
        """
        self.ox_Cd = Cd
        self.ox_A = 0.25e-6 * np.pi * (OD**2 - ID**2) * n
        self.ox_CdA = self.ox_A * Cd

    def size_fuel_holes(self, Cd, d, n = 1):
        """
        Sizes the fuel injector for a number of identical holes.

        Parameters
        ----------
        Cd : float
            Discharge coefficient for the fuel holes.
        d : float
            Hole diameter in millimeters.
        n : int, optional
            Number of fuel holes (default 1).
        """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25e-6 * np.pi * (d**2) * n
        self.fuel_CdA = self.fuel_A * Cd
    
    def size_ox_holes(self, Cd, d, n = 1):
        """
        Sizes the oxidizer injector for a number of identical holes.

        Parameters
        ----------
        Cd : float
            Discharge coefficient for the oxidizer holes.
        d : float
            Hole diameter in millimeters.
        n : int, optional
            Number of oxidizer holes (default 1).
        """
        self.ox_Cd = Cd
        self.ox_A = 0.25e-6 * np.pi * (d**2) * n
        self.ox_CdA = self.ox_A * Cd

    def spi_fuel_mdot(self, dp, fuel_rho):
        """
        Calculates the fuel mass flow rate through the injector using the single phase incompressible model.

        Parameters
        ----------
        dp : float
            Pressure differential across the injector orifice (bar)
        fuel_rho : float
            Density of the fuel (kg/m^3)

        Returns
        -------
        float
            Fuel mass flow rate (kg/s)
        """
        return self.fuel_CdA * np.sqrt(2e5 * dp * fuel_rho)
    
    def spi_fuel_dp(self, mdot, fuel_rho):
        """
        Calculates the pressure differential across the fuel injector orifice using the single phase incompressible model.

        Parameters
        ----------
        mdot : float
            Fuel mass flow rate (kg/s)
        fuel_rho : float
            Density of the fuel (kg/m^3)

        Returns
        -------
        float
            Pressure differential across the injector orifice (bar)
        """
        return ((mdot / self.fuel_CdA)**2) / (2e5 * fuel_rho)

    def spi_ox_mdot(self, dp, ox_rho):
        """
        Calculates the oxidiser mass flow rate through the injector using the single phase incompressible model.

        Parameters
        ----------
        dp : float
            Pressure differential across the injector orifice (bar)
        ox_rho : float
            Density of the oxidiser (kg/m^3)

        Returns
        -------
        float
            Oxidiser mass flow rate (kg/s)
        """
        return self.ox_CdA * np.sqrt(2e5 * dp * ox_rho)

    def spi_ox_dp(self, mdot, ox_rho):
        """
        Calculates the pressure differential across the oxidizer injector orifice using the single phase incompressible model.

        Parameters
        ----------
        mdot : float
            Oxidizer mass flow rate (kg/s)
        ox_rho : float
            Density of the oxidizer (kg/m^3)

        Returns
        -------
        float
            Pressure differential across the injector orifice (bar)
        """
        return ((mdot / self.ox_CdA)**2) / (2e5 * ox_rho)

    def calc_start_mdot(self, fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, ox_temp=15, fuel_gas_class=None, fuel_temp=15):
        """
        Calculates the starting mdots for the injector (venting to atm).
        ----------
        fuel_inj_p : float
            Fuel injector pressure (bar)
        ox_inj_p : float
            Oxidizer injector pressure (bar)
        fuel_rho : float, optional
            Fuel density in kg/m³, defaults to 786
        ox_rho : float, optional
            Oxidizer density in kg/m³, defaults to 860 (used only if oxclass is None)
        oxclass : object, optional
            pyfluids object for oxidizer
            If provided, compressible flow calculations will be used for oxidizer
        ox_temp : float, optional
            Oxidizer temperature in °C, defaults to 15 (used only if oxclass is provided)
        Returns
        -------
        None
            Results are printed directly:
            - Total mass flow rate (g/s)
            - Oxidizer mass flow rate (g/s) and whether flow is choked
            - Fuel mass flow rate (g/s)
            - Oxidizer to fuel ratio (OF)
        """
        ox_chokedstate = 'Unchoked'
        fuel_chokedstate = 'Unchoked'
        if ox_gas_class != None:
            ox_gas_class.update(Input.temperature(ox_temp), Input.pressure(ox_inj_p*1e5))
            ox_R = 8.31447/ox_gas_class.molar_mass
            ox_gamma = (ox_gas_class.specific_heat)/(ox_gas_class.specific_heat-ox_R)
            ox_rho = ox_gas_class.density
            # choking_ratio = ((ox_gamma + 1)/2)**(ox_gamma/(ox_gamma-1))
            ox_k = (2/(ox_gamma+1))**((ox_gamma+1)/(ox_gamma-1))
            min_choked_p = 2 * 1.01325 / (2-ox_gamma*ox_k)
            if ox_inj_p >= min_choked_p:
                ox_mdot_start = self.ox_CdA * np.sqrt(ox_gamma*ox_rho*ox_inj_p*1e5*ox_k)
                ox_chokedstate = 'Choked'
            else:
                ox_mdot_start = self.ox_CdA * np.sqrt(2*(ox_inj_p-1.01325) * 1e5 * ox_rho)
        else:
            ox_mdot_start = self.ox_CdA * np.sqrt(2*(ox_inj_p-1.01325) * 1e5 * ox_rho)

        if fuel_gas_class != None:
            fuel_gas_class.update(Input.temperature(fuel_temp), Input.pressure(fuel_inj_p*1e5))
            fuel_R = 8.31447/fuel_gas_class.molar_mass
            fuel_gamma = (fuel_gas_class.specific_heat)/(fuel_gas_class.specific_heat-fuel_R)
            fuel_rho = fuel_gas_class.density
            fuel_k = (2/(fuel_gamma+1))**((fuel_gamma+1)/(fuel_gamma-1))
            min_choked_p = 2 * 1.01325 / (2-fuel_gamma*fuel_k)
            if fuel_inj_p >= min_choked_p:
                fuel_mdot_start = self.fuel_CdA * np.sqrt(fuel_gamma*fuel_rho*fuel_inj_p*1e5*fuel_k)
                fuel_chokedstate = 'Choked'
            else:
                fuel_mdot_start = self.fuel_CdA * np.sqrt(2*(fuel_inj_p-1.01325) * 1e5 * fuel_rho)
        else:
            fuel_mdot_start = self.fuel_CdA * np.sqrt(2*(fuel_inj_p-1.01325) * 1e5 * fuel_rho)

        print(f'Total Start mdot: {(ox_mdot_start+fuel_mdot_start)*1e3:.3f} g/s')
        if ox_gas_class != None:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.3f} g/s ({ox_chokedstate})')
        else:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.3f} g/s')
        if fuel_gas_class != None:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.3f} g/s ({fuel_chokedstate})')
        else:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.3f} g/s')
        print(f'Start OF: {ox_mdot_start/fuel_mdot_start:.3f}')

class subplot:
    def __init__(self, yn, xn, title, engine):
        self.fig = plt.figure(constrained_layout=True)
        self.fig.suptitle(title)
        self.xn = xn
        self.yn = yn
        self.ax = {}
        self.ax2 = {}
        self.x = engine.x
        self.r = engine.r
        self.rc = engine.rc

    def plt(self, loc, x, y, title, xlabel, ylabel, colour, draw_engine_contour=False, **label):
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
        self.ax[loc].set_xlim(0, self.x[-1]*1e3)
        self.ax[loc].xaxis.grid(AutoMinorLocator())
        self.ax[loc].yaxis.grid(AutoMinorLocator())
        if draw_engine_contour == True:
            self.ax2[loc] = self.ax[loc].twinx()
            self.ax2[loc].plot(self.x*1e3, self.r*1e3, color='gray')
            self.ax2[loc].set_ylim(0, self.rc*5e3)

    def addline(self, loc, x, y, colour, label = None):
        self.ax[loc].plot(x, y, colour, label=label)
        self.ax[loc].legend()
        
class cea_fuel_water_mix:
    def __init__(self, alcohol, water_perc):
        self.alcohol = alcohol
        self.water_perc = water_perc
        if self.alcohol == 'Methanol':
            self.fuel_str = 'C 1 H 4 O 1'
        elif self.alcohol == 'Ethanol':
            self.fuel_str = 'C 2 H 6 O 1'
        card_str = f"""
        fuel {self.alcohol}   {self.fuel_str} 
        h,cal=-57040.0      t(k)=298.15       wt%={100-self.water_perc:.2f}
        oxid water H 2 O 1  wt%={self.water_perc:.2f}
        h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
        """
        add_new_fuel(f'{100-self.water_perc:.3g}% {self.alcohol} {self.water_perc:.3g}% Water', card_str)
    def str(self):
        return f'{100-self.water_perc:.3g}% {self.alcohol} {self.water_perc:.3g}% Water'

if __name__ == '__main__':
    system('cls')

    plt.ion()

    # water_perc = 15
    # alcohol = 'Isopropanol'
    # watermix = cea_fuel_water_mix(alcohol, water_perc)

    hopper = engine('configs/hopperengine.cfg')
    coax_igniter = engine('configs/coax_igniter1.cfg')

    ambient_T = 15

    fuelCd = 0.4
    oxCd = 0.65

    coax_inj_tank_p = injector()
    coax_inj_tank_p.size_fuel_holes(Cd = fuelCd, d = 0.2)
    coax_inj_tank_p.size_ox_anulus(Cd = oxCd, ID = 0.5, OD = 0.9)
    
    evan_daniel_inj = injector()
    evan_daniel_inj.size_fuel_holes(Cd = fuelCd, d = in2mm(0.006))
    evan_daniel_inj.size_ox_holes(Cd = oxCd, d = in2mm(1/32))
    # evan_daniel_inj.size_ox_anulus(Cd = oxCd, ID = in2mm(1/16), OD = in2mm(0.08))

    coax_igniter.dt = 2.5e-3
    coax_igniter.update()

###############################################################

    fuel_reg_p = 27
    # ox_reg_p = psi2bar(330)

    nitrous = Fluid(FluidsList.NitrousOxide)

    n2o_vent_p = 30
    nitrous.update(Input.pressure(n2o_vent_p*1e5), Input.quality(0))
    n2o_sat_p = (nitrous.pressure-100)/1e5
    n2o_temp = nitrous.temperature

    ambient_T = n2o_temp
    # coax_igniter.dt = 2e-3
    # coax_igniter.update()

    # coax_igniter.inj_p_combustion_sim(
    #     injector = coax_inj_tank_p,
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     fuel_inj_p = fuel_reg_p,
    #     ox_inj_p = n2o_sat_p,
    #     fuel_rho = 786,
    #     ox_gas_class = Fluid(FluidsList.NitrousOxide),
    #     ox_temp = ambient_T,
    # )
    # coax_igniter.print_data()
    # coax_inj_tank_p.calc_start_mdot(
    #     fuel_inj_p = fuel_reg_p,
    #     ox_inj_p = n2o_sat_p,
    #     fuel_rho = 786,
    #     ox_gas_class = Fluid(FluidsList.NitrousOxide),
    #     ox_temp = ambient_T,
    # )

    ambient_T = 15

    propane_inj = injector()
    propane_inj.size_fuel_holes(Cd = 0.5, d = 0.7)
    propane_inj.size_ox_anulus(Cd = 0.65, ID = 1, OD = 1.5)

    propane = Fluid(FluidsList.nPropane)
    propane.update(Input.temperature(ambient_T), Input.quality(0))
    propane_sat_p = (propane.pressure-100)/1e5

    nitrous_reg_p = 9

    coax_igniter.inj_p_combustion_sim(
        injector = propane_inj,
        fuel = 'Propane',
        ox = 'N2O',
        fuel_inj_p = propane_sat_p,
        ox_inj_p = nitrous_reg_p,
        ox_gas_class = Fluid(FluidsList.NitrousOxide),
        ox_temp = ambient_T,
        fuel_gas_class = Fluid(FluidsList.nPropane),
        fuel_temp = ambient_T,
    )
    coax_igniter.print_data()
    propane_inj.calc_start_mdot(
        fuel_inj_p = propane_sat_p,
        ox_inj_p = nitrous_reg_p,
        ox_gas_class = Fluid(FluidsList.NitrousOxide),
        ox_temp = ambient_T,
        fuel_gas_class=Fluid(FluidsList.nPropane),
        fuel_temp=ambient_T,
    )

    coax_inj_ox_reg = injector()
    coax_inj_ox_reg.size_fuel_holes(Cd = fuelCd, d = 0.2)
    coax_inj_ox_reg.size_ox_anulus(Cd = oxCd, ID = 0.5, OD = 1.2)

    fuel_reg_p = 8
    ox_reg_p = 8

    # coax_igniter.inj_p_combustion_sim(
    #     injector = coax_inj_ox_reg,
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     fuel_inj_p = fuel_reg_p,
    #     ox_inj_p = ox_reg_p,
    #     fuel_rho = 786,
    #     ox_gas_class = Fluid(FluidsList.NitrousOxide),
    #     ox_temp = ambient_T,
    # )
    # coax_igniter.print_data()
    # coax_inj_ox_reg.calc_start_mdot(
    #     fuel_inj_p = fuel_reg_p,
    #     ox_inj_p = ox_reg_p,
    #     fuel_rho = 786,
    #     ox_gas_class = Fluid(FluidsList.NitrousOxide),
    #     ox_temp = ambient_T,
    # )

    # OFsweep(
    #     fuel = 'Propane',
    #     ox = 'N2O',
    #     OFstart = 0.5,
    #     OFend = 12,
    #     pc = 25,
    #     pe = 1,
    #     cr = 16,
    # )

    # OFsweep(
    #     fuel = 'Isopropanol',
    #     ox = 'N2O',
    #     OFstart = 0.5,
    #     OFend = 10,
    #     pc = 25,
    #     pe = 1,
    #     cr = 16,
    # )

    plt.show(block=True)