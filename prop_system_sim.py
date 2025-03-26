import enginesim as es
import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Mixture, Input

# Unit Conversions
bar2psi = lambda x: x * 14.5038
psi2bar = lambda x: x / 14.5038
bar2pa = lambda x: x * 100000
pa2bar = lambda x: x / 100000
psi2pa = lambda x: x * 6894.76
pa2psi = lambda x: x / 6894.76
degC2K = lambda x: x + 273.15
K2degC = lambda x: x - 273.15
f2degC = lambda x: (x - 32) * 5/9
degC2f = lambda x: x * 9/5 + 32
f2K = lambda x: degC2K(f2degC(x))
K2f = lambda x: degC2f(K2degC(x))
l2m3 = lambda x: x / 1000
m32l = lambda x: x * 1000

def total_CdA(*vals):
    return (sum([CdA**-2 for CdA in vals]))**-0.5

class ox_system:
    def __init__(self, fluid_class, fluid_name, tank_volume, system_CdA, p_supercharge=None, vp=None, T=None, ullage = 0):
        initial_V_l = tank_volume * (1 - ullage)
        initial_V_g = tank_volume * ullage
        self.pf = Fluid(fluid_class)
        self.CdA = system_CdA
        self.name = fluid_name

        if p_supercharge is None:
            self.set_saturated_state(vp=vp, T=T)
            self.tank_p = vp
        else:
            self.set_supercharge_state(p_supercharge, vp, T)
            self.tank_p = p_supercharge

        self.m = self.rho * initial_V_l

    def update(self, dm):
        if dm > 0:
            self.m -= dm
        if self.m < 0:
            self.m = 0
        
    def get_mass(self):
        return self.m
    
    def get_CdA(self):
        return self.CdA

    def set_supercharge_state(self, p_supercharge, vp=None, T=None):
        if vp is None and T is None:
            raise ValueError("Exactly one of vapour pressure or temperature must be specified")
        else: 
            if vp is not None:
                self.pf.update(Input.pressure(vp), Input.quality(0))
                T = self.pf.temperature
            self.pf.update(Input.pressure(p_supercharge), Input.temperature(T))
        self.rho = self.pf.density

    def set_saturated_state(self, vp=None, T=None):
        if vp is None and T is None:
            raise ValueError("Either one of vapour pressure or temperature must be specified")
        else:
            if vp is not None:
                self.pf.update(Input.pressure(vp), Input.quality(0))
            else:
                self.pf.update(Input.temperature(T), Input.quality(0))
        self.rho = self.pf.density

class fuel_system:
    def __init__(self, fluid_name, density, tank_volume, system_CdA, tank_p, ullage = 0):
        initial_V_l = tank_volume * (1 - ullage)
        initial_V_g = tank_volume * ullage
        self.name = fluid_name
        self.CdA = system_CdA
        self.rho = density
        self.tank_p = tank_p
        self.m = self.rho * initial_V_l

    def update(self, dm):
        if dm > 0:
            self.m -= dm
        if self.m < 0:
            self.m = 0

    def get_mass(self):
        return self.m
    
    def get_CdA(self):
        return self.CdA

class prop_system:
    def __init__(self, fuel_system, ox_system, injector, engine):
        self.fuel = fuel_system
        self.ox = ox_system
        self.injector = injector
        self.engine = engine
        self.system_CdAs = es.injector()

        self.total_fuel_CdA = total_CdA(injector.fuel_CdA, fuel_system.CdA)
        self.total_ox_CdA = total_CdA(injector.ox_CdA, ox_system.CdA)
        self.system_CdAs.set_fuel_CdA(self.total_fuel_CdA)
        self.system_CdAs.set_ox_CdA(self.total_ox_CdA)

        # Pre-allocate arrays with estimated size to avoid frequent resizing
        # Estimate max iterations based on propellant mass and average flow rate
        fuel_mass = fuel_system.get_mass()
        ox_mass = ox_system.get_mass()
        est_max_iterations = 1000  # Default estimate
        
        # Initialize arrays with estimated size
        self.time = np.zeros(est_max_iterations)
        self.fuel_mdot = np.zeros(est_max_iterations)
        self.fuel_mass = np.zeros(est_max_iterations)
        self.fuel_rho = np.zeros(est_max_iterations)
        self.fuel_tank_p = np.zeros(est_max_iterations)
        self.fuel_inj_p = np.zeros(est_max_iterations)
        self.fuel_inj_dp = np.zeros(est_max_iterations)
        self.ox_mdot = np.zeros(est_max_iterations)
        self.ox_mass = np.zeros(est_max_iterations)
        self.ox_rho = np.zeros(est_max_iterations)
        self.ox_tank_p = np.zeros(est_max_iterations)
        self.ox_inj_p = np.zeros(est_max_iterations)
        self.ox_inj_dp = np.zeros(est_max_iterations)
        self.pc = np.zeros(est_max_iterations)
        self.pe = np.zeros(est_max_iterations)
        self.OF = np.zeros(est_max_iterations)
        self.thrust = np.zeros(est_max_iterations)
        self.isp = np.zeros(est_max_iterations)
        
        # Initialize index counter
        self.idx = 0

    def engine_sim(self):
        self.engine.inj_p_combustion_sim(
            injector = self.system_CdAs,
            fuel = self.fuel.name,
            ox = self.ox.name,
            fuel_inj_p = pa2bar(self.fuel.tank_p),
            ox_inj_p = pa2bar(self.ox.tank_p),
            fuel_rho = self.fuel.rho,
            ox_rho = self.ox.rho)

    def run_sim(self, dt, sim_time = None, verbose = False):
        first_loop = True
        time_flag = True
        
        # Reduce print frequency to improve performance
        print_interval = max(1, int(1.0 / dt))  # Print approximately once per second
        print_counter = 0
        
        while (self.fuel.m > 0 and self.ox.m > 0 and time_flag):
            self.engine_sim()

            if not first_loop:
                self.time[self.idx] = self.time[self.idx-1] + dt
                
                self.ox.update(self.engine.ox_mdot * dt)
                self.fuel.update(self.engine.fuel_mdot * dt)
            else:
                first_loop = False
                self.time[self.idx] = 0

            # Store values directly in pre-allocated arrays
            self.fuel_mass[self.idx] = self.fuel.m
            self.fuel_mdot[self.idx] = self.engine.fuel_mdot
            self.fuel_rho[self.idx] = self.fuel.rho
            self.fuel_tank_p[self.idx] = pa2bar(self.fuel.tank_p)
            
            fuel_inj_dp_val = self.injector.spi_fuel_dp(self.engine.fuel_mdot, self.fuel.rho)
            self.fuel_inj_dp[self.idx] = fuel_inj_dp_val
            self.fuel_inj_p[self.idx] = self.engine.pc + fuel_inj_dp_val

            self.ox_mass[self.idx] = self.ox.m
            self.ox_mdot[self.idx] = self.engine.ox_mdot
            self.ox_rho[self.idx] = self.ox.rho
            self.ox_tank_p[self.idx] = pa2bar(self.ox.tank_p)
            
            ox_inj_dp_val = self.injector.spi_ox_dp(self.engine.ox_mdot, self.ox.rho)
            self.ox_inj_dp[self.idx] = ox_inj_dp_val
            self.ox_inj_p[self.idx] = self.engine.pc + ox_inj_dp_val

            self.pc[self.idx] = self.engine.pc
            self.pe[self.idx] = self.engine.pe
            self.OF[self.idx] = self.engine.OF
            self.thrust[self.idx] = self.engine.thrust
            self.isp[self.idx] = self.engine.ispsea

            # Only print at specified intervals
            print_counter += 1
            if print_counter >= print_interval:
                print_counter = 0

            if sim_time is not None and self.time[self.idx] >= sim_time:
                time_flag = False

            if verbose and print_counter == 0:
                print(f'Time: {self.time[self.idx]:.2f} s, Fuel Mass: {self.fuel_mass[self.idx]:.3f} kg, '
                      f'Ox Mass: {self.ox_mass[self.idx]:.3f} kg, O/F: {self.OF[self.idx]:.2f}, '
                      f'pc: {self.pc[self.idx]:.2f} bar, pe: {self.pe[self.idx]:.2f} bar, '
                      f'Thrust: {self.thrust[self.idx]:.2f} N, Isp: {self.isp[self.idx]:.2f} s')
            
            self.idx += 1
            
            # Resize arrays if needed
            if self.idx >= len(self.time):
                self._resize_arrays()

        # Trim arrays to actual size used
        self._trim_arrays()
        
        # Calculate system pressure drops after simulation completes
        self.fuel_system_dp = self.fuel_tank_p - self.fuel_inj_p
        self.ox_system_dp = self.ox_tank_p - self.ox_inj_p
    
    def _resize_arrays(self):
        """Resize all arrays when they fill up"""
        current_size = len(self.time)
        new_size = current_size * 2
        
        self.time = np.resize(self.time, new_size)
        self.fuel_mdot = np.resize(self.fuel_mdot, new_size)
        self.fuel_mass = np.resize(self.fuel_mass, new_size)
        self.fuel_rho = np.resize(self.fuel_rho, new_size)
        self.fuel_tank_p = np.resize(self.fuel_tank_p, new_size)
        self.fuel_inj_p = np.resize(self.fuel_inj_p, new_size)
        self.fuel_inj_dp = np.resize(self.fuel_inj_dp, new_size)
        self.ox_mdot = np.resize(self.ox_mdot, new_size)
        self.ox_mass = np.resize(self.ox_mass, new_size)
        self.ox_rho = np.resize(self.ox_rho, new_size)
        self.ox_tank_p = np.resize(self.ox_tank_p, new_size)
        self.ox_inj_p = np.resize(self.ox_inj_p, new_size)
        self.ox_inj_dp = np.resize(self.ox_inj_dp, new_size)
        self.pc = np.resize(self.pc, new_size)
        self.pe = np.resize(self.pe, new_size)
        self.OF = np.resize(self.OF, new_size)
        self.thrust = np.resize(self.thrust, new_size)
        self.isp = np.resize(self.isp, new_size)
    
    def _trim_arrays(self):
        """Trim arrays to actual size used"""
        self.time = self.time[:self.idx]
        self.fuel_mdot = self.fuel_mdot[:self.idx]
        self.fuel_mass = self.fuel_mass[:self.idx]
        self.fuel_rho = self.fuel_rho[:self.idx]
        self.fuel_tank_p = self.fuel_tank_p[:self.idx]
        self.fuel_inj_p = self.fuel_inj_p[:self.idx]
        self.fuel_inj_dp = self.fuel_inj_dp[:self.idx]
        self.ox_mdot = self.ox_mdot[:self.idx]
        self.ox_mass = self.ox_mass[:self.idx]
        self.ox_rho = self.ox_rho[:self.idx]
        self.ox_tank_p = self.ox_tank_p[:self.idx]
        self.ox_inj_p = self.ox_inj_p[:self.idx]
        self.ox_inj_dp = self.ox_inj_dp[:self.idx]
        self.pc = self.pc[:self.idx]
        self.pe = self.pe[:self.idx]
        self.OF = self.OF[:self.idx]
        self.thrust = self.thrust[:self.idx]
        self.isp = self.isp[:self.idx]

    def plot_sim(self):
        fig, axs = plt.subplots(2, 3, figsize=(14, 15))

        # Mass Flow Rates
        axs[0, 0].plot(self.time, self.fuel_mdot, label='Fuel', color='r')
        axs[0, 0].plot(self.time, self.ox_mdot, label='Ox', color='b')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Mass flow rate (kg/s)')
        axs[0, 0].set_title('Mass Flow Rates')
        axs[0, 0].grid(True)
        axs[0, 0].grid(which="minor", alpha=0.5)
        axs[0, 0].minorticks_on()
        axs[0, 0].legend()

        # Propellant Masses
        axs[0, 1].plot(self.time, self.fuel_mass, label='Fuel Mass', color='r')
        axs[0, 1].plot(self.time, self.ox_mass, label='Ox Mass', color='b')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Mass (kg)')
        axs[0, 1].set_title('Propellant Masses')
        axs[0, 1].grid(True)
        axs[0, 1].grid(which="minor", alpha=0.5)
        axs[0, 1].minorticks_on()
        axs[0, 1].legend()

        # O/F Ratio
        axs[1, 0].plot(self.time, self.OF, color='b')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('OF Ratio')
        axs[1, 0].set_title('OF Ratio')
        axs[1, 0].grid(True)
        axs[1, 0].grid(which="minor", alpha=0.5)
        axs[1, 0].minorticks_on()

        # Combined Pressures Plot
        ax_pres = axs[1, 1]
        ax_pres.plot(self.time, self.fuel_tank_p, label='Fuel Tank', color="tab:red", linestyle='--')
        ax_pres.plot(self.time, self.ox_tank_p, label='Ox Tank', color="tab:blue", linestyle='--')
        ax_pres.plot(self.time, self.fuel_inj_p, label='Fuel Inj', color="tab:red")
        ax_pres.plot(self.time, self.ox_inj_p, label='Ox Inj', color="tab:blue")
        ax_pres.plot(self.time, self.pc, label='Chamber', color="tab:orange")
        ax_pres.set_xlabel('Time (s)')
        ax_pres.set_ylabel('Pressure (bar)')
        ax_pres.set_title('System Pressures')
        ax_pres.grid(True)
        ax_pres.grid(which="minor", alpha=0.5)
        ax_pres.minorticks_on()
        ax_pres.set_ylim(bottom=0)
        ax_pres.legend()

        # Thrust Plot
        ax_thrust = axs[1, 2]
        ax_thrust.plot(self.time, self.thrust, color="tab:green")
        ax_thrust.set_xlabel('Time (s)')
        ax_thrust.set_ylabel('Thrust (N)')
        ax_thrust.set_title('Engine Thrust')
        ax_thrust.grid(True)
        ax_thrust.grid(which="minor", alpha=0.5)
        ax_thrust.minorticks_on()
        ax_thrust.set_ylim(bottom=0)

        # Pressure Drops Plot
        ax_dp = axs[0, 2]
        ax_dp.plot(self.time, self.fuel_inj_dp, label='Fuel Inj', color="tab:red")
        ax_dp.plot(self.time, self.ox_inj_dp, label='Ox Inj', color="tab:blue")
        ax_dp.plot(self.time, self.fuel_system_dp, label='Fuel System', color="tab:red", linestyle='--')
        ax_dp.plot(self.time, self.ox_system_dp, label='Ox System', color="tab:blue", linestyle='--')
        ax_dp.grid(True)
        ax_dp.grid(which="minor", alpha=0.5)
        ax_dp.minorticks_on()
        ax_dp.set_ylim(bottom=0)
        ax_dp.legend(loc='best')
        ax_dp.set_xlabel("Time (s)")
        ax_dp.set_ylabel("Pressure Drop (bar)")
        ax_dp.set_title("Pressure Drop")

        plt.tight_layout()

ox = ox_system(fluid_class = FluidsList.NitrousOxide,
               fluid_name = 'N2O',
               tank_volume = l2m3(5),
               system_CdA = 6e-6,
               p_supercharge = bar2pa(32),
               T = 0,
               ullage = 0.2)

fuel = fuel_system(fluid_name = 'Isopropanol',
                   density = 790,
                   tank_volume = l2m3(9),
                   system_CdA = 4e-6, # 150 deg fuel main valve angle
                   tank_p = bar2pa(30),
                   ullage = 0.2)

hopper_inj = es.injector()
# hopper_inj.size_fuel_anulus(Cd = 0.75, ID = 5.565, OD = 6)
# hopper_inj.size_ox_holes(Cd = 0.23, d = 0.7, n = 24)
hopper_inj.set_fuel_CdA(3e-6)
hopper_inj.set_ox_CdA(1.75e-6)

hopper_engine = es.engine('configs/hopperengine.cfg')

system = prop_system(fuel, ox, hopper_inj, hopper_engine)

system.run_sim(0.1, 3)
system.plot_sim()

plt.show()