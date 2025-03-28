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
        self.V = tank_volume
        self.pf = Fluid(fluid_class)
        self.CdA = system_CdA
        self.name = fluid_name

        if p_supercharge is None:
            self.set_saturated_state(vp=vp, T=T)
            self.tank_p = vp
            self.saturated = True
        else:
            self.set_supercharge_state(p_supercharge, vp, T)
            self.tank_p = p_supercharge
            self.saturated = False

        self.m = self.rho * initial_V_l

    def update(self, dm):
        if dm > 0:
            self.m -= dm
        if self.m < 0:
            self.m = 0
    
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
        self.cstar_eff = 1.0  # Default value, can be set later

        self.total_fuel_CdA = total_CdA(injector.fuel_total_CdA, fuel_system.CdA)
        self.total_ox_CdA = total_CdA(injector.ox_CdA, ox_system.CdA)
        self.film_frac = injector.film_frac

        # Pre-allocate arrays with estimated size to avoid frequent resizing
        # Estimate max iterations based on propellant mass and average flow rate
        fuel_mass = fuel_system.m
        ox_mass = ox_system.m
        est_max_iterations = 1000  # Default estimate
        
        # Initialize arrays with estimated size
        self.time = np.zeros(est_max_iterations)
        self.fuel_core_mdot = np.zeros(est_max_iterations)
        self.fuel_total_mdot = np.zeros(est_max_iterations)
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
        self.core_OF = np.zeros(est_max_iterations)
        self.total_OF = np.zeros(est_max_iterations)
        self.thrust = np.zeros(est_max_iterations)
        self.isp = np.zeros(est_max_iterations)
        
        # Initialize index counter
        self.idx = 0

    def engine_sim(self):
        self.engine.system_combustion_sim(
            fuel = self.fuel.name,
            ox = self.ox.name,
            fuel_CdA = self.total_fuel_CdA,
            ox_CdA = self.total_ox_CdA,
            film_frac = self.film_frac,
            fuel_upstream_p = pa2bar(self.fuel.tank_p),
            ox_upstream_p = pa2bar(self.ox.tank_p),
            fuel_rho = self.fuel.rho,
            ox_rho = self.ox.rho,
            cstar_eff = self.cstar_eff,)

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
                self.fuel.update(self.engine.fuel_mdot * dt * (1 + self.film_frac))
            else:
                first_loop = False
                self.time[self.idx] = 0

            # Store values directly in pre-allocated arrays
            self.fuel_mass[self.idx] = self.fuel.m
            self.fuel_core_mdot[self.idx] = self.engine.fuel_mdot
            self.fuel_total_mdot[self.idx] = self.fuel_core_mdot[self.idx] * (1 + self.film_frac)
            self.fuel_rho[self.idx] = self.fuel.rho
            self.fuel_tank_p[self.idx] = pa2bar(self.fuel.tank_p)
            
            fuel_inj_dp_val = self.injector.spi_fuel_dp(self.fuel_core_mdot[self.idx], self.fuel.rho)
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
            self.core_OF[self.idx] = self.engine.OF
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
                      f'Ox Mass: {self.ox_mass[self.idx]:.3f} kg, Core OF: {self.core_OF[self.idx]:.2f}, '
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
        self.total_OF = self.core_OF / (1 + self.film_frac)
        self.ox_stiffness = 100 * self.ox_inj_dp / self.pc
        self.fuel_stiffness = 100 * self.fuel_inj_dp / self.pc
    
    def _resize_arrays(self):
        """Resize all arrays when they fill up"""
        current_size = len(self.time)
        new_size = current_size * 2
        
        self.time = np.resize(self.time, new_size)
        self.fuel_core_mdot = np.resize(self.fuel_core_mdot, new_size)
        self.fuel_total_mdot = np.resize(self.fuel_total_mdot, new_size)
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
        self.core_OF = np.resize(self.core_OF, new_size)
        self.thrust = np.resize(self.thrust, new_size)
        self.isp = np.resize(self.isp, new_size)
    
    def _trim_arrays(self):
        """Trim arrays to actual size used"""
        self.time = self.time[:self.idx]
        self.fuel_core_mdot = self.fuel_core_mdot[:self.idx]
        self.fuel_total_mdot = self.fuel_total_mdot[:self.idx]
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
        self.core_OF = self.core_OF[:self.idx]
        self.thrust = self.thrust[:self.idx]
        self.isp = self.isp[:self.idx]

    def set_cstar_eff(self, cstar_eff):
        self.cstar_eff = cstar_eff

    def plot_sim(self):
        fig, axs = plt.subplots(2, 3, figsize=(14, 15))

        # Mass Flow Rates
        ax_mdot = axs[0, 0]
        ax_mdot.plot(self.time, self.fuel_total_mdot, label='Total Fuel', color='r', linestyle='-.')
        ax_mdot.plot(self.time, self.fuel_core_mdot, label='Core Fuel', color='r')
        ax_mdot.plot(self.time, self.ox_mdot, label='Ox', color='b')
        ax_mdot.set_xlabel('Time (s)')
        ax_mdot.set_ylabel('Mass flow rate (kg/s)')
        ax_mdot.set_title('Mass Flow Rates')
        ax_mdot.grid(True)
        ax_mdot.grid(which="minor", alpha=0.5)
        ax_mdot.minorticks_on()
        ax_mdot.legend()
        ax_mdot.set_ylim(bottom=0)

        # Propellant Masses
        ax_mass = axs[0, 1]
        ax_mass.plot(self.time, self.fuel_mass, label='Fuel Mass', color='r')
        ax_mass.plot(self.time, self.ox_mass, label='Ox Mass', color='b')
        ax_mass.set_xlabel('Time (s)')
        ax_mass.set_ylabel('Mass (kg)')
        ax_mass.set_title('Propellant Masses')
        ax_mass.grid(True)
        ax_mass.grid(which="minor", alpha=0.5)
        ax_mass.minorticks_on()
        ax_mass.legend()
        ax_mass.set_ylim(bottom=0)

        # OF Ratio
        ax_of = axs[1, 0]
        ax_of.plot(self.time, self.core_OF, color='b', label='Core OF')
        ax_of.plot(self.time, self.total_OF, color='r', label='Total OF')
        ax_of.set_xlabel('Time (s)')
        ax_of.set_ylabel('OF Ratio')
        ax_of.set_title('OF Ratio')
        ax_of.grid(True)
        ax_of.grid(which="minor", alpha=0.5)
        ax_of.minorticks_on()
        ax_of.set_ylim(bottom=0)
        ax_of.legend()
        ax_of.set_ylim(0,3)

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
        # Add secondary y-axis for stiffness
        ax_stiff = ax_dp.twinx()
        ax_stiff.plot(self.time, self.fuel_stiffness, label='Fuel Stiffness', color="darkred", linestyle='-.')
        ax_stiff.plot(self.time, self.ox_stiffness, label='Ox Stiffness', color="darkblue", linestyle='-.')
        ax_stiff.set_ylabel('Stiffness (%)')
        ax_stiff.set_ylim(bottom=0)

        # Combine both legends into one
        lines1, labels1 = ax_dp.get_legend_handles_labels()
        lines2, labels2 = ax_stiff.get_legend_handles_labels()
        ax_dp.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax_dp.grid(True)
        ax_dp.grid(which="minor", alpha=0.5)
        ax_dp.minorticks_on()
        ax_dp.set_ylim(bottom=0)
        ax_dp.set_xlabel("Time (s)")
        ax_dp.set_ylabel("Pressure Drop (bar)")
        ax_dp.set_title("Pressure Drop")

        plt.tight_layout()

ox = ox_system(fluid_class = FluidsList.NitrousOxide,
               fluid_name = 'N2O',
               tank_volume = l2m3(5.5),
               system_CdA = 4.8e-6,
               p_supercharge = bar2pa(31.5),
               vp = bar2pa(30),
               ullage = 0.2)

fuel = fuel_system(fluid_name = 'Isopropanol',
                   density = 790,
                   tank_volume = l2m3(9),
                   system_CdA = 3.75e-6, # 150 deg fuel main valve angle
                   tank_p = bar2pa(30.3),
                   ullage = 0.2)

hopper_inj = es.injector()
hopper_inj.size_fuel_anulus(Cd = 0.75, ID = 5.569, OD = 6)
hopper_inj.size_film_holes(Cd = 0.75, d = 0.2, n = 42)
# hopper_inj.size_ox_holes(Cd = 0.4, d = 0.8, n = 24)
# hopper_inj.set_fuel_CdA(3e-6)
hopper_inj.set_ox_CdA(1.8e-6)

hopper_engine = es.engine('configs/hopperengine.cfg')

system = prop_system(fuel, ox, hopper_inj, hopper_engine)
# system.set_cstar_eff(0.85)

system.run_sim(0.5)
system.plot_sim()

plt.show()