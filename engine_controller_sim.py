import propulsion_system_sim as ps
from thermo.chemical import Chemical
import unit_converter as uc
import numpy as np
import time

n2o = ps.thermo_fluid(Chemical("nitrous oxide"), temperature = 273, pressure = 40e5, name = "N2O", cea_name = "N2O") # Cold nitrous
ipa = ps.thermo_fluid(Chemical("isopropanol"), temperature = 290, pressure = 40e5, name = "IPA", cea_name = "Isopropanol")

fuel_tank_p = 60e5  # Pa
ox_tank_p = 60e5  # Pa

fuel_feed = ps.feed_system(fuel_tank_p, "Fuel Feed System")
ox_feed = ps.feed_system(ox_tank_p, "Ox Feed System")

pipe_id_1_2 = uc.in2m(0.5 - 2*0.036)
pipe_id_3_8 = uc.in2m(0.375 - 2*0.036)
abs_roughness = 0.015e-3  # m

valve_speed = 10 # 10 pos / s rate

fuel_pipes = ps.pipe(id = pipe_id_3_8, L=0.5, abs_roughness = abs_roughness, name = "Fuel Feed System Pipes")
fuel_valve = ps.needle_valve(open_CdA = uc.Cv2CdA(1.8), name = '1/2" Needle Valve', max_rate = valve_speed)
regen_channels = ps.orifice(CdA = 24.4e-6, name = "Regen Channels")
fuel_injector = ps.orifice(CdA = 17.4e-6, name = "Fuel Injector") # Measured
fuel_feed.add_component(fuel_pipes, fuel_valve, regen_channels, fuel_injector)

fuel_feed.set_fluid(ipa)
# fuel_feed.set_fluid(ethanol)

ox_pipes = ps.pipe(id = pipe_id_1_2, L=1.5, abs_roughness = abs_roughness, name = "Ox Feed System Pipes")
ox_valve = ps.needle_valve(open_CdA = uc.Cv2CdA(2.4), name = '3/4" Needle Valve', max_rate = valve_speed)
ox_injector = ps.orifice(CdA = 78e-6, name = "N2O Injector")
ox_feed.add_component(ox_pipes, ox_valve, ox_injector)

ox_feed.set_fluid(n2o)

engine = ps.engine("configs/l9.cfg")

coupled_system = ps.propulsion_system(fuel_feed, ox_feed, engine)

dt = 0.02

ox_kp = 0.03
ox_ki = 0.2
ox_kd = 0.0
ox_ki_lim = 1

fuel_kp = 0.4
fuel_ki = 0.8
fuel_kd = 0.0
fuel_ki_lim = 1

ox_valve_controller = ps.PID(Kp=ox_kp*1e-5, Ki=ox_ki*1e-5, Kd=ox_kd*1e-5, setpoint=0.0, ki_lim=ox_ki_lim)
fuel_valve_controller = ps.PID(Kp=fuel_kp, Ki=fuel_ki, Kd=fuel_kd, setpoint=0.0, ki_lim=fuel_ki_lim)

engine_controller = ps.engine_controller(
    fuel_valve=fuel_valve,
    ox_valve=ox_valve,
    fuel_injector=fuel_injector,
    propulsion_system=coupled_system,
    ox_PID=ox_valve_controller,
    fuel_PID=fuel_valve_controller,
    dt=dt
)

# t_end = 5
# pc_trace_vals = np.array([20, 20, 25, 25, 20, 20]) * 1e5
# pc_trace_times = np.array([0, 1, 2, 3, 4, 5])

startup_time = 1
valve_ramp_time = 0.8
pc_trace_vals = np.array([25, 25, 30, 30, 25, 25, 20, 20, 15, 15, 10, 10]) * 1e5
pc_trace_times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]) + startup_time

t_end = pc_trace_times[-1]

OF_trace_vals = np.ones_like(pc_trace_vals) * 3.0  # Constant OF of 3.0
OF_trace_times = pc_trace_times.copy()

times = np.arange(0, t_end+dt, dt)
ox_mdot = np.zeros_like(times)
fuel_mdot = np.zeros_like(times)
pc = np.zeros_like(times)
OF = np.zeros_like(times)
isp = np.zeros_like(times)
thrust = np.zeros_like(times)
cstar = np.zeros_like(times)

ox_control_p = np.zeros_like(times)
ox_control_i = np.zeros_like(times)
ox_control_d = np.zeros_like(times)

fuel_control_p = np.zeros_like(times)
fuel_control_i = np.zeros_like(times)
fuel_control_d = np.zeros_like(times)

ox_valve_position = np.zeros_like(times)
fuel_valve_position = np.zeros_like(times)

tank_pressure = np.zeros_like(times)
fuel_injector_pressure = np.zeros_like(times)
fuel_valve_pressure = np.zeros_like(times)
ox_injector_pressure = np.zeros_like(times)
ox_valve_pressure = np.zeros_like(times)

pc_setpoint = np.zeros_like(times)
OF_setpoint = np.zeros_like(times)

pc_err = np.zeros_like(times)
OF_err = np.zeros_like(times)

ox_pid_contribution = np.zeros_like(times)
fuel_pid_contribution = np.zeros_like(times)

ox_ff = np.zeros_like(times)
fuel_ff = np.zeros_like(times)

# Initialise system
ox_pos_init = 0.5
fuel_pos_init = 0.5

fuel_valve.set_position(0.5)  # Start closed
ox_valve.set_position(0.5)  # Start closed
t = time.time()
coupled_system.solve()
print(f"Coupled systems solved in {1e3*(time.time() - t):.2f} ms")
pc_init = engine.get_pc()
OF_init = ox_feed.get_mdot() / fuel_feed.get_mdot()

print(f"Initial conditions: pc = {pc_init/1e5:.2f} Bar, OF = {OF_init:.2f}")

for i, t in enumerate(times):
    it_t = time.time()

    # Engine controller

    if t < startup_time:
        pc_setpoint[i] = np.interp(t, [0, startup_time], [pc_init, pc_trace_vals[0]])
        OF_setpoint[i] = np.interp(t, [0, startup_time], [OF_init, OF_trace_vals[0]])
    else:
        pc_setpoint[i] = np.interp(t, pc_trace_times, pc_trace_vals)
        OF_setpoint[i] = np.interp(t, OF_trace_times, OF_trace_vals)

    engine_controller.set_setpoints(desired_pc=pc_setpoint[i], desired_OF=OF_setpoint[i])
    engine_controller.run()

    ox_valve_position[i] = ox_valve.get_position()
    fuel_valve_position[i] = fuel_valve.get_position()

    if t < valve_ramp_time:
        ox_valve.set_position((ox_valve_position[i] * t / valve_ramp_time) + (ox_pos_init * (1 - t / valve_ramp_time)))
        fuel_valve.set_position((fuel_valve_position[i] * t / valve_ramp_time) + (fuel_pos_init * (1 - t / valve_ramp_time)))
        ox_valve_position[i] = ox_valve.get_position()
        fuel_valve_position[i] = fuel_valve.get_position()

    ox_control_p[i], ox_control_i[i], ox_control_d[i] = ox_valve_controller.output_terms()
    fuel_control_p[i], fuel_control_i[i], fuel_control_d[i] = fuel_valve_controller.output_terms()

    coupled_system.solve()

    ox_pid_contribution[i], fuel_pid_contribution[i], ox_ff[i], fuel_ff[i] = engine_controller.get_pid_ff()

    tank_pressure[i] = fuel_feed._inlet_pressure
    ox_valve_pressure[i] = ox_valve.get_inlet_pressure()
    fuel_valve_pressure[i] = fuel_valve.get_inlet_pressure()
    fuel_injector_pressure[i] = fuel_injector.get_inlet_pressure()
    ox_injector_pressure[i] = ox_injector.get_inlet_pressure()

    fuel_mdot[i] = fuel_feed.get_mdot()
    ox_mdot[i] = ox_feed.get_mdot()
    pc[i] = engine.get_pc()
    OF[i] = ox_mdot[i] / fuel_mdot[i]
    isp[i] = engine.isp
    thrust[i] = engine.thrust
    cstar[i] = engine.cstar

    pc_err[i] = pc_setpoint[i] - pc[i]
    OF_err[i] = OF_setpoint[i] - OF[i]
    print(f"Iteration {i+1}/{len(times)} solved in {1e3*(time.time() - it_t):.2f} ms\n")

class PlotManager:
    """Class to manage subplots with automatic layout"""
    def __init__(self, fig_num, rows, cols, figsize=(12, 8)):
        self.fig_num = fig_num
        self.rows = rows
        self.cols = cols
        self.current_subplot = 1
        
        plt.figure(fig_num, figsize=figsize)
        plt.tight_layout()
    
    def add_subplot(self, times, data, xlabel, ylabel, title, label=None, ylim_bottom=None, ylim_scale=1.1):
        """Add a subplot to the current figure"""
        plt.figure(self.fig_num)
        plt.subplot(self.rows, self.cols, self.current_subplot)
        
        if label:
            for i, (d, l) in enumerate(zip(data, label)):
                plt.plot(times, d, label=l)
            plt.legend()
        else:
            plt.plot(times, data)

        if isinstance(data, list):
            max_val = np.max([np.max(d) for d in data])
            min_val = np.min([np.min(d) for d in data])
        else:
            max_val = np.max(data)
            min_val = np.min(data)

        range = np.abs(max_val - min_val)

        if ylim_bottom is None:
            ylim_bottom = min_val - range * (ylim_scale-1)

        ylim_top = max_val + range * (ylim_scale-1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.ylim(bottom=ylim_bottom, top=ylim_top)
        plt.xlim(times[0], times[-1])
        
        self.current_subplot += 1

import matplotlib.pyplot as plt

# First figure - System performance
system_plots = PlotManager(1, 2, 3, figsize=(15, 10))

system_plots.add_subplot(times, [fuel_mdot, ox_mdot], 'Time (s)', 'Mass Flow Rate (kg/s)', 
                        'Mass Flow Rates vs Time', label=['Fuel', 'Ox'], ylim_bottom=0)

system_plots.add_subplot(times, [pc / 1e5, pc_setpoint / 1e5], 'Time (s)', 'Chamber Pressure (bar)', 
                        'Chamber Pressure vs Time', label=['Actual', 'Setpoint'], ylim_bottom=0)

system_plots.add_subplot(times, [OF, OF_setpoint], 'Time (s)', 'OF Ratio', 'OF Ratio vs Time', 
                        label=['Actual', 'Setpoint'], ylim_bottom=0)

system_plots.add_subplot(times, [tank_pressure/1e5, fuel_valve_pressure/1e5, ox_valve_pressure/1e5, fuel_injector_pressure/1e5, ox_injector_pressure/1e5, pc/1e5], 
                        'Time (s)', 'Pressure (bar)', 'System Pressures vs Time', 
                        label=['Tank', 'Fuel Valve', 'Ox Valve', 'Fuel Injector', 'Ox Injector', 'Chamber'], ylim_bottom=0)

system_plots.add_subplot(times, [ox_valve_position, fuel_valve_position],
                        'Time (s)', 'Valve Position', 'Valve Positions vs Time',
                        label=['Ox Valve', 'Fuel Valve'], ylim_bottom=0)

# Second figure - Control outputs
control_plots = PlotManager(2, 2, 3, figsize=(15, 10))

control_plots.add_subplot(times, [ox_control_p, fuel_control_p], 'Time (s)', 'P Control Output', 
                            label=['Ox', 'Fuel'],
                            title='Control Proportional Outputs')

control_plots.add_subplot(times, [ox_control_i, fuel_control_i], 'Time (s)', 'Control Output', 
                            label=['Ox', 'Fuel'],
                            title='Control Integral Outputs')

control_plots.add_subplot(times, pc_err/1e5, 'Time (s)', 'Error (bar)',
                            'Chamber Pressure Error vs Time')

control_plots.add_subplot(times, OF_err, 'Time (s)', 'Error',
                            'OF Ratio Error vs Time')

control_plots.add_subplot(times, [ox_pid_contribution, fuel_pid_contribution], 'Time (s)', 'PID Contribution',
                            'PID Contributions vs Time', label=['Ox', 'Fuel'])

control_plots.add_subplot(times, [ox_ff, fuel_ff], 'Time (s)', 'Feedforward Contribution',
                            'Feedforward Contributions vs Time', label=['Ox', 'Fuel'])

plt.show()