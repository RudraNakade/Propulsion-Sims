from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt
import enginesim as es
import numpy as np
import os

os.system('cls' if os.name == 'nt' else 'clear')

t_ambient = 13 # deg C

fuel_rho = 0.98*786 + 0.02*1030 # kg/m^3
fuel_mdot = 0.12 # kg/s
ox_mdot = 0.2 # kg/s

ullage = 0.8

nitrous  = Fluid(FluidsList.NitrousOxide)
nitrous.update(Input.temperature(t_ambient), Input.quality(0))
hopper = es.engine("configs/hopperengine.cfg")

fuel_anulus = 0.216
hopper.fuel_CdA = 0.75 * 1e-6 * np.pi * 0.25 * (6**2 - (6-2*fuel_anulus)**2)
hopper.ox_CdA = 0.4 * np.pi * (0.8 * 1e-3)**2 * 24 * 0.25
hopper.cstar_eff = 1

reg_p = 35

ox_inj = np.linspace(20, 40, 50)
fuel_inj = reg_p
pc = np.array([])
pe = np.array([])
of = np.array([])
isp = np.array([])
cf = np.array([])
thrust = np.array([])
fuel_times = np.array([])
ox_times = np.array([])

# Nitrous
ox_tank_d = 95 # mm
ox_tank_h = 700 # mm
ox_v = 5e-3 # np.pi * ox_tank_d**2 * ox_tank_h * 0.25 * 1e-9
ox_mass = nitrous.density * ox_v * 0.8
ox_time = ox_mass / ox_mdot

# Fuel
fuel_v = 9 # L
fuel_mass = fuel_rho * ullage * fuel_v * 1e-3
fuel_time = fuel_mass / fuel_mdot

for i in range(len(ox_inj)):
    hopper.pressure_combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        fuel_upstream_p = fuel_inj,
        ox_upstream_p = ox_inj[i],
        fuel_rho = 750,
        ox_rho = nitrous.density,
        ox_gas = False,
        ox_gas = FluidsList.NitrousOxide,
    )
    pc = np.append(pc,hopper.pc)
    pe = np.append(pe,hopper.pe)
    of = np.append(of,hopper.OF)
    isp = np.append(isp,hopper.ispsea)
    cf = np.append(cf,hopper.cf)
    thrust = np.append(thrust,hopper.thrust)
    fuel_times = np.append(fuel_times,(fuel_mass/(1.3*hopper.fuel_mdot)))
    ox_times = np.append(ox_times,(ox_mass/hopper.ox_mdot))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
plt1 = ax[0,0]
plt2 = ax[0,1]
plt3 = ax[1,0]
plt4 = ax[1,1]

plt1.plot(pc, thrust,'b', label='Thrust')
plt1.set_xlabel('Chamber Pressure (bar)')
plt1.set_ylabel('Thrust (N)')
plt1.grid()
ax2 = plt1.twinx()
ax2.plot(pc, ox_inj,'r', label='Ox Inj Pressure')
ax2.set_ylabel('Pressure (bar)')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=plt1.transAxes)

plt2.plot(pc, of,'b', label='O/F Ratio  ')
plt2.set_xlabel('Chamber Pressure (bar)')
plt2.set_ylabel('O/F')
plt2.grid(which='both', linestyle='--', linewidth=0.5)
plt2.legend()

plt3.plot(pc, (1e2*(fuel_inj-pc)/pc),'r',label='Fuel Stiffness')
plt3.plot(pc, (1e2*(ox_inj-pc)/pc),'b', label='Ox Stiffness')
plt3.set_xlabel('Chamber Pressure (bar)')
plt3.set_ylabel('Injector Stiffness (%)')
plt3.set_ylim([0, None])
plt3.grid(which='both', linestyle='--', linewidth=0.5)
plt3.legend()

plt4.plot(pc, fuel_times,'r',label='Fuel Tank')
plt4.plot(pc, ox_times,'b', label='Ox Tank')
plt4.set_xlabel('Chamber Pressure (bar)')
plt4.set_ylabel("Firing Time (s)")
plt4.legend()
plt4.grid(which='both', linestyle='--', linewidth=0.5)

print(f"OX Volume: {ox_v*1e3:.2f} L")
print(f"Ox Mass: {ox_mass:.2f} kg")

print(f"\nFuel Volume: {fuel_v:.2f} L")
print(f"Fuel Mass: {fuel_mass:.2f} kg")

print(f"\nFuel burn time: {fuel_time:.2f} s")
print(f"Ox burn time: {ox_time:.2f} s")

plt.show()