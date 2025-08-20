import enginesim as es
import numpy as np
import matplotlib.pyplot as plt
from os import system

system('cls')

engine = es.engine("configs/l9.yaml")

fuel = 'Isopropanol'
ox = 'N2O'

ox_mdot = 2.33
fuel_mdot_arr = np.arange(0.3, 3, 0.01)

pc = np.zeros_like(fuel_mdot_arr)
OF = np.zeros_like(fuel_mdot_arr)
cstar = np.zeros_like(fuel_mdot_arr)
isp = np.zeros_like(fuel_mdot_arr)
ox_isp = np.zeros_like(fuel_mdot_arr)
fuel_isp = np.zeros_like(fuel_mdot_arr)
thrust = np.zeros_like(fuel_mdot_arr)

for i, fuel_mdot in enumerate(fuel_mdot_arr):
    engine.mdot_combustion_sim(
        fuel = fuel,
        ox = ox,
        fuel_mdot = fuel_mdot,
        ox_mdot = ox_mdot,
        cstar_eff=0.96,
        cf_eff=0.905,
        full_sim=False
    )
    pc[i] = engine.pc
    OF[i] = engine.OF
    cstar[i] = engine.cstar
    isp[i] = engine.isp if engine.isp > 0 else None
    thrust[i] = engine.thrust if engine.thrust > 0 else None
    ox_isp[i] = thrust[i] / (ox_mdot * 9.81)
    fuel_isp[i] = thrust[i] / (fuel_mdot * 9.81)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 8))

fig.suptitle(f'Engine Fuel Mass Flow Dependency (Ox Mass Flow = {ox_mdot:.2f} kg/s - constant)')

ax1.plot(OF, pc)
ax1.set_xlabel('O/F Ratio')
ax1.set_ylabel('Chamber Pressure (Bar)')
ax1.set_title('Chamber Pressure vs O/F Ratio')
ax1.grid(True)

ax2.plot(OF, cstar)
ax2.set_xlabel('O/F Ratio')
ax2.set_ylabel('C* (m/s)')
ax2.set_title('C* vs O/F Ratio')
ax2.grid(True)

ax3.plot(OF, isp)
ax3.set_xlabel('O/F Ratio')
ax3.set_ylabel('Isp (s)')
ax3.set_title('Isp vs O/F Ratio')
ax3.grid(True)

ax4.plot(OF, ox_isp)
ax4.set_xlabel('O/F Ratio')
ax4.set_ylabel('Oxidizer ISP (s)')
ax4.set_title('Oxidizer-Based ISP vs O/F Ratio')
ax4.grid(True)

ax5.plot(OF, fuel_isp)
ax5.set_xlabel('O/F Ratio')
ax5.set_ylabel('Fuel ISP (s)')
ax5.set_title('Fuel-Based ISP vs O/F Ratio')
ax5.grid(True)

ax6.plot(OF, thrust)
ax6.set_xlabel('O/F Ratio')
ax6.set_ylabel('Thrust (N)')
ax6.set_title('Thrust vs O/F Ratio')
ax6.grid(True)

plt.tight_layout()
plt.show()