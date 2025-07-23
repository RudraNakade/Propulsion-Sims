import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import matplotlib.pyplot as plt
import mplcursors
from os import system
system('cls')

card_str = """
fuel nitroMethane C 1. H 3. N 1. O 2. wt%= 50.00
h,cal=-27030. t(k)=298.15  rho,g/cc =1.1371
fuel CH3OH(L)   C 1 H 4 O 1
h,cal=-57040.0      t(k)=298.15       wt%=50.00
"""
add_new_fuel('nitromethanol', card_str )
fuels = []

# ox = 'LOX'
ox = 'N2O'
pc = 30
pe = 1
nozzle_eff = 1

# OF_lims = [0.05, 4]
OF_lims = [0.05, 10]
n_points = 250

fuels.append('Methanol')
fuels.append('Ethanol')
fuels.append('Isopropanol')
fuels.append('Propane')
fuels.append('RP-1')
# fuels.append('Ethane')
# fuels.append('CH4')
# fuels.append('C2H6')
# fuels.append('NITROMETHANE')
# fuels.append('nitromethanol')

n_fuels = len(fuels)

OF_arr = np.linspace(OF_lims[0], OF_lims[1], n_points)

fuel_isp_eq = np.zeros((n_fuels, n_points))
fuel_isp_fr = np.zeros((n_fuels, n_points))
fuel_tcs = np.zeros((n_fuels, n_points))
fuel_cstars = np.zeros((n_fuels, n_points))
fuel_OFs = np.zeros((n_fuels, n_points))

for i, fuel in enumerate(fuels):
    cea = CEA_Obj(
        oxName = ox,
        fuelName = fuel,
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

    isp_eq = np.zeros(n_points)  # Pre-allocate a numpy array instead of using a list
    isp_fr = np.zeros(n_points)
    cstar = np.zeros(n_points)
    tc = np.zeros(n_points)

    for j, OF in enumerate(OF_arr):
        eps = cea.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
        [isp_temp_eq, _] = cea.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=1.01325, frozen=0, frozenAtThroat=0)
        [isp_temp_fr, _] = cea.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=1.01325, frozen=1, frozenAtThroat=0)
        [_, cstar_temp_eq, tc_temp_eq] = cea.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=0, frozenAtThroat=0)
        isp_eq[j] = isp_temp_eq if isp_temp_eq > 0 else np.nan
        isp_fr[j] = isp_temp_fr if isp_temp_fr > 0 else np.nan
        cstar[j] = cstar_temp_eq if cstar_temp_eq > 0 else np.nan
        tc[j] = tc_temp_eq if tc_temp_eq > 0 else np.nan
    fuel_isp_eq[i, :] = isp_eq
    fuel_isp_fr[i, :] = isp_fr
    fuel_cstars[i, :] = cstar
    fuel_tcs[i, :] = tc

fuel_isp_eq = fuel_isp_eq * nozzle_eff
fuel_isp_fr = fuel_isp_fr * nozzle_eff

colors = ['b', 'r', 'g', 'm', 'k', 'orange', 'purple', 'brown']

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
fig.suptitle(f'Oxidizer: {ox}, Chamber Pressure: {pc:.3f} bar, Exit Pressure: {pe:.3f} bar - Fuel Comparison')
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
# Equilibrium ISP vs OF Ratio
for i in range(n_fuels):
    ax1.plot(OF_arr, fuel_isp_eq[i], label=fuels[i], color=colors[i % len(colors)])
ax1.set_ylabel('Isp (s)')
ax1.set_title('Equilibrium ISP vs OF Ratio')
ax1.legend()
ax1.grid()
ax1.grid(which='minor', alpha=0.5)
ax1.minorticks_on()
ax1.set_xlim(0, None)

# Frozen ISP vs OF Ratio
for i in range(n_fuels):
    ax2.plot(OF_arr, fuel_isp_fr[i], label=fuels[i], color=colors[i % len(colors)])
ax2.set_ylabel('Isp (s)')
ax2.set_title('Frozen ISP vs OF Ratio')
ax2.legend()
ax2.grid()
ax2.grid(which='minor', alpha=0.5)
ax2.minorticks_on()
ax2.set_xlim(0, None)
ax2.set_ylim(ax1.get_ylim())

# C* vs OF Ratio
for i in range(n_fuels):
    ax3.plot(OF_arr, fuel_cstars[i], label=fuels[i], color=colors[i % len(colors)])
ax3.set_ylabel('C* (m/s)')
ax3.set_title('C* vs OF Ratio')
ax3.legend()
ax3.grid()
ax3.grid(which='minor', alpha=0.5)
ax3.minorticks_on()
ax3.set_xlim(0, None)

# Chamber Temperatures vs OF Ratio
ax4 = axs[1, 1]
for i in range(n_fuels):
    ax4.plot(OF_arr, fuel_tcs[i], label=fuels[i], color=colors[i % len(colors)])
ax4.set_xlabel('O/F ratio')
ax4.set_ylabel('Chamber temp (K)')
ax4.set_title('Chamber Temperatures vs OF Ratio')
ax4.legend()
ax4.grid()
ax4.grid(which='minor', alpha=0.5)
ax4.minorticks_on()
ax4.set_xlim(0, None)


for ax in axs[1, :]:
    ax.set_xlabel('O/F ratio')
    ax.set_xlim(OF_lims[0], OF_lims[1])

plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()