import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import matplotlib.pyplot as plt

card_str = """
fuel nitroMethane C 1. H 3. N 1. O 2. wt%= 50.00
h,cal=-27030. t(k)=298.15  rho,g/cc =1.1371
fuel CH3OH(L)   C 1 H 4 O 1
h,cal=-57040.0      t(k)=298.15       wt%=50.00
"""
add_new_fuel('nitromethanol', card_str )
fuels = []

ox = 'N2O'
pc = 30
pe = 1.01325
nozzle_eff = 1

OF_lims = [0.1, 6]
n_points = 250

fuels.append('Methanol')
fuels.append('Ethanol')
fuels.append('Isopropanol')
# fuels.append('Propane')
# fuels.append('RP-1')
# fuels.append('CH4')
# fuels.append('C2H6')
# fuels.append('NITROMETHANE')
# fuels.append('nitromethanol')

n_fuels = len(fuels)

OF_arr = np.linspace(OF_lims[0], OF_lims[1], n_points)

fuel_isp = np.zeros((n_fuels, n_points))
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

    isp = np.zeros(n_points)  # Pre-allocate a numpy array instead of using a list
    cstar = np.zeros(n_points)
    tc = np.zeros(n_points)

    for j, OF in enumerate(OF_arr):
        eps = cea.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
        [isp_temp, _] = cea.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=1.01325, frozen=0, frozenAtThroat=0)
        [_, cstar_temp, tc_temp] = cea.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=0, frozenAtThroat=0)
        isp[j] = isp_temp if isp_temp > 0 else np.nan
        cstar[j] = cstar_temp if cstar_temp > 0 else np.nan
        tc[j] = tc_temp if tc_temp > 0 else np.nan
    fuel_isp[i, :] = isp
    fuel_cstars[i, :] = cstar
    fuel_tcs[i, :] = tc

fuel_isp = fuel_isp * nozzle_eff

colors = ['b', 'r', 'g', 'm', 'k', 'orange', 'purple', 'brown']

plt.figure()
for i in range(n_fuels):
    plt.plot(OF_arr, fuel_isp[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('Isp (s)')
plt.title('ISP vs OF Ratio')
plt.legend()
plt.grid()
plt.grid(which='minor', alpha=0.5)
plt.tight_layout()
plt.xlim(0, None)

plt.figure()
for i in range(n_fuels):
    plt.plot(OF_arr, fuel_cstars[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('C* (m/s)')
plt.title('C* vs OF Ratio')
plt.legend()
plt.tight_layout()
plt.grid()
plt.grid(which='minor', alpha=0.5)
plt.xlim(0, None)

plt.figure()
for i in range(n_fuels):
    plt.plot(OF_arr, fuel_tcs[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('Chamber temp (K)')
plt.title('Chamber Temperatures vs OF Ratio')
plt.legend()
plt.tight_layout()
plt.grid()
plt.grid(which='minor', alpha=0.5)
plt.xlim(0, None)

plt.show()