import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from pyfluids import Fluid, FluidsList, Mixture, Input
import matplotlib.pyplot as plt

pc = 100

n_points = 100

OFrange = np.linspace(.1, 7, 100)

fuels = ['Methanol', 'Ethanol', 'Propane', 'Isopropanol', 'RP-1', 'CH4', 'C2H6']
n_fuels = len(fuels)
ox = 'LOX'
cr = 5
pe = 1

nozzle_eff = 1

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
        fac_CR=cr,
        make_debug_prints=False)
    isp = np.zeros(n_points)  # Pre-allocate a numpy array instead of using a list
    cstar = np.zeros(n_points)
    tc = np.zeros(n_points)

    for j, OF in enumerate(OFrange):
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

for i in range(n_fuels):
    plt.plot(OFrange, fuel_isp[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('Isp (s)')
plt.title('ISP vs OF Ratio')
plt.legend()
plt.grid()
plt.grid(which='minor', alpha=0.5)
plt.tight_layout()

plt.figure()
for i in range(n_fuels):
    plt.plot(OFrange, fuel_cstars[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('C* (m/s)')
plt.title('C* vs OF Ratio')
plt.legend()
plt.tight_layout()
plt.grid()
plt.grid(which='minor', alpha=0.5)

plt.figure()
for i in range(n_fuels):
    plt.plot(OFrange, fuel_tcs[i], label=fuels[i], color=colors[i % len(colors)])
plt.xlabel('O/F ratio')
plt.ylabel('Chamber temp (K)')
plt.title('Chamber Temperatures vs OF Ratio')
plt.legend()
plt.tight_layout()
plt.grid()
plt.grid(which='minor', alpha=0.5)


plt.show()