import numpy as np

psi_in_bar = 14.503773773

in_to_mm = lambda x: float(x) * 25.4
mm2in = lambda x: float(x) / 25.4

in_to_m = lambda x: float(x) * 25.4e-3
m_to_in = lambda x: float(x) / 25.4e-3

m_to_ft = lambda x: float(x) * 3.280839895
ft_to_m = lambda x: float(x) / 3.280839895

bar_to_psi = lambda x: float(x) * psi_in_bar
psi_to_bar = lambda x: float(x) / psi_in_bar

bar_to_pa = lambda x: float(x) * 100000
pa_to_bar = lambda x: float(x) / 100000

psi_to_pa = lambda x: float(x) * 1e5 / psi_in_bar
pa_to_psi = lambda x: float(x) * psi_in_bar / 1e5

degC_to_K = lambda x: float(x) + 273.15
K_to_degC = lambda x: float(x) - 273.15

f_to_degC = lambda x: (float(x) - 32) * 5/9
deg_to_f = lambda x: float(x) * 9/5 + 32

f_to_R = lambda x: float(x) + 459.67
R_to_f = lambda x: float(x) - 459.67

f_to_K = lambda x: degC_to_K(f_to_degC(float(x)))
K_to_f = lambda x: deg_to_f(K_to_degC(float(x)))

K_to_R = lambda x: float(x) * 9/5
R_to_K = lambda x: float(x) * 5/9

# flow coefficients
gal_to_L = lambda x: float(x) * 3.78541178

Cv_to_CdA = lambda Cv: float(Cv) * gal_to_L(1) / (60 * np.sqrt(2 * 1000 * psi_to_pa(1)))
Kv_to_CdA = lambda Kv: float(Kv) * 1000 / (3600 * np.sqrt(2 * 1000 * 1e5))

CdA_to_Cv = lambda CdA: float(CdA) / (gal_to_L(1) / (60 * np.sqrt(2 * 1000 * psi_to_pa(1))))
CdA_to_Kv = lambda CdA: float(CdA) * 3600 * np.sqrt(2 * 1000 * 1e5) / 1000

Cv_to_Kv = lambda Cv: CdA_to_Kv(Cv_to_CdA(Cv))
Kv_to_Cv = lambda Kv: CdA_to_Cv(Kv_to_CdA(Kv))