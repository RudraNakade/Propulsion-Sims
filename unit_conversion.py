import numpy as np

psi_in_bar = 14.503773773

in2mm = lambda x: float(x) * 25.4
mm2in = lambda x: float(x) / 25.4

in2m = lambda x: float(x) * 25.4e-3
m2in = lambda x: float(x) / 25.4e-3

bar2psi = lambda x: float(x) * psi_in_bar
psi2bar = lambda x: float(x) / psi_in_bar

bar2pa = lambda x: float(x) * 100000
pa2bar = lambda x: float(x) / 100000

psi2pa = lambda x: float(x) * 1e5 / psi_in_bar
pa2psi = lambda x: float(x) * psi_in_bar / 1e5

degC2K = lambda x: float(x) + 273.15
K2degC = lambda x: float(x) - 273.15

f2degC = lambda x: (float(x) - 32) * 5/9
degC2f = lambda x: float(x) * 9/5 + 32

f2K = lambda x: degC2K(f2degC(float(x)))
K2f = lambda x: degC2f(K2degC(float(x)))

# flow coefficients
gal2L = lambda x: float(x) * 3.78541178

Cv2CdA = lambda Cv: float(Cv) * gal2L(1) / (60 * np.sqrt(2 * 1000 * psi2pa(1)))
Kv2CdA = lambda Kv: float(Kv) * 1000 / (3600 * np.sqrt(2 * 1000 * 1e5))