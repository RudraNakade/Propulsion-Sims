from numpy import interp, array

class material:
    def __init__(self, name, v):
        self.name = name
        self.v = v
        self.cte_eq = lambda T: None
        self.conductivity_eq = lambda T: None
        self.modulus_eq = lambda T: None
        self.yield_strength_eq = lambda T: None

    def set_cte_eq(self, a, T_offset, b):
        # a * (T - offset) + b
        self.cte_eq = lambda T: a * (T + T_offset) + b

    def set_constant_cte(self, value):
        self.cte_eq = lambda T: value

    def set_conductivity_eq(self, a, T_offset, b):
        # a * (T - offset) + b
        self.conductivity_eq = lambda T: a * (T + T_offset) + b

    def set_constant_conductivity(self, value):
        self.conductivity_eq = lambda T: value

    def set_yield_strength(self, values, temps):
        self.yield_strength_eq = lambda T: interp(T, temps, values, right=0)

    def set_modulus_eq(self, values, temps):
        self.modulus_eq = lambda T: interp(T, temps, values, right=0)

    def set_roughness(self, values, angles):
        self.roughness_eq = lambda angle: interp(angle, angles, values)

    def cte(self, T):
        return self.cte_eq(T)

    def k(self, T):
        return self.conductivity_eq(T)

    def E(self, T):
        return self.modulus_eq(T)

    def ys(self, T):
        return self.yield_strength_eq(T)
    
    def Ra(self, angle):
        return self.roughness_eq(angle)

AlSi10Mg = material("AlSi10Mg", 0.33)
AlSi10Mg.set_constant_cte(27e-6)
AlSi10Mg.set_constant_conductivity(130)
AlSi10Mg.set_yield_strength(array([204, 204, 182, 158, 132, 70, 30, 12])*1e6, array([200, 298, 423, 473, 523, 573, 623, 673]))
AlSi10Mg.set_modulus_eq(array([77.6, 77.6, 63.2, 60, 55, 45, 37, 28])*1e9, array([-73.15, 25, 150, 200, 250, 300, 350, 400]) + 273.15)
AlSi10Mg.set_roughness(array([8, 12, 24, 54, 68])*1e-6, array([0, 15, 30, 45, 60]))

Al6082_T6 = material("Aluminium 6082-T6", 0.3)
Al6082_T6.set_cte_eq(0.2e-7, -273.15, 22.5e-6)
Al6082_T6.set_conductivity_eq(0.07, -273.15, 190)
Al6082_T6.set_yield_strength(array([1, 1, 0.90, 0.79, 0.65, 0.38, 0.20, 0.11, 0])*260e6, array([-200, 20, 100, 150, 200, 250, 300, 350, 550]) + 273.15)
Al6082_T6.set_modulus_eq(array([70, 70, 69.3, 67.9, 65.1, 60.2, 54.6, 47.6, 37.8, 28.0, 0])*1e9, array([-200, 20, 50, 100, 150, 200, 250, 300, 350, 400, 550]) + 273.15)
Al6082_T6.set_roughness(array([3.2, 3.2])*1e-6, array([0.0, 90.0]))

Inconel718 = material("Inconel 718", 0.28)
Inconel718.set_constant_cte(16e-6)
Inconel718.set_constant_conductivity(12)
Inconel718.set_yield_strength(array([1172, 1172, 1124, 1096, 1076, 1069, 1027, 758])*1e6, array([0, 93, 204, 316, 427, 538, 649, 760]) + 273.15)
Inconel718.set_modulus_eq(array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78])*1e9, array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15)

ABD_900 = material("ABD-900", 0.28)
ABD_900.set_constant_cte(16.3e-6)
ABD_900.set_constant_conductivity(24)
ABD_900.set_yield_strength(array([1090, 1028, 976, 937, 897, 883, 836, 711])*1e6, array([29, 225, 440, 599, 755, 843, 873, 917]) + 273.15)
ABD_900.set_modulus_eq(array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78])*1e9, array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15)

GRCop42 = material("GRCop-42", 0.33)
GRCop42.set_constant_cte(20e-6)
GRCop42.set_constant_conductivity(250)
GRCop42.set_yield_strength(array([175, 170, 160, 150, 135, 120, 95, 70])*1e6, array([300, 400, 500, 600, 700, 800, 900, 1000]))
GRCop42.set_modulus_eq(array([78.9, 78.9])*1e9, array([25, 750]) + 273.15)
GRCop42.set_roughness(array([8, 12, 24, 54, 68])*1e-6, array([0, 15, 30, 45, 60]))