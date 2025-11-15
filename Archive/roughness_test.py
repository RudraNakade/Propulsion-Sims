import numpy as np
import custom_materials
import matplotlib.pyplot as plt

angle_arr = np.linspace(0, 90, 100)
roughness = custom_materials.AlSi10Mg.Ra(angle_arr)

plt.figure()
plt.plot(angle_arr, roughness)
plt.xlabel("Angle (degrees)")
plt.ylabel("Roughness (m)")
plt.title("Roughness of AlSi10Mg vs Angle")
plt.grid()
plt.show()