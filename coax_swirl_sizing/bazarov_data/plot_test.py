import numpy as np
import matplotlib.pyplot as plt

ubar_an_input = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\ubar_an.csv", delimiter=",", names=True)
phi_input = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\phi.csv", delimiter=",", names=True)
mu_input = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\mu.csv", delimiter=",", names=True)
alpha_input = np.genfromtxt("coax_swirl_sizing\\bazarov_data\\alpha.csv", delimiter=",", names=True)

A = np.linspace(0.7, 12, 100)

ubar_an = np.interp(A, ubar_an_input["x"], ubar_an_input["y"])
phi = np.interp(A, phi_input["x"], phi_input["y"])
mu = np.interp(A, mu_input["x"], mu_input["y"])
alpha = np.interp(A, alpha_input["x"], alpha_input["y"])

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot on the primary y-axis
ax1.plot(A, ubar_an, label="u_an", color='blue')
ax1.plot(A, phi, label="phi", color='green')
ax1.plot(A, mu, label="mu", color='red')
ax1.set_xlabel("A")
ax1.set_ylabel("Values for u_an, phi, mu")
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second y-axis for alpha
ax2 = ax1.twinx()
ax2.plot(A, alpha, label="alpha", color='purple')
ax2.set_ylabel("Values for alpha")
ax2.legend(loc='upper right')

ax1.set_xlim(0,14)
ax1.set_ylim(0, 1.2)

ax2.set_ylim(0, 80)

plt.title("Bazarov Parameters vs A")
plt.show()