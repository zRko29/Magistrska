import numpy as np
import matplotlib.pyplot as plt
import functions

# functions parameters
N = 8  # Number of particles
timesteps = 1000 # Number of functions time steps
dt = 1e-1  # Timestep
alpha = 0.5  # cubic coupling
beta = 0.1  # quartic coupling

# for dt, timesteps in zip([1, 5*1e-1, 1e-1, 5*1e-2, 1e-2, 5*1e-3, 1e-3, 5*1e-4, 1e-4, 5*1e-5, 1e-5], 5*np.array([1e1, 1e2/2, 1e2, 1e3/2, 1e3, 1e4/2, 1e4, 1e5/2, 1e5, 1e6/2, 1e6]).astype(int)):
qs, ps = functions.integration(N, timesteps, dt, alpha, beta)
energy = functions.energy(ps, qs, alpha, beta)
mean_energy = np.mean(energy)
max_energy = np.max(energy)
print(dt, timesteps, np.round((energy[0]-mean_energy)/mean_energy*100, 5),"%")

#------------------------------------------
# Energy(t) plot
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# ax.plot(range(timesteps),(energy - mean_energy)/mean_energy)
ax.plot(range(timesteps + 1), np.concatenate((np.array([0]), energy)))

# ax.set_xlim(0, N-1)
# ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('t')
ax.set_ylabel(r'E-$\overline{E}$')
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()

plt.show()