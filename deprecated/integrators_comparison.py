import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import simulation

# Simulation parameters
N = 25  # Number of particles
timesteps = 10000  # Number of simulation time steps
dt = 1e-2  # Timestep
alpha = 0.01  # cubic coupling
beta = 0.01  # quartic coupling
noise_amplitude = 0.01  # Adjust the noise amplitude as needed

frames_skip = 200 # Show every n-th frame in animation

#------------------------------------------

# Initialize positions and momenta
q_i = np.sin(np.linspace(0, np.pi*2, N))  # Sine function
p_i = np.zeros(N)

# Compute all q values
scipy_method= "Radau"

qs_euler, ps_euler = simulation.euler(q_i, p_i, dt, timesteps, N, alpha, beta)
qs_runge_kutta, ps_runge_kutta = simulation.runge_kutta(q_i, p_i, dt, timesteps, N, alpha, beta)
qs_scipy_solve, ps_scipy_solve = simulation.scipy_solve(q_i, p_i, dt, timesteps, scipy_method, N, alpha, beta)

#------------------------------------------
# Animation of chains

def update(frame):
    dots1.set_offsets(np.column_stack((np.arange(N), np.ravel(qs_euler[frame * frames_skip, :]))))
    line1.set_ydata(np.ravel(qs_euler[frame * frames_skip, :]))

    dots2.set_offsets(np.column_stack((np.arange(N), np.ravel(qs_runge_kutta[frame * frames_skip, :]))))
    line2.set_ydata(np.ravel(qs_runge_kutta[frame * frames_skip, :]))
    
    dots3.set_offsets(np.column_stack((np.arange(N), np.ravel(qs_scipy_solve[frame * frames_skip, :]))))
    line3.set_ydata(np.ravel(qs_scipy_solve[frame * frames_skip, :]))

    if frame*frames_skip % 1000 == 0:
        plt.gca().texts.clear()
        plt.text(1, 1, "frame= "+str(frame*frames_skip))

# Create animation
fig, ax = plt.subplots(1, 1, figsize=(9, 7))
plt.suptitle("Comparison of integrators")

dots1 = ax.scatter(np.arange(N), q_i, c='blue', zorder=2)
line1, = ax.plot(np.arange(N), q_i, color='blue', zorder=1)
dots2 = ax.scatter(np.arange(N), q_i, c='orange', zorder=2)
line2, = ax.plot(np.arange(N), q_i, color='orange', zorder=1)
dots3 = ax.scatter(np.arange(N), q_i, c='green', zorder=2)
line3, = ax.plot(np.arange(N), q_i, color='green', zorder=1)

animation = FuncAnimation(fig, update, frames=timesteps//frames_skip, interval=1)

legend_labels = ['Euler', 'Runge-Kutta', scipy_method]
legend_colors = ['blue', 'orange', 'green']
ax.legend(legend_labels, loc='upper right')

ax.set_xlim(0, N-1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('n')
ax.set_ylabel('q')
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()

plt.show()