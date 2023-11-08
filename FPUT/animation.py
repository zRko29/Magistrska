import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 8  # Number of particles
timesteps = 1000  # Number of simulation time steps
dt = 1e-1  # Timestep
alpha = 0.5  # cubic coupling
beta = 0.1  # quartic coupling
frames_skip = 1

folder_path = r"D:\School\Magistrska\data"

q_i = np.sin(np.linspace(0, np.pi*2, N))
q_i[0] = 0
q_i[1] = 0
p_i = np.zeros(N)

qs = np.load(folder_path+r"\data_qs_1.npy")
ps = np.load(folder_path+r"\data_ps_1.npy")

# Update function for animation
def update(frame):

    dots1.set_offsets(np.column_stack((np.arange(N), np.ravel(qs[frame * frames_skip, :]))))
    line1.set_ydata(np.ravel(qs[frame * frames_skip, :]))
    for i in range(N):
        lines2[i].set_data(ps[:frame * frames_skip+1, i], qs[:frame * frames_skip+1, i])
    dots2.set_offsets(np.column_stack((np.ravel(ps[frame * frames_skip]), np.ravel(qs[frame * frames_skip]))))
    
    if frame*frames_skip % 1000 == 0:
        plt.gca().texts.clear()
        plt.text(-0.36, 3.9, "frame= "+str(frame*frames_skip))

# Create animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("Animation")
ax1.set_xlim(0, N-1)
ax1.set_ylim(-4.1, 4.1)
ax1.set_xlabel('n')
ax1.set_ylabel('q')
ax1.grid(True, alpha=0.25, linestyle="--")
dots1 = ax1.scatter(np.arange(N), q_i, c='b', zorder=2)
line1, = ax1.plot(np.arange(N), q_i, color='black', zorder=1)

ax2.set_xlim(-np.max(np.abs(ps))*1.2, np.max(np.abs(ps))*1.2)
ax2.set_ylim(-np.max(np.abs(qs))*1.2, np.max(np.abs(qs))*1.2)
ax2.set_xlabel('p')
ax2.set_ylabel('q')
ax2.grid(True, alpha=0.25, linestyle="--")
dots2 = ax2.scatter(p_i, q_i, c=np.arange(N), cmap='rainbow', edgecolors='none', zorder=2)
lines2 = [ax2.plot([], [], '-', c=c, alpha=0.5, linewidth=0.15, zorder=1)[0] for c in plt.cm.rainbow(np.linspace(0, 1, N))]

animation = FuncAnimation(fig, update, frames=timesteps // frames_skip, interval=1)

# animation.save('animation2.gif', writer='imagemagick', fps=80)

plt.show()
