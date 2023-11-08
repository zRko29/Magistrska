import simulation
import numpy as np
import simulation

# Simulation parameters
N = 5  # Number of particles
timesteps = 10000  # Number of simulation time steps
dt = 1e-2  # Timestep
alpha = 0.01  # cubic coupling
beta = 0.01  # quartic coupling

for alpha in np.linspace(0, 0.6, ):
    for beta in np.linspace(0, 0.6, 5):
        
        qs, _ = simulation.integration(N, timesteps, dt, alpha, beta)
        
        ab = np.append(np.array([alpha, beta]), np.zeros(N - 2))
        
        qs = np.concatenate((ab[np.newaxis], qs), axis = 0)

        np.save("D:\School\Magistrska\data\\"+str(N)+"_"+str(timesteps)+"_"+str(dt)+"_"+str(alpha)+"_"+str(beta), qs)
