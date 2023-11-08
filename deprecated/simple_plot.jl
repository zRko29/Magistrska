using LinearAlgebra
using Plots
include("functions.jl")

# Simulation parameters
N = 8  # Number of particles
timesteps = 1000  # Number of simulation time steps
dt = 1e-1  # Timestep
alpha = 0.5  # cubic coupling
beta = 0.1  # quartic coupling

# Specify the folder path
folder_path = "D:\\School\\Magistrska\\data"

q_i = sin.(range(0, stop=π*2, length=N))
q_i[1] = 0
q_i[end] = 0
p_i = zeros(N)

sim = Simulation(q_i, p_i, N, timesteps, dt, alpha, beta)

saving = "no"

qs, ps = integrate(sim, q_i, p_i)

energy = energy(sim, ps, qs) / N

# println("init H:", energy[1], " init H/N:", energy[1] / N)
# plot(collect(0:dt:timesteps-dt), (energy - energy[1]) / energy[1], legend=false)
# # savefig("energy.pdf")
# display(plot)

# Plot first 300 steps
limit = 500

for particles in 2:N-1
    plot(q_i = range(0, stop=limit, length=limit), qs[1:limit, particles], color="blue")

    # Set labels and title
    xlabel!("X")
    ylabel!("Y")
    ylims!(-1.1*abs(minimum(qs[1:limit, particles])), 1.1*maximum(qs[1:limit, particles]))

    # Show the plot
    display(plot)
end
