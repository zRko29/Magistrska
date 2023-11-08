using LinearAlgebra
using DifferentialEquations
using OrdinaryDiffEq

function make_sequences(data, window_size)
    X = []
    y = []

    for i in 1:length(data)-window_size
        push!(X, data[i:i+window_size-1])
        push!(y, data[i+window_size])
    end

    return hcat(X...)', y
end

struct Simulation
    q_i::Vector{Float64}
    p_i::Vector{Float64}
    N::Int
    timesteps::Float64
    dt::Float64
    alpha::Float64
    beta::Float64
end

function forces(sim::Simulation, q)
    d1 = diff(q[2:end])
    d2 = diff(q[1:end-1])

    force = d1 - d2 + sim.alpha * (d1.^2 - d2.^2) + sim.beta * (d1.^3 - d2.^3)
    
    force = vcat(0, force, 0)

    return force
end

function func(sim::Simulation, t, y)
    q, p = split(y, 2)
    force = forces(sim, q)

    return vcat(p, force)
end

function integrate(sim::Simulation, q_i, p_i)
    y0 = vcat(q_i, p_i)
    times = 0:sim.dt:sim.timesteps
    result = solve(prob_func(sim, y0, times), Tsit5(), y0, times)
    
    qs = result.u[1:sim.N, :]
    ps = result.u[sim.N+1:end, :]
    
    return qs', ps'
end

function energy(sim::Simulation, ps, qs)
    d = diff(qs, dims=2)

    energy = sum(ps.^2 ./ 2, dims=2) .+ sum(d.^2 ./ 2 .+ sim.alpha .* d.^3 ./ 3 .+ sim.beta .* d.^4 ./ 4, dims=2)

    return energy
end

function prob_func(sim::Simulation, y0, times)
    tspan = (minimum(times), maximum(times))  # Construct tspan from the minimum and maximum values of times
    
    function prob_func!(dy, y, p, t)
        dy[1:sim.N] .= func(sim, t, y)
    end
    
    ODEProblem(prob_func!, y0, tspan)  # Pass tspan as the third argument
end