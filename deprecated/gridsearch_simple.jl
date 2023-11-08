using Random
using LinearAlgebra
using DifferentialEquations
using Flux
using Flux: @epochs, batch, mse
using Flux.Optimise: update!
using Plots

Random.seed!(42)

function preprocess(X)
    X = (X .- mean(X)) ./ std(X)
    return X
end

function postprocess(X)
    X = X .* std(X) .+ mean(X)
    return X
end

function make_sequences(data, window_size)
    X = []
    y = []

    for i in 1:length(data) - window_size
        push!(X, data[i:i + window_size])
        push!(y, data[i + window_size])
    end

    return convert(Array{Float64, 3}, X), convert(Array{Float64, 1}, y)
end

function forces(q, p, alpha, beta)
    d1 = diff(q[2:end])
    d2 = diff(q[1:end-1])

    force = d1 - d2 + alpha * (d1.^2 - d2.^2) + beta * (d1.^3 - d2.^3)

    return vcat([0], force, [0])
end

function func(du, u, p, t)
    q, p = split(u, 2)
    force = forces(q, p, p[1], p[2])
    du .= vcat(p, force)
end

function integrate(q_i, p_i, timesteps, dt, N, alpha, beta)
    y0 = vcat(q_i, p_i)

    times = 0:dt:timesteps
    prob_func = ODEProblem((du, u, p, t) -> func(du, u, p, t), 0.0, timesteps, y0)
    result = solve(prob_func, Tsit5(), saveat=times)

    qs = result[1:N, :]
    ps = result[N+1:end, :]

    return qs', ps'
end

#---------------------------------------------------
# Parameters
const N = 8
const timesteps = 1000
const dt = 0.1
const alpha = 0.5
const beta = 0.1
const hidden_units_per_layer = 10
const window_size = 5
const dropout = 0
const train_size = 0.7
const val_size = 0.2
const test_size = 0.1
Random.seed!(42)
const epochs = 5
const batch_size = 20
const learning_rate = 0.01

const optimization_steps = 200
const hyperparameters = Dict("hidden_units_per_layer" => [6, 7, 8, 9], "window_size" => [5, 6, 7, 8], "epochs" => [50], "batch_size" => [15, 20, 25, 30], "learning_rate" => [0.01, 0.005], "dropout" => [0, 0.01, 0.1])

#---------------------------------------------------
# Import data

Random.seed!(42)
const folder_path = "D:\\School\\Magistrska\\data_julia"

q_i = sin.(range(0, stop=2π, length=N))
p_i = zeros(N)

# Numerically solve the differential equation
qs, ps = integrate(q_i, p_i, timesteps, dt, N, alpha, beta)

# Boundary particles are stationary, we don't need them
qs = qs[:, 2:end-1]
ps = ps[:, 2:end-1]

#---------------------------------------------------
# Make a hyperparameter grid for gridsearch, then shuffle it

using Random: sample

grid = Iterators.product(hyperparameters["hidden_units_per_layer"], hyperparameters["window_size"], hyperparameters["epochs"], hyperparameters["batch_size"], hyperparameters["learning_rate"], hyperparameters["dropout"])
all_parameter_comb = sample(list(grid), optimization_steps, replace=false)

#---------------------------------------------------
# Make model in Flux

function build_model(learning_rate, hidden_units_per_layer, dropout, window_size)
    model = Chain(
        RNN(GRU(hidden_units_per_layer)),
        Dense(hidden_units_per_layer, N-2),
    )

    optimizer = ADAM(learning_rate)
    loss(x, y) = mse(model(x), y)

    return model, optimizer, loss
end

#---------------------------------------------------
# Loop through hyperparameter combinations and keep updating the last best result

# Make initial "best-" variables to later fill in
best_mse = Inf
best_params = Dict()
best_model = Nothing
best_pred = []

iteration = 1

# Loop
for params in all_parameter_comb
    # Reshape input data for GRU
    X, y = make_sequences(preprocess.(qs), params[2])

    # Train/test/validation split
    n = size(X, 1)
    train_end = Int(floor(train_size * n))
    val_end = train_end + Int(floor(val_size * n))

    X_train, y_train = X[1:train_end, :, :], y[1:train_end]
    X_val, y_val = X[train_end+1:val_end, :, :], y[train_end+1:val_end]
    X_test, y_test = X[val_end+1:end, :, :], y[val_end+1:end]

    model, optimizer, loss = build_model(params[5], params[1], params[6], params[2])

    best_val_loss = Inf
    best_model_file = "gridsearch_temp.bson"

    @epochs params[3] Flux.train!((X_train, y_train), loss, optimizer, cb = function ()
        if val_loss < best_val_loss
            best_val_loss = val_loss
            Flux.@save best_model_file model
        end
    end)

    # Use model to make predictions, iterative process
    test_pred = copy(X_test[1, :, :])
    for k in 1:size(X_test, 1)
        pred = best_model(test_pred[end, :, :])
        test_pred = cat(test_pred, pred, dims=1)
    end

    test_pred = test_pred[params[2]+1:end, :, :]

    # Compare predicted data with y_test
    mse_value = mse(postprocess.(y_test), postprocess.(test_pred))

    # Update best variables if mse<best_mse
    if mse_value < best_mse
        best_mse = mse_value
        best_params = Dict(
            "hidden_units_per_layer" => params[1],
            "window_size" => params[2],
            "epochs" => params[3],
            "batch_size" => params[4],
            "learning_rate" => params[5],
            "dropout" => params[6],
        )
        best_model_file = "best_model_RNN/gridsearch_1.bson"
        best_pred = test_pred

        println("Best Parameters:", best_params)
        println("Best mse:", best_mse)

        # Plot 3rd particle
        plot(postprocess.(y_test)[:, 3], color=:blue)
        plot!(postprocess.(test_pred)[:, 3], color=:orange)
        ylims!(-1.2, 1.2)
        # savefig("plot_particle_3.pdf")
        display(plot)

        # Save on each step in case of crash
        Flux.@save best_model_file model

        save("best_model_RNN/best_pred_1.bson", "best_pred", best_pred)
        open("best_model_RNN/best_params_julia_1.json", "w") do config_file
            JSON.print(config_file, best_params)
        end
    end

    println("\n Completed: ", iteration, "/", optimization_steps)
    println("-------------------------------------------------------------")

    iteration += 1
end

#---------------------------------------------------
# Import optimal variables to build best_model

best_model_file = "best_model_RNN/gridsearch_1.bson"
best_model = Flux.load(best_model_file)
best_params = open("best_model_RNN/best_params_1.json") do config_file
    JSON.parse(config_file)
end
best_pred = load("best_model_RNN/best_pred_1.bson")["best_pred"]

#---------------------------------------------------
# Use best_model to get best_pred

X, y = make_sequences(preprocess.(qs), best_params["window_size"])

n = size(X, 1)
train_end = Int(floor(train_size * n))
val_end = train_end + Int(floor(val_size * n))

X_train, y_train = X[1:train_end, :, :], y[1:train_end]
X_val, y_val = X[train_end+1:val_end, :, :], y[train_end+1:val_end]
X_test, y_test = X[val_end+1:end, :, :], y[val_end+1:end]

test_pred = copy(X_test[1, :, :])
for k in 1:size(X_test, 1)
    pred = best_model(test_pred[end, :, :])
    test_pred = cat(test_pred, pred, dims=1)
end

test_pred = test_pred[best_params["window_size"]+1:end, :, :]

mse_value = mse(postprocess.(y_test), postprocess.(test_pred))

println("Optimal Parameters:", best_params)
println("Optimal mse:", mse_value)

#---------------------------------------------------
# Plot q(t) for each particle separately

for particles in 1:N-2
    plot(preprocess.(y_test)[:, particles], color=:blue)
    plot!(postprocess.(test_pred)[:, particles], color=:orange)
    # savefig("plot_particle_$(particles).pdf")
    display(plot)
end