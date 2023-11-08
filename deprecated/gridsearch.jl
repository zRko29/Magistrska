using YAML

# Read the configuration from the YAML file
config = YAML.load(open("config.yaml", "r"))

N = config["rnn"]["simulation"]["N"]
timesteps = config["rnn"]["simulation"]["timesteps"]
dt = config["rnn"]["simulation"]["dt"]
alpha = config["rnn"]["simulation"]["alpha"]
beta = config["rnn"]["simulation"]["beta"]

seed = config["rnn"]["model"]["seed"]
train_size = config["rnn"]["model"]["train_size"]
val_size = config["rnn"]["model"]["val_size"]
test_size = config["rnn"]["model"]["test_size"]

loss = config["rnn"]["training"]["loss"]
save = config["rnn"]["data"]["save"]
load = config["rnn"]["data"]["load"]

hyperparameters = config["gridsearch"]["hyperparameters"]

# Set the random seed
Random.seed!(seed)

# Create an instance of the Simulation class
using LinearAlgebra
using Statistics
import Base.getindex
import Base.setindex!
import Base.size

mutable struct Simulation
    q_i::Array{Float64,1}
    p_i::Array{Float64,1}
    N::Int
    timesteps::Int
    dt::Float64
    alpha::Float64
    beta::Float64
end

function Simulation(q_i, p_i, N, timesteps, dt, alpha, beta)
    return Simulation(q_i, p_i, N, timesteps, dt, alpha, beta)
end

function integrate(sim::Simulation)
    qs = zeros(Float64, sim.N, sim.timesteps)
    ps = zeros(Float64, sim.N, sim.timesteps)
    
    qs[:, 1] = sim.q_i
    ps[:, 1] = sim.p_i
    
    for t in 2:sim.timesteps
        qs[:, t] = qs[:, t-1] + sim.dt * (ps[:, t-1] .+ sim.alpha * sin.(qs[:, t-1]))
        ps[:, t] = ps[:, t-1] - sim.dt * sim.beta * (sin.(qs[:, t-1]) .+ sim.alpha * cos.(qs[:, t-1]))
    end
    
    return qs, ps
end

folder_path = "D:\\School\\Magistrska\\data"

q_i = sin.(LinRange(0, 2 * pi, N))
p_i = zeros(N)

if isempty(readdir(folder_path))
    println("No saved data, running simulation.")
    
    qs, ps = integrate(Simulation(q_i, p_i, N, timesteps, dt, alpha, beta))
else
    println("Found some saved data, skipping simulation.")
    
    qs = np.load(joinpath(folder_path, "data_qs_1.npy"))
    ps = np.load(joinpath(folder_path, "data_ps_1.npy"))
end

qs = qs[:, 2:end-1]
ps = ps[:, 2:end-1]

qs_mean = mean(qs)
qs_std = std(qs)

function preprocess(X)
    X = (X .- qs_mean) ./ qs_std
    return X
end

function postprocess(X)
    X = X .* qs_std .+ qs_mean
    return X
end

using Random
Random.seed!(seed)

# Randomize the order of the gridsearch
all_parameter_comb = Random.shuffle(hyperparameters)

using Flux
using Flux: @epochs, batch
using Flux.Optimise: update!
using Flux: throttle

function build_model(learning_rate, hidden_units_per_layer, dropout, window_size)
    initializer = Flux.initializer(RandomNormal(0.0, 0.05, seed))
    
    model = Chain(
        GRU(window_size, hidden_units_per_layer; init = initializer),
        dropout > 0 ? Dropout(dropout) : identity,
        Dense(hidden_units_per_layer, N-2; init = initializer)
    )
    
    loss(x, y) = Flux.mse(model(x), y)
    opt = ADAM(learning_rate)
    
    return model, loss, opt
end

import Statistics.mean
import Statistics.std

best_mse = Inf
best_params = Dict()
best_model = Nothing
best_pred = Nothing

iteration = 1

for params in all_parameter_comb
    X, y = functions.make_sequences(preprocess(qs), params["window_size"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=false)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size/(1-val_size), shuffle=false)

    model, loss, opt = build_model(params["learning_rate"], params["hidden_units_per_layer"], params["dropout"], params["window_size"])

    best_loss = Inf

    @epochs params["epochs"] begin
        for (x, y) in batch(X_train, y_train, params["batch_size"])
            gs = gradient(Flux.params(model)) do
                l = loss(x, y)
                Flux.reset!(opt)
                back!(l)
                return l
            end
            update!(opt, gs)
        end
        
        val_loss = mean(loss(X_val, y_val))
        
        if val_loss < best_loss
            best_loss = val_loss
            
            if val_loss < best_mse
                best_mse = val_loss
                best_params = params
                best_model = deepcopy(model)
                
                test_pred = [X_test[1, :]]
                
                for k in 1:size(X_test, 1)
                    pred = model([test_pred[k][end-params["window_size"]+1:end]])
                    test_pred = vcat(test_pred, pred)
                end
                
                test_pred = test_pred[params["window_size"]+1:end]
                
                plt.plot(postprocess(y_test)[:, 2], color="tab:blue")
                plt.plot(postprocess(test_pred)[:, 2], color="tab:orange")
                plt.ylim(-1.2, 1.2)
                plt.show()
                
                np.save("D:\\School\\Magistrska\\best_model_RNN\\best_pred_1.npy", test_pred)
                
                open("D:\\School\\Magistrska\\best_model_RNN\\best_params_1.json", "w") do config_file
                    JSON.json(best_params, config_file)
                end
            end
        end
        
        println("\n Completed: ", iteration, "/", length(all_parameter_comb))
        println("-------------------------------------------------------------")
        
        iteration += 1
    end
end
