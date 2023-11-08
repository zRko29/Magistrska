seed = 42
import numpy as np
np.random.seed(seed)
from scipy.integrate import solve_ivp

def preprocess(X):
  X = (X - np.mean(X)) / np.std(X)
  return X

def postprocess(X):
  X = X * np.std(X) + np.mean(X)
  return X

def make_sequences(data, window_size):
  X = []
  y = []

  for i in range(len(data)-window_size):
    X.append(data[i : i + window_size])
    y.append(data[i + window_size])

  return np.array(X), np.array(y)
  
def forces(q, alpha, beta):
  d1 = np.diff(q[1:])
  d2 = np.diff(q[:-1])
  
  force = d1 - d2 + alpha * (d1**2 - d2**2) + beta * (d1**3 - d2**3)
  
  force = np.append(np.insert(force, 0, 0),0)

  return force
    
def func(t, y):
  q, p = np.array_split(y, 2)
  force = forces(q)

  return np.concatenate((p, force))

def integrate(q_i, p_i, timesteps, dt, N, alpha, beta):
  y0 = np.concatenate((q_i, p_i))
  
  times = np.arange(0, timesteps, dt)
  result = solve_ivp(func, [times[0], times[-1]], y0, method="Radau", t_eval=times, params=[alpha, beta], rtol=1e-8, atol=1e-8, max_step=0.01)

  qs = result.y[:N, :].T
  ps = result.y[N:, :].T
  
  return qs, ps

#---------------------------------------------------
# Parameters
N = 8
timesteps = 1000
dt = 0.1
alpha = 0.5
beta = 0.1
hidden_units_per_layer = 10
window_size = 5
dropout = 0
train_size = 0.7
val_size = 0.2
test_size = 0.1
seed = 42
epochs = 5
batch_size = 20
loss = "mse"
learning_rate = 0.01

optimization_steps = 200
hyperparameters = {"hidden_units_per_layer": [6, 7, 8, 9], "window_size": [5, 6, 7, 8], 'epochs': [50], 'batch_size': [15, 20, 25, 30], 'learning_rate': [0.01, 0.005], "dropout": [0, 0.01, 0.1]}

#---------------------------------------------------
# Import data

import random
random.seed(seed)
import functions
import os

folder_path = "D:\\School\\Magistrska\\data"

q_i = np.sin(np.linspace(0, np.pi*2, N))
p_i = np.zeros(N)

# Numerically solve the differential equation
qs, ps = integrate(q_i, p_i, timesteps, dt, alpha, beta)
    
# Boundary particles are stationary, we don't need them
qs = qs[:, 1:-1]
ps = ps[:, 1:-1]

#---------------------------------------------------
# Make a hyperparameter grid for gridsearch, then shuffle it

from sklearn.model_selection import ParameterGrid

grid = ParameterGrid(hyperparameters)
all_parameter_comb = random.sample(list(grid), optimization_steps)

#---------------------------------------------------
# Make model in Tensorflow with Keras api

import tensorflow as tf
tf.random.set_seed(seed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, GRU

def build_model(learning_rate, hidden_units_per_layer, dropout, window_size):
    # Initialize weights, for reproducibility
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)

    # Single GRU layer, dropout and final dense layer
    model = Sequential()
    model.add(InputLayer((window_size, N-2)))
    model.add(GRU(hidden_units_per_layer, activation="tanh", kernel_initializer=initializer)) 
    model.add(Dropout(dropout, seed=seed))
    model.add(Dense(N-2, kernel_initializer=initializer))

    # Gradient descent optimizer Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model

#---------------------------------------------------
# Loop through hyperparameter combinations and keep updating the last best result

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.callbacks import ModelCheckpoint

# Make initial "best-" variables to later fill in
best_mse = np.inf
best_params = {}
best_model = None
best_pred = []

iteration = 1

# Loop
for params in all_parameter_comb:
    
    # Reshape input data for GRU
    X, y = make_sequences(preprocess(qs), params["window_size"])

    # Train/test/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size/(1-val_size), shuffle=False)

    model = build_model(params["learning_rate"], params["hidden_units_per_layer"], params["dropout"], params["window_size"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params["epochs"], batch_size=params["batch_size"], verbose = 0)
    
    # Use model to make predictions, iterative process
    test_pred = np.copy(X_test[0])
    for k in range(len(X_test)):
        pred = model.predict(test_pred[np.newaxis, -params["window_size"]:], verbose=0)
        test_pred = np.concatenate((test_pred, pred), axis = 0)
        
    test_pred = test_pred[params["window_size"]:]

    # Compare predicted data with y_test
    mse = mean_squared_error(postprocess(y_test), postprocess(test_pred))

    # Update best variables if mse<best_mse
    if mse < best_mse:
        best_mse = mse
        best_params = params
        best_model = model
        best_pred = test_pred

        print("Best Parameters:", best_params)
        print("Best mse:", best_mse)
        
        # Plot 3rd particle
        plt.plot(postprocess(y_test)[:, 2], color="tab:blue")
        plt.plot(postprocess(test_pred)[:, 2], color="tab:orange")
        plt.ylim(-1.2, 1.2)
        plt.show()
        
        # Save on each step in case of crash        
        best_model.save(r'D:\\School\\Magistrska\\best_model_RNN\\gridsearch_1.h5')
        
        np.save(r'D:\\School\\Magistrska\\best_model_RNN\\best_pred_1.npy', best_pred)

        with open(r'D:\\School\\Magistrska\\best_model_RNN\\best_params_1.json', 'w') as config_file:
            json.dump(best_params, config_file)
        
    print("\n Completed: ", iteration, "/", len(all_parameter_comb))
    print("-------------------------------------------------------------")

    iteration += 1

#---------------------------------------------------
# Import optimal variables to build best_model

optimal_model = tf.keras.models.load_model(r"D:\\School\\Magistrska\\best_model_RNN\\gridsearch_1.h5")
# optimal_model.set_weights(best_model.get_weights())

with open(r"D:\\School\\Magistrska\\best_model_RNN\\best_params_1.json", 'r') as config_file:
    best_params = json.load(config_file)
    
best_pred = np.load(r'D:\\School\\Magistrska\\best_model_RNN\\best_pred_1.npy')

#---------------------------------------------------
# Use best_model to get best_pred

X, y = make_sequences(preprocess(qs), best_params["window_size"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size/(1-val_size), shuffle=False)

test_pred = np.copy(X_test[0])
for k in range(len(X_test)):
    pred = optimal_model.predict(test_pred[np.newaxis, -best_params["window_size"]:], verbose=0)
    test_pred = np.concatenate((test_pred, pred), axis=0)
    
test_pred = test_pred[best_params["window_size"]:]

mse = mean_squared_error(postprocess(y_test), postprocess(test_pred))

print("Optimal Parameters:", best_params)
print("Optimal mse:", mse)

#---------------------------------------------------
# Plot q(t) for each particle seperately

for particles in range(N-2):
    plt.plot(preprocess(y_test)[:, particles], color="tab:blue")
    plt.plot(postprocess(test_pred)[:, particles], color="tab:orange")
    # plt.plot(postprocess(best_pred)[:, particles], color="tab:red")
    # plt.ylim(-1.2, 1.2)
    # plt.savefig("plot_particle_#"+str(particles)+".pdf", format='pdf')
    plt.show()