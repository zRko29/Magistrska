import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import simulation
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

# Simulation parameters
N = 5  # Number of particles
timesteps = 10000  # Number of simulation time steps
dt = 1e-2  # Timestep
alpha = 0.1  # cubic coupling
beta = 0.1  # quartic coupling

qs, ps = simulation.integration(N, timesteps, dt, alpha, beta)

# Import simulation
# qs = np.load("D:\School\Magistrska\data\" + "_" + str(N) + "_" + str(timesteps) + "_" + str(dt) + "_" + str(alpha) + "_" + str(beta) + ".npy")

def df_to_X_y(data, window_size=5):
  X = []
  y = []
  
  for i in range(len(data)-window_size):
    X.append(data[i: i + window_size])
    y.append(data[i + window_size])
    
  return np.array(X), np.array(y)

window_size = 15
X, y = df_to_X_y(qs, window_size)
X.shape, y.shape

train_size = 0.7
val_size = 0.2
test_size = 0.1

tr=int(timesteps*train_size)
va=int(timesteps*val_size)
te=int(timesteps*test_size)

X_train, y_train = X[:tr], y[:tr]
X_val, y_val = X[-va:], y[-va:]
X_test, y_test = X[tr: tr +  te], y[tr: tr + te]

#--------------------------------------------------

# Define the RNN model
def build_model(learning_rate, batch_size, epochs, layers, size, window_size):
    model = Sequential()
    model.add(InputLayer((window_size, N)))
    
    if layers != 0:
        for iter in layers - 1:
            model.add(LSTM(size, input_shape=(window_size, N)), return_sequences=True)
        
    model.add(LSTM(size))
    model.add(Dense(N))
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    
    return model

hyperparameters = {
    'learning_rate': [0.01, 0.1],
    'batch_size': [32, 30],  #30
    'epochs': [18, 20],
    'layers': [0],
    "size": [30],
    "window_size": [15, 20]
}

#-------------------------------------------------
#-------------------------------------------------
# Perform GridSearch

model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_model)
grid_search = GridSearchCV(estimator = model, param_grid = hyperparameters, cv = 3, verbose=2, n_jobs=7)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(best_model)
print("best params: ", best_params)