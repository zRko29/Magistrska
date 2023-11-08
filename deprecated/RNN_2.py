import numpy as np

# Simulation parameters
N = 20  # Number of particles
timesteps = 10000  # Number of simulation time steps
timesteps_n = 50  # Number of additional simulation time steps
dt = 1e-2  # Timestep
alpha = 0.1  # cubic coupling
beta = 0.1  # quartic coupling

# Import saved data
qs_imported = np.load("D:\School\Magistrska\data\qs_integration" + "_" + str(N) + "_" + str(timesteps) + "_" + str(dt) + "_" + str(alpha) + "_" + str(beta) + ".npy")

qs = qs_imported[:timesteps - timesteps_n]

from sklearn.preprocessing import StandardScaler
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
scaler = StandardScaler()
scaler = scaler.fit(qs)
qs_scaled = scaler.transform(qs)

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features (N particles). 

trainX = []
trainY = []

n_past = 14 # Number of past days we want to use to predict the future.
n_future = 1 # Number of days we want to look into the future based on the past days.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
for i in range(n_past, len(qs_scaled) - n_future + 1):
    trainX.append(qs_scaled[i - n_past : i])
    trainY.append(qs_scaled[i + n_future - 1 : i + n_future])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_past, N), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(N))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
epochs = 5
batch_size = 16

history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#-------------------------------------------------
#-------------------------------------------------
print("\n Retrieving more simulation data.")
# Do further integration and predictions
qs = qs_imported[timesteps - timesteps_n:]
scaler = scaler.fit(qs)

qs_reshaped = np.empty((1, n_past, N))

#Reformat input data into a shape: (n_samples x timesteps x n_features)
for i in range(n_past, len(qs_scaled) - n_future + 1):
    qs_reshaped = np.concatenate((qs_reshaped, qs_scaled[np.newaxis, i - n_past : i]))

print("\n Predicting dynamics one step into future.")
# Predict only one future_steps into future
predictions = np.concatenate((qs_reshaped[0], model.predict(qs_reshaped)), axis=0)

print("\n Predicting whole dynamics.")
# Predict the whole evolution from initial past_steps
predictions_self= np.copy(qs_reshaped[0])

for frame in range(1, predictions.shape[0] - n_past + 1):
    
    if frame % 10 == 0:
        print("frame= "+str(frame//(predictions.shape[0] - n_past + 1)*100)+"%")
    
    prediction = model.predict(predictions_self[np.newaxis, -n_past:])
    predictions_self = np.concatenate((predictions_self, prediction), axis=0)

#Perform inverse transformation to rescale back to original range
predictions = scaler.inverse_transform(predictions)
predictions_self = scaler.inverse_transform(predictions_self)

import matplotlib.animation as animation

print("\n Animating simulation and predictions.")
# Create the figure and axes for animation
fig, ax = plt.subplots()

# Initialize the lines for simulation and predictions
simulation_line, = ax.plot([], [], 'r-', label='Simulation')
predictions_line, = ax.plot([], [], color='red', linestyle="--", label='Prediction')
prediction_self_line, = ax.plot([], [], color='green', linestyle="--", label='Prediction_self')

# Convert the predictions and predictions_self lists to numpy arrays
predictions = np.array(predictions)
predictions_self = np.array(predictions_self)

# Set up the axis labels and legend
ax.set_xlabel('Particle')
ax.set_ylabel('Position')
ax.set_title('Animation')
ax.legend()
ax.grid()

# Set the axis limits
ax.set_xlim(0, N-1)
ax.set_ylim(-1.5, 1.5)

# Function to update the animation frames
def update(frame):   
    simulation_line.set_data(range(N), qs[frame])
    predictions_line.set_data(range(N), predictions[frame])
    prediction_self_line.set_data(range(N), predictions_self[frame])
    
    if frame % 100 == 0:
        plt.gca().texts.clear()
        plt.text(0.2, 1.53, "frame= "+str(frame))
    
    return simulation_line, predictions_line, prediction_self_line

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=timesteps_n-1, interval = 1, blit=True)

# Show the animation
plt.show()