import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split
import simulation

tf.keras.backend.set_floatx('float64')

# Simulation parameters
N = 20  # Number of particles
timesteps = 10000  # Number of simulation time steps
timesteps_n = 1000  # Number of additional simulation time steps
dt = 1e-2  # Timestep
alpha = 0.1  # cubic coupling
beta = 0.1  # quartic coupling

# Import simulation
qss = np.load("D:\School\Magistrska\data\qs_integration" + "_" + str(N) + "_" + str(timesteps) + "_" + str(dt) + "_" + str(alpha) + "_" + str(beta) + ".npy")

qs = qss[:timesteps - timesteps_n]

# Split the data into training and testing sets
past_steps = 10
future_steps = 1
steps = timesteps - timesteps_n - past_steps - future_steps

data, target = simulation.split_sequences(qs, past_steps, future_steps)

target = target[:, 0]

# t = int(0.8 * steps)
# x_train, x_test = data[:t], data[t:]
# y_train, y_test = target[:t], target[t:]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(LSTM(N, input_shape=(past_steps, N)))
optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
epochs = 5
batch_size = 8
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print('Test Loss:', loss)

#-------------------------------------------------
#-------------------------------------------------

# Do further integration and predictions
qs = qss[timesteps - timesteps_n:]

steps = timesteps_n - past_steps - future_steps

qs_reshaped, _ = simulation.split_sequences(qs, past_steps, future_steps)

# Predict only one future_steps into future
predictions = np.concatenate((qs_reshaped[0], model.predict(qs_reshaped)), axis=0)

# Predict the whole evolution from initial past_steps
predictions_self= np.copy(qs_reshaped[0])

for frame in range(1, steps + 1):
    prediction = model.predict(predictions_self[np.newaxis,-past_steps:])
    
    predictions_self = np.concatenate((predictions_self, prediction), axis=0)

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
    
    # if frame*frames_skip % 100 == 0:
    plt.gca().texts.clear()
    plt.text(0.2, 1.53, "frame= "+str(frame))
    
    return simulation_line, predictions_line, prediction_self_line

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=timesteps_n-1, interval = 1, blit=True)

# Show the animation
plt.show()
