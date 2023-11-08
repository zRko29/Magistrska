import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import simulation

# Generate FPU data
N = 40  # Number of particles
time_steps = 1000  # Number of simulation time steps
k = 1.0  # Spring constant
alpha = 1.0  # Nonlinearity parameter
dt = 1e-1  # Timestep

#--------------------------------------------------
# Initialize positions and momenta
q_i = np.random.uniform(-0.1, 0.1, N)
p_i = np.zeros(N)

qs, ps = simulation.euler(q_i, p_i, k, alpha, dt, time_steps, N)

for future_steps in [1, 5, 10, 20]:
    # Convert data to numpy arrays with future_steps
    data = np.array(qs[:-future_steps])  # Input to NN
    target = np.array(qs[future_steps:])  # Wanted output of NN

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42)

    # Define the PINN model
    model = tf.keras.Sequential()
    model.add(Dense(64, input_shape=(N,)))
    model.add(Dense(64, input_shape=(N,)))
    model.add(Dense(64, input_shape=(N,)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(N))

    # Define the PINN loss function
    def hamiltonian_loss(y_true, y_pred):
        q_true, p_true = tf.split(y_true, num_or_size_splits=2, axis=1)
        q_pred, p_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

        # Define the parameters
        k = 1.0  # Spring constant

        # Compute Hamiltonian for true and predicted values
        H_true = tf.reduce_sum(p_true**2) / 2 + tf.reduce_sum(k * (q_true[2:] - q_true[:-2])**2) / 2
        H_pred = tf.reduce_sum(p_pred**2) / 2 + tf.reduce_sum(k * (q_pred[2:] - q_pred[:-2])**2) / 2

        # Compute the mean squared error
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Combine Hamiltonian and MSE terms in the loss
        loss = mse + tf.abs(H_true - H_pred)

        return loss

    # Compile the model
    model.compile(loss=hamiltonian_loss, optimizer='adam')

    # Train the model
    epochs = 20
    batch_size = 32
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print('Test Loss:', loss)

    # Predict the future state of the FPU system
    predictions = model.predict(x_test)

    # Plot the predicted and ground truth trajectories
    plt.figure(figsize=(9, 6))
    for i in range(1):  # Plotting 10 random trajectories
        plt.plot(range(N), y_test[i], 'b-', label='Simulation')
        plt.plot(range(N), predictions[i][:N], 'r-', linewidth=1.5, label='Prediction')
        plt.xlabel('Particle Index')
        plt.ylabel('Displacement')
        plt.title(f'Prediction vs. Simulation (Future Steps: {future_steps})')
        plt.legend()
        plt.grid(True)
        plt.show()
