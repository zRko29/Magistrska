import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

class Simulation():      
	def __init__(self, N, timesteps, dt, alpha, beta):
		self.N = N
		self.timesteps = timesteps
		self.dt = dt
		self.alpha = alpha
		self.beta = beta
		self.q_i = np.sin(np.linspace(0, np.pi*2, N))
		self.p_i = np.zeros(N)
    
	_forces = lambda self, q: np.concatenate(([0], np.diff(q[1:]) - np.diff(q[:-1]) + self.alpha * (np.diff(q[1:])**2 - np.diff(q[:-1])**2) + self.beta * (np.diff(q[1:])**3 - np.diff(q[:-1])**3), [0]))
  
	# def _forces(self, q):
	# 	d1 = np.diff(q[1:])
	# 	d2 = np.diff(q[:-1])

	# 	force = d1 - d2 + np.multiply(self.alpha, np.square(d1) - np.square(d2)) + np.multiply(self.beta, np.power(d1, 3) - np.power(d2, 3))
	# 	force = np.concatenate(([0], force, [0]))
	# 	return force
      
	def _func(self, t, y):
		q, p = np.array_split(y, 2)
		force = self._forces(q)
		return np.concatenate((p, force))

	def integrate(self):
		y0 = np.concatenate((self.q_i, self.p_i))

		times = np.arange(0, self.timesteps, self.dt)
		result = solve_ivp(self._func, [times[0], times[-1]], y0, method="Radau", t_eval=times, rtol=1e-8, atol=1e-8, max_step=0.01)

		qs = result.y[:self.N, :].T
		ps = result.y[self.N:, :].T
		
		return qs, ps
    
	def energy(ps, qs):
		d = np.diff(qs, axis=1)

		energy = np.sum(np.divide(np.square(ps), 2), axis=1) + np.sum(np.divide(np.square(d), 2) + np.multiply(self.alpha/3, np.power(d, 3)) + np.multiply(self.beta/4, np.power(d, 4)), axis = 1)

		return energy


class MachineLearning():
	def __init__(self, window_size, step_size, train_size, val_size, test_size):
		self.window_size = window_size
		self.step_size = step_size
		self.train_size = train_size
		self.val_size = val_size
		self.test_size = test_size

	def make_sequences(self, qs):
		X, y = [], []

		qs_temp = self.preprocess(qs[::self.step_size])

		for i in range(len(qs_temp)-self.window_size):
			X.append(qs_temp[i:i+self.window_size])
			y.append(qs_temp[i+self.window_size])

		return np.array(X), np.array(y)

	def make_split(self, X, y):

		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, shuffle=False)

		# mind the test_size
		X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=self.test_size/(1-self.val_size), shuffle=False)

		return X_train, X_test, X_val, y_train, y_test, y_val

	def preprocess(self, X):
		self.mean = np.mean(X)
		self.dev = np.std(X)

		X = (X - self.mean) / self.dev
		return X

	def postprocess(self, X):
		X = X * self.dev + self.mean
		return X
  
	@staticmethod
	def gradient_descent(model, epochs, train_loader, optimizer,  device, val_loader, train_dataset, val_dataset, criterion, verbose=1, patience=100):
		train_losses = [np.inf]
		validation_losses = [np.inf]

		best_loss = np.inf
		best_model = model
		patience_counter = 0
		best_epoch = 0
		hidden = None

		# Create a progress bar for the training loop
		progress_bar = tqdm(total=epochs, desc="Training Epochs")

		for epoch in range(epochs):
			model.train()
			train_loss = 0.0
			for inputs, labels in train_loader:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				outputs, _ = model(inputs, hidden)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * inputs.size(0)

			model.eval()
			val_loss = 0.0
			with torch.no_grad():
				for inputs, labels in val_loader:
					inputs = inputs.to(device)
					labels = labels.to(device)
					outputs, _ = model(inputs, hidden)
					loss = criterion(outputs, labels)
					val_loss += loss.item() * inputs.size(0)
			
			train_loss /= len(train_dataset)
			val_loss /= len(val_dataset)

			if val_loss < best_loss:
				best_loss = val_loss
				best_model = model
				best_epoch = epoch
				patience_counter = 0

			train_losses.append(train_loss)
			validation_losses.append(val_loss)

			patience_counter += 1

			if verbose == 1:
				if val_loss > validation_losses[-2]:
					val_arrow = "↑"
				else:
					val_arrow = "↓"

				if train_loss > train_losses[-2]:
					train_arrow = "↑"
				else:
					train_arrow = "↓"

				progress_bar.set_postfix(Train_loss=f"{train_loss:.3e} {train_arrow}", Val_loss=f"{val_loss:.3e} {val_arrow}")
				progress_bar.update(1)
			
			if patience_counter == patience and epoch < epochs:
				print("\nPatience exceeded. Stopping training.")
				progress_bar.close()
				print(f"Best epoch: {best_epoch}")
				return train_losses[1:], validation_losses[1:], best_model

		progress_bar.close()
		print(f"Best epoch: {best_epoch}")
		return train_losses[1:], validation_losses[1:], best_model

	@staticmethod
	def predictions(model, device, X_test):
		model.eval()

		window_size = X_test.shape[1]
		test_pred = np.copy(X_test[0])
		hidden = None

		with torch.no_grad():
			for k in range(len(X_test)):
				inputs = torch.Tensor(test_pred[np.newaxis, -window_size:]).to(device)
				pred, hidden = model(inputs, hidden)
				pred = pred.cpu().numpy()
				test_pred = np.concatenate((test_pred, pred), axis=0)

		return test_pred[window_size:] 

class Model(torch.nn.Module):
	def __init__(self, hidden_size, num_layers, dropout, model_type="RNN"):
		super(Model, self).__init__()
		self.model_type = model_type

		if self.model_type == "RNN":
			self.RNN = torch.nn.RNN(input_size=2, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
		elif self.model_type == "GRU":
			self.GRU = torch.nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)

		self.dense = torch.nn.Linear(in_features=hidden_size, out_features=2)

	def forward(self, input, hidden):
		if self.model_type == "RNN":
			out, hidden = self.RNN(input, hidden)
		elif self.model_type == "GRU":
			out, hidden = self.GRU(input, hidden)
		out = self.dense(out[:, -1, :])
		return out, hidden