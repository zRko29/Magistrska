import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm


def standard_map(init_points, steps, K, seed=None, sampling="linear"):
	if seed != None:
		np.random.seed(seed=seed)	

	if sampling == "random":
		theta = np.random.uniform(0, 2 * np.pi, init_points)
		p = np.random.uniform(-1, 1, init_points)

	elif sampling == "linear":
		theta = np.linspace(0, 2 * np.pi, init_points)
		p = np.linspace(-1, 1, init_points)

	theta_values = np.zeros((steps, init_points))
	p_values = np.zeros((steps, init_points))

	for iter in range(steps):
		theta = (theta + p) % (2 * np.pi)
		p = p + K * np.sin(theta)

		theta_values[iter] = theta
		p_values[iter] = p

	return theta_values, p_values


class MachineLearning:
	def __init__(self, window_size, step_size, train_size, val_size):
		self.window_size = window_size
		self.step_size = step_size
		self.train_size = train_size
		self.val_size = val_size

	def make_sequences(self, thetas, ps, sequences="linear", predict=False):

		thetas_temp = thetas[::self.step_size]
		ps_temp = ps[::self.step_size]
		matrix_temp = np.copy(thetas_temp)

		for i in range(thetas_temp.shape[1]):
			matrix_temp = np.insert(matrix_temp, 2 * i + 1, ps_temp[:,i], axis=1)

		X, y = [], []

		if predict == False:
			if sequences == "linear":
				for k in range(thetas_temp.shape[1]):
					for i in range(thetas_temp.shape[0] - self.window_size):
						X.append(np.stack((thetas_temp[i:i+self.window_size, k], ps_temp[i:i+self.window_size, k]), axis=1))
						y.append(np.stack([thetas_temp[i+self.window_size, k], ps_temp[i+self.window_size, k]], axis=-1))
			
			elif sequences == "parallel":
				for k in range(thetas_temp.shape[0] - self.window_size):
					X.append(matrix_temp[k:k+self.window_size])
					y.append(matrix_temp[k+self.window_size])
					
		elif predict == True:
			if sequences == "linear":
				for k in range(thetas_temp.shape[1]):
					X.append(np.stack((thetas_temp[:self.window_size, k], ps_temp[:self.window_size, k]), axis=1))
					y.append(np.stack([thetas_temp[self.window_size:, k], ps_temp[self.window_size:, k]], axis=-1))

			elif sequences == "parallel":				
				X.append(matrix_temp[:self.window_size])
				y.append(matrix_temp[self.window_size:])

		return np.array(X), np.array(y)

	@staticmethod
	def preprocess(X):
		return X / np.pi - 1

	@staticmethod
	def postprocess(X):
		return (X + 1) * np.pi
  
	@staticmethod
	def gradient_descent(model, epochs, train_loader, optimizer, device, val_loader, train_dataset, val_dataset, criterion, verbose=1, patience=100):
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
				progress_bar.set_postfix(Train_loss=f"{train_loss:.3e}", Val_loss=f"{val_loss:.3e}")
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
	def predictions(model, device, seed, num_predictions, sequences="linear"):
		model.eval()

		window_size = seed.shape[1]
		final_preds = []

		# Create a progress bar for predictions
		if sequences == "linear":
			progress_bar = tqdm(total=seed.shape[0], desc="Predictions")
		
		for i in range(seed.shape[0]):
			hidden = None
			temp_pred = np.copy(seed[i])

			with torch.no_grad():

				if sequences == "parallel":
					progress_bar = tqdm(total=num_predictions, desc="Predictions")

				for k in range(num_predictions):			
					inputs = torch.Tensor(temp_pred[np.newaxis, -window_size:]).to(device)
					pred, hidden = model(inputs, hidden)
					pred = pred.cpu().numpy()
					temp_pred = np.concatenate((temp_pred, pred), axis=0)

					if sequences == "parallel":
						progress_bar.set_postfix()
						progress_bar.update(1)

			if sequences == "linear":
				progress_bar.set_postfix()
				progress_bar.update(1)

			final_preds.append(temp_pred[window_size:])
		
		progress_bar.close()

		return np.array(final_preds)


class Model(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout, model_type="RNN"):
		super(Model, self).__init__()
		self.model_type = model_type

		if self.model_type == "RNN":
			self.RNN = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
		elif self.model_type == "GRU":
			self.GRU = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)

		self.dense = torch.nn.Linear(in_features=hidden_size, out_features=input_size)

	def forward(self, input, hidden):
		if self.model_type == "RNN":
			out, hidden = self.RNN(input, hidden)
		elif self.model_type == "GRU":
			out, hidden = self.GRU(input, hidden)
		out = self.dense(out[:, -1, :])
		return out, hidden