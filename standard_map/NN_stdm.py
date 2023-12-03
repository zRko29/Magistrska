# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Colab Notebooks")

from utils.mapping_helper import StandardMap
from utils.training_helper import ModelTrainer
from utils.Model import Model

import torch

# --------------------------------------------------------------
# Training phase

map = StandardMap(seed=42)
map.generate_data()
map.plot_data()

thetas, ps = map.retrieve_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = ModelTrainer(device=device)

trainer.prepare_data(thetas, ps, shuffle=True)

model = Model(device=device)

trainer.set_model(model)
trainer.set_criterion()
trainer.set_optimizer()

trainer.train_model(verbose=True)

trainer.plot_losses()

# --------------------------------------------------------------
# Testing phase

map = StandardMap(init_points=20, steps=200, seed=42)
map.generate_data()

thetas, ps = map.retrieve_data()

trainer.do_autoregression(thetas, ps, regression_seed=10)

trainer.plot_2d()
