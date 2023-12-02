# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Colab Notebooks")

from utils.mapping_helper import StandardMap
from utils.training_helper import ModelTrainer
from utils.Model import Model

# --------------------------------------------------------------
# Training phase

map = StandardMap(init_points=100, steps=300, seed=42)
map.generate_data()
map.plot_data()

thetas, ps = map.retrieve_data()

trainer = ModelTrainer()

trainer.prepare_data(thetas, ps, shuffle=True)

model = Model()

trainer.set_model(model)
trainer.set_criterion("mse")
trainer.set_optimizer("adam")
trainer.set_device()

trainer.train_model(verbose=True)
trainer.plot_losses()


# --------------------------------------------------------------
# Testing phase

map = StandardMap(init_points=20, steps=200, seed=42)
map.generate_data()

thetas, ps = map.retrieve_data()

trainer.do_autoregression(thetas, ps)

trainer.plot_2d()
