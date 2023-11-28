# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Colab Notebooks")

from utils.mapping_helper import StandardMap
from utils.machine_learning_helper import Training, Validation
from utils.Model import Model


# --------------------------------------------------------------
# Training

map = StandardMap(seed=42)
map.do_mapping()

thetas_train, ps_train = map.get_data()

train = Training()
train.set_training_parameters(thetas_train, ps_train)

train.prepare_data(shuffle=True)

model = Model()
model.get_total_number_of_params()
train.set_model(model)

train.set_criterion(loss="huber")
train.set_optimizer(optimizer="radam")

train.set_device()

train.train_model(verbose=True, patience=50)

train.plot_losses()

# --------------------------------------------------------------
# Validation

map_test = StandardMap(init_points=10, steps=500, seed=42)
map_test.do_mapping()

thetas_test, ps_test = map_test.get_data()

validate = Validation()
validate.set_validation_parameters(thetas_test, ps_test)
validate.set_model(train.model)

validate.set_device()

validate.prepare_data()

validate.validate_model(verbose=True)

validate.plot_2d()

validate.plot_1d()
