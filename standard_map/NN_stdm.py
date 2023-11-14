# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Colab Notebooks")

from stdm_helper import StandardMap
from machine_learning_helper import Training, Validation, Model

# --------------------------------------------------------------
# Training

map_train = StandardMap()
map_train.do_mapping()

thetas_train, ps_train = map_train.get_data()

model = Model()
model.get_total_number_of_params()

train = Training(thetas_train, ps_train, model)

train.set_criterion()
train.set_optimizer()
train.set_device()

train.prepare_data(shuffle=True)

train.train_model(verbose=True, patience=50)

train.plot_losses()

# --------------------------------------------------------------
# Validation

map_test = StandardMap(init_points=10, steps=500)
map_test.do_mapping()

thetas_test, ps_test = map_test.get_data()

validate = Validation(thetas_test, ps_test, train.model)

validate.set_device()

validate.prepare_data()

test_preds = validate.validate_model(verbose=True)

validate.plot_2d()

validate.plot_1d()
