import torch
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import gpytorch

from utils.mapping_helper import StandardMap
from utils.machine_learning_helper import Training, Validation
from utils.Model import Model

# --------------------------------------------------------------

map_train = StandardMap()
map_train.do_mapping()
thetas_train, ps_train = map_train.get_data()

map_test = StandardMap()
map_test.do_mapping()
thetas_test, ps_test = map_test.get_data()


def objective_function(kwargs):
    # Update your model architecture with the new hyperparameters
    updated_model = Model(**kwargs)

    # Perform training with the updated model
    train = Training(thetas_train, ps_train, updated_model)
    train.set_criterion(loss="huber")
    train.set_optimizer(optimizer="adamw")
    train.set_device()
    train.prepare_data(shuffle=True)
    train.train_model(verbose=True, patience=50)

    validate = Validation(thetas_test, ps_test, train.model)
    validate.set_device()
    validate.prepare_data()

    validate.validate_model(verbose=True)

    validate.compare_predictions()

    # Assuming you want to minimize the validation loss
    return validate.loss.item()


# --------------------------------------------------------------
# Bayesian Optimization

# Define the search space for hyperparameters
bounds = torch.tensor(
    [[0.1, 0.9], [1, 5], [32, 256]]
)  # dropout, num_layers, hidden_units

# Initial random points for experimentation
initial_random_points = 5
initial_points = (
    torch.rand(initial_random_points, 3) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
)

# Initialize GP model
gp = FixedNoiseGP(
    initial_points, torch.tensor([objective_function(p) for p in initial_points])
)

# Fit the GP model
mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Bayesian optimization loop
num_steps = 10
for step in range(num_steps):
    # Define the acquisition function
    acquisition_function = ExpectedImprovement(gp, best_f=gp.train_targets.min())

    # Optimize the acquisition function to get the next set of hyperparameters
    candidate, acq_value = optimize_acqf(
        acquisition_function, bounds=bounds, q=1, num_restarts=5, raw_samples=20
    )

    # Evaluate the objective function at the candidate
    new_point = candidate.detach().view(1, -1)
    new_value = objective_function(new_point)

    # Update the GP model with the new observation
    gp = FixedNoiseGP(
        torch.cat([gp.train_inputs[0], new_point]),
        torch.cat([gp.train_targets, new_value.view(-1)]),
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

# Get the best hyperparameters
best_hyperparameters = gp.train_inputs[0][gp.train_targets.argmin()]
best_dropout, best_num_layers, best_hidden_units = best_hyperparameters.numpy()

# Print or use the best hyperparameters in your model
print("Best Hyperparameters:")
print(
    f"Dropout: {best_dropout}, Num Layers: {best_num_layers}, Hidden Units: {best_hidden_units}"
)
