import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MSDLoss(_Loss):
    """
    Computes the mean squared distance between the input and target points.

    Note: It is different from torch.nn.MSELoss in that it sums square differences along dim=-1 before taking the mean.
    It can be shown that MSDLoss(x) = 2 * MSELoss(x) for some tensor x.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input.shape = target.shape = (num paths, path length, dimensionality=2)

        # squared differences between coordinates in inputs and targets
        squared_diff = (input - target).pow(2)
        # squared_diff.shape = (num paths, path length, 2)

        # sum squared differences to get squared distances between
        # points in input and target
        squared_distances = squared_diff.sum(dim=-1)
        # squared_distances.shape = (num paths, path length)

        # mean squared distance over all points
        mean_squared_distance = squared_distances.mean()
        # mean_squared_distance.shape = (1,)

        return mean_squared_distance


class PathAccuracy(_Loss):
    """
    Computes the accuracy of prediction by considering the mean squared distances along each path separately.

    Note: Since some paths might be harder to predict, measuring mean squared distance for each path separately is sensible.
    """

    def __init__(self, threshold: float = 1e-5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input.shape = target.shape = (num paths, path length, dimensionality=2)

        # squared differences between coordinates in inputs and targets
        squared_diff = (input - target).pow(2)
        # squared_diff.shape = (num paths, path length, 2)

        # sum squared differences to get squared distances between
        # points in input and target
        squared_distances = squared_diff.sum(dim=-1)
        # squared_distances.shape = (num paths, path length)

        # mean squared distance for each path
        mean_squared_distance_per_path = squared_distances.mean(dim=1)
        # mean_squared_distance_per_path.shape = (num paths,)

        # label paths as correct if mean squared distance is below threshold
        labels = (mean_squared_distance_per_path < self.threshold).float()
        # labels.shape = (num paths,)

        # accuracy is the mean of the labels
        accuracy = labels.mean()
        # accuracy.shape = (1,)

        return accuracy


if __name__ == "__main__":

    mse = torch.nn.MSELoss(reduction="mean")
    msd = MSDLoss()
    accuracy = PathAccuracy(threshold=3e-1)

    data1 = torch.rand(5, 10, 2)
    data2 = torch.rand(5, 10, 2)

    mse_value = mse(data1, data2)
    msd_value = msd(data1, data2)
    accuracy_value = accuracy(data1, data2)

    print(f"MSE: {mse_value:.3f}, MSD: {msd_value:.3f}, Accuracy: {accuracy_value:.3f}")
