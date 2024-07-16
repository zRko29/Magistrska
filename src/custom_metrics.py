import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


# class MSDLoss(_Loss):
#     """
#     Computes the mean squared distance (MSD) between the input and target points.

#     Note: It is different from torch.nn.MSELoss in that it sums square differences along dim=-1 before taking the mean.
#     It can be shown that MSDLoss(x) = 2 * MSELoss(x) for some tensor x.
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.squared_difference = torch.nn.MSELoss(reduction="none")

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         # input.shape = target.shape = (num paths, path length, dimensionality=2)

#         # squared differences between coordinates in inputs and targets
#         # squared_diff = (input - target).pow(2)
#         squared_diff = self.squared_difference(input, target)
#         # squared_diff.shape = (num paths, path length, 2)

#         # sum squared differences to get squared distances between
#         # points in input and target
#         squared_distances = squared_diff.sum(dim=-1)
#         # squared_distances.shape = (num paths, path length)

#         # MSD over all points
#         MSD = squared_distances.mean()
#         # MSD.shape = (1,)

#         return MSD


class MMSDLoss(_Loss):
    """
    Computes the mean squared distance (MSD) between the input and target points.

    Note: It is different from torch.nn.MSELoss in that it sums square differences along dim=-1 before taking the mean.
    It can be shown that MSDLoss(x) = 2 * MSELoss(x) for some tensor x.
    Note: Takes into account the modular nature of the data.
    """

    def __init__(self, mod_value: float = 1.0) -> None:
        super().__init__()
        self.mod_value = mod_value
        self.half_mod_value = mod_value / 2

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input.shape = target.shape = (num paths, path length, dimensionality=2)

        # differeces between inputs and targets
        diff = input - target
        # diff.shape = (num paths, path length, 2)

        # modulus of differences
        mod_diff = (
            torch.remainder(diff + self.half_mod_value, self.mod_value)
            - self.half_mod_value
        )
        # mod_diff.shape = (num paths, path length, 2)

        # squared differences between coordinates in inputs and targets
        squared_diff = mod_diff.pow(2)
        # squared_diff.shape = (num paths, path length, 2)

        # sum squared differences to get squared distances between
        # points in input and target
        squared_distances = squared_diff.sum(dim=-1)
        # squared_distances.shape = (num paths, path length)

        # MSD over all points
        MSD = squared_distances.mean()
        # MSD.shape = (1,)

        return MSD


class MSDLoss(_Loss):
    """
    Computes the mean squared distance (MSD) between the input and target points.

    Note: Is more efficient than previous implementation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.half_msd = torch.nn.MSELoss(reduction="mean")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        half_msd = self.half_msd(input, target)
        msd = torch.mul(half_msd, 2)
        return msd


class PathAccuracy(_Loss):
    """
    Computes the accuracy of prediction by considering the MSD along each path separately.

    Note: Since some paths might be harder to predict, measuring MSD for each path separately is sensible.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.threshold = threshold

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input.shape = target.shape = (num paths, path length, dimensionality=2)

        # squared differences between coordinates in inputs and targets
        # squared_diff = (input - target).pow(2)
        squared_diff = self.squared_difference(input, target)
        # squared_diff.shape = (num paths, path length, 2)

        # sum squared differences to get squared distances between
        # points in input and target
        squared_distances = squared_diff.sum(dim=-1)
        # squared_distances.shape = (num paths, path length)

        # MSD for each path
        MSD_per_path = squared_distances.mean(dim=1)
        # MSD_per_path.shape = (num paths,)

        # label paths as correct if MSD is below threshold
        # labels = (MSD_per_path < self.threshold).float()
        labels = torch.where(MSD_per_path < self.threshold, 1.0, 0.0)
        # labels.shape = (num paths,)

        # accuracy is the mean of the labels
        accuracy = labels.mean()
        # accuracy.shape = (1,)

        return accuracy


class ModPathAccuracy(_Loss):
    """
    Computes the accuracy of prediction by considering the MSD along each path separately.

    Note: Since some paths might be harder to predict, measuring MSD for each path separately is sensible.
    Note: Takes into account the modular nature of the data.
    """

    def __init__(self, threshold: float, mod_value: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold

        self.mod_value = mod_value
        self.half_mod_value = mod_value / 2

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input.shape = target.shape = (num paths, path length, dimensionality=2)

        # differeces between inputs and targets
        diff = input - target
        # diff.shape = (num paths, path length, 2)

        # modulus of differences
        mod_diff = (
            torch.remainder(diff + self.half_mod_value, self.mod_value)
            - self.half_mod_value
        )
        # mod_diff.shape = (num paths, path length, 2)

        # squared differences between coordinates in inputs and targets
        squared_diff = mod_diff.pow(2)
        # squared_diff.shape = (num paths, path length, 2)

        # sum squared differences to get squared distances between
        # points in input and target
        squared_distances = squared_diff.sum(dim=-1)
        # squared_distances.shape = (num paths, path length)

        # MSD for each path
        MSD_per_path = squared_distances.mean(dim=1)
        # MSD_per_path.shape = (num paths,)

        # label paths as correct if MSD is below threshold
        labels = torch.where(MSD_per_path < self.threshold, 1.0, 0.0)
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
