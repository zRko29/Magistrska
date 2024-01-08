import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

with open(os.path.join(CONFIG_DIR, "parameters.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)


class StandardMap:
    """
    A class representing the Standard Map dynamical system.
    """

    def __init__(
        self,
        init_points: int = None,
        steps: int = None,
        K: float = None,
        sampling: str = None,
        seed: bool = None,
        inner_outer: str = None,
    ):
        params = PARAMETERS.get("stdm_parameters")

        self.init_points = init_points or params.get("init_points")
        self.steps = steps or params.get("steps")
        self.K = K or params.get("K")
        self.sampling = sampling or params.get("sampling")
        self.inner_outer = inner_outer or params.get("inner_outer")

        self.theta_values = np.array([])
        self.p_values = np.array([])

        if seed is not None:
            np.random.seed(seed=seed)

    def retrieve_data(self):
        return self.theta_values, self.p_values

    def generate_data(self):
        theta, p = self._get_initial_points()

        self.theta_values = np.zeros((self.steps, self.init_points))
        self.p_values = np.zeros((self.steps, self.init_points))

        for iter in range(self.steps):
            theta = (theta + p) % (2 * np.pi)
            p = p + self.K * np.sin(theta)

            self.theta_values[iter] = theta
            self.p_values[iter] = p

    def plot_data(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.theta_values, self.p_values, "bo", markersize=0.3)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.1, 2.05 * np.pi)
        plt.ylim(-1.5, 1.5)
        plt.show()

    def _get_initial_points(self):
        edge = 1e-1
        shift = 0.05
        init_points_lower = self.init_points // 2 + self.init_points % 2
        init_points_upper = self.init_points // 2

        if self.inner_outer == "outer":
            theta_params_lower = [0, np.pi / 2 - edge, init_points_lower]
            theta_params_upper = [3 / 2 * np.pi + edge, 2 * np.pi, init_points_upper]
            p_params_lower = [-1, -0.5 - edge, init_points_lower]
            p_params_upper = [0.5 + edge, 1, init_points_upper]

            if self.sampling == "random":
                theta_init_lower = np.random.uniform(*theta_params_lower)
                theta_init_upper = np.random.uniform(*theta_params_upper)
                p_init_lower = np.random.uniform(*p_params_lower)
                p_init_upper = np.random.uniform(*p_params_upper)
            elif self.sampling == "linear":
                theta_init_lower = np.linspace(*theta_params_lower) + shift
                theta_init_upper = np.linspace(*theta_params_upper) + shift
                p_init_lower = np.linspace(*p_params_lower) + shift
                p_init_upper = np.linspace(*p_params_upper) + shift

            theta_init = np.concatenate((theta_init_lower, theta_init_upper))
            p_init = np.concatenate((p_init_lower, p_init_upper))

            return theta_init, p_init

        elif self.inner_outer == "inner":
            theta_params = [np.pi / 2 + edge, 3 / 2 * np.pi - edge, self.init_points]
            p_params = [-0.5 + edge, 0.5 - edge, self.init_points]
        elif self.inner_outer == "full":
            theta_params = [0, 2 * np.pi, self.init_points]
            p_params = [-1, 1, self.init_points]

        if self.sampling == "random":
            theta_init = np.random.uniform(*theta_params)
            p_init = np.random.uniform(*p_params)

        elif self.sampling == "linear":
            theta_init = np.linspace(*theta_params) + shift
            p_init = np.linspace(*p_params) + shift

        return theta_init, p_init


if __name__ == "__main__":
    map = StandardMap(
        init_points=100, steps=500, sampling="random", K=0.2, inner_outer="full"
    )
    map.generate_data()
    map.plot_data()
