import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

ROOT_DIR = os.getcwd()
MAIN_DIR = os.path.join(ROOT_DIR, "standard_map")
DATA_DIR = os.path.join(MAIN_DIR, "data")
CONFIG_DIR = os.path.join(MAIN_DIR, "config")

with open(os.path.join(CONFIG_DIR,  "parameters.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)


class StandardMap:
    """
    A class representing the Standard Map, a two-dimensional area-preserving map.

    Parameters:
    -----------
    init_points : int
        The number of initial points to use for the map.
    steps : int
        The number of iterations to perform.
    K : float
        The value of the control parameter for the map.
    seed : int, optional
        The random seed to use for the map.
    sampling : str, optional
        The method to use for selecting initial points. Can be "random" or "linear".

    Attributes:
    -----------
    theta_values : numpy.ndarray
        The values of the theta variable for each iteration of the map.
    p_values : numpy.ndarray
        The values of the p variable for each iteration of the map.
    """

    def __init__(self, init_points: int = None, steps: int = None, K: float = None, sampling: str = None, seed: bool = None):
        params = PARAMETERS.get("stdm_parameters")

        self.init_points = init_points or params.get("init_points")
        self.steps = steps or params.get("steps")
        self.K = K or params.get("K")
        self.sampling = sampling or params.get("sampling")
        self.seed = seed or params.get("seed")

        self.theta_values = np.array([])
        self.p_values = np.array([])

        if self.seed is not None:
            np.random.seed(seed=self.seed)

    def get_data(self):
        return self.theta_values, self.p_values

    def do_mapping(self):
        theta, p = self._select_initial_points()

        self.theta_values = np.zeros((self.steps, self.init_points))
        self.p_values = np.zeros((self.steps, self.init_points))

        for iter in range(self.steps):
            theta = (theta + p) % (2 * np.pi)
            p = p + self.K * np.sin(theta)

            self.theta_values[iter] = theta
            self.p_values[iter] = p

    def save_data(self):
        thetas_path = os.path.join(DATA_DIR, "theta_values.npy")
        ps_path = os.path.join(DATA_DIR, "p_values.npy")

        np.save(thetas_path, self.theta_values)
        np.save(ps_path, self.p_values)

    def plot_data(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.theta_values, self.p_values, "bo", markersize=0.3)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.1, 2.05 * np.pi)
        plt.ylim(-1.5, 1.5)
        plt.show()

    def _select_initial_points(self):
        if self.sampling == "random":
            theta_init = np.random.uniform(0, 2 * np.pi, self.init_points)
            p_init = np.random.uniform(-1, 1, self.init_points)

        elif self.sampling == "linear":
            theta_init = np.linspace(0, 2 * np.pi, self.init_points)
            p_init = np.linspace(-1, 1, self.init_points)

        elif self.sampling == "normal":
            mu, sigma = 0, 0.5
            theta_init = np.random.normal(mu + np.pi, sigma, self.init_points)
            p_init = np.random.normal(mu, sigma, self.init_points)

        return theta_init, p_init


if __name__ == "__main__":
    map = StandardMap()
    map.do_mapping()
    map.save_data()
    map.plot_data()
