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
    A class representing the Standard Map dynamical system.
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
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta_values, self.p_values, "bo", markersize=0.3)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.1, 2.05 * np.pi)
        plt.ylim(-1.5, 1.5)
        plt.show()

    def _get_initial_points(self):
        if self.sampling == "random":
            theta_init = np.random.uniform(0, 2 * np.pi, self.init_points)
            p_init = np.random.uniform(-1, 1, self.init_points)

        elif self.sampling == "linear":
            theta_init = np.linspace(0, 2 * np.pi, self.init_points)
            p_init = np.linspace(-1, 1, self.init_points)

        elif self.sampling == "normal":
            mu, sigma = 0, 0.8
            theta_init = np.random.normal(mu + np.pi, 2*sigma, self.init_points)
            p_init = np.random.normal(mu, sigma, self.init_points)

        return theta_init, p_init


if __name__ == "__main__":
    map = StandardMap(init_points=100, steps=100, sampling="normal")
    map.generate_data()
    map.plot_data()
