import numpy as np
import matplotlib.pyplot as plt


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
        params: dict = None,
    ):
        self.init_points = init_points or params.get("init_points")
        self.steps = steps or params.get("steps")
        self.K = K or params.get("K")
        self.sampling = sampling or params.get("sampling")

        self.rng = np.random.default_rng(seed=seed)

    def retrieve_data(self):
        return self.theta_values, self.p_values

    def retrieve_spectrum(self):
        return self.spectrum

    def generate_data(self, lyapunov: bool = False, n: int = 10**4):
        theta, p = self._get_initial_points()
        steps = n if lyapunov else self.steps

        self.theta_values = np.empty((steps, self.init_points))
        self.p_values = np.empty((steps, self.init_points))

        for step in range(steps):
            theta = np.mod(theta + p, 2 * np.pi)
            p = p + self.K * np.sin(theta)
            self.theta_values[step] = theta
            self.p_values[step] = p

        if lyapunov:
            self.spectrum = self._lyapunov(n)

        self.theta_values = self.theta_values[: self.steps]
        self.p_values = self.p_values[: self.steps]
        self.lyapunov = lyapunov

    def _jaccobi(self, row, column):
        der = self.K * np.cos(
            self.theta_values[row, column] + self.p_values[row, column]
        )
        return np.array([[1, 1], [der, 1 + der]])

    def _lyapunov(self, n, treshold=1e3):
        spectrum = np.empty(self.init_points)
        for column in range(self.init_points):
            M = np.identity(2)
            exp = np.zeros(2)

            for row in range(n):
                M = self._jaccobi(row, column) @ M

                if np.linalg.norm(M) > treshold:
                    Q, R = np.linalg.qr(M)
                    exp += np.log(np.abs(R.diagonal()))
                    M = Q

            _, R = np.linalg.qr(M)
            exp += np.log(np.abs(R.diagonal()))

            spectrum[column] = exp[0] / n

        return spectrum

    def _get_initial_points(self):
        theta_params = [0, 2 * np.pi, self.init_points]
        p_params = [-1, 1, self.init_points]

        if self.sampling == "random":
            theta_init = self.rng.uniform(*theta_params)
            p_init = self.rng.uniform(*p_params)

        elif self.sampling == "linear":
            shift = 0.05  # to avoid duplication
            theta_init = np.linspace(*theta_params) + shift
            p_init = np.linspace(*p_params) + shift

        return theta_init, p_init

    def plot_data(self):
        plt.figure(figsize=(7, 4))
        if self.lyapunov:
            chaotic_indices = np.where(self.spectrum > 1e-4)[0]
            regular_indices = np.where(self.spectrum <= 1e-4)[0]
            plt.plot(
                self.theta_values[chaotic_indices],
                self.p_values[chaotic_indices],
                "ro",
                markersize=0.3,
            )
            plt.plot(
                self.theta_values[regular_indices],
                self.p_values[regular_indices],
                "bo",
                markersize=0.3,
            )
            legend_handles = [
                plt.scatter([], [], color="red", marker=".", label="Chaotic"),
                plt.scatter([], [], color="blue", marker=".", label="Regular"),
            ]
            plt.legend(handles=legend_handles)
        else:
            plt.plot(self.theta_values, self.p_values, "bo", markersize=0.3)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.1, 2.05 * np.pi)
        plt.ylim(-1.8, 1.8)
        plt.show()


if __name__ == "__main__":
    map = StandardMap(init_points=100, steps=300, sampling="random", K=1.5)
    map.generate_data(lyapunov=True, n=10**4 * 5)
    map.plot_data()
