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

    def generate_data(self, lyapunov: bool = False, n: int = 10**5):
        theta, p = self._get_initial_points()
        steps = n if lyapunov else self.steps

        self.theta_values = np.empty((steps, self.init_points))
        self.p_values = np.empty((steps, self.init_points))

        for step in range(steps):
            theta = np.mod(theta + p, 1)
            p = np.mod(p + self.K / (2 * np.pi) * np.sin(2 * np.pi * theta), 1)
            self.theta_values[step] = theta
            self.p_values[step] = p

        if lyapunov:
            self.spectrum = self._lyapunov(n)

        self.theta_values = self.theta_values[: self.steps]
        self.p_values = self.p_values[: self.steps]
        self.lyapunov = lyapunov

    def _jaccobi(self, row, column):
        der = self.K * np.cos(
            2 * np.pi * (self.theta_values[row, column] + self.p_values[row, column])
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

        # make binary labels
        spectrum = (spectrum > 1e-4).astype(int)

        return spectrum

    def _get_initial_points(self):
        params = [0, 1, self.init_points]

        if self.sampling == "random":
            theta_init = self.rng.uniform(*params)
            p_init = self.rng.uniform(*params)

        elif self.sampling == "linear":
            theta_init = np.linspace(*params)
            p_init = np.linspace(*params)

        return theta_init, p_init

    def plot_data(self):
        plt.figure(figsize=(7, 4))
        if self.lyapunov:
            chaotic_indices = np.where(self.spectrum == 1)[0]
            regular_indices = np.where(self.spectrum == 0)[0]
            plt.plot(
                self.theta_values[:, chaotic_indices],
                self.p_values[:, chaotic_indices],
                "ro",
                markersize=0.5,
            )
            plt.plot(
                self.theta_values[:, regular_indices],
                self.p_values[:, regular_indices],
                "bo",
                markersize=0.5,
            )
            legend_handles = [
                plt.scatter([], [], color="red", marker=".", label="Chaotic"),
                plt.scatter([], [], color="blue", marker=".", label="Regular"),
            ]
            plt.legend(handles=legend_handles)
        else:
            plt.plot(self.theta_values, self.p_values, "bo", markersize=0.5)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()


if __name__ == "__main__":
    map = StandardMap(init_points=30, steps=5000, sampling="random", K=1.0, seed=42)
    map.generate_data(lyapunov=True)
    map.plot_data()
