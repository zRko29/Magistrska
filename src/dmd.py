import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


class DMD:
    def __init__(self, data: np.ndarray | list) -> None:
        if not isinstance(data, list):
            data = [data]
        self.len_of_data: int = len(data)

        self.D: dict = {}
        self.X: dict = {}
        self.Y: dict = {}

        for i in range(self.len_of_data):
            self.D[i] = np.array(
                [data[i][:, :, j].flatten() for j in range(data[i].shape[2])]
            )

            self.X[i] = np.array(self.D[i][:-1]).T
            self.Y[i] = np.array(self.D[i][1:]).T

    def _generate_dmd_results(self) -> None:
        self.res = self._dmd()

    # DMD classical version
    def _dmd(self) -> dict:
        results: dict = {}

        for i in range(self.len_of_data):
            U, S, Vh = LA.svd(self.X[i], full_matrices=False)
            A = (U.T) @ self.Y[i] @ (Vh.T) * (1 / S)

            eigen_vals, eigen_vecs = LA.eig(A)
            proj_DMD_modes = U @ eigen_vecs

            exact_DMD_modes = (
                (1 / eigen_vals) * (self.Y[i] @ (Vh.T) * (1 / S)) @ eigen_vecs
            )

            # index of angles of eigen_vals
            ids = np.argsort(np.abs(np.angle(eigen_vals)))

            results[i] = {
                "A": A,
                "projDMD": proj_DMD_modes,
                "exactDMD": exact_DMD_modes,
                "eigen_vals": eigen_vals,
                "eigen_vecs": eigen_vecs,
                "ids": ids,
            }

        return results

    def plot_source_matrix(self, titles: list = None) -> None:
        fig, axs = plt.subplots(
            1, self.len_of_data, figsize=(self.len_of_data * 6, 6), sharey=True
        )
        if self.len_of_data == 1:
            axs = [axs]

        for i in range(self.len_of_data):
            axs[i].imshow(self.D[i].T, aspect="auto")
            axs[i].set_xlabel("step")
            if i == 0:
                axs[i].set_ylabel("point index")
            if titles is None:
                axs[i].set_title(f"Data {i}")
            else:
                axs[i].set_title(titles[i])

        plt.tight_layout()
        plt.show()

    def plot_eigenvalues(self, titles: list = None) -> None:
        fig, axs = plt.subplots(
            1, self.len_of_data, figsize=(self.len_of_data * 6, 6), sharey=True
        )
        if self.len_of_data == 1:
            axs = [axs]

        for i in range(self.len_of_data):
            phi = np.linspace(0, 2 * np.pi, 1000)
            axs[i].plot(np.sin(phi), np.cos(phi), "r-", label="unit circle")
            axs[i].plot(
                self.res[i]["eigen_vals"].real,
                self.res[i]["eigen_vals"].imag,
                ".",
                markersize=10,
                label="data",
            )
            axs[i].axhline(y=0, color="k")
            axs[i].axvline(x=0, color="k")
            axs[i].set_xlabel("re")
            if i == 0:
                axs[i].set_ylabel("im")
            if titles is None:
                axs[i].set_title(f"Data {i}")
            else:
                axs[i].set_title(titles[i])
            axs[i].legend(loc="upper right")
            axs[i].grid()

        plt.tight_layout()
        plt.show()

    # plot abs. values, phase and spacing of phases distribution
    def plot_abs_values(self, titles: list = None) -> None:
        fig, axs = plt.subplots(
            1, self.len_of_data, figsize=(self.len_of_data * 10, 6), sharey=True
        )
        if self.len_of_data == 1:
            axs = [axs]

        for i in range(self.len_of_data):
            norms = np.sort(np.abs(self.res[i]["eigen_vals"]))

            axs[i].hist(norms, density=True)
            axs[i].set_xlabel("$|\lambda|$")
            if i == 0:
                axs[i].set_ylabel("pdf")
            if titles is None:
                axs[i].set_title(f"Data {i}")
            else:
                axs[i].set_title(titles[i])
            axs[i].grid()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from mapping_helper import StandardMap

    map1 = StandardMap(init_points=1000, steps=500, sampling="grid", K=[0.1])
    map1.generate_data()
    thetas, ps = map1.retrieve_data()
    data1 = np.stack([thetas.T, ps.T], axis=1)

    map2 = StandardMap(init_points=1000, steps=500, sampling="grid", K=[1.0])
    map2.generate_data()
    thetas, ps = map2.retrieve_data()
    data2 = np.stack([thetas.T, ps.T], axis=1)

    dmd = DMD([data1, data2])
    dmd.plot_source_matrix()
    dmd._generate_dmd_results()
    dmd.plot_eigenvalues()
    dmd.plot_abs_values()

"""
# function for create a gridded data
def G(x, f, nQ, nP):
    v = np.reshape(x, (nQ * nP, 2))
    fv = np.apply_along_axis(f, 1, v)
    return np.reshape(fv, (nQ, nP))


# show dynamics through observable f(x,y) = ||(x,y)||
# f just an arbitrary choice of observable, but gives nice results

qs = np.linspace(0, 1, np.sqrt(init_points).astype(int))
ps = np.linspace(0, 1, np.sqrt(init_points).astype(int))

qq, pp = np.meshgrid(qs, ps, indexing="ij")

size = init_points
nx = 10
ny = int(np.round(steps / nx))

fig, axs = plt.subplots(
    nx, ny, sharex="col", sharey="row", figsize=(nx, ny), layout="constrained"
)

for idx in range(nx * ny):
    ii = int(idx / nx)
    jj = idx % nx
    axs[ii, jj].contourf(
        qq,
        pp,
        G(
            X[:, idx],
            np.linalg.norm,
            np.sqrt(init_points).astype(int),
            np.sqrt(init_points).astype(int),
        ),
    )

plt.show()

# plotting modes through observable f(x,y) = ||(x,y)|| == just an arbitrary choice of observable
# more precisely we do f(f_1(x,y), f_2(x,y)) for presentation purpose

qq, pp = np.meshgrid(qs, ps, indexing="ij")

size = init_points
nx = 10
ny = int(np.round(steps / nx))

fig, axs = plt.subplots(
    nx, ny, sharex="col", sharey="row", figsize=(nx, ny), layout="constrained"
)

for idx in range(nx * ny):
    ii = int(idx / nx)
    jj = idx % nx

    target_idx = res["ids"][idx]
    lam = res["eigen_vals"][target_idx]

    s = (
        f"$\lambda$ = {lam.real:.4f} + {lam.imag:.4f} i"
        if lam.imag >= 0
        else f"$\lambda$ = {lam.real:.4f} - i{-lam.imag:.4f}"
    )
    # s = s + f", $|\lambda|=${np.abs(lam):.4f}"

    axs[ii, jj].contourf(
        qq,
        pp,
        G(
            res["exactDMD"][:, target_idx],
            np.linalg.norm,
            np.sqrt(init_points).astype(int),
            np.sqrt(init_points).astype(int),
        ),
    )
    axs[ii, jj].text(
        1 / 5.8, 6 / 5, s, fontsize=3, bbox=dict(facecolor="white", alpha=0.5)
    )

plt.show()
"""
