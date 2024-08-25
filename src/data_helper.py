import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np
from typing import Tuple, List, Optional
import warnings
import os

from src.mapping_helper import StandardMap


class Data(pl.LightningDataModule):

    def __init__(
        self,
        map_object: Optional[StandardMap] = None,
        data_path: Optional[str] = None,
        K: List[float] | float = None,
        binary: bool = False,
        params: dict = None,
        train_size: float = 1.0,
        plot_data: bool = False,
    ) -> None:
        super(Data, self).__init__()
        self.seq_len: int = params.get("seq_length")
        self.batch_size: int = params.get("batch_size")
        self.every_n_step: int = params.get("every_n_step")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_within_batches: bool = params.get("shuffle_within_batches")
        self.drop_last: bool = params.get("drop_last")
        val_reg_preds: int = params.get("val_reg_preds")
        self.rng: np.random.Generator = np.random.default_rng(seed=42)

        # generate new data
        if map_object is not None:
            map_object.generate_data()
            thetas, ps = map_object.retrieve_data()

            # fake spectrum
            self.spectrum = self.rng.choice([0, 1], size=thetas.shape[1])
            self.reverse_indices = None

        # load data
        elif data_path is not None:
            thetas, ps, self.spectrum = self._load_data(data_path, K, binary)

            indices = np.arange(len(self.spectrum))
            self.rng.shuffle(indices)
            thetas = thetas[:, indices]
            ps = ps[:, indices]
            self.spectrum = self.spectrum[indices]

            self.reverse_indices = np.empty_like(indices)
            self.reverse_indices[indices] = np.arange(len(self.spectrum))

            init_points = params.get("init_points")
            steps = params.get("steps")
            thetas = thetas[:steps, :init_points]
            ps = ps[:steps, :init_points]

        if plot_data:
            map_object.plot_data()

        # data.shape = [init_points, steps, 2]
        self.data = np.stack([thetas.T, ps.T], axis=-1)

        # take every n-th step
        # assert (self.data.shape[1] // self.every_n_step) >= (
        # self.seq_len + val_reg_preds
        # ), f"take steps >= {(self.seq_len + val_reg_preds)*self.every_n_step}"
        self.data = self.data[:, :: self.every_n_step]

        # shuffle trajectories
        if self.shuffle_trajectories:
            self.rng.shuffle(self.data)

        t = int(len(self.data) * train_size)

        if train_size > 0.0:
            train_sequences = self._make_sequences(self.data[:t], 1)
            self.train_pairs = self._make_input_output_pairs(train_sequences, 1)

        if train_size < 1.0:
            validation_sequences = self._make_sequences(self.data[t:], val_reg_preds)
            self.validation_pairs = self._make_input_output_pairs(
                validation_sequences, val_reg_preds
            )
        else:
            self.validation_pairs = np.array([])

        self.print_info(train_size)

    def print_info(self, train_size: float) -> None:
        if 0.0 < train_size < 1.0:
            if (
                len(self.train_pairs) < self.batch_size
                or len(self.validation_pairs) < self.batch_size
            ):
                warnings.warn(
                    f"Batch size ({self.batch_size}) is larger than the number of training or validation pairs. Is drop_last set to True?"
                )

            print(
                f"{len(self.train_pairs)} training pairs of shape ({self.train_pairs[0][0].shape[0]}, {self.train_pairs[0][1].shape[0]})."
            )
            print(
                f"{len(self.validation_pairs)} validation pairs of shape ({self.validation_pairs[0][0].shape[0]}, {self.validation_pairs[0][1].shape[0]})."
            )
        elif train_size == 0.0:
            if len(self.validation_pairs) < self.batch_size:
                warnings.warn(
                    f"Batch size ({self.batch_size}) is larger than the number of training or validation pairs. Is drop_last set to True?"
                )
            print(
                f"{len(self.validation_pairs)} validation pairs of shape ({self.validation_pairs[0][0].shape[0]}, {self.validation_pairs[0][1].shape[0]})."
            )
        else:
            if len(self.train_pairs) < self.batch_size:
                warnings.warn(
                    "Batch size is larger than the number of training pairs. Is drop_last set to True?"
                )

            print(
                f"{len(self.train_pairs)} training pairs of shape ({self.train_pairs[0][0].shape[0]}, {self.train_pairs[0][1].shape[0]})."
            )

    def _make_sequences(self, data: np.ndarray, val_reg_preds: int) -> np.ndarray:
        init_points: int
        steps: int
        features: int
        init_points, steps, features = data.shape

        if self.seq_len >= steps:
            sequences = data
        else:
            # sequences.shape = [init_points * (steps - seq_len), seq_len + val_reg_preds, features]
            sequences = np.lib.stride_tricks.sliding_window_view(
                data, (1, self.seq_len + val_reg_preds, features)
            )
            sequences = sequences.reshape(
                init_points * (steps - self.seq_len - val_reg_preds + 1),
                self.seq_len + val_reg_preds,
                features,
            )

        return sequences

    def _make_input_output_pairs(
        self, sequences, val_reg_preds: int
    ) -> List[Tuple[np.ndarray]]:
        return [(seq[:-val_reg_preds], seq[-val_reg_preds:]) for seq in sequences]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.train_pairs),
            batch_size=self.batch_size,
            shuffle=self.shuffle_within_batches,
            drop_last=self.drop_last,
            # pin_memory=True,
            # num_workers=8,
            # persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.validation_pairs),
            batch_size=self.batch_size * 5,
            drop_last=False,
            # pin_memory=True,
            # num_workers=8,
            # persistent_workers=True,
        )

    def predict_dataloader(self) -> torch.Tensor:
        return DataLoader(
            InferenceDataset(torch.tensor(self.data), self.spectrum),
            batch_size=int(1e6),
            shuffle=False,
        )

    @staticmethod
    def _load_data(
        path: str, K: List[float] | float, binary: bool
    ) -> Tuple[np.ndarray]:
        if not isinstance(K, list):
            K = [K]

        directories: List[str] = _get_subdirectories(path, K)

        thetas_list = [
            np.load(os.path.join(directory, "theta_values.npy"))
            for directory in directories
        ]
        ps_list = [
            np.load(os.path.join(directory, "p_values.npy"))
            for directory in directories
        ]
        spectrum_list = [
            np.load(os.path.join(directory, "spectrum.npy"))
            for directory in directories
        ]

        thetas = np.concatenate(thetas_list, axis=1)
        ps = np.concatenate(ps_list, axis=1)
        spectrum = np.concatenate(spectrum_list)

        if binary:
            spectrum = (spectrum * 1e5 > 11).astype(int)

        return thetas, ps, spectrum


def _get_subdirectories(directory: str, K: List[float]) -> List[str]:
    subdirectories = []
    for d in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, d)):
            if float(d) in K:
                subdirectories.append(os.path.join(directory, d))

    subdirectories.sort()
    return subdirectories


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Tuple[np.ndarray]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x, y = self.data[idx]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Tuple[np.ndarray]], spectrum: np.ndarray):
        self.data = data
        self.spectrum = spectrum

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x, y = self.data[idx], self.spectrum[idx]
        # x = torch.tensor(x)
        # y = torch.tensor(y)
        return x, y
