import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np
from typing import Tuple

from src.mapping_helper import StandardMap


class Data(pl.LightningDataModule):
    def __init__(
        self,
        map_object: StandardMap,
        params: dict,
        train_size: float = 1.0,
        plot_data: bool = False,
    ) -> None:
        super(Data, self).__init__()
        self.seq_len: int = params.get("seq_length")
        self.batch_size: int = params.get("batch_size")
        self.take_every_nth_step: int = params.get("take_every_nth_step")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_within_batches: bool = params.get("shuffle_within_batches")
        self.drop_last: bool = params.get("drop_last")
        take_last_n: int = params.get("take_last_n")
        self.rng: np.random.Generator = np.random.default_rng(seed=42)

        map_object.generate_data()

        thetas: np.ndarray
        ps: np.ndarray
        thetas, ps = map_object.retrieve_data()

        if plot_data:
            map_object.plot_data()

        # data.shape = [init_points, steps, 2]
        self.data = np.stack([thetas.T, ps.T], axis=-1)
        # duplicate data to make it longer
        self.data = np.concatenate([self.data, self.data], axis=0)

        # take every n-th step
        assert (self.data.shape[1] // self.take_every_nth_step) >= (
            self.seq_len + take_last_n
        )
        self.data = self.data[:, :: self.take_every_nth_step]

        # shuffle trajectories
        if self.shuffle_trajectories:
            self.rng.shuffle(self.data)

        t = int(len(self.data) * train_size)

        train_sequences = self._make_sequences(self.data[:t], 1)
        self.train_pairs = self._make_input_output_pairs(train_sequences, 1)

        if train_size < 1.0:
            validation_sequences = self._make_sequences(self.data[t:], take_last_n)
            self.validation_pairs = self._make_input_output_pairs(
                validation_sequences, take_last_n
            )

    def _make_sequences(self, data: np.ndarray, take_last_n: int) -> np.ndarray:
        init_points: int
        steps: int
        features: int
        init_points, steps, features = data.shape

        if self.seq_len >= steps:
            sequences = data

        else:
            # sequences.shape = [init_points * (steps - seq_len), seq_len + take_last_n, features]
            sequences = np.lib.stride_tricks.sliding_window_view(
                data, (1, self.seq_len + take_last_n, features)
            )
            sequences = sequences.reshape(
                init_points * (steps - self.seq_len - take_last_n + 1),
                self.seq_len + take_last_n,
                features,
            )

        return sequences

    def _make_input_output_pairs(
        self, sequences, take_last_n: int
    ) -> list[tuple[np.ndarray]]:
        return [(seq[:-take_last_n], seq[-take_last_n:]) for seq in sequences]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.train_pairs),
            batch_size=self.batch_size,
            shuffle=self.shuffle_within_batches,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.validation_pairs),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self) -> torch.Tensor:
        return torch.tensor(self.data).unsqueeze(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x, y = self.data[idx]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
