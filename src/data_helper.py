import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np
from typing import Tuple

from src.mapping_helper import StandardMap
from src.utils import plot_split


class Data(pl.LightningDataModule):
    def __init__(
        self,
        map_object: StandardMap,
        params: dict,
        train_size: float = 1.0,
        plot_data: bool = False,
        plot_data_split: bool = False,
    ) -> None:
        super(Data, self).__init__()
        self.seq_len: int = params.get("seq_length")
        self.batch_size: int = params.get("batch_size")
        self.take_every_nth_step: int = params.get("take_every_nth_step")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_within_batches: bool = params.get("shuffle_within_batches")
        self.drop_last: bool = params.get("drop_last")
        sequence_type: str = params.get("sequence_type")
        self.rng: np.random.Generator = np.random.default_rng(seed=42)

        map_object.generate_data()

        thetas: np.ndarray
        ps: np.ndarray
        thetas, ps = map_object.retrieve_data()

        if plot_data:
            map_object.plot_data()

        # data.shape = [init_points, steps, 2]
        self.data = np.stack([thetas.T, ps.T], axis=-1)

        # take every n-th step
        self.data = self.data[:, :: self.take_every_nth_step]

        # shuffle trajectories
        if self.shuffle_trajectories:
            self.rng.shuffle(self.data)

        # many-to-many or many-to-one
        sequences = self._make_sequences(self.data, type=sequence_type)

        if plot_data_split:
            plot_split(sequences, train_size)

        self.input_output_pairs = self._make_input_output_pairs(
            sequences, type=sequence_type
        )
        self.t = int(len(self.input_output_pairs) * train_size)

    def _make_sequences(self, data: np.ndarray, type: str) -> np.ndarray:
        init_points: int
        steps: int
        features: int
        init_points, steps, features = data.shape

        if self.seq_len >= steps:
            sequences = data

        elif type == "many-to-many":
            # sequences.shape = [init_points*(steps//seq_len), seq_len, features]
            sequences = np.split(data, steps // self.seq_len, axis=1)

            sequences = np.array(
                [seq[i] for i in range(init_points) for seq in sequences]
            )

        elif type == "many-to-one":
            # sequences.shape = [init_points * (steps - seq_len), seq_len + 1, features]
            sequences = np.lib.stride_tricks.sliding_window_view(
                data, (1, self.seq_len + 1, features)
            )
            sequences = sequences.reshape(
                init_points * (steps - self.seq_len), self.seq_len + 1, features
            )
        else:
            raise ValueError(f"Invalid sequence type: {type}")

        return sequences

    def _make_input_output_pairs(self, sequences, type: str) -> list[tuple[np.ndarray]]:
        if type == "many-to-many":
            # i.shape = (seq_len - 1, seq_len - 1)
            return [(seq[:-1], seq[1:]) for seq in sequences]
        elif type == "many-to-one":
            # i.shape = (seq_len - 1, 1)
            return [(seq[:-1], seq[-1:]) for seq in sequences]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[: self.t]),
            batch_size=self.batch_size,
            shuffle=self.shuffle_within_batches,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[self.t :]),
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
