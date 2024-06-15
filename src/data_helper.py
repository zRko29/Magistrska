import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np
from typing import Tuple, List
import warnings

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
        self.every_n_step: int = params.get("every_n_step")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_within_batches: bool = params.get("shuffle_within_batches")
        self.drop_last: bool = params.get("drop_last")
        val_reg_preds: int = params.get("val_reg_preds")
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
        assert (self.data.shape[1] // self.every_n_step) >= (
            self.seq_len + val_reg_preds
        ), f"take steps >= {(self.seq_len + val_reg_preds)*self.every_n_step}"
        self.data = self.data[:, :: self.every_n_step]

        # shuffle trajectories
        if self.shuffle_trajectories:
            self.rng.shuffle(self.data)

        t = int(len(self.data) * train_size)

        train_sequences = self._make_sequences(self.data[:t], 1)
        self.train_pairs = self._make_input_output_pairs(train_sequences, 1)

        if train_size < 1.0:
            validation_sequences = self._make_sequences(self.data[t:], val_reg_preds)
            self.validation_pairs = self._make_input_output_pairs(
                validation_sequences, val_reg_preds
            )
        else:
            self.validation_pairs = np.array([])

        if (
            len(self.train_pairs) < self.batch_size
            or len(self.validation_pairs) < self.batch_size
        ):
            warnings.warn(
                "Batch size is larger than the number of training or validation pairs. Is drop_last set to True?"
            )

        print(
            f"{len(self.train_pairs)} training pairs of shape ({self.train_pairs[0][0].shape[0]}, {self.train_pairs[0][1].shape[0]})."
        )
        print(
            f"{len(self.validation_pairs)} validation pairs of shape ({self.validation_pairs[0][0].shape[0]}, {self.validation_pairs[0][1].shape[0]})."
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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.validation_pairs),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self) -> torch.Tensor:
        return torch.tensor(self.data).unsqueeze(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Tuple[np.ndarray]]):
        self.data: List[Tuple[np.ndarray]] = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x, y = self.data[idx]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
