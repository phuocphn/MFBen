from typing import Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class GeometryDataLoader:
    """Custom DataLoader for Geometry Datasets

    This allow you to have a fast dataloader
    """

    def __init__(
        self,
        geometries: torch.Tensor,
        metadata: torch.Tensor,
        batch_size: int = None,
        ignore: bool = False,
        shuffle: bool = False,
    ):
        """Initialize a GeometryDataLoader

        :param geometries: a matrix with shape [number of computional domain, num cells, 5]
        :param metadata: a metadata with shape [number of computional domain, 1]
        :param ignore: Whether to ignore incomplete batches (default is False).
        :param shuffle: Whether to shuffle the dataset (default is False)
        """
        super().__init__()
        self.geometries = geometries
        self.metadata = metadata

        self.batch_size = batch_size
        self.ignore = ignore
        self.shuffle = shuffle

        assert self.metadata.shape[0] == self.geometries.shape[0]
        self.dataset_size = self.metadata.shape[0]

        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        else:
            self.indices = torch.arange(self.dataset_size)

    def __len__(self) -> int:
        """Get the number of batches in the dataloader

        :return: The number of batches
        """
        if self.batch_size is None:
            return 1
        if self.ignore:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size // self.dataset_size) + 1

    def __iter__(self):
        """Initialize the data loader iterator.

        :return: The data loader iterator
        """

        self.current_index = 0
        return self

    def __next__(self):
        """Generate the next batch of data

        :return: The next batch of data, together with metadata.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration

        # If batch_size is None, return the entire dataset as a single batch
        if self.batch_size is None:
            self.current_index += self.dataset_size
            return (self.geometries, self.metadata)

        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        batch_geo = self.geometries[batch_indices, :, :]
        batch_meta = self.metadata[batch_indices, :, :]
        self.current_index += self.batch_size
        return (batch_geo, batch_meta)
