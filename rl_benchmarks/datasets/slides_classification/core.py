# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SlideFeaturesDataset module. This module is at the core of data loading for
downstream experiments. It allows to sample over features and labels."""

import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from torch.utils.data import Dataset


class SlideFeaturesDataset(Dataset):
    """From either a list of path to numpy files or loaded numpy features,
    create a `torch.utils.data.Dataset` that samples over the features and labels.

    Parameters
    ----------
    features: Union[np.ndarray, List[Path], List[np.ndarray]]
        Either a list of path to numpy files or loaded numpy features.
    labels: Union[np.ndarray, List[Any]]
        Slide labels (one label per slide).
    n_tiles: Optional[int] = None
        Maximum number of tiles. If ``n_tiles`` is not
        ``None`` and a slide has less than `n_tiles` tiles, all the tiles will
        be used.
    shuffle: bool = True
        Useful if ``n_tiles`` is not None to not sample the same tiles every epoch.
    transform: Optional[Callable] = None
        If not ``None``, function to apply to the slide features. This function
        should take a tensor (``torch.Tensor``) as input and return a tensor
        (``torch.Tensor``). The function will only *not* be applied to the
        first three columns of the slide features.
    """

    def __init__(
        self,
        features: Union[np.ndarray, List[Path], List[np.ndarray]],
        labels: Union[np.ndarray, List[Any]],
        n_tiles: Optional[int] = None,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length.\
            Given {len(features)} and {len(labels)}."
            )

        if n_tiles is None and shuffle:
            warnings.warn("n_tiles is None and shuffle is True.")

        self.features = features
        self.labels = labels
        self.n_tiles = n_tiles
        self.shuffle = shuffle
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        item: int
            Index of item

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (N_tiles, N_FEATURES), ()
        """
        slide_features = self.features[item]
        slide_label = self.labels[item]

        if self.n_tiles is not None:
            # Using memory map to not load the entire np array when we
            # only want `n_tiles <= len(slide_features)` tiles' features.
            if isinstance(slide_features, Path) or isinstance(
                slide_features, str
            ):
                slide_features = np.load(slide_features, mmap_mode="r")

            indices = np.arange(len(slide_features))

            if self.shuffle:
                # We do not shuffle inplace using `np.random.shuffle(slide_features)`
                # as this will load the whole umpy array, removing all benefits
                # of above `mmap_mode='r'`. Instead we shuffle indices and slice into
                # the numpy array.
                np.random.shuffle(indices)

            # Take the desired amount of tiles.
            indices = indices[: self.n_tiles]

            # Indexing will make the array contiguous by loading it in RAM.
            slide_features = slide_features[indices]

        else:
            # Load the whole np.array.
            if isinstance(slide_features, Path) or isinstance(
                slide_features, str
            ):
                slide_features = np.load(slide_features)

            if self.shuffle:
                # Shuffle inplace.
                np.random.shuffle(slide_features)

        slide_features = torch.from_numpy(slide_features.astype(np.float32))

        return slide_features, torch.tensor([slide_label])
