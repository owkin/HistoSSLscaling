# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions related to features extraction. Those functions are
essential to run features extraction process, as done in
``"rl_benchmarks/tools/extract_features/"`` scripts."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openslide
import pandas as pd
import torch
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from ..models import Extractor


class TilesMap(Dataset):
    """BaseTilesMap Dataset.
    From a slide given by slide_path, create a map-style PyTorch Dataset object
    that returns n_tiles tiles sampled, either randomly or sequentially.

    This dataset returns PIL.Image samples. In order to be used in addition with
    a `torch.utils.data.DataLoader`, you must transforms the samples to `torch.Tensor`.
    `torchvision.transforms.ToTensor` can by use for this purpose.

    Parameters
    ----------
    slide_path: str
        Path to slide from which to extract tiles and features.
    tissue_coords: np.ndarray
        Coordinates of tiles in matter.
    tiles_level: int
        Level to extract the tiles from.
    tiles_size: int
        Size of the tiles. The returned tiles will be PIL.Image
        object of size (tiles_size, tiles_size, 3)
    n_tiles: Optional[int] = None
        Number of tiles to sample.
    random_sampling: bool = False
        Sample either randomly or sequentially.
    transform: Optional[Callable] = None
        Transformation to apply to the PIL.Image after sampling.
    """

    def __init__(
        self,
        slide_path: Union[Path, str],
        tissue_coords: np.ndarray,
        tiles_level: int,
        tiles_size: int = 224,
        n_tiles: Optional[int] = None,
        random_sampling: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        if n_tiles is None:
            n_tiles = len(tissue_coords)
        else:
            n_tiles = min(n_tiles, len(tissue_coords))

        if tissue_coords.ndim == 3:
            raise ValueError("tissue_coords must have only two dimensions.")

        self.slide_path = Path(slide_path)
        self.n_tiles = n_tiles
        self.transform = transform
        self.random_sampling = random_sampling

        self.tiles_level = tiles_level
        self.tissue_coords = tissue_coords.astype(int)
        self.tiles_size = [tiles_size, tiles_size]

        self.wsi = openslide.open_slide(self.slide_path)
        self.dz = DeepZoomGenerator(
            self.wsi, tile_size=self.tiles_size[0], overlap=0
        )
        self.build_indices()

    def sample_new_tiles(self) -> None:
        """Permute tile indices to sample new tiles.
        Should be called at the end of every epoch.

        Raises
        ------
        ValueError: samples_new_tiles should only be called if random_sampling is set.
        """
        if not self.random_sampling:
            raise ValueError(
                "samples_new_tiles should only be called if random_sampling is set."
            )

    def build_indices(self) -> None:
        """Build indices for __getitem__ function."""
        if self.random_sampling:
            indices = np.random.permutation(
                np.arange(0, len(self.tissue_coords))
            )
            self.indices = indices[: self.n_tiles]
        else:
            self.indices = np.arange(0, len(self.tissue_coords))[
                : self.n_tiles
            ]

    def __getitem__(self, item: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """Retrieve a tile from a ``openslide.deepzoom.DeepZoomGenerator``
        object, coordinates and tile level (in the DeepZoomGenerator system).

        Returns
        -------
        Tuple[Image.Image, Dict[str, Any]]
            ``tile``: histology tile with shape ``self.tiles_size``
            ``metadata``: dictionary with metadata to fully retrieve the tile.
        """
        # True index of the tile. If random_sampling is False, same index.
        index = self.indices[item]
        coords = self.tissue_coords[index, :]
        tile = self.dz.get_tile(level=int(self.tiles_level), address=coords)

        if tile.size != self.tiles_size:
            # If the tile is on a border, we need to pad it.
            tile = np.array(tile)
            tile = np.pad(
                tile,
                pad_width=(
                    (0, self.tiles_size[0] - tile.shape[0]),
                    (0, self.tiles_size[1] - tile.shape[1]),
                    (0, 0),
                ),
            )
            tile = Image.fromarray(tile)

        if self.transform:
            tile = self.transform(tile)

        metadata = {
            "slide_name": self.slide_path.name,
            "coords": coords,
            "level": self.tiles_level,
            "slide_length": len(self),
        }
        return (tile, metadata)

    def __len__(self) -> int:
        """Return number of tiles."""
        return self.n_tiles


def extract_from_slide(
    slide: openslide.OpenSlide,
    level: int,
    coords: np.ndarray,
    feature_extractor: Extractor,
    tile_size: int = 224,
    n_tiles: Optional[int] = None,
    random_sampling: bool = False,
    num_workers: int = 8,
    batch_size: int = 64,
):
    """Use a Feature Extractor to embed tiles sampled from the coordinates.

    Parameters
    ----------
    slide: openslide.OpenSlide
        Slide to extract.
    level: int
        DeepZoom level.
    coords: np.ndarray
        Array of tile coordinates to extract, shape (N_tiles, 2).
    feature_extractor: Extractor
        A feature extractor.
    tile_size: int = 224
        Tile size (pixels).
    n_tiles: Optional[int] = None
        Number of tiles. If None (default), all tiles are extracted.
    random_sampling: bool = False
        Sample either randomly or sequentially.
    num_workers: int = 8
        Number of workers for the slides torch.utils.data.DataLoader. Useful to parallelize
        reads on several slides at the same time.
    batch_size: int = 64
        Batch size for the extractor.

    Returns
    -------
    features: np.ndarray
        Array of shape (N_TILES, 3 + N_FEATURES) where the 3 first coordinates
        are (``level``, x_coordinate, y_coordinate).
    """
    tile_map = TilesMap(
        slide,
        tissue_coords=coords,
        tiles_level=level,
        tiles_size=tile_size,
        n_tiles=n_tiles,
        random_sampling=random_sampling,
        transform=feature_extractor.transform,
    )

    dataloader = DataLoader(
        tile_map,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )

    slide_features = []
    for images, metadata in dataloader:
        # The tiles dataset provide metadata on the tiles.
        batch_coords = metadata["coords"]
        batch_levels = metadata["level"]

        # Extracts the features.
        features = feature_extractor.extract_features_as_numpy(images)

        # Concatenate the level, coords and features.
        features = np.concatenate(
            (
                batch_levels.unsqueeze(1).float().numpy(),
                batch_coords.float().numpy(),
                features,
            ),
            axis=1,
        )

        slide_features.append(features)

    return np.concatenate(slide_features, axis=0)


class TileImagesDataset(Dataset):
    """From a panda dataframe `dataset` containing the following information:
    "image_id": image ID
    "image_path": path to the tile
    "center_id": center ID (optional)
    "label": tissue class (0 to 8, NCT-CRC) or presence of tumor (0 or 1, Camelyon17-WILDS),

    create a `torch.utils.data.Dataset` that samples over the tile images and labels.

    Parameters
    ----------
    dataset: pd.DataFrame
        Input dataframe with image ids, paths and corresponding labels.
    size: int = 224
        Tile size (pixels) after resizing.
    transform: Optional[Callable] = None
        Function to be applied to tile images.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        tile_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self._tile_size = [tile_size, tile_size]

    def __getitem__(self, item: int) -> Image.Image:
        """Retrieve an image from ``self.dataset``."""
        row = self.dataset.iloc[item]

        image = Image.open(row.image_path).convert("RGB")

        if image.size != self._tile_size:
            image = image.resize(self._tile_size)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        """Returns length of the dataset."""
        return len(self.dataset)


def extract_from_tiles(
    dataset_tiles: pd.DataFrame,
    feature_extractor: Extractor,
    tile_size: int = 224,
    num_workers: int = 8,
    batch_size: int = 64,
) -> Tuple[List[np.array], np.array]:
    """Use a Feature Extractor to embed tiles sampled from the a tiles dataset
    such as Camelyon17-WILDS or NCT-CRC.

    Parameters
    ----------
    dataset_tiles: pd.DataFrame
        Data frame containing the following columns:
            "image_id"   : image ID
            "image_path" : path to the tile
            "center_id"  : center ID (optional)
            "label"      : tissue class (0 to 8, NCT-CRC) or presence of
                           tumor (0 or 1, Camelyon17-WILDS)

    feature_extractor: Extractor
        A feature extractor.
    tile_size: int = 224
        Tile size (pixels).
    num_workers: int = 8
        Number of workers for the tiles torch.utils.data.DataLoader.
        Useful to parallelize reads on several tiles at the same time.
    batch_size: int = 64
        Batch size for the extractor.

    Returns
    -------
    Tuple[List[np.array], np.array]
        List of tiles features (BS, N_FEATURES) and
        corresponding ids (N_TILES_DATASETS,). Length of features list times
        batch size roughly give the size of the tiles dataset.
    """
    tile_map = TileImagesDataset(
        dataset_tiles,
        tile_size=tile_size,
        transform=feature_extractor.transform,
    )

    dataloader = DataLoader(
        tile_map,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )

    tile_features = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        features = feature_extractor.extract_features_as_numpy(batch)
        tile_features.append(features)

    return (
        np.concatenate(tile_features),
        dataset_tiles.image_id.values,
    )


def preload_features(
    fpaths: List[Union[str, Path]],
    n_tiles: int = 1_000,
    shuffle: bool = False,
    with_memory_map: bool = True,
) -> List:
    """Preload all features from a list of features paths.

    Parameters
    ----------
    fpaths: List[Union[str, Path]]
        List of features paths or features numpy arrays.
    n_tiles: int = 1_000
        Number of tiles to keep for all slides.
    shuffle: bool = False
        If True, shuffle tiles in the input list ``fpaths``.
    with_memory_map: bool = True
        Use ``mmap_mode='r'`` when loading slides features (recommended).
    """
    features = []
    indices_features = []

    for i, slide_features in tqdm(enumerate(fpaths), total=len(fpaths)):
        # Using memory map not to load the entire np.array when we
        # only want `n_tiles <= len(slide_features)` tiles' features.
        mmap_mode = "r" if with_memory_map else None
        slide_features = np.load(slide_features, mmap_mode=mmap_mode)

        if n_tiles is not None:
            indices = np.arange(len(slide_features))
            if shuffle:
                # We do not shuffle inplace using `np.random.shuffle(slide_features)`
                # as this will load the whole numpy array, removing all benefits
                # of above `mmap_mode='r'`. Instead we shuffle indices and slice
                # into the numpy array.
                np.random.shuffle(indices)

            # Take the desired amount of tiles.
            indices = indices[:n_tiles]

            # Indexing will make the array contiguous by loading it in RAM.
            slide_features = slide_features[indices]

        else:
            if shuffle:
                # Shuffle inplace
                np.random.shuffle(slide_features)

        features.append(slide_features)
        indices_features.append(i)

    return features, indices_features


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.BoolTensor, Any]:
    """Pad together sequences of arbitrary lengths.
    Add a mask of the padding to the samples that can later be used
    to ignore padding in activation functions.

    Expected to be used in combination of a torch.utils.datasets.DataLoader.

    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using ``torch.utils.data.dataloader.default_collate``.

    Parameters
    ----------
    batch: List[Tuple[torch.Tensor, Any]]
        List of tuples (features, Any). Features have shape (N_slides_tiles, F)
        with ``N_slides_tiles`` being specific to each slide depending on the
        number of extractable tiles in the tissue matter. ``F`` is the feature
        extractor output dimension.
    batch_first: bool = True
        Either return (B, N_TILES, F) or (N_TILES, B, F)
    max_len: Optional[int] = None
        Pre-defined maximum length for elements inside a batch.

    Returns
    -------
    padded_sequences, masks, Any: Tuple[torch.Tensor, torch.BoolTensor, Any]
        - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
        - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]

        with N_TILES = max_len if max_len is not None
        or N_TILES = max length of the training samples.

    """
    # Expect the sequences to be the first one in the sample tuples
    sequences = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[1:])

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    # Batching other members of the tuple using default_collate
    others = default_collate(others)

    return (padded_sequences, masks, *others)
