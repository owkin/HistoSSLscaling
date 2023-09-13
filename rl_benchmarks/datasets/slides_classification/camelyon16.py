# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Camelyon16 dataset loading."""

from pathlib import Path
from typing import Union

import pandas as pd

from ...constants import CAMELYON16_PATHS
from ...utils import merge_multiple_dataframes


def load_camelyon16(
    features_root_dir: Union[str, Path],
    tile_size: int = 224,
    cohort: str = "TRAIN",
    load_slide: bool = False,
) -> pd.DataFrame:
    """Load data from Camelyon16 dataset [1]_.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/. If no
        features have been extracted yet, `features_path` is made of NaNs.
    cohort: str
        The subset of Camelyon16 cohort to use, either ``'TRAIN'`` or ``'TEST'``.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "patient_id": patient ID (is slide ID for Camelyon16)
        "slide_id": slide ID
        "slide_path": path to the slide
        "coords_path": path to the coordinates
        "label": values of the outcome to predict
    
    References
    ----------
    .. [1] https://camelyon17.grand-challenge.org/Data/ (CC0 1.0 License).
    """
    # Get paths.
    labels_path = Path(CAMELYON16_PATHS["LABELS"][cohort])
    slides_root_dir = Path(CAMELYON16_PATHS["SLIDES"])
    coords_root_dir = Path(CAMELYON16_PATHS["COORDS"](tile_size))

    # Get labels.
    dataset_labels = pd.read_csv(labels_path)
    dataset_labels = dataset_labels.rename(
        columns={
            "Unnamed: 0": "patient_id",
            "target": "label",
        }
    )
    dataset_labels["label"] = (dataset_labels.label == "Tumor").astype(float)
    dataset_labels = dataset_labels.drop_duplicates()
    dataset_labels["label"] = dataset_labels["label"].astype(float)
    dataset_labels = dataset_labels[["patient_id", "label"]]

    # Get slides paths.
    spaths = list(slides_root_dir.glob("*.tif"))
    dataset_slides = pd.DataFrame({"slide_path": spaths})
    dataset_slides["slide_id"] = dataset_slides.slide_path.apply(
        lambda x: x.name.split(".")[0]
    )
    dataset_slides = dataset_slides[["slide_id", "slide_path"]]

    # Get paths of tiles coordinates arrays.
    cpaths = list(coords_root_dir.glob("*/coords.npy"))
    dataset_coords = pd.DataFrame({"coords_path": cpaths})
    dataset_coords["slide_id"] = dataset_coords.coords_path.apply(
        lambda x: x.parent.name.split(".")[0]
    )
    dataset_coords = dataset_coords[["slide_id", "coords_path"]]

    # Get feature paths (if available).
    fpaths = list(
        (features_root_dir / "CAMELYON16_FULL").glob("*/features.npy")
    )
    dataset_features = pd.DataFrame({"features_path": fpaths})
    dataset_features["slide_id"] = dataset_features.features_path.apply(
        lambda x: x.parent.name.split(".")[0]
    )
    dataset_features = dataset_features[["slide_id", "features_path"]]

    # Merge dataframes.
    if load_slide:
        dataset = merge_multiple_dataframes(
            dfs=[dataset_slides, dataset_coords, dataset_features],
            on=["slide_id", "slide_id"],
            how=["outer", "outer"],
        )
    else:
        dataset = merge_multiple_dataframes(
            dfs=[dataset_coords, dataset_features],
            on=["slide_id"],
            how=["outer"],
        )
    dataset["patient_id"] = dataset["slide_id"]
    dataset = pd.merge(
        left=dataset,
        right=dataset_labels,
        on="patient_id",
        how="right",
        sort=False,
    )
    dataset["center_id"] = None
    return dataset
