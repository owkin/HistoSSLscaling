# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Camelyon17-WILDS dataset loading."""

from pathlib import Path
import pandas as pd

from ...constants import CAMELYON17_WILDS_PATHS


def load_camelyon17_wilds(
    cohort: str = "TRAIN",
) -> pd.DataFrame:
    """Load data and labels associated with the Camelyon17-WILDS Challenge [1]_.
    Parameters
    ----------
    cohort: str
        The subset of Camelyon17-WILDS cohort to use, either
        ``'TEST'``, ``'TRAIN'``, ``'VALID'`` or ``'FULL'``.

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "image_id": image ID
        "image_path": path to the tile
        "center_id": center ID (0 to 4)
        "label": presence (1) / absence (0) of tumor

    Raises
    ------
    ValueError
        If ``cohort`` is not ``'TEST'``, ``'TRAIN'``, ``'VALID'`` or ``'FULL'``.

    References
    ----------
    .. [1] https://wilds.stanford.edu/datasets/ (CC0 1.0 License).
    """

    # Select train, val or test tiles
    if cohort == "TRAIN":
        centers = [0, 3, 4]
    elif cohort == "VALID":
        centers = [1]
    elif cohort == "TEST":
        centers = [2]
    elif cohort == "FULL":
        centers = [0, 1, 2, 3, 4]
    else:
        raise ValueError(f"Split {cohort} not recognized")

    # Load dataframe with metadata.
    dataset = pd.read_csv(CAMELYON17_WILDS_PATHS["METADATA"], index_col=0)

    # Filter patients.
    mask = dataset.loc[:, "center"].isin(centers)
    dataset = dataset.loc[mask, :]

    # Create return dataframe (standardized column names as for NCT-CRC).
    dataset.loc[:, "image_id"] = [
        f"patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}"
        for patient, node, x, y in dataset.loc[
            :, ["patient", "node", "x_coord", "y_coord"]
        ].itertuples(index=False, name=None)
    ]

    dataset.loc[:, "image_path"] = [
        Path(CAMELYON17_WILDS_PATHS["PATCHES"])
        / f"patient_{patient:03d}_node_{node}"
        / f"patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}.png"
        for patient, node, x, y in dataset.loc[
            :, ["patient", "node", "x_coord", "y_coord"]
        ].itertuples(index=False, name=None)
    ]

    dataset = dataset.rename(columns={"tumor": "label", "center": "center_id"})
    dataset = dataset[["image_id", "image_path", "label", "center_id"]]
    dataset.index = dataset["image_id"]
    return dataset
