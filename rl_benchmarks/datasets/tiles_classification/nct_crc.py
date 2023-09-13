# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NCT-CRC dataset loading."""

import pickle
from pathlib import Path

import pandas as pd

from ...constants import NCT_CRC_PATHS


def load_nct_crc(
    cohort: str = "TRAIN",
) -> pd.DataFrame:
    """Load data and labels associated with NCT-CRC dataset [1]_.
    Parameters
    ----------
    cohort: str
        The subset of NCT-CRC dataset to use, either "TRAIN", "VALID",
        "TEST" or "FULL".

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "image_id": image ID
        "image_path": path to the tile
        "label": tissue class (0 to 8)

    Raises
    ------
    ValueError
        If ``cohort`` is not ``'VALID'`` or ``'TRAIN'``.

    References
    ----------
    .. [1] https://zenodo.org/record/1214456#.YVrmANpBwRk (CC-BY 4.0 License).
    """
    # Select train, validation or testing tiles
    if cohort in ["TRAIN", "VALID"]:
        path = Path(NCT_CRC_PATHS[cohort])
    else:
        raise ValueError(f"Split {cohort} not recognized")

    tpaths = list(path.glob("*/*.tif"))
    dataset = pd.DataFrame({"image_path": tpaths})

    dataset["image_id"] = dataset.image_path.apply(
        lambda x: x.name.split(".")[0]
    )
    dataset["label"] = dataset.image_path.apply(lambda x: x.parent.name)

    with open(NCT_CRC_PATHS["LABELS"], "rb") as file:
        dict_labels = pickle.load(file)

    dataset["label"] = dataset.label.map(dict_labels)
    dataset.index = dataset["image_id"]
    dataset["center_id"] = None
    return dataset
