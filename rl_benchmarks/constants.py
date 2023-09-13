# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Paths to all raw and preprocessed data (weights included).
Those paths are directly computed from the ``'conf.yaml'`` file at the root
of this repository. Once the installation and data download steps are completed,
you finally need to edit the `conf.yaml` file so that to specify:
- `data_dir`: root directory that contains the downloaded data.
- `logs_save_dir`: directory for cross-validation experiments logs
If you downloaded the data in `/home/user/downloaded_data/data/` folder,
then this should be the `data_dir`."""

from typing import Tuple
from pathlib import Path
import yaml


def get_data_roots() -> Tuple[Path, Path]:
    """Return the path to:
    - the root directory which contains raw and preprocessed data
    - the root directory which contains experiments logs
    """
    parent_dir = list(Path(__file__).parents)[1].resolve()
    with open(str(parent_dir / "conf.yaml"), "r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
            return (
                Path(data["data_dir"]),
                Path(data["logs_save_dir"]),
            )
        except yaml.YAMLError as exc:
            print(exc)
            raise


# Get data directories.
DATA_ROOT_DIR, LOGS_ROOT_DIR = get_data_roots()
# Raw data directory.
RAW_DATA_DIR = DATA_ROOT_DIR / "raw"
# Preprocessed data root directory.
PREPROCESSED_DATA_DIR = DATA_ROOT_DIR / "preprocessed"
# TCGA raw slides directory. See main README.md (data structure section) for
# details.
TCGA_SLIDES_DIR = DATA_ROOT_DIR.joinpath(
    "raw/slides_classification/TCGA/slides/parafine"
)
# Model weights directory.
WEIGHTS_DATA_DIR = DATA_ROOT_DIR / "weights"


# Slide classification tasks.
# Camelyon16 dataset.
CAMELYON16_PATHS = {
    "SLIDES": RAW_DATA_DIR.joinpath(
        "slides_classification/CAMELYON16_FULL/slides"
    ),
    "COORDS": (
        lambda tile_size: PREPROCESSED_DATA_DIR.joinpath(
            f"slides_classification/coords/coords_{tile_size}/CAMELYON16_FULL/"
        )
    ),
    "LABELS": {
        "TRAIN": RAW_DATA_DIR.joinpath(
            "slides_classification/CAMELYON16_FULL/clinical/train_clinical_data.csv"
        ),
        "TEST": RAW_DATA_DIR.joinpath(
            "slides_classification/CAMELYON16_FULL/clinical/test_clinical_data.csv"
        ),
    },
}

# TCGA cohorts.
TCGA_PATHS = {
    "SLIDES": RAW_DATA_DIR.joinpath(
        "slides_classification/TCGA/slides/parafine"
    ),
    "COORDS": (
        lambda tile_size: PREPROCESSED_DATA_DIR.joinpath(
            f"slides_classification/coords/coords_{tile_size}/TCGA/"
        )
    ),
    "LABELS": {
        "SURVIVAL": (
            lambda cohort: RAW_DATA_DIR.joinpath(
                f"slides_classification/TCGA/clinical/survival/survival_labels_{cohort.lower()}.csv"
            )
        ),
        "MSI": (
            lambda cohort: RAW_DATA_DIR.joinpath(
                f"slides_classification/TCGA/clinical/msi/msi_labels_{cohort.lower()}.csv"
            )
        ),
        "HRD": (
            lambda cohort: RAW_DATA_DIR.joinpath(
                f"slides_classification/TCGA/clinical/hrd/hrd_labels_{cohort.lower()}.csv"
            )
        ),
        "SUBTYPES": (
            lambda cohort: RAW_DATA_DIR.joinpath(
                "slides_classification/TCGA/clinical/subtypes/"
                f"{cohort.lower()}_pan_can_atlas_2018_clinical_data.tsv.gz"
            )
        ),
    },
}

# Tiles classification tasks.
# NCT-CRC dataset.
NCT_CRC_PATHS = {
    "TRAIN": RAW_DATA_DIR.joinpath(
        "tiles_classification/NCT-CRC_FULL/patches/NCT-CRC-HE-100K-NONORM/"
    ),
    "VALID": RAW_DATA_DIR.joinpath(
        "tiles_classification/NCT-CRC_FULL/patches/CRC-VAL-HE-7K/"
    ),
    "LABELS": RAW_DATA_DIR.joinpath(
        "tiles_classification/NCT-CRC_FULL/clinical/dict_labels.pkl"
    ),
}

# Camelyon17 dataset.
CAMELYON17_WILDS_PATHS = {
    "METADATA": RAW_DATA_DIR.joinpath(
        "tiles_classification/CAMELYON17-WILDS_FULL/clinical/metadata.csv"
    ),
    "PATCHES": RAW_DATA_DIR.joinpath(
        "tiles_classification/CAMELYON17-WILDS_FULL/patches/"
    ),
}

# Model weights
MODEL_WEIGHTS = {
    "iBOTViTBasePANCAN": WEIGHTS_DATA_DIR.joinpath("ibot_vit_base_pancan.pth"),
}

# Mean and standard deviation used for ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tiles sizes of reference (on which the feature extractor was trained and
# should be used to transform images).
TILE_SIZES = {
    "iBOTViTBasePANCAN": 224,
}

# Link datasets to tasks.
TASKS = {
    # NCT-CRC and Camelyon17 cohorts
    "NCT-CRC_FULL": "tiles_classification",
    "NCT-CRC_TRAIN": "tiles_classification",
    "NCT-CRC_VALID": "tiles_classification",
    "CAMELYON17-WILDS_FULL": "tiles_classification",
    "CAMELYON17-WILDS_TRAIN": "tiles_classification",
    "CAMELYON17-WILDS_VALID": "tiles_classification",
    "CAMELYON17-WILDS_TEST": "tiles_classification",
    # Camelyon16 cohorts
    "CAMELYON16_FULL": "slides_classification",
    "CAMELYON16_TRAIN": "slides_classification",
    "CAMELYON16_TEST": "slides_classification",
    # TCGA cohorts
    "TCGA_BRCA": "slides_classification",
    "TCGA_COAD": "slides_classification",
    "TCGA_CRC": "slides_classification",
    "TCGA_NSCLC": "slides_classification",
    "TCGA_RCC": "slides_classification",
    "TCGA_KICH": "slides_classification",
    "TCGA_KIRP": "slides_classification",
    "TCGA_KIRC": "slides_classification",
    "TCGA_LUAD": "slides_classification",
    "TCGA_LUSC": "slides_classification",
    "TCGA_OV": "slides_classification",
    "TCGA_PAAD": "slides_classification",
    "TCGA_READ": "slides_classification",
    "TCGA_STAD": "slides_classification",
}

# Available cohorts.
AVAILABLE_COHORTS = TASKS.keys()

# TCGA cohorts, labels and cohorts dimensions.
TCGA_COHORTS = [
    "TCGA_BRCA",
    "TCGA_COAD",
    "TCGA_CRC",
    "TCGA_KICH",
    "TCGA_KIRC",
    "TCGA_KIRP",
    "TCGA_LUAD",
    "TCGA_LUSC",
    "TCGA_NSCLC",
    "TCGA_OV",
    "TCGA_PAAD",
    "TCGA_RCC",
    "TCGA_READ",
    "TCGA_STAD",
]
TCGA_TASKS = {
    "TCGA_BRCA": [
        None,
        "HRD",
        "OS",
        "MOLECULAR_SUBTYPE",
        "HISTOLOGICAL_SUBTYPE",
    ],
    "TCGA_COAD": [None, "MSI", "OS"],
    "TCGA_READ": [None, "MSI", "OS"],
    "TCGA_CRC": [None, "MSI", "OS"],
    "TCGA_KICH": [None, "CANCER_SUBTYPE"],
    "TCGA_KIRC": [None, "CANCER_SUBTYPE"],
    "TCGA_KIRP": [None, "CANCER_SUBTYPE"],
    "TCGA_RCC": [None, "CANCER_SUBTYPE"],
    "TCGA_LUAD": [None, "OS", "CANCER_SUBTYPE"],
    "TCGA_LUSC": [None, "OS", "CANCER_SUBTYPE"],
    "TCGA_NSCLC": [None, "OS", "CANCER_SUBTYPE"],
    "TCGA_OV": [None, "HRD"],
    "TCGA_PAAD": [None, "OS"],
    "TCGA_STAD": [None, "MSI"],
}
