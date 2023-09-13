# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading module."""

from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .slides_classification.camelyon16 import load_camelyon16
from .slides_classification.core import SlideFeaturesDataset
from .slides_classification.tcga import load_tcga
from .tiles_classification.camelyon17_wilds import load_camelyon17_wilds
from .tiles_classification.nct_crc import load_nct_crc
from ..constants import AVAILABLE_COHORTS
from ..utils import preload_features


def load_dataset(
    cohort,
    features_root_dir: Optional[str] = None,
    label: Optional[str] = None,
    tile_size: int = 224,
    load_slide: bool = True,
) -> pd.DataFrame:
    """For each dataset (and outcome if applicable), load the following data:

    * Slides datasets *
    "patient_id": patient ID
    "slide_id": slide ID
    "slide_path": path to the slide (raw slide)
    "coords_path": path to the existing tiles coordinates from the slide (numpy arrays)
    "label": outcome to predict

    * Tiles datasets *
    "image_id": image ID
    "image_path": path to the tile
    "center_id": center ID (optional)
    "label": tissue class (0 to 8, NCT-CRC) or presence of tumor (0 or 1, Camelyon17-WILDS)

    Parameters
    ----------
    cohort: str
        Name of the cohort:
        - For TCGA cohorts:
            ``'TCGA_COAD'``,``'TCGA_READ'``, etc.
            See ``slides_classification/tcga.py`` for details.
        - For Camelyon16 dataset:
            ``'CAMELYON16_TRAIN'``, ``'CAMELYON16_TEST'`` or``'CAMELYON16_FULL'``
            See ``slides_classification/camelyon16.py`` for details.
        - For Camelyon17-WILDS dataset:
            ``'CAMELYON17-WILDS_TRAIN'``, ``'CAMELYON17-WILDS_VALID'``, ``'CAMELYON17-WILDS_TEST'``
            or ``'CAMELYON17-WILDS_FULL'``.
            See ``tiles_classification/camelyon17_wilds.py`` for details.
        - For NCT-CRC dataset:
            ``'NCT-CRC_TRAIN'``, ``'NCT-CRC_VALID'`` or ``'NCT-CRC_FULL'``.
            See ``tiles_classification/nct_crc.py`` for details.

    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        * Slides datasets *
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/

        or

        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/TCGA/


        * Tiles datasets *
        /home/user/data/rl_benchmarks_data/preprocessed/
        tiles_classification/features/iBOTViTBasePANCAN/NCT-CRC_FULL/

        or

        /home/user/data/rl_benchmarks_data/preprocessed/
        tiles_classification/features/iBOTViTBasePANCAN/CAMELYON17-WILDS_FULL/

        If no features have been extracted yet, `features_path` is made of NaNs.

    label: Optional[str] = None
        Only needed for TCGA cohorts.
        The task-specific label. Valid labels are: ``'MOLECULAR_SUBTYPE'``,
        ``'HISTOLOGICAL_SUBTYPE'``, ``'TUMOR_TYPE'``, ``'CANCER_SUBTYPE'``,
        ``'SURVIVAL'``, ``'MSI'`` and ``'HRD'``.

    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
        This parameter is automatically picked up during feature extraction
        depending on the feature extractor at stake.
        See ``rl_benchmarks.constants.TILE_SIZES``.

    cohort : str = "TCGA_COAD"
        Name of the TCGA cohort to consider. Valid TCGA cohorts are: ``'COAD'`` (colon
        adenocarcinoma), ``'READ'`` (rectum adenocarcinoma), ``'LUAD'`` (lung
        adenocarcinoma), ``'LUSC'`` (lung squamous cell carcinoma), ``'BRCA'``
        (breast invasive carcinoma), ``'KIRC'`` (kidney renal clear cell carcinoma),
        ``'KIRP'`` (kidney renal papillary cell carcinoma), ``'KICH'`` (kidney
        chromophobe), ``'OV'`` (ovarian serous cystadenocarcinoma), ``'MESO'``
        (mesothelioma), ``'PAAD'`` (pancreatic adenocarcinoma), ``'PRAD'``
        (prostate adenocarcinoma).

    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).
    """
    if cohort not in AVAILABLE_COHORTS:
        raise ValueError(
            f"Please specify a cohort in {AVAILABLE_COHORTS}. Cohort: {cohort} is not supported."
        )

    kwargs = {
        "features_root_dir": features_root_dir,
        "tile_size": tile_size,
        "load_slide": load_slide,
    }
    # Slide-level datasets.
    # TCGA cohorts.
    if "TCGA" in cohort:
        if cohort == "TCGA_NSCLC":
            df_luad = load_tcga(cohort="TCGA_LUAD", label=label, **kwargs)
            df_lusc = load_tcga(cohort="TCGA_LUSC", label=label, **kwargs)
            dataset = pd.concat([df_luad, df_lusc], axis=0)
        elif cohort == "TCGA_CRC":
            df_coad = load_tcga(cohort="TCGA_COAD", label=label, **kwargs)
            df_read = load_tcga(cohort="TCGA_READ", label=label, **kwargs)
            dataset = pd.concat([df_coad, df_read], axis=0)
        elif cohort == "TCGA_RCC":
            df_kirc = load_tcga(cohort="TCGA_KIRC", label=label, **kwargs)
            df_kirp = load_tcga(cohort="TCGA_KIRP", label=label, **kwargs)
            df_kirch = load_tcga(cohort="TCGA_KICH", label=label, **kwargs)
            dataset = pd.concat([df_kirc, df_kirp, df_kirch], axis=0)
        else:
            dataset = load_tcga(cohort=cohort, label=label, **kwargs)
        drop_na_columns = ["coords_path", "label"]
        if load_slide:
            drop_na_columns += ["slide_path"]

    # CAMELYON16 cohorts.
    elif "CAMELYON16" in cohort:
        if cohort == "CAMELYON16_FULL":
            df_train = load_camelyon16(cohort="TRAIN", **kwargs)
            df_test = load_camelyon16(cohort="TEST", **kwargs)
            dataset = pd.concat([df_train, df_test], axis=0)
        else:
            cohort = cohort.split("_")[-1]
            dataset = load_camelyon16(cohort=cohort, **kwargs)
        drop_na_columns = ["coords_path", "label"]
        if load_slide:
            drop_na_columns += ["slide_path"]

    # Tile-level datasets.
    # NCT-CRC.
    elif "NCT-CRC" in cohort:
        features_root_dir = Path(
            str(features_root_dir).replace(cohort, "NCT-CRC_FULL")
        )
        kwargs.update({"features_root_dir": features_root_dir})
        if cohort == "NCT-CRC_FULL":
            df_train = load_nct_crc(cohort="TRAIN")
            df_val = load_nct_crc(cohort="VALID")
            dataset = pd.concat([df_train, df_val], axis=0)
        else:
            cohort = cohort.split("_")[1]
            dataset = load_nct_crc(cohort)
        drop_na_columns = ["image_path"]

    # CAMELYON17-WILDS.
    elif "CAMELYON17-WILDS" in cohort:
        features_root_dir = Path(
            str(features_root_dir).replace(cohort, "CAMELYON17-WILDS")
        )
        kwargs.update({"features_root_dir": features_root_dir})
        cohort = cohort.split("_")[1]
        dataset = load_camelyon17_wilds(cohort)
        drop_na_columns = ["image_path"]

    dataset = dataset.dropna(subset=drop_na_columns, inplace=False)
    return dataset


class SlideClassificationDataset(SlideFeaturesDataset):
    """Data loader for slide-classification downstream experiments based on
    ``SlideFeaturesDataset`` module. See ``load_dataset`` above function for
    a detailed documentation. Contrarily to slide-level tasks, data loading for
    classification (i.e. linear evaluation) is handled directly in
    ``rl_benchmarks/tools/tile_level_tasks/linear_evaluation.py``.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/
    cohort: str = None
        Name of the cohort, e.g ``'TCGA_READ'`` or ``'CAMELYON16_FULL'``.
    label: str = None
        Only needed for TCGA cohorts.
    n_tiles: int = 1000
        Number of tiles per slide.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
        This parameter is automatically picked up during feature extraction
        depending on the feature extractor at stake.
        See ``rl_benchmarks.constants.TILE_SIZES``.
    """

    def __init__(
        self,
        features_root_dir: Union[str, Path],
        cohort: str = None,
        label: str = None,
        n_tiles: int = 1_000,
        tile_size: int = 224,
    ):
        dataset = load_dataset(
            cohort=cohort,
            features_root_dir=features_root_dir,
            label=label,
            tile_size=tile_size,
            load_slide=False,
        )
        dataset = dataset[~dataset["features_path"].isna()].reset_index(
            drop=True
        )
        if dataset.shape[0] == 0:
            log_cohort = f"TCGA/{cohort}" if "TCGA" in cohort else cohort
            raise AttributeError(
                f"No features exist at {str(features_root_dir)} for {log_cohort}."
            )
        # Preload features
        features = dataset["features_path"].values
        features, indices = preload_features(
            fpaths=features,
            n_tiles=n_tiles,
        )
        self.dataset = dataset.iloc[indices]
        labels = self.dataset.iloc[indices].label.values
        super().__init__(features, labels=labels, n_tiles=n_tiles)

        self.patient_id = self.dataset.patient_id.values
        self.center_id = self.dataset.center_id.values
        self.n_slides = self.dataset.slide_id.nunique()
        self.n_patients = self.dataset.patient_id.nunique()
        self.n_features = features[0].shape[-1] - 3
        self.n_labels, self.stratified = self._set_stratification(
            cohort=cohort, label=label
        )

    def _set_stratification(
        self, cohort: str, label: str
    ) -> Tuple[int, np.array]:
        """Get number of classes and frequencies for stratification.
        For overall survival prediction, number of events serve as the
        stratification reference."""
        if "CAMELYON16" in cohort:
            n_labels = 1  # tumor presence (1) vs absence (0)
            stratified = self.dataset.label.values
        elif "TCGA" in cohort:
            if label in ["OS", "MSI", "HRD"]:
                n_labels = 1
            elif label == "CANCER_SUBTYPE":
                if cohort in ["TCGA_CRC", "TCGA_NSCLC"]:
                    n_labels = 1
                elif cohort == "TCGA_RCC":
                    n_labels = 3
                else:
                    n_labels = self.dataset.label.nunique()
            elif label == "HISTOLOGICAL_SUBTYPE":
                if cohort == "TCGA_BRCA":
                    n_labels = 1
                else:
                    n_labels = self.dataset.label.nunique()
            elif label == "MOLECULAR_SUBTYPE":
                if cohort == "TCGA_BRCA":
                    n_labels = 5
                else:
                    n_labels = self.dataset.label.nunique()
            else:
                n_labels = self.dataset.label.nunique()

            # Store statistics for further stratification in cross-validation.
            if label == "OS":
                stratified = 1 * (self.dataset.label.values > 0)
            elif label in [
                "MSI",
                "HRD",
                "CANCER_SUBTYPE",
                "HISTOLOGICAL_SUBTYPE",
                "MOLECULAR_SUBTYPE",
            ]:
                stratified = self.dataset.label.values
            else:
                stratified = None
        return n_labels, stratified
