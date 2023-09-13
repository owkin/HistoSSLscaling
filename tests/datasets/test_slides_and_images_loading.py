# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for raw data loading."""

import pytest
import pickle as pkl
import unittest

from loguru import logger
from rl_benchmarks.constants import (
    PREPROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TCGA_COHORTS,
    TCGA_TASKS,
)
from rl_benchmarks.datasets import load_dataset
from rl_benchmarks.utils import get_labels_distribution

SLIDES_FEATURES_DIR = (
    PREPROCESSED_DATA_DIR
    / "slides_classification"
    / "features"
    / "iBOTViTBasePANCAN"
)

TILES_FEATURES_DIR = (
    PREPROCESSED_DATA_DIR
    / "tiles_classification"
    / "features"
    / "iBOTViTBasePANCAN"
)

TCGA_STATISTICS = pkl.load(
    open(
        RAW_DATA_DIR / "slides_classification" / "TCGA" / "tcga_statistics.pk",
        "rb",
    )
)


# Test presence of slides or images.
@pytest.mark.test_raw_data_loading
class TestLoadCamelyon16Images(unittest.TestCase):
    """Test Camelyon16 dataset loading with slides paths."""

    def test_num_training_ids(self):
        """Test number of training patients and samples."""
        df = load_dataset(
            features_root_dir=SLIDES_FEATURES_DIR,
            cohort="CAMELYON16_TRAIN",
            label=None,
            tile_size=224,
            load_slide=True,
        )
        self.assertEqual(df.patient_id.nunique(), 269)
        self.assertEqual(df.slide_id.nunique(), 269)
        self.assertEqual(df.features_path.isna().sum(), 0)
        self.assertEqual(df.coords_path.isna().sum(), 0)

    def test_num_testing_ids(self):
        """Test number of test patients and samples."""
        df = load_dataset(
            features_root_dir=SLIDES_FEATURES_DIR,
            cohort="CAMELYON16_TEST",
            label=None,
            tile_size=224,
            load_slide=True,
        )
        self.assertEqual(df.patient_id.nunique(), 130)
        self.assertEqual(df.slide_id.nunique(), 130)
        self.assertEqual(df.features_path.isna().sum(), 0)
        self.assertEqual(df.coords_path.isna().sum(), 0)


@pytest.mark.test_raw_data_loading
class TestLoadTCGAImages(unittest.TestCase):
    """Test TCGA cohorts loading with slides paths."""

    def test_num_ids(self):
        """Test number of patients and samples."""
        for cohort in TCGA_COHORTS:
            labels = TCGA_TASKS[cohort]
            for label in labels:
                logger.info(f"---- Cohort: {cohort}; Label: {label} ----")
                n_patients, n_slides, distribution = TCGA_STATISTICS[cohort][
                    label
                ]
                df = load_dataset(
                    features_root_dir=SLIDES_FEATURES_DIR,
                    cohort=cohort,
                    label=label,
                    tile_size=224,
                    load_slide=True,
                )
                self.assertEqual(get_labels_distribution(df), distribution)
                self.assertEqual(df.patient_id.nunique(), n_patients)
                self.assertEqual(df.slide_id.nunique(), n_slides)
                self.assertEqual(df.slide_path.isna().sum(), 0)
                self.assertEqual(df.features_path.isna().sum(), 0)
                self.assertEqual(df.coords_path.isna().sum(), 0)


@pytest.mark.test_raw_data_loading
class TestLoadNCTCRCImages(unittest.TestCase):
    """Test NCT-CRC images loading. Requires images to be stored in ``raw```
    directory."""

    def test_num_ids(self):
        """Test number of images in train and validation sets."""
        df = load_dataset(
            features_root_dir=TILES_FEATURES_DIR,
            cohort="NCT-CRC_TRAIN",
        )
        self.assertEqual(df.image_id.nunique(), 100_000)
        df = load_dataset(
            features_root_dir=TILES_FEATURES_DIR,
            cohort="NCT-CRC_VALID",
        )
        self.assertEqual(df.image_id.nunique(), 7_180)


@pytest.mark.test_raw_data_loading
class TestLoadCamelyon17WILDSImages(unittest.TestCase):
    """Test Camelyon17-WILDS images loading. Requires images to be stored in ``raw```
    directory."""

    def test_num_ids(self):
        """Test number of images in train, validation and test sets."""
        df = load_dataset(
            features_root_dir=TILES_FEATURES_DIR,
            cohort="CAMELYON17-WILDS_TRAIN",
        )
        self.assertEqual(df.image_id.nunique(), 302_436 + 33_560)
        df = load_dataset(
            features_root_dir=TILES_FEATURES_DIR,
            cohort="CAMELYON17-WILDS_VALID",
        )
        self.assertEqual(df.image_id.nunique(), 34_904)
        df = load_dataset(
            features_root_dir=TILES_FEATURES_DIR,
            cohort="CAMELYON17-WILDS_TEST",
        )
        self.assertEqual(df.image_id.nunique(), 85_054)
