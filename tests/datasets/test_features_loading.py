# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for features loading."""

import pickle as pkl
import unittest

import numpy as np
from loguru import logger
from rl_benchmarks.constants import (
    PREPROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TCGA_COHORTS,
    TCGA_TASKS,
)
from rl_benchmarks.datasets import SlideClassificationDataset, load_dataset
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


# Test features loading.
class TestLoadCamelyon16Features(unittest.TestCase):
    """Test Camelyon16 dataset loading without slides paths."""

    def test_num_training_ids(self):
        """Test number of training patients and samples."""
        df = load_dataset(
            features_root_dir=SLIDES_FEATURES_DIR,
            cohort="CAMELYON16_TRAIN",
            label=None,
            tile_size=224,
            load_slide=False,
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
            load_slide=False,
        )
        self.assertEqual(df.patient_id.nunique(), 130)
        self.assertEqual(df.slide_id.nunique(), 130)
        self.assertEqual(df.features_path.isna().sum(), 0)
        self.assertEqual(df.coords_path.isna().sum(), 0)


class TestLoadTCGAFeatures(unittest.TestCase):
    """Test TCGA cohorts loading without slides paths."""

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
                    load_slide=False,
                )
                self.assertEqual(get_labels_distribution(df), distribution)
                self.assertEqual(df.patient_id.nunique(), n_patients)
                self.assertEqual(df.slide_id.nunique(), n_slides)
                self.assertEqual(df.features_path.isna().sum(), 0)
                self.assertEqual(df.coords_path.isna().sum(), 0)


class TestLoadNCTCRCFeatures(unittest.TestCase):
    """Test NCT-CRC dataset features."""

    def test_num_ids(self):
        """Test number of features."""
        tiles_features_dir = TILES_FEATURES_DIR / "NCT-CRC_FULL"
        tiles_features = np.load(
            tiles_features_dir / "tile_features.npy", allow_pickle=True
        )
        tiles_ids = np.load(
            tiles_features_dir / "tile_ids.npy", allow_pickle=True
        )
        self.assertEqual(tiles_features.shape[0], 107_180)
        self.assertEqual(tiles_ids.shape[0], 107_180)


class TestLoadCamelyon17WILDSFeatures(unittest.TestCase):
    """Test Camelyon17-WILDS dataset features."""

    def test_num_ids(self):
        """Test number of features."""
        tiles_features_dir = TILES_FEATURES_DIR / "CAMELYON17-WILDS_FULL"
        tiles_features = np.load(
            tiles_features_dir / "tile_features.npy", allow_pickle=True
        )
        tiles_ids = np.load(
            tiles_features_dir / "tile_ids.npy", allow_pickle=True
        )
        self.assertEqual(
            tiles_features.shape[0], 302_436 + 33_560 + 34_904 + 85_054
        )
        self.assertEqual(
            tiles_ids.shape[0], 302_436 + 33_560 + 34_904 + 85_054
        )
        self.assertEqual(tiles_features.shape[0], tiles_ids.shape[0])


class TestSlideClassificationDataset(unittest.TestCase):
    """Test SlideClassificationDataset class."""

    def test_num_ids(self):
        """Test number of patients, coordinates, features, and dimensionality
        of feature matrix."""
        # TCGA cohorts.
        for cohort in TCGA_COHORTS:
            logger.info(f"---- Cohort: {cohort}; Label: {None} ----")
            n_patients, n_slides, distribution = TCGA_STATISTICS[cohort][None]
            slide_dataset = SlideClassificationDataset(
                SLIDES_FEATURES_DIR,
                cohort=cohort,
                label=None,
                n_tiles=1_000,
                tile_size=224,
            )
            df = slide_dataset.dataset
            self.assertEqual(df.patient_id.nunique(), n_patients)
            self.assertEqual(df.slide_id.nunique(), n_slides)
            self.assertEqual(get_labels_distribution(df), distribution)
            self.assertEqual(df.features_path.isna().sum(), 0)
            self.assertEqual(df.coords_path.isna().sum(), 0)
            self.assertEqual(len(slide_dataset.features), n_slides)
            self.assertEqual(slide_dataset.features[0].shape, (1_000, 3 + 768))

        # CAMELYON16_FULL.
        slide_dataset = SlideClassificationDataset(
            SLIDES_FEATURES_DIR,
            cohort="CAMELYON16_FULL",
            label=None,
            tile_size=224,
        )
        df = slide_dataset.dataset
        self.assertEqual(df.patient_id.nunique(), 399)
        self.assertEqual(df.slide_id.nunique(), 399)
        self.assertEqual(df.features_path.isna().sum(), 0)
        self.assertEqual(df.coords_path.isna().sum(), 0)
        self.assertEqual(len(slide_dataset.features), 399)
        self.assertEqual(slide_dataset.features[0].shape, (1_000, 3 + 768))
