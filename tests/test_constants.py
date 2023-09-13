# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for constants."""

import unittest
from glob import glob
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Union

import pytest
from loguru import logger
from rl_benchmarks.constants import (
    CAMELYON16_PATHS,
    CAMELYON17_WILDS_PATHS,
    DATA_ROOT_DIR,
    LOGS_ROOT_DIR,
    MODEL_WEIGHTS,
    NCT_CRC_PATHS,
    PREPROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TCGA_PATHS,
    WEIGHTS_DATA_DIR,
)


class TestDictPaths(unittest.TestCase):
    """Recursively iterate over a nested Python dictionary."""

    def _test_dict_paths(self, dict_paths: Dict[str, Any]):
        """Core function for recursive testing. Here is tested for each value
        found in the dictionary `dict_paths`:
        - If it's a path: check if this corresponds either to an existing directory or file
        - If it's a lambda function (lambda cohort: path_to_cohort): check if the lambda function is returning existing files.
        Parameters
        ----------
        dict_paths: Dict[str, Any]
            Dataset-specific dictionary describing path to dataset-
            specific inputs (masks, coordinates, annotations, etc).
            See ``constants.py`` for example.
        """
        for _, value in dict_paths.items():
            if isinstance(value, FunctionType):
                # example with TCGA_PATHS:
                # TCGA_PATHS["LABELS"]["MSI"] := func = lambda cohort: ROOT_DIR_DATA.joinpath(
                #    f"shared/widy/data/tcga_tasks/msi/msi_status_{cohort.lower()}_tcgabiolinks.csv"
                # )
                # so glob(str(func("*"))) returns all files matching the glob pattern
                files_list = glob(str(value("*")))
                self.assertGreater(len(files_list), 0)

            elif isinstance(value, dict):
                self._test_dict_paths(value)
            else:
                with self.subTest(path=value):
                    self.assertTrue(
                        Path(value).is_dir() or Path(value).is_file()
                    )

    def _test_paths(self, path: Union[str, Path]):
        """Function to assess a path to a file or directory actually exists."""
        self.assertTrue(Path(path).is_dir() or Path(path).is_file())

    def _test_constants(self, obj: Union[Dict[str, Any], Union[str, Path]]):
        """Function testing objects that are either direct paths to files or
        directories, and nested dictionaries."""
        if isinstance(obj, (str, Path)):
            self._test_paths(obj)
        elif isinstance(obj, dict):
            self._test_dict_paths(obj)
        else:
            raise TypeError("Object must be a dictionary, or string, or Path.")

    def _test_data_exist(self, test_raw_data: bool = False):
        """Iterate over objects containing paths to files or raw directories
        in ``constants.py`` file. Also test raw patches and slides directories
        if ``test_raw_data=True``."""
        for obj, obj_name in zip(
            [
                CAMELYON16_PATHS,
                CAMELYON17_WILDS_PATHS,
                NCT_CRC_PATHS,
                TCGA_PATHS,
                DATA_ROOT_DIR,
                LOGS_ROOT_DIR,
                WEIGHTS_DATA_DIR,
                RAW_DATA_DIR,
                PREPROCESSED_DATA_DIR,
                MODEL_WEIGHTS,
            ],
            [
                "Camelyon16 dataset",
                "Camelyon17-WILDS dataset",
                "NCT-CRC dataset",
                "TCGA dataset",
                "Whole data root dir",
                "Logs root dir",
                "Weights root dir",
                "Raw data dir",
                "Preprocessed data dir",
                "Model weights dir",
            ],
        ):
            logger.info(f"Processing {obj_name}...")
            raw_data_keys = ["SLIDES", "PATCHES", "TRAIN", "VALID"]
            if not test_raw_data and isinstance(obj, dict):
                if any(k in obj.keys() for k in raw_data_keys):
                    obj = {
                        key: value
                        for (key, value) in obj.items()
                        if key not in raw_data_keys
                    }
            self._test_constants(obj)

    @pytest.mark.test_raw_data_loading
    def test_all_data_exist(self):
        """Iterate over all objects containing paths to files or directories in
        ``constants.py``."""
        self._test_data_exist(test_raw_data=True)

    def test_prep_data_only_exist(self):
        """Iterate over objects containing paths to files or directories in
        ``constants.py`` file which do not need to download the raw data, and
        only need to download the preprocessed data (Drive folder)."""
        self._test_data_exist(test_raw_data=False)
