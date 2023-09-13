# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility function for storing parameters """

from loguru import logger
from torch.utils.data import Dataset


def log_slide_dataset_info(dataset: Dataset) -> None:
    """Log information about a slide-level dataset."""
    logger.info(f"  Number of slides: {dataset.n_slides}")
    logger.info(f"  Number of patients: {dataset.n_patients}")
    logger.info(f"  Number of features: {dataset.n_features}")
    logger.info(f"  Number of labels: {dataset.n_labels}")
    logger.info("\n")
