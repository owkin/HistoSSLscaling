# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for testing."""

from typing import Dict
import pandas as pd


def get_labels_distribution(dataset: pd.DataFrame) -> Dict[int, float]:
    """Get labels distribution. If survival task: get censoring and no. events.
    Parameters
    ----------
    dataset: pd.DataFrame
        Output dataframe from ``rl_benchmarks.datasets.load_dataset``.
    Returns
    -------
    Dict[int, float]
        Labels distribution for classification tasks, censoring distribution
        for survival tasks.
    """
    counts = dataset.label.value_counts().to_dict()
    if len(counts) > 10:
        counts = {0: (dataset.label <= 0).sum(), 1: (dataset.label > 0).sum()}
    return counts
