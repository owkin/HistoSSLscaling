# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading for slide-level experiments:
- features extraction: ``camelyon16.py`` and ``tcga.py`` files implement data
loading for Camelyon16 dataset and TCGA cohorts, respectively.
- classification: ``core.py`` file implements the ``SlideFeaturesDataset`` module.
This module is at the core of data loading for downstream experiments.
It allows to sample over features and labels."""
