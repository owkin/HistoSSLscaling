"""Data loading for slide-level experiments:
- features extraction: ``camelyon16.py`` and ``tcga.py`` files implement data
loading for Camelyon16 dataset and TCGA cohorts, respectively.
- classification: ``core.py`` file implements the ``SlideFeaturesDataset`` module.
This module is at the core of data loading for downstream experiments.
It allows to sample over features and labels."""
