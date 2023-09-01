"""Data loading for tile-level experiments:
- features extraction: ``camelyon17_wilds.py`` and ``nct_crc.py`` files implement data
loading for Camelyon16 dataset and TCGA cohorts, respectively.
Contrarily to slide-level tasks, data loading for classification (i.e. linear evaluation)
is handled directly in ``rl_benchmarks/tools/tile_level_tasks/linear_evaluation.py``. There
is no equivalent for ``SlideFeaturesDataset``."""
