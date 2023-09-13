# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module covering all utility functions related to:
``cross_validation.py``: cross-validation (slide-level experiments).
``features.py``: features extraction.
``functional.py``: Sigmoid and Softmax activation functions.
``linear_evaluation.py``: linear evaluation (tile-level experiments).
``loading.py``: data loading, saving and processing.
``logging.py``: logging (slide-level experiments).
``serialization.py``: configurations saving for features extraction.
``testing.py``: utility function for testing in ``tests`` folder.
"""

from .cross_validation import (
    params2experiment_name,
    resume_training,
    set_seed,
)
from .features import (
    extract_from_tiles,
    extract_from_slide,
    pad_collate_fn,
    preload_features,
)
from .functional import sigmoid, softmax
from .linear_evaluation import (
    remove_labels,
    bootstrap,
    dict_to_dataframe,
    get_binary_class_metrics,
    get_bootstrapped_metrics,
    build_datasets,
)
from .loading import (
    load_pickle,
    load_yaml,
    merge_multiple_dataframes,
    save_pickle,
)
from .logging import (
    log_slide_dataset_info,
)
from .serialization import get_slide_config, get_tile_config, store_params
from .testing import get_labels_distribution
from ..trainers.utils import (
    slide_level_train_step,
    slide_level_val_step,
)
