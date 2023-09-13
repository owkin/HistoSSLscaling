# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-validation utility functions."""

import itertools
from typing import Dict, List, Union

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset, Subset

from ..datasets import SlideClassificationDataset


def split_cv(
    dataset: Union[Dataset, Subset, SlideClassificationDataset],
    n_splits: int = 5,
    stratified: bool = True,
    split_mode: str = "patient_split",
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedGroupKFold.split:
    """Utility function to split a ``torch.utils.data.Dataset`` into
    ``n_splits`` given a ``split_mode``.

    Parameters
    ----------
    dataset: Union[Dataset, Subset, SlideClassificationDataset]
        Dataset to split into. This dataset can either be a ``torch.utils.data.Subset``
        class or ``SlideClassificationDataset``. The latter is a "raw" dataset
        built in ``rl_benchmarks.datasets.__init__`` module. The former ``Subset``
        is built such that, for instance:
        >>> train_val_dataset = Subset(dataset, indices=train_val_indices)
        >>> test_dataset = Subset(dataset, indices=test_indices)
        where ``dataset`` is a ``SlideClassificationDataset`` class.
        As such, ``train_val_dataset`` and ``test_dataset`` attributes can now
        be accessed through ``train_val_dataset.dataset.attr`` where ``attr``
        is an attribute, e.g. ``stratified``.
        ``Dataset`` class is used for testing.

    n_splits: int = 5
        Number of folds.

    stratified: bool = True
        Whether to stratify the splits.

    split_mode: str = "patient_split"
        Which mode of stratification. Other modes are ``'random_split'`` and
        ``'center_split'``. Default is ``'patient_split'`` to avoid any data
        leaking between folds.

    shuffle: bool = True
        Whether to shuffle the dataset, default to True.

    random_state: int = 42
        Random state for stratified splitting.

    Returns
    -------

    Raises
    ------
    TypeError
        If ``dataset`` has wront type.
    ValueError
        If ``split_mode`` split mode is not supported.
    """
    indices = np.arange(len(dataset))
    # If stratification is asked.
    if stratified:
        if isinstance(dataset, Subset):
            y = dataset.dataset.stratified[dataset.indices]
            patient_ids = dataset.dataset.patient_id[dataset.indices]
            center_ids = dataset.dataset.center_id[dataset.indices]
        elif isinstance(dataset, (Dataset, SlideClassificationDataset)):
            y = dataset.stratified
            patient_ids = dataset.patient_id
            center_ids = dataset.center_id
        else:
            raise TypeError(
                f"``dataset`` has wrong type: {type(dataset)}. "
                "Accepted types: ``SlideClassificationDataset``, "
                "``torch.utils.data.Subset`` and ``torch.utils.data.Dataset``."
            )
    else:
        y = [0] * len(dataset)

    if split_mode == "random_split":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        splits = splitter.split(indices, y=y)

    elif split_mode == "patient_split":
        groups = None
        if stratified and (len(set(patient_ids)) >= n_splits):
            groups = patient_ids
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        splits = splitter.split(indices, y=y, groups=groups)

    elif split_mode == "center_split":
        groups = None
        if stratified and (len(set(center_ids)) >= n_splits):
            groups = center_ids
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        splits = splitter.split(indices, y=y, groups=groups)

    else:
        raise ValueError(
            f"{split_mode} split mode not supported. Please choose "
            " between 'random_split', 'patient_split' or 'center_split'."
        )
    return splits


def generate_permutations(
    grid_search_params: Dict[str, List[float]],
) -> List[Dict[str, float]]:
    """From a dictionary of type
    >>> {'learning_rate': [0.001, 0.0001], 'weight_decay': [0.0, 0.0001]}

    returns all possible configurations:
    >>> [
    >>>     {'learning_rate': 0.001, 'weight_decay': 0.0},
    >>>     {'learning_rate': 0.001, 'weight_decay': 0.0001},
    >>>     {'learning_rate': 0.0001, 'weight_decay': 0.0},
    >>>     {'learning_rate': 0.0001, 'weight_decay': 0.0001}
    >>> ]

    Parameters
    ----------
    grid_search_params: Dict[str, List[float]]
        Dictionary describing different sets of parameters to train on.

    Returns
    -------
    List[Dict[str, float]]
        List with all combinations of parameters possible.
    """
    keys, values = [], []
    sub_keys, sub_values = [], []
    ignored_keys, ignored_values = [], []
    for k, v in grid_search_params.items():
        if isinstance(v, dict):
            sub_keys.append(k)
            sub_values.append(generate_permutations(v))
        elif isinstance(v, list):
            keys.append(k)
            values.append(v)
        else:
            ignored_keys.append(k)
            ignored_values.append(v)

    permutations_dicts = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]
    sub_permutations_dicts = [
        dict(zip(sub_keys, v)) for v in itertools.product(*sub_values)
    ]
    ignored_dict = dict(zip(ignored_keys, ignored_values))

    result = []
    for d1 in permutations_dicts:
        for d2 in sub_permutations_dicts:
            z = d1.copy()
            z.update(d2)
            result.append(z)

    for d in result:
        d.update(ignored_dict)
    return result


def update_trainer_cfg(
    trainer_cfg: DictConfig, n_features: int, n_labels: int
) -> None:
    """Update trainer configuration with the correct number of features and
    labels. This is useful to correctly instantiate the Multiple Instance
    Learning algorithms, all taking ``in_featurets`` and ``out_features`` as
    input parameters."""
    dimensions = {
        "in_features": n_features,
        "out_features": n_labels,
    }
    trainer_cfg["model"].update(dimensions)
