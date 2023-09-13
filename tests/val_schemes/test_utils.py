# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test utility functions for nested cross-validation."""

from typing import Optional
import numpy as np
from torch.utils.data import Dataset, Subset

from rl_benchmarks.val_schemes import (
    generate_permutations,
    split_cv,
)


class SyntheticDataset(Dataset):
    """Create a synthetic dataset.

    Parameters
    ----------
    features: np.array
        Array of features with shape (N_slides, d), making the simplistic
        assumption that there is one feature vector per slide of dimension ``d``.
    stratified: Optional[np.array] = None
        Array of labels to optionally stratify on.
    center_id: Optional[np.array] = None
        Array of centers ids to optionally stratify on.
    patient_id: Optional[np.array] = None
    Array of patients ids to optionally stratify on.
    """

    def __init__(
        self,
        features: np.array,
        stratified: Optional[np.array] = None,
        center_id: Optional[np.array] = None,
        patient_id: Optional[np.array] = None,
    ) -> None:
        super(SyntheticDataset, self).__init__()
        self.features = features
        self.stratified = stratified
        self.center_id = center_id
        self.patient_id = patient_id

    def __len__(self) -> int:
        """Get number of slides."""
        return len(self.features)

    def __getitem__(self, index: int) -> np.array:
        """Get current features (``d``-dimensional vector)."""
        return self.features[index]


def test_split_cv() -> None:
    """Test data splitting in nested cross-validation."""
    # Test basic CV splitting.
    n_slides = 1000
    n_splits = 5
    dataset = SyntheticDataset(np.random.rand(n_slides, 256))
    splits = split_cv(
        dataset=dataset,
        n_splits=n_splits,
        stratified=False,
        split_mode="random_split",
    )
    for i, (train_indices, val_indices) in enumerate(splits):
        train_dataset = Subset(dataset, indices=train_indices)
        val_dataset = Subset(dataset, indices=val_indices)
        assert len(train_dataset) == 800
        assert len(val_dataset) == 200
    assert i == n_splits - 1  # pylint: disable=undefined-loop-variable

    # Test stratified CV splitting.
    n_slides = 1000
    n_splits = 5
    dataset = SyntheticDataset(
        features=np.random.rand(n_slides, 256),
        stratified=np.random.randint(0, 2, n_slides),
    )
    splits = split_cv(
        dataset=dataset,
        n_splits=n_splits,
        stratified=True,
        split_mode="random_split",
    )
    for i, (train_indices, val_indices) in enumerate(splits):
        train_dataset = Subset(dataset, indices=train_indices)
        val_dataset = Subset(dataset, indices=val_indices)

        assert (
            abs(
                train_dataset.dataset.stratified[train_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.05
        )
        assert (
            abs(
                val_dataset.dataset.stratified[val_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.05
        )

    # Test center-stratified CV splitting.
    n_slides = 1000
    n_splits = 5
    dataset = SyntheticDataset(
        features=np.random.rand(n_slides, 256),
        stratified=np.random.randint(0, 2, n_slides),
        center_id=np.random.randint(0, 5, n_slides),
    )
    splits = split_cv(
        dataset=dataset,
        n_splits=n_splits,
        stratified=True,
        split_mode="center_split",
    )
    for i, (train_indices, val_indices) in enumerate(splits):
        train_dataset = Subset(dataset, indices=train_indices)
        val_dataset = Subset(dataset, indices=val_indices)

        assert (
            abs(
                train_dataset.dataset.stratified[train_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.10
        )
        assert (
            abs(
                val_dataset.dataset.stratified[val_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.10
        )

        train_center_ids = set(
            train_dataset.dataset.center_id[train_dataset.indices]
        )
        val_center_ids = set(
            val_dataset.dataset.center_id[val_dataset.indices]
        )
        assert train_center_ids.intersection(val_center_ids) == set()

    # Test patients-stratified CV splitting.
    n_slides = 1000
    n_splits = 5
    dataset = SyntheticDataset(
        features=np.random.rand(n_slides, 256),
        stratified=np.random.randint(0, 2, n_slides),
        patient_id=np.random.randint(0, 950, n_slides),
    )
    splits = split_cv(
        dataset=dataset,
        n_splits=n_splits,
        stratified=True,
        split_mode="patient_split",
    )
    for i, (train_indices, val_indices) in enumerate(splits):
        train_dataset = Subset(dataset, indices=train_indices)
        val_dataset = Subset(dataset, indices=val_indices)

        assert (
            abs(
                train_dataset.dataset.stratified[train_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.10
        )
        assert (
            abs(
                val_dataset.dataset.stratified[val_dataset.indices].mean()
                - dataset.stratified.mean()
            )
            < 0.10
        )

        train_patient_ids = set(
            train_dataset.dataset.patient_id[train_dataset.indices]
        )
        val_patient_ids = set(
            val_dataset.dataset.patient_id[val_dataset.indices]
        )
        assert train_patient_ids.intersection(val_patient_ids) == set()


def test_generate_permutations():
    """Test ``generate_permutations`` function."""
    grid_search_params = {
        "num_epochs": [10],
        "learning_rate": [1.0e-2, 1.0e-3],
        "weight_decay": 0.0,
        "model": {
            "_target_": "rl_benchmarks.models.Chowder",
            "n_extreme": [10, 25],
        },
    }

    configurations = generate_permutations(
        grid_search_params=grid_search_params,
    )
    true_configurations = [
        {
            "num_epochs": 10,
            "learning_rate": 0.01,
            "model": {
                "n_extreme": 10,
                "_target_": "rl_benchmarks.models.Chowder",
            },
            "weight_decay": 0.0,
        },
        {
            "num_epochs": 10,
            "learning_rate": 0.01,
            "model": {
                "n_extreme": 25,
                "_target_": "rl_benchmarks.models.Chowder",
            },
            "weight_decay": 0.0,
        },
        {
            "num_epochs": 10,
            "learning_rate": 0.001,
            "model": {
                "n_extreme": 10,
                "_target_": "rl_benchmarks.models.Chowder",
            },
            "weight_decay": 0.0,
        },
        {
            "num_epochs": 10,
            "learning_rate": 0.001,
            "model": {
                "n_extreme": 25,
                "_target_": "rl_benchmarks.models.Chowder",
            },
            "weight_decay": 0.0,
        },
    ]
    assert configurations == true_configurations
    assert len(configurations) == 1 * 2 * 1 * 2
