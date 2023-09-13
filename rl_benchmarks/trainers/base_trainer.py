# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base trainer class."""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

from torch.utils.data import Subset


class BaseTrainer:
    """Base trainer class with ``train``, ``evaluate``, ``save`` and ``load``
    methods. ``train`` and ``evaluate`` methods should be overriden."""

    def __init__(self):
        pass

    def train(
        self,
        train_set: Subset,
        val_set: Subset,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Training function."""
        raise NotImplementedError

    def evaluate(
        self,
        test_set: Subset,
    ) -> Dict[str, float]:
        """Inference function."""
        raise NotImplementedError

    def save(self, filepath: Union[Path, str]):
        """Model serialization."""
        filepath = Path(filepath).with_suffix(".pkl")
        with filepath.open("wb") as p:
            pickle.dump(self, p)

    @classmethod
    def load(cls, filepath: Union[Path, str]):
        """Model loading."""
        del cls
        filepath = Path(filepath).with_suffix(".pkl")
        with filepath.open("rb") as p:
            obj = pickle.load(p)
        return obj
