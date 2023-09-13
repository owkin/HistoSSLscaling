# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for feature extractors."""

import unittest

import torch

from rl_benchmarks.models import (
    ParallelExtractor,
    iBOTViT,
)

BATCH_SIZE = 2
IMAGES = torch.rand(BATCH_SIZE, 3, 224, 224)


class TestiBOTViTBase(unittest.TestCase):
    """Test ``iBOTViT`` class with a ``'vit_base_pancan'`` architecture,
    which corresponds to ``'iBOT[ViT-B]PanCancer'`` in our publication."""

    def test_features_shape(self):
        """Test output features shape."""
        extractor = iBOTViT(
            architecture="vit_base_pancan",
            encoder="teacher",
        )
        features = extractor(IMAGES)
        self.assertEqual(features.shape, (BATCH_SIZE, 768))


class TestParallelExtractor(unittest.TestCase):
    """Test ``ParallelExtractor``."""

    def test_features_shape_and_device(self):
        """Test output features shape and device."""
        _extractor = iBOTViT(
            architecture="vit_base_pancan",
            encoder="teacher",
        )
        for gpu in [-1, None, 0, [0, 1]]:
            extractor = ParallelExtractor(_extractor, gpu=gpu)
            features = extractor(IMAGES)
            if gpu == -1:
                self.assertEqual(str(features.device), "cpu")
            else:
                self.assertEqual(str(features.device), "cuda:0")
            self.assertEqual(features.shape, (BATCH_SIZE, 768))
