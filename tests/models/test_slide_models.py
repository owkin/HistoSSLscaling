# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MIL aggregation  models."""

import unittest

import torch

from rl_benchmarks.models import (
    ABMIL,
    Chowder,
    DSMIL,
    HIPTMIL,
    MeanPool,
    TransMIL,
)


class TestABMIL(unittest.TestCase):
    """Test ``ABMIL`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((8, 10, 259))
        mask = torch.zeros((8, 10, 1)).bool()
        model = ABMIL(
            in_features=256,
            out_features=1,
            d_model_attention=128,
            temperature=1.0,
            tiles_mlp_hidden=None,
            mlp_hidden=None,
            mlp_dropout=None,
            mlp_activation=None,
            bias=True,
            metadata_cols=3,
        )
        output = model(features, mask)
        self.assertEqual(output.shape, (8, 1))


class TestChowder(unittest.TestCase):
    """Test ``Chowder`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((8, 20, 259))
        mask = torch.zeros((8, 20, 1)).bool()
        model = Chowder(
            in_features=256,
            out_features=1,
            n_top=10,
            n_bottom=10,
            tiles_mlp_hidden=None,
            mlp_hidden=None,
            mlp_dropout=None,
            mlp_activation=None,
            bias=True,
            metadata_cols=3,
        )
        output = model(features, mask)
        self.assertEqual(output.shape, (8, 1))


class TestDSMIL(unittest.TestCase):
    """Test ``DSMIL`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((8, 10, 259))
        mask = torch.zeros((8, 10, 1)).bool()
        model = DSMIL(in_features=256, out_features=1, metadata_cols=3)
        output = model(features, mask)
        self.assertEqual(output.shape, (8, 1))


class TestHIPTMIL(unittest.TestCase):
    """Test ``HIPTMIL`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((1, 10, 259))
        mask = torch.zeros((1, 10, 1)).bool()
        model = HIPTMIL(in_features=256, out_features=1, metadata_cols=3)
        output = model(features, mask)
        self.assertEqual(output.shape, (1, 1))


class TestMeanPool(unittest.TestCase):
    """Test ``MeanPool`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((8, 10, 259))
        mask = torch.zeros((8, 10, 1)).bool()
        model = MeanPool(
            in_features=256,
            out_features=1,
            metadata_cols=3,
        )
        output = model(features, mask)
        self.assertEqual(output.shape, (8, 1))


class TestTransMIL(unittest.TestCase):
    """Test ``TransMIL`` MIL algorithm."""

    def test_features_shape(self):
        """Test model's output shape and correct execution."""
        features = torch.randn((8, 10, 259))
        mask = torch.zeros((8, 10, 1)).bool()
        model = TransMIL(
            in_features=256,
            out_features=1,
            metadata_cols=3,
        )
        output = model(features, mask)
        self.assertEqual(output.shape, (8, 1))
