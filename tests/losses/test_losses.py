# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for losses."""

import unittest

import torch

from rl_benchmarks.losses import BCEWithLogitsLoss, CrossEntropyLoss, CoxLoss


class TestCoxLoss(unittest.TestCase):
    """Test ``CoxLoss`` survival loss function."""

    def test_shape_output(self):
        """Test output shape."""
        criterion = CoxLoss()
        logits = torch.randn((8, 1))
        labels = torch.randint(-100, 100, (8, 1)).float()
        loss = criterion(logits, labels)
        self.assertEqual(len(loss.shape), 0)


class TestCrossEntropyLoss(unittest.TestCase):
    """Test ``CrossEntropyLoss`` loss function in a multi-class classification
    setting."""

    def test_shape_output(self):
        """Test output shape."""
        criterion = CrossEntropyLoss()
        logits = torch.randn((8, 10))
        labels = torch.randint(10, (8,)).float()
        loss = criterion(logits, labels)
        self.assertEqual(len(loss.shape), 0)


class TestBCEWithLogitsLoss(unittest.TestCase):
    """Test ``BCEWithLogitsLoss`` loss function in a binary classification
    setting."""

    def test_shape_output(self):
        """Test output shape."""
        criterion = BCEWithLogitsLoss()
        logits = torch.randn((8, 1))
        labels = torch.randint(2, (8, 1)).float()
        loss = criterion(logits, labels)
        self.assertEqual(len(loss.shape), 0)
