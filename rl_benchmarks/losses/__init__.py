# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module covering losses for slide-level downstream tasks."""

from .bce_with_logits_loss import BCEWithLogitsLoss
from .cox_loss import CoxLoss
from .cross_entropy_loss import CrossEntropyLoss

# OS prediction.
SURVIVAL_LOSSES = CoxLoss
# Binary and multi-categorial outcome prediction, respectively.
CLASSIFICATION_LOSSES = (BCEWithLogitsLoss, CrossEntropyLoss)
