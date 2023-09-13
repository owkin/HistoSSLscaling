# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-Entropy (CE) loss."""

import torch
from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-Entropy (BC) Loss.
    This criterion computes the cross entropy loss between input logits and target.
    It is useful when training a classification problem with C classes.
    See [1]_ for details.

    References
    ----------
    .. [1] https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    def forward(  # pylint: disable=arguments-renamed
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        Parameters
        ----------
        logits: torch.Tensor
            Non-normalized predictions from the model, shape (BS, N_CLASSES).
        labels: torch.Tensor
            Labels of the outcome, shape (BS, 1).
        """
        return super().forward(logits.squeeze(), labels.long().squeeze())
