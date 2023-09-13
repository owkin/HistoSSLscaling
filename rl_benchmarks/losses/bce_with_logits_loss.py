# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Binary Cross-Entropy (BCE) with Logits Loss."""

import torch


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Binary Cross-Entropy (BCE) with Logits Loss.
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    See [1]_ for details.

    References
    ----------
    .. [1] https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
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
        return super().forward(logits, labels)
