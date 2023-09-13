# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cox Loss function."""

from typing import Tuple

import torch


def _sort_risks(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts risks according to the following key: `lambda x: (abs(x), -x)`
    For instance:

    - `[1, -1, 2, 2, -2, -2, 3]` becomes
    - `[1, -1, 2, -2, 2 -2, 3]`

    Sorting will be done on target and applied to both input and target.

    Parameters
    ----------
    input: torch.Tensor
        Risk prediction from the model (higher score means higher risk and lower survival).
    target: torch.Tensor
        Labels of the event occurences. Negative values are the censored values.

    Returns
    -------
    sorted_y_pred, sorted_y_true: Tuple[torch.Tensor, torch.Tensor]
        Sorted risks.
    """
    # Convert to floats.
    target = target.float()
    input = input.float()

    # Reduce dimension of necessary.
    target = torch.squeeze(target, dim=-1)
    input = torch.squeeze(input, dim=-1)

    # Put non-censored individuals before censored ones.
    _, indices = torch.sort(torch.sign(target), descending=True)
    target = target[indices]
    input = input[indices]

    # Sort by absolute value.
    _, indices = torch.sort(torch.abs(target))
    target = target[indices]
    input = input[indices]

    return input, target


def cox(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cox Loss implemented in PyTorch.
    ..warnings:: There shouldn’t be any zero value (because we couldn’t determine censure).

    Parameters
    ----------
    input: torch.Tensor
        Risk prediction from the model (higher score means higher risk and lower survival)
    target: torch.Tensor
        Labels of the event occurences. Negative values are the censored values.
    reduction: str = "mean"
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.

    Returns
    -------
    loss: torch.Tensor
        The cox loss (scalar)
    """
    input, target = _sort_risks(input, target)

    # The tensors are reversed because the generator gives the target in
    # ascending order, and this implementation assumes a descending order.
    input = input.flip(0)
    target = target.flip(0)

    hazard_ratio = torch.exp(input)
    e = (torch.sign(target) + 1) / 2.0

    log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
    uncensored_likelihood = input - log_risk
    censored_likelihood = -uncensored_likelihood * e

    if reduction != "none":
        censored_likelihood = (
            torch.mean(censored_likelihood)
            if reduction == "mean"
            else torch.sum(censored_likelihood)
        )
    return censored_likelihood


class CoxLoss(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """Main Cox Loss module.
    .. warning:: There shouldn’t be any zero value (because we couldn’t determine censure).

    Parameters
    ----------
    reduction: str = "mean"
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(CoxLoss, self).__init__(reduction=reduction)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Foward pass.
        Parameters
        ----------
        logits: torch.Tensor
            Risk prediction from the model (higher score means higher risk and lower survival).
        labels: torch.Tensor
            Labels of the event occurences. Negative values are the censored values.
        Returns
        -------
        cox_loss: torch.Tensor
        """
        return cox(logits, labels, reduction=self.reduction)
