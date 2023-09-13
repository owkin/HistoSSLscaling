# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Masked Linear implementation used for GatedAttention module, used later on
in TransMIL algorithm."""

from typing import List, Optional, Union

import torch


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )


class TilesMLP(torch.nn.Module):
    """MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(
            torch.nn.Linear(in_features, out_features, bias=bias)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x
