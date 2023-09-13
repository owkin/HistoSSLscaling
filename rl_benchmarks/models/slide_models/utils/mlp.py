# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standard MLP module implementation for MIL aggregation models."""

from typing import Optional, List

import torch


class MLP(torch.nn.Sequential):
    """MLP Module.

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Dimension of hidden layer(s).
    dropout: Optional[List[float]] = None
        Dropout rate(s).
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        MLP activation.
    bias: bool = True
        Add bias to MLP hidden layers.

    Raises
    ------
    ValueError
        If ``hidden`` and ``dropout`` do not share the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)
