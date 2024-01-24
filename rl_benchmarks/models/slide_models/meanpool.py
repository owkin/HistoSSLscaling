# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MeanPool aggregation algorithm."""

from typing import List, Optional
import torch


class MLP(torch.nn.Sequential):
    """MLP model.

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features (int): out_features
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Number of units for layers of the MLP. If `None`, a linear layer is
        used to compute the prediction from the averaged tile features.
        If e.g. `[128, 64]`, the prediction is computed with a
        MLP features_dim -> 128 -> 64 -> 1.
    dropout: Optional[List[float]] = None,
        Dropout that is used for each layer of the MLP. If `None`, no dropout
        is used.
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation that is used after each layer of the MLP.
    bias: bool = True
        Whether to add bias for layers of the MLP.
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
                    "hidden must have a value and have the same length as"
                    "dropout if dropout is given."
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


class MeanPool(torch.nn.Module):
    """MeanPool model.
    A simple model that takes the average of tile features of a slide as input
    (averaged over the tiles dimension).

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features (int): out_features
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Number of units for layers of the MLP. If `None`, a linear layer is
        used to compute the prediction from the averaged tile features.
        If e.g. `[128, 64]`, the prediction is computed with a
        MLP features_dim -> 128 -> 64 -> 1.
    dropout: Optional[List[float]] = None,
        Dropout that is used for each layer of the MLP. If `None`, no dropout
        is used.
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation that is used after each layer of the MLP.
    bias: bool = True
        Whether to add bias for layers of the MLP.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 3,
    ):
        super(MeanPool, self).__init__()
        self.mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )

        self.metadata_cols = metadata_cols

    def _mean(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        if mask is not None:
            # Only mean over non padded features.
            mean_x = torch.sum(x.masked_fill(mask, 0.0), dim=1) / torch.sum(
                (~mask).float(),  # pylint: disable=invalid-unary-operand-type
                dim=1,
            )
        else:
            mean_x = torch.mean(x, dim=1)
        return mean_x

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """
        features = x[..., self.metadata_cols:]
        mean_feats = self._mean(features, mask)
        return self.mlp(mean_feats)
