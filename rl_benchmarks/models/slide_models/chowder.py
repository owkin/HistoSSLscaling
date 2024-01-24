# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Chowder aggregation algorithm."""

import warnings
from typing import Optional, List

import torch
from torch import nn

from .utils.extreme_layer import ExtremeLayer
from .utils.mlp import MLP
from .utils.tile_layers import TilesMLP


class Chowder(nn.Module):
    """Chowder MIL model (See [1]_).

    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int
        Controls the number of scores and, by extension, the number of out_features.
    n_top: int
        Number of tiles with hightest scores that are selected and fed to the MLP.
    n_bottom: int
        Number of tiles with lowest scores that are selected and fed to the MLP.
    tiles_mlp_hidden: Optional[List[int]] = None
        Number of units for layers in the first MLP applied tile wise to compute
        a score for each tiles from the tile features.
        If `None`, a linear layer is used to compute tile scores.
        If e.g. `[128, 64]`, the tile scores are computed with a MLP of dimension
        features_dim -> 128 -> 64 -> 1.
    mlp_hidden: Optional[List[int]] = None
        Number of units for layers of the second MLP that combine top and bottom
        scores and outputs a final prediction at the slide-level. If `None`, a
        linear layer is used to compute the prediction from the extreme scores.
        If e.g. `[128, 64]`, the prediction is computed
        with a MLP n_top + n_bottom -> 128 -> 64 -> 1.
    mlp_dropout: Optional[List[float]] = None
        Dropout that is used for each layer of the MLP. If `None`, no dropout
        is used.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation that is used after each layer of the MLP.
    bias: bool = True
        Whether to add bias for layers of the tiles MLP.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] Pierre Courtiol, Eric W. Tramel, Marc Sanselme, and Gilles Wainrib. Classification
    and disease localization in histopathology using only global labels: A weakly-supervised
    approach. CoRR, abs/1802.02212, 2018.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 3,
    ) -> None:
        super(Chowder, self).__init__()

        if n_extreme is not None:
            warnings.warn(
                DeprecationWarning(
                    f"Use `n_extreme=None, n_top={n_extreme if n_top is None else n_top}, "
                    f"n_bottom={n_extreme if n_bottom is None else n_bottom}` instead."
                )
            )

            if n_top is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_top={n_top}`"
                        f"with `n_top=n_extreme={n_extreme}`."
                    )
                )
            if n_bottom is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_bottom={n_bottom}`"
                        f"with `n_bottom=n_extreme={n_extreme}`."
                    )
                )

            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError(
                "At least one of `n_top` or `n_bottom` must not be None."
            )

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=out_features,
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = n_top + n_bottom
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)

        self.metadata_cols = metadata_cols

    @staticmethod
    def weight_initialization(module: torch.nn.Module) -> None:
        """Initialize weights for the module using Xavier initialization method,
        "Understanding the difficulty of training deep feedforward neural networks",
        Glorot, X. & Bengio, Y. (2010)."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(
            x=features[..., self.metadata_cols:], mask=mask
        )
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        # Apply MLP to the N_TOP + N_BOTTOM scores.
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        return y.squeeze(2)
